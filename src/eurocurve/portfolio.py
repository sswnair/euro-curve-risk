"""The analysis for one valuation date: every bond priced off the curve,
its yield and risk, the portfolio totals, and the six rate shocks.

Market values and DV01s are in euros. A position's DV01 is its bond's DV01
per 100 scaled by nominal / 100.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bonds import accrued_interest, convexity, macaulay_duration, modified_duration, price_from_curve, yield_to_maturity
from .curve import Curve
from .data import Inputs
from .models import Position
from .risk import curve_dv01, key_rate_dv01s
from .scenarios import SCENARIOS, irrbb_shock, key_rate_estimate, scenario_pnl

BP = 1e-4


@dataclass
class BondRisk:
    position: Position
    dirty: float
    accrued: float
    ytm: float
    macaulay: float
    modified: float
    convexity: float
    dv01: float  # euros per basis point
    kr_dv01: np.ndarray  # euros per basis point, by key tenor

    @property
    def clean(self) -> float:
        return self.dirty - self.accrued

    @property
    def market_value(self) -> float:
        return self.position.nominal / 100.0 * self.dirty


@dataclass
class ScenarioResult:
    name: str
    pnl_by_bond: np.ndarray
    estimate: float

    @property
    def pnl(self) -> float:
        return float(self.pnl_by_bond.sum())


@dataclass
class Analysis:
    inputs: Inputs
    date: dt.date
    curve: Curve
    bonds: list[BondRisk]
    scenarios: list[ScenarioResult]

    @property
    def tenors(self) -> tuple[float, ...]:
        return self.inputs.settings.key_tenors

    @property
    def market_value(self) -> float:
        return sum(b.market_value for b in self.bonds)

    @property
    def nominal(self) -> float:
        return sum(b.position.nominal for b in self.bonds)

    @property
    def dv01(self) -> float:
        return sum(b.dv01 for b in self.bonds)

    @property
    def kr_dv01(self) -> np.ndarray:
        return np.sum([b.kr_dv01 for b in self.bonds], axis=0)

    @property
    def curve_duration(self) -> float:
        """Years. The percentage change in value for a 1 point parallel move in the zero curve."""
        return self.dv01 / (self.market_value * BP)

    def weighted(self, attribute: str) -> float:
        """A bond measure averaged over the portfolio, weighted by market value."""
        return sum(getattr(b, attribute) * b.market_value for b in self.bonds) / self.market_value

    @property
    def worst(self) -> ScenarioResult:
        return min(self.scenarios, key=lambda s: s.pnl)

    def shock(self, name: str):
        sizes = self.inputs.settings.shock_sizes
        return lambda t: irrbb_shock(t, name, sizes)

    def bond_table(self) -> pd.DataFrame:
        rows = []
        for b in self.bonds:
            bond = b.position.bond
            rows.append(
                {
                    "bond_id": bond.bond_id,
                    "name": bond.name,
                    "coupon": bond.coupon,
                    "maturity": bond.maturity,
                    "nominal": b.position.nominal,
                    "clean_price": b.clean,
                    "accrued": b.accrued,
                    "dirty_price": b.dirty,
                    "market_value": b.market_value,
                    "ytm": b.ytm,
                    "macaulay_duration": b.macaulay,
                    "modified_duration": b.modified,
                    "convexity": b.convexity,
                    "dv01": b.dv01,
                }
            )
        return pd.DataFrame(rows)

    def key_rate_table(self) -> pd.DataFrame:
        frame = pd.DataFrame([b.kr_dv01 for b in self.bonds], columns=[_tenor_label(t) for t in self.tenors])
        frame.insert(0, "bond_id", [b.position.bond.bond_id for b in self.bonds])
        return frame

    def scenario_table(self) -> pd.DataFrame:
        rows = []
        for s in self.scenarios:
            row = {"scenario": s.name, "pnl": s.pnl, "pnl_pct": s.pnl / self.market_value, "key_rate_estimate": s.estimate}
            for b, value in zip(self.bonds, s.pnl_by_bond):
                row[b.position.bond.bond_id] = value
            rows.append(row)
        return pd.DataFrame(rows)


def _tenor_label(t: float) -> str:
    if t < 1:
        return f"{round(t * 12)}M"
    return f"{t:g}Y"


def analyze_bond(position: Position, curve: Curve, tenors) -> BondRisk:
    bond, settle = position.bond, curve.date
    scale = position.nominal / 100.0
    dirty = price_from_curve(bond, curve)
    ytm = yield_to_maturity(bond, settle, dirty)
    return BondRisk(
        position=position,
        dirty=dirty,
        accrued=accrued_interest(bond, settle),
        ytm=ytm,
        macaulay=macaulay_duration(bond, settle, ytm),
        modified=modified_duration(bond, settle, ytm),
        convexity=convexity(bond, settle, ytm),
        dv01=scale * curve_dv01(bond, curve),
        kr_dv01=scale * key_rate_dv01s(bond, curve, tenors),
    )


def analyze(inputs: Inputs, date: dt.date | None = None) -> Analysis:
    """Value and measure the portfolio on `date`, or on the last date in the history."""
    history = inputs.history
    day = history.dates[-1] if date is None else date
    if day not in history.frame.index:
        raise ValueError(f"no curve for {day}; the nearest earlier one is {history.on_or_before(day)}")
    matured = [p.bond.bond_id for p in inputs.positions if p.bond.maturity <= day]
    if matured:
        raise ValueError(f"matured by {day}: {', '.join(matured)}")
    curve = history.curve_on(day)
    bonds = [analyze_bond(p, curve, inputs.settings.key_tenors) for p in inputs.positions]
    analysis = Analysis(inputs=inputs, date=day, curve=curve, bonds=bonds, scenarios=[])
    for name in SCENARIOS:
        shift = analysis.shock(name)
        analysis.scenarios.append(
            ScenarioResult(
                name=name,
                pnl_by_bond=scenario_pnl(inputs.positions, curve, shift),
                estimate=key_rate_estimate(analysis.kr_dv01, analysis.tenors, shift),
            )
        )
    return analysis
