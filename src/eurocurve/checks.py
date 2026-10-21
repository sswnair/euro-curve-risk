"""Checks on the inputs and on the results, each with a status:

    ok        nothing to look at
    warning   worth a look before the numbers go anywhere
    fail      the numbers can't be trusted until it is fixed
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from .bonds import price_from_yield
from .data import CurveHistory
from .dates import target_days
from .ecb import maturity_of
from .models import SvenssonParams
from .portfolio import Analysis
from .svensson import fit, zero_rate

# The grid for refitting the curve: tau1 from 0.1 to 5 years, tau2 from 0.5 to 30.
TAU1_GRID = tuple(round(0.1 * i, 1) for i in range(1, 51))
TAU2_GRID = tuple(round(0.5 * i, 1) for i in range(1, 61))

REBUILD_TOLERANCE_BP = 0.01
FIT_WARNING_BP = 1.0
BIG_MOVE_BP = 15.0


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _fmt(day: dt.date) -> str:
    return f"{day:%b} {day.day}, {day.year}"


def rebuild_check(history: CurveHistory, day: dt.date) -> Check:
    """The curve rebuilt from the day's parameters against the zero rates published with them."""
    name = "Curve matches the published zero rates"
    published = history.spot_rates_on(day)
    if not published:
        return Check(name, "warning", "The curve file has no published zero rates to compare with.")
    t = np.array(list(published))
    gap = np.abs(zero_rate(t, history.params_on(day)) - np.array(list(published.values()))) * 100
    worst = int(np.argmax(gap))
    span = f"{len(t)} maturities from {_maturity(t.min())} to {_maturity(t.max())}"
    if gap.max() < 0.00005:
        detail = f"Matches to within 0.0001bp at all {span}."
    else:
        detail = f"Largest difference {gap[worst]:.4f}bp, at {_maturity(t[worst])}, across {span}."
    return Check(name, "ok" if gap.max() <= REBUILD_TOLERANCE_BP else "fail", detail)


def fit_check(history: CurveHistory, day: dt.date) -> tuple[Check, SvenssonParams | None, float]:
    """A fresh Svensson fit to the published zero rates, as a test of the fitting code."""
    name = "Refit to the published zero rates"
    published = history.spot_rates_on(day)
    if len(published) < 4:
        return Check(name, "warning", "Needs at least four published zero rates."), None, float("nan")
    params, rmse = fit(list(published), list(published.values()), TAU1_GRID, TAU2_GRID)
    status = "ok" if rmse <= FIT_WARNING_BP else "warning"
    detail = (
        f"Root mean squared error {rmse:.3f}bp over {len(published)} maturities, "
        f"with tau1 {params.tau1:g} and tau2 {params.tau2:g} years against the published "
        f"{history.params_on(day).tau1:.3f} and {history.params_on(day).tau2:.3f}."
    )
    return Check(name, status, detail), params, rmse


def gaps_check(history: CurveHistory, start: dt.date, end: dt.date) -> Check:
    name = "A curve for every TARGET business day"
    have = set(history.dates)
    missing = [d for d in target_days(start, end) if d not in have]
    if not missing:
        return Check(name, "ok", f"No gaps from {_fmt(start)} to {_fmt(end)}.")
    listed = ", ".join(_fmt(d) for d in missing[:5]) + (" and more" if len(missing) > 5 else "")
    days = "business day" if len(missing) == 1 else "business days"
    return Check(name, "warning", f"{len(missing)} {days} without a curve: {listed}.")


def big_moves_check(history: CurveHistory, start: dt.date, end: dt.date) -> Check:
    """Days when a published zero rate moved more than BIG_MOVE_BP from the previous curve."""
    name = f"One-day moves over {BIG_MOVE_BP:g}bp"
    codes = history.spot_codes
    if not codes:
        return Check(name, "warning", "The curve file has no published zero rates to check.")
    frame = history.frame.loc[[d for d in history.dates if start <= d <= end], codes]
    moves = frame.diff().iloc[1:] * 100
    flagged = moves[(moves.abs() > BIG_MOVE_BP).any(axis=1)]
    if flagged.empty:
        return Check(name, "ok", f"None from {_fmt(start)} to {_fmt(end)}.")
    parts = []
    for day, row in flagged.iterrows():
        code = row.abs().idxmax()
        parts.append(f"{_fmt(day)}, {_rate_name(maturity_of(code))} rate {row[code]:+.1f}bp")
    return Check(name, "warning", "Check the source before relying on these: " + "; ".join(parts) + ".")


def key_rate_check(analysis: Analysis) -> Check:
    """The key-rate DV01s should add up to the parallel DV01."""
    total, parallel = float(analysis.kr_dv01.sum()), analysis.dv01
    gap = abs(total - parallel) / abs(parallel)
    size = "less than 0.01%" if gap < 1e-4 else f"{gap:.2%}"
    detail = f"Key-rate DV01s add up to EUR {total:,.0f} against EUR {parallel:,.0f} for a parallel move, a gap of {size}."
    return Check("Key rates add up", "ok" if gap < 1e-3 else "fail", detail)


def yield_check(analysis: Analysis) -> Check:
    """Every bond should reprice from its own yield."""
    worst = max(abs(price_from_yield(b.position.bond, analysis.date, b.ytm) - b.dirty) for b in analysis.bonds)
    detail = f"The largest difference between a bond's curve price and its price from its yield is {worst:.2e} per 100."
    return Check("Yields reprice the bonds", "ok" if worst < 1e-7 else "fail", detail)


def _maturity(t: float) -> str:
    return f"{round(t * 12)} months" if t < 1 else f"{t:g} years"


def _rate_name(t: float) -> str:
    return f"{round(t * 12)}-month" if t < 1 else f"{t:g}-year"


def run_checks(analysis: Analysis) -> list[Check]:
    history, day = analysis.inputs.history, analysis.date
    start = history.dates[0]
    fit_result, _, _ = fit_check(history, day)
    return [
        rebuild_check(history, day),
        fit_result,
        gaps_check(history, start, day),
        big_moves_check(history, start, day),
        key_rate_check(analysis),
        yield_check(analysis),
    ]
