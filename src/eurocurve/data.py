"""Reading the curve history, the portfolio and the settings.

A data folder holds three files:

    curve.csv       date, BETA0 to TAU2, then any zero rates (SR_3M ... SR_30Y),
                    one row per day in the ECB's units
    portfolio.csv   bond_id, name, coupon, maturity, nominal
    settings.json   names for the portfolio and the curve, key tenors, shock sizes

`eurocurve download` writes curve.csv straight from the ECB.
"""

from __future__ import annotations

import bisect
import datetime as dt
import json
import pathlib
from dataclasses import dataclass, field

import pandas as pd

from .curve import Curve
from .ecb import PARAMS, is_spot_rate, maturity_of
from .models import Bond, Position, Settings, ShockSizes, SvenssonParams

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


def _date(text) -> dt.date:
    return dt.date.fromisoformat(str(text)[:10])


@dataclass
class CurveHistory:
    """Daily curves, one row per date, in the ECB's units."""

    frame: pd.DataFrame
    dates: list[dt.date] = field(init=False)
    spot_codes: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.frame = self.frame.sort_index()
        self.dates = list(self.frame.index)
        self.spot_codes = [c for c in self.frame.columns if is_spot_rate(c)]

    def __len__(self) -> int:
        return len(self.dates)

    def on_or_before(self, day: dt.date) -> dt.date:
        """The latest date with a curve, no later than `day`."""
        i = bisect.bisect_right(self.dates, day)
        if i == 0:
            raise ValueError(f"no curve on or before {day}; the history starts on {self.dates[0]}")
        return self.dates[i - 1]

    def params_on(self, day: dt.date) -> SvenssonParams:
        if day not in self.frame.index:
            raise KeyError(f"no curve for {day}")
        row = self.frame.loc[day]
        return SvenssonParams(*(float(row[p]) for p in PARAMS))

    def curve_on(self, day: dt.date) -> Curve:
        return Curve(day, self.params_on(day))

    def spot_rates_on(self, day: dt.date) -> dict[float, float]:
        """The published zero rates for `day` by maturity in years. Empty when the file has none."""
        row = self.frame.loc[day]
        return {maturity_of(c): float(row[c]) for c in self.spot_codes if pd.notna(row[c])}


def load_curve(path: pathlib.Path | str) -> CurveHistory:
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip().upper() if str(c).strip().lower() != "date" else "date" for c in frame.columns]
    missing = [c for c in ("date", *PARAMS) if c not in frame.columns]
    if missing:
        raise ValueError(f"{pathlib.Path(path).name} is missing {', '.join(missing)}")
    frame["date"] = frame["date"].map(_date)
    if frame["date"].duplicated().any():
        raise ValueError(f"{pathlib.Path(path).name} has more than one row for {frame.loc[frame['date'].duplicated(), 'date'].iloc[0]}")
    frame = frame.set_index("date")
    frame = frame[[c for c in frame.columns if c in PARAMS or is_spot_rate(c)]].astype(float)
    incomplete = frame[list(PARAMS)].isna().any(axis=1)
    if incomplete.any():
        raise ValueError(f"{pathlib.Path(path).name} is missing parameters on {incomplete[incomplete].index[0]}")
    if ((frame["TAU1"] <= 0) | (frame["TAU2"] <= 0)).any():
        raise ValueError(f"{pathlib.Path(path).name} has a tau that is not positive")
    if frame.empty:
        raise ValueError(f"{pathlib.Path(path).name} has no curves")
    return CurveHistory(frame)


def load_portfolio(path: pathlib.Path | str) -> list[Position]:
    frame = pd.read_csv(path, dtype={"bond_id": str, "name": str})
    needed = ["bond_id", "name", "coupon", "maturity", "nominal"]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"{pathlib.Path(path).name} is missing {', '.join(missing)}")
    if frame["bond_id"].duplicated().any():
        raise ValueError(f"bond {frame.loc[frame['bond_id'].duplicated(), 'bond_id'].iloc[0]} is listed twice")
    positions = []
    for r in frame.itertuples(index=False):
        coupon, nominal = float(r.coupon), float(r.nominal)
        if coupon < 0:
            raise ValueError(f"bond {r.bond_id} has a negative coupon")
        if nominal <= 0:
            raise ValueError(f"bond {r.bond_id} needs a positive nominal")
        positions.append(Position(Bond(str(r.bond_id), str(r.name), coupon, _date(r.maturity)), nominal))
    if not positions:
        raise ValueError(f"{pathlib.Path(path).name} has no bonds")
    return positions


def load_settings(path: pathlib.Path | str) -> Settings:
    raw = json.loads(pathlib.Path(path).read_text())
    tenors = tuple(float(t) for t in raw["key_tenors"])
    if any(b <= a for a, b in zip(tenors, tenors[1:])) or not tenors or tenors[0] <= 0:
        raise ValueError("key_tenors must be positive and strictly increasing")
    sizes = raw["shock_sizes_bp"]
    return Settings(
        name=raw["name"],
        curve_name=raw.get("curve_name", "ECB euro area government bond curve"),
        key_tenors=tenors,
        shock_sizes=ShockSizes(float(sizes["parallel"]), float(sizes["short"]), float(sizes["long"])),
        shock_sizes_note=raw.get("shock_sizes_note") or "these sizes",
    )


@dataclass
class Inputs:
    settings: Settings
    history: CurveHistory
    positions: list[Position]
    folder: pathlib.Path


def load_inputs(folder: pathlib.Path | str = SAMPLE_DIR, curve_file: pathlib.Path | str | None = None) -> Inputs:
    """Everything in a data folder. curve_file, when given, replaces the folder's curve.csv."""
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"no such folder: {folder}")
    return Inputs(
        settings=load_settings(folder / "settings.json"),
        history=load_curve(curve_file or folder / "curve.csv"),
        positions=load_portfolio(folder / "portfolio.csv"),
        folder=folder,
    )
