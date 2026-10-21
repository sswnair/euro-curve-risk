"""Write the synthetic curve history in data/sample/curve.csv.

The curves are made up. They are in the ECB's format and units, with
parameters of the same size as the ECB's published ones, but they are not
ECB data. Use `eurocurve download` for the real thing.

The path runs from one curve on September 30, 2025 to another a year later,
with a rise in the middle of the year, daily noise that fades back toward
the path, and two things for the data checks to find: a day when the
3-month rate jumps by about 20 basis points, and a business day with no
curve at all.
Run from the project root after `pip install -e .`:

    python scripts/make_samples.py
"""

from __future__ import annotations

import datetime as dt
import pathlib

import numpy as np
import pandas as pd

from eurocurve.dates import target_days
from eurocurve.ecb import PARAMS, SPOT_RATES, maturity_of
from eurocurve.models import SvenssonParams
from eurocurve.svensson import zero_rate

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample" / "curve.csv"

SEED = 20260930
START, END = dt.date(2025, 9, 30), dt.date(2026, 9, 30)
JUMP_DAY = dt.date(2026, 3, 19)
MISSING_DAY = dt.date(2026, 6, 17)

# Betas in percent, taus in years. Both ends share their taus.
FIRST = np.array([0.930850, 0.982273, -0.186953, 7.446408])
LAST = np.array([0.783970, 1.238966, 0.401882, 8.277565])
TAU1, TAU2 = 2.64, 15.4

# The mid-year rise, added to beta 0 and beta 1 at its peak.
RISE = np.array([0.15, 0.10, 0.0, 0.0])
# Daily noise on each beta in percent, and how fast it fades (half-life in business days).
NOISE = np.array([0.030, 0.026, 0.040, 0.040])
HALF_LIFE = 30
# The jump: mostly beta 1, so the short end moves most.
JUMP = np.array([0.05, 0.16, 0.0, 0.0])


def main() -> None:
    rng = np.random.default_rng(SEED)
    days = target_days(START, END)
    n = len(days)
    keep = 0.5 ** (1 / HALF_LIFE)
    corr = np.array(
        [
            [1.0, -0.6, 0.0, 0.2],
            [-0.6, 1.0, 0.1, 0.0],
            [0.0, 0.1, 1.0, -0.3],
            [0.2, 0.0, -0.3, 1.0],
        ]
    )
    chol = np.linalg.cholesky(corr)
    drift = np.zeros(4)
    tau_drift = np.zeros(2)
    rows = []
    for i, day in enumerate(days):
        s = i / (n - 1)
        if day == JUMP_DAY:
            drift = keep * drift + JUMP
        elif i:
            drift = keep * drift + NOISE * (chol @ rng.standard_normal(4))
            tau_drift = keep * tau_drift + np.array([0.008, 0.006]) * rng.standard_normal(2)
        betas = (1 - s) * FIRST + s * LAST + RISE * np.sin(np.pi * s) + drift
        tau1, tau2 = TAU1 * np.exp(tau_drift[0]), TAU2 * np.exp(tau_drift[1])
        params = SvenssonParams(*(round(float(b), 6) for b in betas), round(float(tau1), 6), round(float(tau2), 6))
        if day == MISSING_DAY:
            continue
        spots = zero_rate([maturity_of(c) for c in SPOT_RATES], params)
        row = {"date": day.isoformat()}
        row.update(zip(PARAMS, (params.beta0, params.beta1, params.beta2, params.beta3, params.tau1, params.tau2)))
        row.update((code, round(float(v), 6)) for code, v in zip(SPOT_RATES, spots))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT, index=False, float_format="%.6f")
    print(f"{len(frame)} curves from {frame['date'].iloc[0]} to {frame['date'].iloc[-1]} written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
