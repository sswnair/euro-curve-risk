"""The six standard interest rate shocks, and what they do to a portfolio.

The shapes are the Basel Committee's standardized scenarios for interest
rate risk in the banking book (BCBS 368, 2016). Each currency gets three
sizes: a parallel shock, a short rate shock that fades with maturity, and a
long rate shock that builds with it. The steepener and the flattener combine
the short and long shocks.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .bonds import price_from_curve
from .curve import Curve
from .models import Position, ShockSizes

SCENARIOS = ("parallel up", "parallel down", "steepener", "flattener", "short up", "short down")
DECAY = 4.0  # years, the x in the Basel formulas

EUR_2016 = ShockSizes(parallel=200.0, short=250.0, long=100.0)
EUR_2024 = ShockSizes(parallel=225.0, short=350.0, long=200.0)


def irrbb_shock(t, scenario: str, sizes: ShockSizes, decay: float = DECAY) -> np.ndarray:
    """The shift in basis points at each maturity in t for one scenario.

    With short(t) = sizes.short * exp(-t / decay) and
    long(t) = sizes.long * (1 - exp(-t / decay)):

        parallel up      +sizes.parallel
        parallel down    -sizes.parallel
        short up         +short(t)
        short down       -short(t)
        steepener        -0.65 * short(t) + 0.9 * long(t)
        flattener        +0.8 * short(t) - 0.6 * long(t)

    Raises ValueError for any other scenario name.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    short = sizes.short * np.exp(-t / decay)
    long = sizes.long * (1.0 - np.exp(-t / decay))
    if scenario == "parallel up":
        return np.full_like(t, sizes.parallel)
    if scenario == "parallel down":
        return np.full_like(t, -sizes.parallel)
    if scenario == "short up":
        return short
    if scenario == "short down":
        return -short
    if scenario == "steepener":
        return -0.65 * short + 0.9 * long
    if scenario == "flattener":
        return 0.8 * short - 0.6 * long
    raise ValueError(f"unknown scenario {scenario!r}")


def scenario_pnl(positions: Sequence[Position], curve: Curve, shift: Callable) -> np.ndarray:
    """The change in market value of each position, in euros, when the zero
    curve moves by shift(t) basis points, with every bond repriced in full:

        nominal / 100 * (dirty price on the shifted curve - dirty price today)

    One value per position, in the order given.
    """
    moved = curve.shifted(shift)
    return np.array(
        [p.nominal / 100.0 * (price_from_curve(p.bond, moved) - price_from_curve(p.bond, curve)) for p in positions]
    )


def key_rate_estimate(kr_dv01s, tenors, shift: Callable) -> float:
    """The first-order estimate of the same P&L from key-rate DV01s.

    kr_dv01s are in euros for a 1 basis point fall at each tenor, so a rise
    of shift(tenor) basis points loses about kr_dv01 * shift(tenor) there:

        -sum over the tenors of kr_dv01s[k] * shift(tenors[k])

    The gap to scenario_pnl has two sources: convexity, which a first-order
    estimate leaves out, and the shock's shape between key tenors, which the
    tents can only follow in straight lines.
    """
    move = np.asarray(shift(np.asarray(tenors, dtype=float)), dtype=float)
    return float(-np.dot(np.asarray(kr_dv01s, dtype=float), move))
