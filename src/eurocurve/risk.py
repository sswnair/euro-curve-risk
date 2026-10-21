"""Sensitivity to the zero curve: DV01 and key-rate DV01s.

Both come from repricing the bond on bumped copies of the curve, 1 basis
point down and 1 basis point up, and taking half the difference. Values are
per 100 nominal and positive for a bond that gains when rates fall.

A 1 basis point move in a continuously compounded zero rate is a (1 + y)
basis point move in an annual yield y, so DV01 here sits close to Macaulay
duration * price / 10,000 rather than modified duration * price / 10,000.
With positive yields that makes it a little larger.
"""

from __future__ import annotations

import numpy as np

from .bonds import price_from_curve
from .curve import Curve
from .models import Bond

KEY_TENORS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)


def key_rate_shift(t, tenors, k: int, bp: float = 1.0) -> np.ndarray:
    """A tent-shaped shift of `bp` basis points centered on tenors[k].

    It is bp at tenors[k] and falls in a straight line to zero at the
    neighboring tenors. The first tent stays at bp all the way down to
    maturity 0 and the last one stays at bp beyond the last tenor, so the
    tents for every k add up to a parallel shift of bp at any maturity.

    Raises ValueError unless tenors is strictly increasing, and IndexError
    when k is not a position in tenors.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    tenors = np.asarray(tenors, dtype=float)
    if len(tenors) < 1 or np.any(np.diff(tenors) <= 0):
        raise ValueError("tenors must be strictly increasing")
    if not 0 <= k < len(tenors):
        raise IndexError(f"no tenor at position {k}")
    unit = np.zeros(len(tenors))
    unit[k] = 1.0
    return bp * np.interp(t, tenors, unit)


def curve_dv01(bond: Bond, curve: Curve) -> float:
    """Price change per 100 nominal for a 1 basis point parallel fall in
    the zero curve, from a bump each way:

        (price with every zero rate 1bp lower - price 1bp higher) / 2
    """
    down = price_from_curve(bond, curve.parallel(-1.0))
    up = price_from_curve(bond, curve.parallel(1.0))
    return (down - up) / 2.0


def key_rate_dv01s(bond: Bond, curve: Curve, tenors=KEY_TENORS) -> np.ndarray:
    """DV01 split by tenor: for each key tenor, the price change per 100
    nominal for a 1 basis point fall in that tenor's tent, measured the same
    way as curve_dv01. Returns one value per tenor, in tenor order. They add
    up to curve_dv01 to within a tiny higher-order difference."""
    out = []
    for k in range(len(tenors)):
        down = price_from_curve(bond, curve.shifted(lambda t, k=k: -key_rate_shift(t, tenors, k)))
        up = price_from_curve(bond, curve.shifted(lambda t, k=k: key_rate_shift(t, tenors, k)))
        out.append((down - up) / 2.0)
    return np.array(out)
