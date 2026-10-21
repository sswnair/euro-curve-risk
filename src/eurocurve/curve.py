"""A zero curve for one date, and shifted copies of it.

Pricing a bond needs two things from a Curve: how many years away each
payment date is, and the discount factor at that maturity. Rate shocks and
key-rate bumps are shifted copies of the same curve, so the code that
prices a bond doesn't need to know whether the curve it was handed has
been moved.
"""

from __future__ import annotations

import datetime as dt
from typing import Callable

import numpy as np

from .dates import year_fraction
from .models import SvenssonParams
from .svensson import discount_factor, forward_rate, zero_rate

Shift = Callable[[np.ndarray], np.ndarray]


class Curve:
    """The Svensson curve for `date`, plus an optional shift in basis points.

    The shift is a function of maturity in years, so a parallel move, a
    Basel shock and a key-rate tent are all just shifts.
    """

    def __init__(self, date: dt.date, params: SvenssonParams, shift: Shift | None = None):
        self.date = date
        self.params = params
        self._shift = shift

    def __repr__(self) -> str:
        moved = ", shifted" if self._shift else ""
        return f"Curve({self.date.isoformat()}{moved})"

    @property
    def is_shifted(self) -> bool:
        return self._shift is not None

    def time(self, day: dt.date) -> float:
        """Years from the curve date to `day`, actual/365."""
        return year_fraction(self.date, day)

    def shift_at(self, t) -> np.ndarray:
        """The shift in basis points at each maturity, zero for an unshifted curve."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if self._shift is None:
            return np.zeros_like(t)
        return np.broadcast_to(np.asarray(self._shift(t), dtype=float), t.shape).copy()

    def zero(self, t) -> np.ndarray:
        """Zero rates in percent, continuously compounded, shift included."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return zero_rate(t, self.params) + self.shift_at(t) / 100.0

    def discount(self, t) -> np.ndarray:
        """Discount factors at each maturity, shift included."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return discount_factor(t, self.zero(t))

    def forward(self, t) -> np.ndarray:
        """Instantaneous forward rates in percent. Only for an unshifted curve."""
        if self.is_shifted:
            raise ValueError("forward rates are only available on an unshifted curve")
        return forward_rate(t, self.params)

    def shifted(self, shift: Shift) -> "Curve":
        """A copy of this curve with `shift` (basis points, a function of maturity) added on top."""
        if self._shift is None:
            return Curve(self.date, self.params, shift)
        current = self._shift
        return Curve(self.date, self.params, lambda t: np.asarray(current(t), dtype=float) + np.asarray(shift(t), dtype=float))

    def parallel(self, bp: float) -> "Curve":
        """A copy with every zero rate moved by `bp` basis points."""
        return self.shifted(lambda t: np.full(np.shape(t), float(bp)))
