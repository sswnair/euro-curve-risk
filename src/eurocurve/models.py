"""The small objects the rest of the code passes around."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class SvenssonParams:
    """One day's curve in the ECB's units: the betas in percent, the taus in years."""

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float

    @property
    def betas(self) -> tuple[float, float, float, float]:
        return (self.beta0, self.beta1, self.beta2, self.beta3)


@dataclass(frozen=True)
class Bond:
    """A fixed-coupon bond paying once a year, redeemed at 100.

    coupon is the annual rate in percent of nominal, so 2.5 pays 2.50 per 100
    a year. Coupons fall on the maturity date's day and month.
    """

    bond_id: str
    name: str
    coupon: float
    maturity: dt.date


@dataclass(frozen=True)
class Position:
    """A holding of one bond. nominal is the face amount in euros."""

    bond: Bond
    nominal: float


@dataclass(frozen=True)
class ShockSizes:
    """The three sizes the standard rate shocks are built from, in basis points."""

    parallel: float
    short: float
    long: float


@dataclass(frozen=True)
class Settings:
    name: str
    curve_name: str
    key_tenors: tuple[float, ...]
    shock_sizes: ShockSizes
    shock_sizes_note: str
