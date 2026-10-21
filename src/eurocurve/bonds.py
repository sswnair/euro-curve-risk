"""Fixed-coupon bonds: coupon dates, accrued interest, cash flows, prices,
yields, duration and convexity.

The bonds pay one coupon a year on the maturity date's day and month and are
redeemed at 100, like Bunds and most other euro government bonds. Interest
accrues on an actual/actual (ICMA) basis. Prices are per 100 nominal, and
coupon dates are not moved for weekends or holidays.

Yields follow the ICMA convention for annual bonds: a cash flow k coupon
periods after the next coupon date is discounted over w + k years, where w is
the part of the current coupon period still to run.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from .curve import Curve
from .dates import add_years
from .models import Bond


def coupon_dates(maturity: dt.date, settle: dt.date) -> tuple[dt.date, list[dt.date]]:
    """The coupon date on or before settle, and every coupon date after
    settle up to and including maturity.

    Coupon dates step back from maturity one year at a time, and a February
    29 maturity pays on February 28 in the other years. A bond settling on a
    coupon date gets that date back as the previous coupon, and the coupon
    paid that day is not in the list.

    Raises ValueError when settle is on or after maturity.
    """
    if settle >= maturity:
        raise ValueError(f"the bond matured on {maturity}")
    years = 0
    while add_years(maturity, -(years + 1)) > settle:
        years += 1
    previous = add_years(maturity, -(years + 1))
    upcoming = [add_years(maturity, -k) for k in range(years, -1, -1)]
    return previous, upcoming


def accrued_interest(bond: Bond, settle: dt.date) -> float:
    """Accrued interest per 100 nominal, actual/actual (ICMA):

        coupon * days since the previous coupon / days in the coupon period

    Zero on a coupon date.
    """
    previous, upcoming = coupon_dates(bond.maturity, settle)
    return bond.coupon * (settle - previous).days / (upcoming[0] - previous).days


def cash_flows(bond: Bond, settle: dt.date) -> list[tuple[dt.date, float]]:
    """The payments still to come per 100 nominal, as (date, amount) in date order.

    Each coupon date after settle pays the coupon, and maturity pays 100 on
    top. Zero coupons are left out, so a zero-coupon bond has a single cash
    flow of 100.
    """
    _, upcoming = coupon_dates(bond.maturity, settle)
    flows = []
    for day in upcoming:
        amount = bond.coupon + (100.0 if day == bond.maturity else 0.0)
        if amount:
            flows.append((day, amount))
    return flows


def price_from_curve(bond: Bond, curve: Curve) -> float:
    """Dirty price per 100 nominal for settlement on the curve date: each
    cash flow times the curve's discount factor at its payment date."""
    flows = cash_flows(bond, curve.date)
    t = np.array([curve.time(day) for day, _ in flows])
    amounts = np.array([amount for _, amount in flows])
    return float(np.sum(amounts * curve.discount(t)))


def price_from_yield(bond: Bond, settle: dt.date, ytm: float) -> float:
    """Dirty price per 100 nominal at an annual yield ytm (0.025 for 2.5%):

        sum over the coupon dates after settle of  cash flow / (1 + ytm) ** (w + k)

    where k is 0 for the next coupon date, 1 for the one after and so on, and
    w is the days from settle to the next coupon over the days in the
    current coupon period.
    """
    previous, upcoming = coupon_dates(bond.maturity, settle)
    w = (upcoming[0] - settle).days / (upcoming[0] - previous).days
    price = 0.0
    for k, day in enumerate(upcoming):
        amount = bond.coupon + (100.0 if day == bond.maturity else 0.0)
        price += amount / (1.0 + ytm) ** (w + k)
    return price


def yield_to_maturity(bond: Bond, settle: dt.date, dirty_price: float) -> float:
    """The annual yield at which price_from_yield gives dirty_price, as a
    decimal and accurate to at least 1e-10. Negative yields are fine: euro
    government bonds had them for years.

    Raises ValueError when dirty_price is not positive.
    """
    if dirty_price <= 0:
        raise ValueError("price must be positive")
    lo, hi = -0.5, 1.0
    while price_from_yield(bond, settle, hi) > dirty_price:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if price_from_yield(bond, settle, mid) > dirty_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    return (lo + hi) / 2


def macaulay_duration(bond: Bond, settle: dt.date, ytm: float) -> float:
    """Macaulay duration in years: the present-value-weighted average time
    to each cash flow, with the times (w + k) and the discounting used in
    price_from_yield."""
    previous, upcoming = coupon_dates(bond.maturity, settle)
    w = (upcoming[0] - settle).days / (upcoming[0] - previous).days
    price = weighted = 0.0
    for k, day in enumerate(upcoming):
        amount = bond.coupon + (100.0 if day == bond.maturity else 0.0)
        pv = amount / (1.0 + ytm) ** (w + k)
        price += pv
        weighted += (w + k) * pv
    return weighted / price


def modified_duration(bond: Bond, settle: dt.date, ytm: float) -> float:
    """Macaulay duration / (1 + ytm): the percentage price change for a one
    point change in the annual yield."""
    return macaulay_duration(bond, settle, ytm) / (1.0 + ytm)


def convexity(bond: Bond, settle: dt.date, ytm: float) -> float:
    """Convexity in years squared, with the same times and discounting as
    price_from_yield:

        sum of (w + k) * (w + k + 1) * PV / (price * (1 + ytm) ** 2)

    where PV is the present value of each cash flow.
    """
    previous, upcoming = coupon_dates(bond.maturity, settle)
    w = (upcoming[0] - settle).days / (upcoming[0] - previous).days
    price = weighted = 0.0
    for k, day in enumerate(upcoming):
        amount = bond.coupon + (100.0 if day == bond.maturity else 0.0)
        pv = amount / (1.0 + ytm) ** (w + k)
        price += pv
        weighted += (w + k) * (w + k + 1) * pv
    return weighted / (price * (1.0 + ytm) ** 2)
