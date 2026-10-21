"""Prices off the curve, prices from a yield, and yields from a price."""

import datetime as dt
import math

import pytest

from conftest import TODAY, TYPICAL, flat
from eurocurve.bonds import price_from_curve, price_from_yield, yield_to_maturity
from eurocurve.curve import Curve
from eurocurve.models import Bond

BUND_2035 = Bond("B35", "2.60% Aug 2035", 2.60, dt.date(2035, 8, 15))
ZERO_2031 = Bond("Z31", "0.00% Sep 2031", 0.0, dt.date(2031, 9, 30))  # 1,826 days away
FIVE_YEAR = Bond("B31", "4.00% Sep 2031", 4.0, dt.date(2031, 9, 30))

# ---------------------------------------------------------------------------
# Off the curve
# ---------------------------------------------------------------------------


def test_zero_coupon_price_off_a_flat_curve():
    price = price_from_curve(ZERO_2031, Curve(TODAY, flat(3.0)))
    assert price == pytest.approx(100 * math.exp(-0.03 * 1826 / 365), abs=1e-10)


def test_coupon_bond_price_off_a_flat_curve():
    days = [(dt.date(year, 8, 15) - TODAY).days for year in range(2027, 2036)]
    expected = sum(2.60 * math.exp(-0.025 * d / 365) for d in days) + 100 * math.exp(-0.025 * days[-1] / 365)
    assert price_from_curve(BUND_2035, Curve(TODAY, flat(2.5))) == pytest.approx(expected, abs=1e-10)


def test_price_off_a_typical_curve():
    assert price_from_curve(BUND_2035, Curve(TODAY, TYPICAL)) == pytest.approx(96.5340824435, abs=1e-8)


def test_the_curve_shift_is_part_of_the_price():
    shifted = price_from_curve(BUND_2035, Curve(TODAY, flat(2.0)).parallel(100))
    assert shifted == pytest.approx(price_from_curve(BUND_2035, Curve(TODAY, flat(3.0))))


def test_prices_are_for_settlement_on_the_curve_date():
    later = Curve(dt.date(2027, 9, 30), flat(3.0))
    days = [(dt.date(year, 8, 15) - later.date).days for year in range(2028, 2036)]
    expected = sum(2.60 * math.exp(-0.03 * d / 365) for d in days) + 100 * math.exp(-0.03 * days[-1] / 365)
    assert price_from_curve(BUND_2035, later) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# From a yield
# ---------------------------------------------------------------------------


def test_a_bond_on_a_coupon_date_at_its_coupon_rate_is_at_par():
    assert price_from_yield(FIVE_YEAR, dt.date(2026, 9, 30), 0.04) == pytest.approx(100.0)


def test_between_coupons_a_bond_at_its_coupon_rate_is_100_grown_at_the_yield():
    # Every cash flow is a fraction (1 - w) of a period closer, so the dirty
    # price is 100 * 1.026 ** (46 / 365).
    assert price_from_yield(BUND_2035, TODAY, 0.026) == pytest.approx(100.3240077055, abs=1e-9)


def test_zero_coupon_price_from_a_yield():
    # Next notional coupon date Feb 15, 2027: w = 138 / 365, then five more years.
    bond = Bond("Z32", "0.00% Feb 2032", 0.0, dt.date(2032, 2, 15))
    assert price_from_yield(bond, TODAY, 0.03) == pytest.approx(100 / 1.03 ** (138 / 365 + 5), abs=1e-10)


def test_worked_example_between_coupons():
    w = 319 / 365
    expected = sum(2.60 / 1.03 ** (w + k) for k in range(9)) + 100 / 1.03 ** (w + 8)
    assert price_from_yield(BUND_2035, TODAY, 0.03) == pytest.approx(expected, abs=1e-10)


def test_price_falls_as_the_yield_rises():
    prices = [price_from_yield(BUND_2035, TODAY, y) for y in (-0.005, 0.0, 0.02, 0.04)]
    assert prices == sorted(prices, reverse=True)


def test_at_a_zero_yield_the_price_is_the_sum_of_the_cash_flows():
    assert price_from_yield(BUND_2035, TODAY, 0.0) == pytest.approx(100 + 9 * 2.60)


# ---------------------------------------------------------------------------
# Yield from a price
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ytm", [-0.005, 0.0, 0.0125, 0.026, 0.08])
def test_yield_round_trip(ytm):
    price = price_from_yield(BUND_2035, TODAY, ytm)
    assert yield_to_maturity(BUND_2035, TODAY, price) == pytest.approx(ytm, abs=1e-10)


def test_a_par_bond_on_a_coupon_date_yields_its_coupon():
    assert yield_to_maturity(FIVE_YEAR, dt.date(2026, 9, 30), 100.0) == pytest.approx(0.04, abs=1e-10)


def test_a_zero_coupon_bond_above_100_has_a_negative_yield():
    bond = Bond("Z32", "0.00% Feb 2032", 0.0, dt.date(2032, 2, 15))
    years = 138 / 365 + 5
    assert yield_to_maturity(bond, TODAY, 101.0) == pytest.approx((100 / 101) ** (1 / years) - 1, abs=1e-10)


def test_yield_of_a_bond_priced_off_the_curve():
    price = price_from_curve(BUND_2035, Curve(TODAY, TYPICAL))
    assert yield_to_maturity(BUND_2035, TODAY, price) == pytest.approx(0.0309485602, abs=1e-9)


@pytest.mark.parametrize("price", [0.0, -5.0])
def test_a_price_that_is_not_positive_raises(price):
    with pytest.raises(ValueError):
        yield_to_maturity(BUND_2035, TODAY, price)
