"""Coupon dates, accrued interest and cash flows."""

import datetime as dt

import pytest

from conftest import TODAY
from eurocurve.bonds import accrued_interest, cash_flows, coupon_dates
from eurocurve.models import Bond

BUND_2035 = Bond("B35", "2.60% Aug 2035", 2.60, dt.date(2035, 8, 15))
ZERO_2032 = Bond("Z32", "0.00% Feb 2032", 0.0, dt.date(2032, 2, 15))

# ---------------------------------------------------------------------------
# Coupon dates
# ---------------------------------------------------------------------------


def test_previous_and_upcoming_coupon_dates():
    previous, upcoming = coupon_dates(BUND_2035.maturity, TODAY)
    assert previous == dt.date(2026, 8, 15)
    assert upcoming == [dt.date(year, 8, 15) for year in range(2027, 2036)]


def test_settling_on_a_coupon_date():
    previous, upcoming = coupon_dates(BUND_2035.maturity, dt.date(2026, 8, 15))
    assert previous == dt.date(2026, 8, 15)
    assert upcoming[0] == dt.date(2027, 8, 15)


def test_the_day_before_a_coupon_date():
    previous, upcoming = coupon_dates(BUND_2035.maturity, dt.date(2026, 8, 14))
    assert previous == dt.date(2025, 8, 15)
    assert upcoming[0] == dt.date(2026, 8, 15)


def test_the_last_coupon_period():
    previous, upcoming = coupon_dates(dt.date(2027, 3, 1), TODAY)
    assert previous == dt.date(2026, 3, 1)
    assert upcoming == [dt.date(2027, 3, 1)]


def test_a_february_29_maturity():
    previous, upcoming = coupon_dates(dt.date(2028, 2, 29), TODAY)
    assert previous == dt.date(2026, 2, 28)
    assert upcoming == [dt.date(2027, 2, 28), dt.date(2028, 2, 29)]


@pytest.mark.parametrize("settle", [dt.date(2035, 8, 15), dt.date(2035, 9, 1)])
def test_a_bond_that_has_matured_raises(settle):
    with pytest.raises(ValueError):
        coupon_dates(BUND_2035.maturity, settle)


# ---------------------------------------------------------------------------
# Accrued interest
# ---------------------------------------------------------------------------


def test_accrued_interest_worked_example():
    # 46 days of a 365-day period: 2.60 * 46 / 365
    assert accrued_interest(BUND_2035, TODAY) == pytest.approx(0.3276712329, abs=1e-10)


def test_accrued_interest_over_a_period_with_february_29():
    # March 15, 2027 to March 15, 2028 has 366 days, 199 of them gone by September 30.
    bond = Bond("B30", "3.00% Mar 2030", 3.0, dt.date(2030, 3, 15))
    assert accrued_interest(bond, dt.date(2027, 9, 30)) == pytest.approx(1.631147541, abs=1e-9)


def test_no_accrued_interest_on_a_coupon_date():
    assert accrued_interest(BUND_2035, dt.date(2026, 8, 15)) == 0.0


def test_a_day_before_the_coupon_accrues_almost_all_of_it():
    assert accrued_interest(BUND_2035, dt.date(2026, 8, 14)) == pytest.approx(2.60 * 364 / 365)


def test_a_zero_coupon_bond_accrues_nothing():
    assert accrued_interest(ZERO_2032, TODAY) == 0.0


# ---------------------------------------------------------------------------
# Cash flows
# ---------------------------------------------------------------------------


def test_coupons_then_coupon_plus_redemption():
    flows = cash_flows(BUND_2035, TODAY)
    assert len(flows) == 9
    assert flows[0] == (dt.date(2027, 8, 15), pytest.approx(2.60))
    assert flows[-2] == (dt.date(2034, 8, 15), pytest.approx(2.60))
    assert flows[-1] == (dt.date(2035, 8, 15), pytest.approx(102.60))


def test_cash_flows_are_in_date_order():
    dates = [day for day, _ in cash_flows(BUND_2035, TODAY)]
    assert dates == sorted(dates)


def test_a_coupon_paid_on_the_settlement_date_is_not_included():
    flows = cash_flows(BUND_2035, dt.date(2026, 8, 15))
    assert flows[0][0] == dt.date(2027, 8, 15)


def test_a_zero_coupon_bond_has_one_cash_flow():
    assert cash_flows(ZERO_2032, TODAY) == [(dt.date(2032, 2, 15), 100.0)]


def test_cash_flows_add_up_to_the_coupons_and_the_redemption():
    total = sum(amount for _, amount in cash_flows(BUND_2035, TODAY))
    assert total == pytest.approx(100 + 9 * 2.60)
