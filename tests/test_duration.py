"""Macaulay and modified duration, and convexity, from the yield."""

import datetime as dt

import pytest

from conftest import TODAY
from eurocurve.bonds import convexity, macaulay_duration, modified_duration, price_from_yield
from eurocurve.models import Bond

THREE_YEAR = Bond("B29", "5.00% Sep 2029", 5.0, dt.date(2029, 9, 30))  # three coupons left on TODAY
BUND_2035 = Bond("B35", "2.60% Aug 2035", 2.60, dt.date(2035, 8, 15))
ZERO_2032 = Bond("Z32", "0.00% Feb 2032", 0.0, dt.date(2032, 2, 15))
LONG = Bond("B56", "2.90% Aug 2056", 2.90, dt.date(2056, 8, 15))


def test_macaulay_duration_worked_example():
    # (1 * 5/1.05 + 2 * 5/1.05**2 + 3 * 105/1.05**3) / 100
    assert macaulay_duration(THREE_YEAR, TODAY, 0.05) == pytest.approx(2.8594104308, abs=1e-9)


def test_modified_duration_worked_example():
    assert modified_duration(THREE_YEAR, TODAY, 0.05) == pytest.approx(2.7232480294, abs=1e-9)


def test_convexity_worked_example():
    # (1*2 * 5/1.05 + 2*3 * 5/1.05**2 + 3*4 * 105/1.05**3) / (100 * 1.05**2)
    assert convexity(THREE_YEAR, TODAY, 0.05) == pytest.approx(10.2056241998, abs=1e-9)


def test_a_zero_coupon_bond_has_a_duration_equal_to_its_life():
    # w = 138 / 365 of the current notional period, then five more years
    assert macaulay_duration(ZERO_2032, TODAY, 0.03) == pytest.approx(138 / 365 + 5)


def test_zero_coupon_duration_does_not_depend_on_the_yield():
    assert macaulay_duration(ZERO_2032, TODAY, 0.01) == pytest.approx(macaulay_duration(ZERO_2032, TODAY, 0.06))


def test_modified_duration_divides_by_one_plus_the_yield():
    for ytm in (-0.002, 0.03):
        expected = macaulay_duration(BUND_2035, TODAY, ytm) / (1 + ytm)
        assert modified_duration(BUND_2035, TODAY, ytm) == pytest.approx(expected)


@pytest.mark.parametrize("bond", [THREE_YEAR, BUND_2035, ZERO_2032, LONG], ids=lambda b: b.bond_id)
def test_modified_duration_matches_a_small_move_in_the_yield(bond):
    y, h = 0.03, 1e-6
    price = price_from_yield(bond, TODAY, y)
    slope = (price_from_yield(bond, TODAY, y - h) - price_from_yield(bond, TODAY, y + h)) / (2 * h * price)
    assert modified_duration(bond, TODAY, y) == pytest.approx(slope, rel=1e-7)


@pytest.mark.parametrize("bond", [THREE_YEAR, BUND_2035, ZERO_2032, LONG], ids=lambda b: b.bond_id)
def test_convexity_matches_a_small_move_in_the_yield(bond):
    y, h = 0.03, 1e-4
    price = price_from_yield(bond, TODAY, y)
    curvature = (price_from_yield(bond, TODAY, y + h) + price_from_yield(bond, TODAY, y - h) - 2 * price) / (h * h * price)
    assert convexity(bond, TODAY, y) == pytest.approx(curvature, rel=1e-5)


def test_longer_bonds_have_longer_durations():
    durations = [modified_duration(b, TODAY, 0.03) for b in (THREE_YEAR, ZERO_2032, BUND_2035, LONG)]
    assert durations == sorted(durations)


def test_duration_is_shorter_than_life_for_a_coupon_bond():
    life = (BUND_2035.maturity - TODAY).days / 365
    assert macaulay_duration(BUND_2035, TODAY, 0.03) < life
