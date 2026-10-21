"""Key-rate tents, DV01 and key-rate DV01s."""

import datetime as dt
import math

import numpy as np
import pytest

from conftest import TODAY, TYPICAL, flat
from eurocurve.curve import Curve
from eurocurve.models import Bond
from eurocurve.risk import KEY_TENORS, curve_dv01, key_rate_dv01s, key_rate_shift

FIVE_YEARS = 3  # position of 5.0 in KEY_TENORS
BUND_2035 = Bond("B35", "2.60% Aug 2035", 2.60, dt.date(2035, 8, 15))
LONG = Bond("B56", "2.90% Aug 2056", 2.90, dt.date(2056, 8, 15))
ZERO_4Y = Bond("Z30", "0.00% Sep 2030", 0.0, dt.date(2030, 9, 29))  # exactly 4.0 years: 1,460 days

# ---------------------------------------------------------------------------
# Tents
# ---------------------------------------------------------------------------


def test_a_tent_peaks_at_its_own_tenor():
    assert key_rate_shift(5.0, KEY_TENORS, FIVE_YEARS)[0] == pytest.approx(1.0)


def test_a_tent_is_zero_at_the_neighboring_tenors_and_beyond():
    assert key_rate_shift([1.0, 3.0, 7.0, 30.0], KEY_TENORS, FIVE_YEARS) == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_a_tent_falls_in_a_straight_line():
    assert key_rate_shift([4.0, 4.5, 6.0], KEY_TENORS, FIVE_YEARS) == pytest.approx([0.5, 0.75, 0.5])


def test_the_first_tent_stays_flat_down_to_zero():
    assert key_rate_shift([0.0, 0.25, 1.0, 1.5], KEY_TENORS, 0) == pytest.approx([1.0, 1.0, 1.0, 0.5])


def test_the_last_tent_stays_flat_beyond_the_last_tenor():
    assert key_rate_shift([25.0, 30.0, 40.0], KEY_TENORS, len(KEY_TENORS) - 1) == pytest.approx([0.5, 1.0, 1.0])


def test_the_tents_add_up_to_a_parallel_shift():
    t = np.linspace(0, 45, 181)
    total = sum(key_rate_shift(t, KEY_TENORS, k, bp=2.5) for k in range(len(KEY_TENORS)))
    assert total == pytest.approx(np.full_like(t, 2.5))


def test_a_tent_scales_with_its_size():
    assert key_rate_shift(4.0, KEY_TENORS, FIVE_YEARS, bp=-10.0)[0] == pytest.approx(-5.0)


def test_tenors_must_be_strictly_increasing():
    with pytest.raises(ValueError):
        key_rate_shift(1.0, [1.0, 5.0, 5.0, 10.0], 0)


@pytest.mark.parametrize("k", [-1, len(KEY_TENORS)])
def test_k_must_point_at_a_tenor(k):
    with pytest.raises(IndexError):
        key_rate_shift(1.0, KEY_TENORS, k)


# ---------------------------------------------------------------------------
# DV01
# ---------------------------------------------------------------------------


def test_dv01_of_a_zero_coupon_bond():
    t = 1460 / 365
    expected = 100 * (math.exp(-0.0299 * t) - math.exp(-0.0301 * t)) / 2
    assert curve_dv01(ZERO_4Y, Curve(TODAY, flat(3.0))) == pytest.approx(expected, rel=1e-9)


def test_dv01_is_positive():
    assert curve_dv01(BUND_2035, Curve(TODAY, TYPICAL)) > 0


def test_dv01_worked_example():
    assert curve_dv01(BUND_2035, Curve(TODAY, TYPICAL)) == pytest.approx(0.0771313839, abs=1e-9)


def test_dv01_is_about_price_times_curve_duration():
    # For a 1bp parallel move in continuously compounded rates, DV01 is close
    # to the PV-weighted time of the cash flows times price / 10,000.
    curve = Curve(TODAY, flat(3.0))
    days = [(dt.date(year, 8, 15) - TODAY).days / 365 for year in range(2027, 2036)]
    flows = [2.60] * 8 + [102.60]
    pv = [f * math.exp(-0.03 * t) for f, t in zip(flows, days)]
    expected = sum(t * v for t, v in zip(days, pv)) / 10_000
    assert curve_dv01(BUND_2035, curve) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Key-rate DV01s
# ---------------------------------------------------------------------------


def test_one_key_rate_dv01_per_tenor():
    assert key_rate_dv01s(BUND_2035, Curve(TODAY, TYPICAL)).shape == (len(KEY_TENORS),)


@pytest.mark.parametrize("bond", [BUND_2035, LONG, ZERO_4Y], ids=lambda b: b.bond_id)
def test_key_rate_dv01s_add_up_to_the_dv01(bond):
    curve = Curve(TODAY, TYPICAL)
    assert key_rate_dv01s(bond, curve).sum() == pytest.approx(curve_dv01(bond, curve), rel=1e-6)


def test_a_four_year_zero_splits_evenly_between_three_and_five_years():
    curve = Curve(TODAY, flat(3.0))
    krs = key_rate_dv01s(ZERO_4Y, curve)
    half = curve_dv01(ZERO_4Y, curve) / 2
    assert krs[2] == pytest.approx(half, rel=1e-6)  # 3 years
    assert krs[3] == pytest.approx(half, rel=1e-6)  # 5 years
    assert np.delete(krs, [2, 3]) == pytest.approx(np.zeros(len(KEY_TENORS) - 2), abs=1e-12)


def test_key_rates_follow_the_tenors_given():
    curve = Curve(TODAY, TYPICAL)
    krs = key_rate_dv01s(BUND_2035, curve, tenors=(2.0, 10.0))
    assert krs.shape == (2,)
    assert krs.sum() == pytest.approx(curve_dv01(BUND_2035, curve), rel=1e-6)


def test_a_bond_has_no_risk_beyond_its_last_cash_flow():
    krs = key_rate_dv01s(BUND_2035, Curve(TODAY, TYPICAL))
    assert krs[-3:] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)  # 15, 20 and 30 years
