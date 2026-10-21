"""The Svensson curve: loadings, zero and forward rates, discount factors, par yields."""

import math

import numpy as np
import pytest

from conftest import TYPICAL, flat
from eurocurve.svensson import discount_factor, forward_rate, loadings, par_yield, zero_rate

# ---------------------------------------------------------------------------
# Loadings
# ---------------------------------------------------------------------------


def test_one_row_of_loadings_per_maturity():
    assert loadings([1.0, 2.0, 3.0], 1.0, 5.0).shape == (3, 4)


def test_a_single_maturity_still_gives_a_row():
    assert loadings(5.0, 1.0, 5.0).shape == (1, 4)


def test_loadings_worked_example():
    # t = 2, tau1 = 1, tau2 = 4, so x1 = 2 and x2 = 0.5
    row = loadings(2.0, 1.0, 4.0)[0]
    assert row == pytest.approx([1.0, 0.4323323584, 0.2969970751, 0.1804080209], abs=1e-9)


def test_loadings_at_maturity_zero_are_the_limits():
    assert loadings(0.0, 2.0, 8.0)[0] == pytest.approx([1.0, 1.0, 0.0, 0.0])


def test_loadings_just_above_zero_are_close_to_the_limits():
    assert loadings(1e-9, 2.0, 8.0)[0] == pytest.approx([1.0, 1.0, 0.0, 0.0], abs=1e-8)


def test_loadings_at_very_long_maturities():
    # At t = 1000 the exponentials vanish and the slope terms are tau / t.
    assert loadings(1000.0, 1.0, 5.0)[0] == pytest.approx([1.0, 0.001, 0.001, 0.005], abs=1e-12)


def test_negative_maturity_raises():
    with pytest.raises(ValueError):
        loadings([1.0, -0.5], 1.0, 5.0)


@pytest.mark.parametrize("tau1, tau2", [(0.0, 5.0), (1.0, 0.0), (-1.0, 5.0)])
def test_taus_must_be_positive(tau1, tau2):
    with pytest.raises(ValueError):
        loadings(1.0, tau1, tau2)


# ---------------------------------------------------------------------------
# Zero and forward rates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "t, expected",
    [(0.25, 2.1021501773), (1.0, 2.5046655549), (2.0, 2.6419152137), (5.0, 2.7800684706), (10.0, 3.1478800224), (30.0, 3.4594153613)],
)
def test_zero_rates_on_a_typical_curve(t, expected):
    assert zero_rate(t, TYPICAL)[0] == pytest.approx(expected, abs=1e-9)


def test_zero_rate_at_maturity_zero_is_beta0_plus_beta1():
    assert zero_rate(0.0, TYPICAL)[0] == pytest.approx(1.15 + 0.68)


def test_zero_rates_for_several_maturities_at_once():
    rates = zero_rate([1.0, 5.0, 10.0], TYPICAL)
    assert rates.shape == (3,)
    assert rates == pytest.approx([2.5046655549, 2.7800684706, 3.1478800224], abs=1e-9)


def test_a_flat_curve_has_the_same_zero_rate_everywhere():
    assert zero_rate([0.0, 0.5, 7.0, 30.0], flat(3.0)) == pytest.approx([3.0] * 4)


@pytest.mark.parametrize(
    "t, expected",
    [(0.25, 2.3274272953), (1.0, 2.7851628096), (2.0, 2.7532597148), (5.0, 3.1132407191), (10.0, 3.8063047688), (30.0, 3.0007076699)],
)
def test_forward_rates_on_a_typical_curve(t, expected):
    assert forward_rate(t, TYPICAL)[0] == pytest.approx(expected, abs=1e-9)


def test_forward_rate_at_maturity_zero_equals_the_zero_rate():
    assert forward_rate(0.0, TYPICAL)[0] == pytest.approx(zero_rate(0.0, TYPICAL)[0])


@pytest.mark.parametrize("t", [0.5, 3.0, 7.0, 20.0])
def test_forward_rate_is_the_slope_of_rate_times_maturity(t):
    # f(t) = d/dt [t z(t)], checked with a small central difference.
    h = 1e-5
    up, down = (t + h) * zero_rate(t + h, TYPICAL)[0], (t - h) * zero_rate(t - h, TYPICAL)[0]
    assert forward_rate(t, TYPICAL)[0] == pytest.approx((up - down) / (2 * h), abs=1e-6)


# ---------------------------------------------------------------------------
# Discount factors and par yields
# ---------------------------------------------------------------------------


def test_discount_factor_worked_example():
    assert discount_factor(2.0, 3.0)[0] == pytest.approx(0.9417645336, abs=1e-10)  # exp(-0.06)


def test_discount_factor_at_maturity_zero_is_one():
    assert discount_factor(0.0, 3.0)[0] == 1.0


def test_discount_factors_for_several_maturities():
    t = np.array([1.0, 2.0, 5.0])
    assert discount_factor(t, [2.0, 2.5, 3.0]) == pytest.approx(np.exp([-0.02, -0.05, -0.15]))


def test_negative_rates_give_discount_factors_above_one():
    assert discount_factor(1.0, -0.5)[0] == pytest.approx(math.exp(0.005))


def test_par_yield_on_a_flat_curve():
    # With every discount factor exp(-r t), the par coupon is exp(r) - 1 at any maturity.
    for years in (1, 5, 30):
        assert par_yield(years, flat(3.0)) == pytest.approx(3.0454533954, abs=1e-9)


def test_one_year_par_yield():
    assert par_yield(1, TYPICAL) == pytest.approx(2.536295828, abs=1e-8)


def test_ten_year_par_yield():
    assert par_yield(10, TYPICAL) == pytest.approx(3.1603122622, abs=1e-9)


def test_par_yield_prices_a_bond_at_100():
    coupon = par_yield(10, TYPICAL)
    t = np.arange(1.0, 11.0)
    df = discount_factor(t, zero_rate(t, TYPICAL))
    assert coupon * df.sum() + 100 * df[-1] == pytest.approx(100.0)


@pytest.mark.parametrize("years", [0, -1, 2.5])
def test_par_yield_needs_whole_years(years):
    with pytest.raises(ValueError):
        par_yield(years, TYPICAL)
