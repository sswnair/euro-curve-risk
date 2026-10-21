"""The six standard rate shocks, full revaluation and the key-rate estimate."""

import datetime as dt
import math

import numpy as np
import pytest

from conftest import TODAY, TYPICAL, flat
from eurocurve.curve import Curve
from eurocurve.models import Bond, Position, ShockSizes
from eurocurve.risk import KEY_TENORS, curve_dv01, key_rate_dv01s
from eurocurve.scenarios import EUR_2016, EUR_2024, SCENARIOS, irrbb_shock, key_rate_estimate, scenario_pnl

ZERO_4Y = Bond("Z30", "0.00% Sep 2030", 0.0, dt.date(2030, 9, 29))  # exactly 4.0 years: 1,460 days
BUND_2035 = Bond("B35", "2.60% Aug 2035", 2.60, dt.date(2035, 8, 15))
LONG = Bond("B56", "2.90% Aug 2056", 2.90, dt.date(2056, 8, 15))

# ---------------------------------------------------------------------------
# Shock shapes
# ---------------------------------------------------------------------------


def test_six_scenarios():
    assert len(SCENARIOS) == 6


@pytest.mark.parametrize("scenario, sign", [("parallel up", 1), ("parallel down", -1)])
def test_parallel_shocks_are_the_same_at_every_maturity(scenario, sign):
    assert irrbb_shock([0.0, 5.0, 30.0], scenario, EUR_2024) == pytest.approx([sign * 225.0] * 3)


def test_the_short_shock_fades_with_maturity():
    shock = irrbb_shock([0.0, 4.0, 30.0], "short up", EUR_2024)
    assert shock == pytest.approx([350.0, 128.7578044100, 0.1935795296], abs=1e-9)


def test_short_down_mirrors_short_up():
    t = [0.5, 2.0, 10.0]
    assert irrbb_shock(t, "short down", EUR_2024) == pytest.approx(-irrbb_shock(t, "short up", EUR_2024))


def test_steepener_worked_example():
    # At 4 years: short = 350 / e, long = 200 * (1 - 1/e)
    assert irrbb_shock(4.0, "steepener", EUR_2024)[0] == pytest.approx(30.0891277226, abs=1e-9)


def test_flattener_worked_example():
    assert irrbb_shock(4.0, "flattener", EUR_2024)[0] == pytest.approx(27.1517764686, abs=1e-9)


def test_steepener_and_flattener_at_the_short_end():
    assert irrbb_shock(0.0, "steepener", EUR_2024)[0] == pytest.approx(-0.65 * 350)
    assert irrbb_shock(0.0, "flattener", EUR_2024)[0] == pytest.approx(0.8 * 350)


def test_steepener_and_flattener_at_the_long_end():
    t = 200.0  # long enough for the short shock to vanish
    assert irrbb_shock(t, "steepener", EUR_2024)[0] == pytest.approx(0.9 * 200)
    assert irrbb_shock(t, "flattener", EUR_2024)[0] == pytest.approx(-0.6 * 200)


def test_the_2016_euro_sizes():
    assert irrbb_shock(0.0, "parallel up", EUR_2016)[0] == 200.0
    assert irrbb_shock(0.0, "short up", EUR_2016)[0] == pytest.approx(250.0)
    assert irrbb_shock(200.0, "steepener", EUR_2016)[0] == pytest.approx(90.0)


def test_the_decay_can_be_changed():
    assert irrbb_shock(2.0, "short up", EUR_2024, decay=2.0)[0] == pytest.approx(350 / math.e)


def test_shock_shape_works_on_arrays():
    t = np.linspace(0, 30, 7)
    for scenario in SCENARIOS:
        assert irrbb_shock(t, scenario, ShockSizes(100, 100, 100)).shape == (7,)


def test_an_unknown_scenario_raises():
    with pytest.raises(ValueError):
        irrbb_shock(1.0, "twist", EUR_2024)


# ---------------------------------------------------------------------------
# Full revaluation
# ---------------------------------------------------------------------------


def test_full_revaluation_of_a_zero_coupon_bond():
    curve = Curve(TODAY, flat(3.0))
    pnl = scenario_pnl([Position(ZERO_4Y, 10_000_000)], curve, lambda t: np.full_like(t, 100.0))
    expected = 10_000_000 / 100 * 100 * (math.exp(-0.04 * 4.0) - math.exp(-0.03 * 4.0))
    assert pnl == pytest.approx([expected], abs=1e-6)


def test_one_result_per_position_in_order():
    positions = [Position(BUND_2035, 5_000_000), Position(LONG, 5_000_000)]
    pnl = scenario_pnl(positions, Curve(TODAY, TYPICAL), lambda t: irrbb_shock(t, "parallel up", EUR_2024))
    assert pnl.shape == (2,)
    assert pnl[1] < pnl[0] < 0  # the 30-year bond loses more


def test_no_shift_means_no_change():
    pnl = scenario_pnl([Position(BUND_2035, 5_000_000)], Curve(TODAY, TYPICAL), lambda t: np.zeros_like(t))
    assert pnl == pytest.approx([0.0], abs=1e-9)


def test_falling_rates_gain_more_than_rising_rates_lose():
    positions = [Position(LONG, 10_000_000)]
    curve = Curve(TODAY, TYPICAL)
    up = scenario_pnl(positions, curve, lambda t: irrbb_shock(t, "parallel up", EUR_2024))[0]
    down = scenario_pnl(positions, curve, lambda t: irrbb_shock(t, "parallel down", EUR_2024))[0]
    assert down > -up > 0


def test_full_revaluation_worked_example():
    positions = [Position(BUND_2035, 10_000_000)]
    pnl = scenario_pnl(positions, Curve(TODAY, TYPICAL), lambda t: irrbb_shock(t, "steepener", EUR_2024))
    assert pnl == pytest.approx([-932651.9284], abs=1e-3)


# ---------------------------------------------------------------------------
# The key-rate estimate
# ---------------------------------------------------------------------------


def test_key_rate_estimate_worked_example():
    estimate = key_rate_estimate([100.0, 200.0, 300.0], [1.0, 2.0, 5.0], lambda t: 10.0 * np.asarray(t))
    assert estimate == pytest.approx(-(100 * 10 + 200 * 20 + 300 * 50))


def test_a_parallel_rise_loses_about_dv01_per_basis_point():
    curve = Curve(TODAY, TYPICAL)
    krs = key_rate_dv01s(BUND_2035, curve)
    assert key_rate_estimate(krs, KEY_TENORS, lambda t: np.full_like(t, 25.0)) == pytest.approx(-25 * curve_dv01(BUND_2035, curve), rel=1e-6)


def test_the_estimate_is_close_for_a_small_parallel_shock():
    positions = [Position(BUND_2035, 10_000_000)]
    curve = Curve(TODAY, TYPICAL)
    krs = key_rate_dv01s(BUND_2035, curve) * 10_000_000 / 100

    def shift(t):
        return np.full_like(t, 2.0)

    full = scenario_pnl(positions, curve, shift)[0]
    assert key_rate_estimate(krs, KEY_TENORS, shift) == pytest.approx(full, rel=1e-3)


def test_the_estimate_follows_any_shift_drawn_straight_between_key_tenors():
    # Such a shift is exactly a sum of tents, so only convexity is left over.
    positions = [Position(LONG, 10_000_000)]
    curve = Curve(TODAY, TYPICAL)
    krs = key_rate_dv01s(LONG, curve) * 10_000_000 / 100
    points = [0.3, -0.1, 0.05, 0.2, -0.25, 0.1, 0.4, -0.3, 0.15]  # basis points

    def shift(t):
        return np.interp(t, KEY_TENORS, points)

    full = scenario_pnl(positions, curve, shift)[0]
    assert key_rate_estimate(krs, KEY_TENORS, shift) == pytest.approx(full, rel=1e-3)
