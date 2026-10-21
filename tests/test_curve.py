"""The Curve wrapper: shifts, discount factors and the time axis."""

import datetime as dt
import math

import numpy as np
import pytest

from conftest import TODAY, TYPICAL, flat
from eurocurve.curve import Curve
from eurocurve.svensson import zero_rate


def test_an_unshifted_curve_gives_the_svensson_zero_rates():
    t = [0.5, 2.0, 10.0]
    assert Curve(TODAY, TYPICAL).zero(t) == pytest.approx(zero_rate(t, TYPICAL))


def test_a_parallel_shift_moves_every_rate():
    curve = Curve(TODAY, TYPICAL)
    t = np.array([0.5, 2.0, 10.0])
    assert curve.parallel(25).zero(t) == pytest.approx(curve.zero(t) + 0.25)


def test_shifts_add_up():
    curve = Curve(TODAY, flat(3.0)).parallel(10).shifted(lambda t: 2.0 * t)
    assert curve.zero([0.0, 5.0]) == pytest.approx([3.1, 3.2])


def test_shifting_leaves_the_original_alone():
    curve = Curve(TODAY, flat(3.0))
    curve.parallel(50)
    assert not curve.is_shifted
    assert curve.zero(5.0)[0] == pytest.approx(3.0)


def test_discount_factors_include_the_shift():
    curve = Curve(TODAY, flat(3.0)).parallel(-100)
    assert curve.discount(2.0)[0] == pytest.approx(math.exp(-0.04))


def test_time_is_actual_days_over_365():
    assert Curve(TODAY, TYPICAL).time(dt.date(2027, 9, 30)) == 1.0
    assert Curve(TODAY, TYPICAL).time(dt.date(2028, 9, 30)) == pytest.approx(731 / 365)


def test_forward_rates_only_on_an_unshifted_curve():
    curve = Curve(TODAY, TYPICAL)
    assert curve.forward(0.0)[0] == pytest.approx(curve.zero(0.0)[0])
    with pytest.raises(ValueError):
        curve.parallel(1).forward(5.0)
