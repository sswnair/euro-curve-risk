"""Fitting a Svensson curve to a set of zero rates."""

import numpy as np
import pytest

from eurocurve.models import SvenssonParams
from eurocurve.svensson import fit, loadings, zero_rate

MATURITIES = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
TRUE = SvenssonParams(beta0=1.2, beta1=0.9, beta2=1.5, beta3=6.0, tau1=1.5, tau2=9.0)


def test_fit_finds_a_curve_whose_taus_are_on_the_grid():
    rates = zero_rate(MATURITIES, TRUE)
    params, rmse = fit(MATURITIES, rates, [0.5, 1.0, 1.5, 2.0], [5.0, 7.0, 9.0, 11.0])
    assert (params.tau1, params.tau2) == (1.5, 9.0)
    assert params.betas == pytest.approx(TRUE.betas, abs=1e-8)
    assert rmse == pytest.approx(0.0, abs=1e-8)


def test_fit_returns_a_svensson_params():
    params, _ = fit(MATURITIES, zero_rate(MATURITIES, TRUE), [1.5], [9.0])
    assert isinstance(params, SvenssonParams)


def test_betas_come_from_least_squares_and_the_error_is_in_basis_points():
    # Rates = curve + noise that the four loadings can't explain, so least
    # squares returns the curve's betas and leaves exactly the noise behind.
    x = loadings(MATURITIES, 1.5, 9.0)
    raw = np.random.default_rng(7).normal(0.0, 0.02, len(MATURITIES))
    noise = raw - x @ np.linalg.lstsq(x, raw, rcond=None)[0]
    rates = zero_rate(MATURITIES, TRUE) + noise
    params, rmse = fit(MATURITIES, rates, [1.5], [9.0])
    assert params.betas == pytest.approx(TRUE.betas, abs=1e-9)
    assert rmse == pytest.approx(np.sqrt(np.mean(noise**2)) * 100, rel=1e-9)


def test_the_best_pair_on_the_grid_wins():
    rates = zero_rate(MATURITIES, TRUE)
    _, close = fit(MATURITIES, rates, [1.4], [9.0])
    _, far = fit(MATURITIES, rates, [3.0], [9.0])
    params, rmse = fit(MATURITIES, rates, [3.0, 1.4], [9.0])
    assert params.tau1 == 1.4
    assert rmse == pytest.approx(close) and close < far


def test_pairs_with_tau2_not_above_tau1_are_skipped():
    # These rates come from tau1 = 6 and tau2 = 2, which would fit exactly,
    # but only tau1 = 2 with tau2 = 6 is allowed.
    rates = zero_rate(MATURITIES, SvenssonParams(1.0, 1.0, 2.0, 3.0, 6.0, 2.0))
    params, rmse = fit(MATURITIES, rates, [2.0, 6.0], [2.0, 6.0])
    assert (params.tau1, params.tau2) == (2.0, 6.0)
    assert rmse > 0.01


def test_a_grid_without_a_valid_pair_raises():
    with pytest.raises(ValueError):
        fit(MATURITIES, zero_rate(MATURITIES, TRUE), [5.0, 6.0], [3.0, 5.0])


def test_fit_needs_four_rates():
    with pytest.raises(ValueError):
        fit([1, 2, 3], [2.0, 2.1, 2.2], [1.0], [5.0])


def test_maturities_and_rates_must_match():
    with pytest.raises(ValueError):
        fit([1, 2, 3, 4, 5], [2.0, 2.1, 2.2, 2.3], [1.0], [5.0])
