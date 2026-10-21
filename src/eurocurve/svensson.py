"""The Svensson curve.

The ECB publishes its euro area curve as six numbers a day. Four betas set the
level, the slope and two humps, and two taus set where along the curve the
slope and the humps act. From those six numbers come the zero rate, the
instantaneous forward rate and the discount factor at any maturity.

Rates are in percent and continuously compounded, as the ECB publishes them.
Maturities are in years.
"""

from __future__ import annotations

import numpy as np

from .models import SvenssonParams


def loadings(t, tau1: float, tau2: float) -> np.ndarray:
    """The four Svensson loadings at each maturity in t, one row per maturity.

    With x1 = t / tau1 and x2 = t / tau2, the columns are

        level   1
        slope   (1 - exp(-x1)) / x1
        hump 1  (1 - exp(-x1)) / x1 - exp(-x1)
        hump 2  (1 - exp(-x2)) / x2 - exp(-x2)

    At t = 0 they take their limits, 1, 1, 0 and 0. t can be a single number
    or an array, and the result always has shape (number of maturities, 4).

    Raises ValueError for a negative maturity or a tau that is not positive.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(t < 0):
        raise ValueError("maturities can't be negative")
    if tau1 <= 0 or tau2 <= 0:
        raise ValueError("tau1 and tau2 must be positive")

    def slope(tau: float) -> np.ndarray:
        x = t / tau
        out = np.ones_like(x)
        positive = x > 0
        out[positive] = -np.expm1(-x[positive]) / x[positive]
        return out

    s1, s2 = slope(tau1), slope(tau2)
    return np.column_stack([np.ones_like(t), s1, s1 - np.exp(-t / tau1), s2 - np.exp(-t / tau2)])


def zero_rate(t, params: SvenssonParams) -> np.ndarray:
    """Zero-coupon rate at each maturity in t, in percent: the loadings times the betas."""
    return loadings(t, params.tau1, params.tau2) @ np.array(params.betas)


def forward_rate(t, params: SvenssonParams) -> np.ndarray:
    """Instantaneous forward rate at each maturity in t, in percent:

        beta0 + beta1 exp(-x1) + beta2 x1 exp(-x1) + beta3 x2 exp(-x2)

    with x1 = t / tau1 and x2 = t / tau2. At t = 0 it equals the zero rate,
    beta0 + beta1.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(t < 0):
        raise ValueError("maturities can't be negative")
    x1, x2 = t / params.tau1, t / params.tau2
    return params.beta0 + params.beta1 * np.exp(-x1) + params.beta2 * x1 * np.exp(-x1) + params.beta3 * x2 * np.exp(-x2)


def discount_factor(t, rate) -> np.ndarray:
    """The value today of 1 paid at maturity t, exp(-rate / 100 * t), where
    rate is the continuously compounded zero rate in percent. t and rate can
    be numbers or arrays of the same length."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(t < 0):
        raise ValueError("maturities can't be negative")
    return np.exp(-np.asarray(rate, dtype=float) / 100.0 * t)


def par_yield(years: int, params: SvenssonParams) -> float:
    """The annual coupon, in percent, that prices a bond at 100 when it pays
    once a year for `years` whole years:

        100 * (1 - DF(years)) / (DF(1) + DF(2) + ... + DF(years))

    Raises ValueError unless years is a whole number of at least 1.
    """
    if years < 1 or years != int(years):
        raise ValueError("par yields are for whole years from 1 up")
    t = np.arange(1, int(years) + 1, dtype=float)
    df = discount_factor(t, zero_rate(t, params))
    return float(100.0 * (1.0 - df[-1]) / df.sum())


def fit(maturities, rates, tau1_grid, tau2_grid) -> tuple[SvenssonParams, float]:
    """Fit a Svensson curve to a set of zero rates by least squares.

    Once the taus are fixed, the zero rate is a linear function of the betas,
    so for each pair on the grid with tau1 < tau2 the betas come from an
    ordinary least squares regression of the rates on the loadings. The
    pair with the smallest sum of squared errors wins.

    Returns the fitted parameters and the root mean squared error in basis
    points. Raises ValueError when maturities and rates differ in length,
    when there are fewer than four rates, or when the grid has no pair with
    tau1 < tau2.
    """
    t = np.atleast_1d(np.asarray(maturities, dtype=float))
    y = np.atleast_1d(np.asarray(rates, dtype=float))
    if len(t) != len(y):
        raise ValueError("need one rate per maturity")
    if len(t) < 4:
        raise ValueError("need at least four rates to fit four betas")
    best = None
    for tau1 in tau1_grid:
        for tau2 in tau2_grid:
            if tau2 <= tau1:
                continue
            x = loadings(t, tau1, tau2)
            betas = np.linalg.lstsq(x, y, rcond=None)[0]
            sse = float(np.sum((x @ betas - y) ** 2))
            if best is None or sse < best[0]:
                best = (sse, betas, float(tau1), float(tau2))
    if best is None:
        raise ValueError("the grid has no pair with tau1 below tau2")
    sse, betas, tau1, tau2 = best
    params = SvenssonParams(*(float(b) for b in betas), tau1, tau2)
    return params, (sse / len(t)) ** 0.5 * 100.0
