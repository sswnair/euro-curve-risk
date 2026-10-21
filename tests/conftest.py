"""Shared fixtures. The path line lets `pytest` run straight from a checkout."""

import datetime as dt
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from eurocurve.models import SvenssonParams  # noqa: E402

TODAY = dt.date(2026, 9, 30)

# An upward-sloping curve: about 2.1% at three months, 3.1% at ten years and
# 3.5% at 30 years, with a hump in the forward rates.
TYPICAL = SvenssonParams(beta0=1.15, beta1=0.68, beta2=2.48, beta3=7.55, tau1=0.88, tau2=13.7)


def flat(rate: float) -> SvenssonParams:
    """A flat curve: the same continuously compounded zero rate, in percent, at every maturity."""
    return SvenssonParams(beta0=rate, beta1=0.0, beta2=0.0, beta3=0.0, tau1=1.0, tau2=5.0)


@pytest.fixture(scope="session")
def sample():
    from eurocurve.data import load_inputs

    return load_inputs()


@pytest.fixture(scope="session")
def sample_analysis(sample):
    from eurocurve.portfolio import analyze

    return analyze(sample)
