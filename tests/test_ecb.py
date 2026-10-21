"""Reading the ECB's yield curve files. Nothing here goes online."""

import datetime as dt
import pathlib

import pandas as pd
import pytest

from eurocurve.ecb import PARAMS, ecb_url, maturity_of, parse_ecb_csv

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "ecb_yc.csv"


@pytest.mark.parametrize("code, years", [("SR_3M", 0.25), ("SR_1Y", 1.0), ("SR_1Y6M", 1.5), ("SR_30Y", 30.0)])
def test_maturity_of_a_zero_rate_series(code, years):
    assert maturity_of(code) == pytest.approx(years)


@pytest.mark.parametrize("code", ["BETA0", "SR_", "IF_10Y", "SR_10"])
def test_other_series_are_not_zero_rates(code):
    with pytest.raises(ValueError):
        maturity_of(code)


def test_url_for_the_aaa_curve():
    url = ecb_url(dt.date(2025, 9, 30), dt.date(2026, 9, 30))
    assert url.startswith("https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.BETA0+BETA1+")
    assert "+SR_30Y?" in url
    assert "startPeriod=2025-09-30&endPeriod=2026-09-30&format=csvdata" in url


def test_url_for_the_all_bonds_curve():
    assert ".G_N_C.SV_C_YM." in ecb_url(dt.date(2026, 1, 1), dt.date(2026, 2, 1), curve="all")


def test_unknown_curve_raises():
    with pytest.raises(ValueError):
        ecb_url(dt.date(2026, 1, 1), dt.date(2026, 2, 1), curve="AA")


def test_parse_gives_one_row_per_day():
    frame = parse_ecb_csv(FIXTURE.read_text())
    assert list(frame["date"]) == [dt.date(2026, 3, 26), dt.date(2026, 3, 27)]
    assert frame.loc[1, "BETA0"] == pytest.approx(1.15)
    assert frame.loc[1, "TAU2"] == pytest.approx(13.7)


def test_parse_keeps_parameters_then_zero_rates_by_maturity():
    frame = parse_ecb_csv(FIXTURE.read_text())
    assert list(frame.columns) == ["date", *PARAMS, "SR_3M", "SR_1Y6M", "SR_10Y"]


def test_parse_drops_other_series():
    assert "IF_10Y" not in parse_ecb_csv(FIXTURE.read_text()).columns


def test_parse_reads_the_series_from_the_key_when_needed():
    raw = pd.read_csv(FIXTURE).drop(columns=["DATA_TYPE_FM"])
    frame = parse_ecb_csv(raw.to_csv(index=False))
    assert frame.loc[0, "SR_10Y"] == pytest.approx(3.146254)


def test_parse_ignores_a_byte_order_mark():
    assert len(parse_ecb_csv("\ufeff" + FIXTURE.read_text())) == 2


def test_a_file_without_all_six_parameters_raises():
    raw = pd.read_csv(FIXTURE)
    raw = raw[raw["DATA_TYPE_FM"] != "TAU2"]
    with pytest.raises(ValueError, match="TAU2"):
        parse_ecb_csv(raw.to_csv(index=False))


def test_something_that_is_not_an_ecb_file_raises():
    with pytest.raises(ValueError):
        parse_ecb_csv("date,rate\n2026-01-02,2.0\n")
