"""Downloading the ECB's euro area yield curve and reading the files it sends.

The ECB fits two curves every TARGET business day, one to AAA-rated euro area
government bonds and one to all euro area government bonds, and publishes
for each the six Svensson parameters and zero rates at fixed maturities
calculated from them. Series keys look like

    YC.B.U2.EUR.4F.G_N_A.SV_C_YM.BETA0     AAA curve, beta 0
    YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y    AAA curve, 10-year zero rate

G_N_A is the AAA curve and G_N_C the all-bonds curve. The betas and the
zero rates are in percent and the taus in years. The zero rates are
continuously compounded; the ECB calls them spot rates, hence SR_.
"""

from __future__ import annotations

import datetime as dt
import io
import re

import pandas as pd

PARAMS = ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2")
SPOT_RATES = ("SR_3M", "SR_6M", "SR_1Y", "SR_2Y", "SR_3Y", "SR_5Y", "SR_7Y", "SR_10Y", "SR_15Y", "SR_20Y", "SR_25Y", "SR_30Y")
CURVES = {"aaa": "G_N_A", "all": "G_N_C"}

ECB_URL = (
    "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.{curve}.SV_C_YM.{series}"
    "?startPeriod={start}&endPeriod={end}&format=csvdata"
)
USER_AGENT = "euro-curve-risk/0.1"

_SPOT = re.compile(r"^SR_(?:(\d+)Y)?(?:(\d+)M)?$")


def is_spot_rate(code: str) -> bool:
    match = _SPOT.match(code)
    return bool(match and any(match.groups()))


def maturity_of(code: str) -> float:
    """Years to maturity for a zero rate series: SR_3M is 0.25, SR_1Y6M is 1.5, SR_10Y is 10."""
    if not is_spot_rate(code):
        raise ValueError(f"not a zero rate series: {code!r}")
    years, months = (int(g) if g else 0 for g in _SPOT.match(code).groups())
    return years + months / 12


def ecb_url(start: dt.date, end: dt.date, curve: str = "aaa", spot_rates=SPOT_RATES) -> str:
    if curve not in CURVES:
        raise ValueError(f"curve is one of {', '.join(CURVES)}, got {curve!r}")
    series = "+".join((*PARAMS, *spot_rates))
    return ECB_URL.format(curve=CURVES[curve], series=series, start=start, end=end)


def parse_ecb_csv(text: str) -> pd.DataFrame:
    """The ECB Data Portal's CSV output, one row per series per day, turned
    into one row per day: a date column, then the parameters, then the zero
    rates in order of maturity.

    The series name comes from the DATA_TYPE_FM column, or from the end of
    the KEY when that column is missing. Days without all six parameters are
    dropped.
    """
    frame = pd.read_csv(io.StringIO(text.lstrip("\ufeff")))
    frame.columns = [str(c).strip().upper() for c in frame.columns]
    missing = {"TIME_PERIOD", "OBS_VALUE"} - set(frame.columns)
    if missing or not ({"DATA_TYPE_FM", "KEY"} & set(frame.columns)):
        raise ValueError(f"not an ECB yield curve file, missing {sorted(missing) or ['DATA_TYPE_FM or KEY']}")
    if "DATA_TYPE_FM" in frame.columns:
        series = frame["DATA_TYPE_FM"].astype(str).str.strip().str.upper()
    else:
        series = frame["KEY"].astype(str).str.strip().str.upper().str.split(".").str[-1]
    long = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["TIME_PERIOD"]).dt.date,
            "series": series,
            "value": pd.to_numeric(frame["OBS_VALUE"], errors="coerce"),
        }
    ).dropna()
    if long.empty:
        raise ValueError("the file has no observations")
    wide = long.pivot_table(index="date", columns="series", values="value", aggfunc="last")
    absent = [p for p in PARAMS if p not in wide.columns]
    if absent:
        raise ValueError(f"the file has no {', '.join(absent)} series")
    spots = sorted((c for c in wide.columns if is_spot_rate(c)), key=maturity_of)
    wide = wide[list(PARAMS) + spots].dropna(subset=list(PARAMS))
    wide.columns.name = None
    return wide.reset_index().sort_values("date").reset_index(drop=True)


def download_curve(start: dt.date, end: dt.date, curve: str = "aaa") -> pd.DataFrame:
    """The curve's parameters and zero rates from start to end, in the layout of data/sample/curve.csv."""
    import requests

    response = requests.get(ecb_url(start, end, curve), headers={"User-Agent": USER_AGENT}, timeout=60)
    response.raise_for_status()
    return parse_ecb_csv(response.text)
