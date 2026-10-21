"""Reading the sample data folder, and the checks on what is read."""

import datetime as dt
import json
import shutil

import pytest

from eurocurve.data import SAMPLE_DIR, load_curve, load_inputs, load_portfolio, load_settings


def test_the_sample_curve_history(sample):
    history = sample.history
    assert len(history) == 255
    assert history.dates[0] == dt.date(2025, 9, 30)
    assert history.dates[-1] == dt.date(2026, 9, 30)


def test_the_sample_portfolio(sample):
    assert len(sample.positions) == 10
    assert sum(p.nominal for p in sample.positions) == 250_000_000
    assert sample.positions[0].bond.maturity == dt.date(2027, 12, 15)


def test_the_sample_settings(sample):
    settings = sample.settings
    assert settings.key_tenors == (1, 2, 3, 5, 7, 10, 15, 20, 30)
    assert (settings.shock_sizes.parallel, settings.shock_sizes.short, settings.shock_sizes.long) == (225, 350, 200)


def test_parameters_for_a_day(sample):
    params = sample.history.params_on(dt.date(2026, 9, 30))
    assert params.tau1 > 0 and params.tau2 > params.tau1


def test_published_zero_rates_for_a_day(sample):
    rates = sample.history.spot_rates_on(dt.date(2026, 9, 30))
    assert list(rates) == [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30]


def test_the_latest_curve_on_or_before_a_date(sample):
    history = sample.history
    assert history.on_or_before(dt.date(2026, 9, 30)) == dt.date(2026, 9, 30)
    assert history.on_or_before(dt.date(2026, 9, 27)) == dt.date(2026, 9, 25)  # a Sunday
    assert history.on_or_before(dt.date(2026, 6, 17)) == dt.date(2026, 6, 16)  # no curve that day


def test_a_date_before_the_history_raises(sample):
    with pytest.raises(ValueError):
        sample.history.on_or_before(dt.date(2025, 9, 1))


def test_a_day_without_a_curve_raises(sample):
    with pytest.raises(KeyError):
        sample.history.params_on(dt.date(2026, 6, 17))


@pytest.fixture
def folder(tmp_path):
    copy = tmp_path / "data"
    shutil.copytree(SAMPLE_DIR, copy)
    return copy


def test_a_copied_folder_loads(folder):
    assert len(load_inputs(folder).positions) == 10


def test_a_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_inputs(tmp_path / "nowhere")


def _edit_portfolio(folder, old, new):
    path = folder / "portfolio.csv"
    path.write_text(path.read_text().replace(old, new, 1))
    return path


def test_a_bond_listed_twice_raises(folder):
    path = _edit_portfolio(folder, "EG29,", "EG27,")
    with pytest.raises(ValueError, match="twice"):
        load_portfolio(path)


def test_a_negative_coupon_raises(folder):
    path = _edit_portfolio(folder, ",2.00,2027-12-15,", ",-2.00,2027-12-15,")
    with pytest.raises(ValueError, match="coupon"):
        load_portfolio(path)


def test_a_zero_nominal_raises(folder):
    path = _edit_portfolio(folder, ",2027-12-15,20000000", ",2027-12-15,0")
    with pytest.raises(ValueError, match="nominal"):
        load_portfolio(path)


def test_a_curve_file_missing_a_parameter_raises(folder):
    path = folder / "curve.csv"
    lines = path.read_text().splitlines()
    lines = [",".join(cell for i, cell in enumerate(line.split(",")) if i != 6) for line in lines]  # drop TAU2
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="TAU2"):
        load_curve(path)


def test_a_curve_file_with_a_repeated_date_raises(folder):
    path = folder / "curve.csv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines + [lines[-1]]))
    with pytest.raises(ValueError, match="more than one row"):
        load_curve(path)


def test_key_tenors_must_increase(folder):
    path = folder / "settings.json"
    raw = json.loads(path.read_text())
    raw["key_tenors"] = [1, 5, 3]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_settings(path)
