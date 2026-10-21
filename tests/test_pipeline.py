"""The whole run on the sample: prices, risk, shocks, checks, the report and the command line."""

import datetime as dt
import shutil

import pytest

from eurocurve.checks import fit_check, run_checks
from eurocurve.cli import main
from eurocurve.data import SAMPLE_DIR
from eurocurve.report import render_html, write_outputs


def bond(analysis, bond_id):
    return next(b for b in analysis.bonds if b.position.bond.bond_id == bond_id)


def test_the_run_values_the_last_day(sample_analysis):
    assert sample_analysis.date == dt.date(2026, 9, 30)
    assert len(sample_analysis.bonds) == 10


def test_portfolio_totals(sample_analysis):
    assert sample_analysis.market_value == pytest.approx(238_975_477.13, abs=0.01)
    assert sample_analysis.dv01 == pytest.approx(203_163.05, abs=0.01)
    assert sample_analysis.curve_duration == pytest.approx(8.5014, abs=1e-4)
    assert sample_analysis.weighted("ytm") == pytest.approx(0.029015, abs=1e-6)
    assert sample_analysis.weighted("modified") == pytest.approx(8.2758, abs=1e-4)


def test_one_bond_in_detail(sample_analysis):
    b = bond(sample_analysis, "EG35")
    assert b.clean == pytest.approx(97.225238, abs=1e-6)
    assert b.accrued == pytest.approx(0.327671, abs=1e-6)
    assert b.ytm == pytest.approx(0.0295958, abs=1e-7)
    assert b.modified == pytest.approx(7.768823, abs=1e-6)
    assert b.convexity == pytest.approx(71.9482, abs=1e-4)
    assert b.dv01 == pytest.approx(27_297.00, abs=0.01)


def test_the_zero_coupon_bond(sample_analysis):
    b = bond(sample_analysis, "EG32")
    assert b.accrued == 0.0
    assert b.clean == pytest.approx(86.58176, abs=1e-5)
    assert b.macaulay == pytest.approx(138 / 365 + 5)


def test_market_values_include_accrued_interest(sample_analysis):
    b = bond(sample_analysis, "EG30")
    assert b.market_value == pytest.approx(30_000_000 / 100 * (b.clean + b.accrued))


def test_key_rate_dv01s(sample_analysis):
    expected = [2473.93, 4949.52, 9626.67, 20264.67, 27034.62, 28418.41, 42050.54, 32169.06, 36175.59]
    assert list(sample_analysis.kr_dv01) == pytest.approx(expected, abs=0.01)
    assert sample_analysis.kr_dv01.sum() == pytest.approx(sample_analysis.dv01, rel=1e-6)


@pytest.mark.parametrize(
    "name, pnl, estimate",
    [
        ("parallel up", -39_165_810.93, -45_711_678.85),
        ("parallel down", 54_522_946.28, 45_711_678.85),
        ("steepener", -23_646_539.72, -26_977_483.16),
        ("flattener", 17_197_381.13, 14_964_239.44),
        ("short up", -7_798_588.24, -8_238_407.29),
        ("short down", 8_130_070.77, 8_238_407.29),
    ],
)
def test_rate_shocks(sample_analysis, name, pnl, estimate):
    result = next(s for s in sample_analysis.scenarios if s.name == name)
    assert result.pnl == pytest.approx(pnl, abs=0.01)
    assert result.estimate == pytest.approx(estimate, abs=0.01)


def test_the_worst_shock_is_a_parallel_rise(sample_analysis):
    assert sample_analysis.worst.name == "parallel up"


def test_the_checks(sample_analysis):
    checks = run_checks(sample_analysis)
    assert [c.status for c in checks] == ["ok", "ok", "warning", "warning", "ok", "ok"]
    assert "Jun 17, 2026" in checks[2].detail
    assert "Mar 19, 2026" in checks[3].detail and "+20.4bp" in checks[3].detail


def test_the_refit_is_within_a_tenth_of_a_basis_point(sample_analysis):
    _, params, rmse = fit_check(sample_analysis.inputs.history, sample_analysis.date)
    assert rmse < 0.1
    assert (params.tau1, params.tau2) == (3.0, 15.5)


def test_an_earlier_valuation_date(sample):
    from eurocurve.portfolio import analyze

    earlier = analyze(sample, dt.date(2026, 3, 31))
    assert earlier.date == dt.date(2026, 3, 31)
    assert earlier.market_value > 0


def test_a_date_without_a_curve_raises(sample):
    from eurocurve.portfolio import analyze

    with pytest.raises(ValueError, match="2026-06-16"):
        analyze(sample, dt.date(2026, 6, 17))


def test_the_report_page(sample_analysis):
    page = render_html(sample_analysis, run_checks(sample_analysis), sample=True, run_at=dt.datetime(2026, 10, 1, 9, 0))
    assert page.startswith("<!doctype html>")
    assert "Synthetic sample data" in page
    assert "EUR 238.98m" in page
    assert "EUR 203,163" in page
    assert "Parallel up" in page
    assert "<script src" not in page and "<link" not in page


def test_chart_hover_data_spans_each_chart(sample_analysis):
    import json
    import re

    page = render_html(sample_analysis, run_checks(sample_analysis))
    payloads = [json.loads(m) for m in re.findall(r'<script type="application/json" class="chart-data">(.*?)</script>', page)]
    assert len(payloads) == 4
    for data in payloads:
        n = len(data["labels"])
        last = data["x0"] + data["step"] * (n if data["kind"] == "bars" else n - 1)
        assert 500 < last <= 720
        assert all(len(series["values"]) == n for series in data["series"])


def test_the_outputs(sample_analysis, tmp_path):
    paths = write_outputs(sample_analysis, run_checks(sample_analysis), tmp_path)
    assert paths["report"].name == "rate_risk_2026-09-30.html"
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0
    assert len(paths["bonds"].read_text().splitlines()) == 11


def test_the_command_line(tmp_path, capsys):
    assert main(["run", "--sample", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Market value EUR 238.98m" in out
    assert (tmp_path / "rate_risk_2026-09-30.html").exists()


def test_the_command_line_moves_to_the_last_curve_before_a_missing_day(tmp_path, capsys):
    assert main(["run", "--sample", "--date", "2026-06-17", "--out", str(tmp_path)]) == 0
    assert "using 2026-06-16" in capsys.readouterr().out


def test_warnings_do_not_fail_the_run(tmp_path):
    assert main(["run", "--sample", "--fail-on-checks", "--out", str(tmp_path)]) == 0


def test_the_command_line_needs_inputs(capsys):
    assert main(["run"]) == 1


def test_the_sample_bonds_on_another_curve_file(tmp_path):
    other = tmp_path / "ecb.csv"
    shutil.copy(SAMPLE_DIR / "curve.csv", other)
    assert main(["run", "--sample", "--curve", str(other), "--out", str(tmp_path / "out")]) == 0
    page = (tmp_path / "out" / "rate_risk_2026-09-30.html").read_text()
    assert "Sample bonds." in page and "Curve from ecb.csv" in page
    assert "Synthetic sample data" not in page
