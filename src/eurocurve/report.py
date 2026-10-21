"""The rate risk report: one HTML page for the valuation date, plus CSV files
of everything behind it.

The page has no external resources, so it can be emailed or archived as is.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import math
import pathlib

import numpy as np
import pandas as pd

from .checks import Check
from .dates import add_months, add_years
from .portfolio import Analysis
from .svensson import par_yield, zero_rate

TABLE_TENORS = (0.25, 1, 2, 3, 5, 7, 10, 15, 20, 30)
HISTORY_TENORS = (2, 10, 30)
CHART_MATURITIES = np.round(np.arange(0.25, 30.001, 0.25), 2)

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def _minus(text: str) -> str:
    return text.replace("-", "−")


def money(value: float, digits: int = 0) -> str:
    if abs(value) < 0.5 * 10 ** -digits:
        value = 0.0  # no "-0"
    return _minus(f"{value:,.{digits}f}")


def millions(value: float, digits: int = 2) -> str:
    return money(value / 1e6, digits) + "m"


def signed_millions(value: float, digits: int = 2) -> str:
    """+54.52m or −39.17m."""
    if abs(value) < 0.5 * 10 ** (6 - digits):
        return millions(0.0, digits)
    return _minus(f"{value / 1e6:+,.{digits}f}m")


def eur(value: float, digits: int = 0) -> str:
    """EUR 1,234 or −EUR 1,234."""
    sign = "−" if value < 0 and abs(value) >= 0.5 * 10 ** -digits else ""
    return f"{sign}EUR {money(abs(value), digits)}"


def eur_m(value: float) -> str:
    sign = "−" if value < 0 and abs(value) >= 5e3 else ""
    return f"{sign}EUR {millions(abs(value))}"


def pct(value: float, digits: int = 2, sign: bool = False) -> str:
    if abs(value) < 0.5 * 10 ** -(digits + 2):
        value = 0.0
    return _minus(f"{value:+.{digits}%}" if sign else f"{value:.{digits}%}")


def rate(value: float, digits: int = 3) -> str:
    """A rate already in percent."""
    return _minus(f"{value:.{digits}f}%")


def bp(value: float, digits: int = 0) -> str:
    if abs(value) < 0.5 * 10 ** -digits:
        return "0"
    return _minus(f"{value:+.{digits}f}")


def fmt_date(day: dt.date) -> str:
    return f"{day:%b} {day.day}, {day.year}"


def plural(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def tenor(t: float) -> str:
    return f"{round(t * 12)}M" if t < 1 else f"{t:g}Y"


def tenor_long(t: float) -> str:
    if t < 1:
        return plural(round(t * 12), "month", "months")
    return "1 year" if t == 1 else f"{t:g} years"


SCENARIO_LABELS = {
    "parallel up": "Parallel up",
    "parallel down": "Parallel down",
    "steepener": "Steepener",
    "flattener": "Flattener",
    "short up": "Short rates up",
    "short down": "Short rates down",
}

# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------


def icon(status: str, size: int = 14) -> str:
    if status == "ok":
        glyph = '<circle cx="8" cy="8" r="7.5" fill="currentColor"/><path d="M4.6 8.3l2.2 2.2 4.6-4.8" fill="none" style="stroke:var(--surface)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
    elif status == "warning":
        glyph = '<path d="M8 .9l7.3 13.4H.7z" fill="currentColor"/><rect x="7.1" y="5.2" width="1.8" height="5" rx=".9" style="fill:var(--surface)"/><circle cx="8" cy="12.2" r="1" style="fill:var(--surface)"/>'
    else:
        glyph = '<path d="M5.1.6h5.8l4.5 4.5v5.8l-4.5 4.5H5.1L.6 10.9V5.1z" fill="currentColor"/><path d="M5.4 5.4l5.2 5.2M10.6 5.4l-5.2 5.2" style="stroke:var(--surface)" stroke-width="1.8" stroke-linecap="round"/>'
    return f'<svg class="icon icon-{status}" width="{size}" height="{size}" viewBox="0 0 16 16" aria-hidden="true">{glyph}</svg>'


def chip(status: str, label: str) -> str:
    return f'<span class="chip">{icon(status)}<span>{_esc(label)}</span></span>'


def table(headers: list[str], rows: list[list[str]], numeric: set[int] = frozenset(), foot: list[str] | None = None, cls: str = "") -> str:
    def cells(row, tag):
        return "".join(f'<{tag} class="num">{c}</{tag}>' if i in numeric else f"<{tag}>{c}</{tag}>" for i, c in enumerate(row))

    head = cells(headers, "th")
    body = "".join(f"<tr>{cells(r, 'td')}</tr>" for r in rows)
    tfoot = f"<tfoot><tr>{cells(foot, 'td')}</tr></tfoot>" if foot else ""
    return f'<div class="table-wrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody>{tfoot}</table></div>'


def _nice_step(span: float, target: int = 5) -> float:
    raw = span / max(target, 1)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= mult * magnitude:
            return mult * magnitude
    return 10 * magnitude


def _ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    step = _nice_step(hi - lo, target)
    v, out = math.ceil(lo / step - 1e-9) * step, []
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def _month_ticks(dates: list[dt.date], x) -> list[str]:
    """A tick and label where each month starts. The first month is labeled
    too, unless the next month starts too close to fit both."""
    starts = [i for i, day in enumerate(dates) if i == 0 or day.month != dates[i - 1].month]
    if len(starts) > 1 and starts[1] < 8:
        starts = starts[1:]
    parts = []
    for i in starts:
        day = dates[i]
        label = f"{day:%b}" if day.month != 1 else f"{day:%b} {day.year}"
        parts.append(f'<line class="tick" x1="{x(i):.1f}" x2="{x(i):.1f}" y1="0" y2="4"/>')
        parts.append(f'<text class="axis-text" x="{x(i):.1f}" y="18" text-anchor="middle">{label}</text>')
    return parts


def _interactive(svg: str, legend: str, data: dict, caption: str = "") -> str:
    payload = json.dumps(data).replace("</", "<\\/")
    cap = f'<p class="axis-caption">{_esc(caption)}</p>' if caption else ""
    return (
        f'<div class="chart" data-interactive>{legend}<div class="chart-scroll">{svg}{cap}</div>'
        f'<script type="application/json" class="chart-data">{payload}</script><div class="tip" hidden></div></div>'
    )


def _legend(items: list[tuple[str, str]], kind: str = "line") -> str:
    key = "key-line" if kind == "line" else "key"
    return '<div class="legend">' + "".join(f'<span><i class="{key} {cls}"></i>{_esc(label)}</span>' for label, cls in items) + "</div>"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _line_chart(
    series: list[dict],
    n: int,
    x_of_index,
    x_axis: list[str],
    labels: list[str],
    *,
    aria: str,
    step: float,
    x0: float,
    width: int = 720,
    height: int = 270,
    left: int = 48,
    right: int = 118,
    top: int = 12,
    bottom: int = 28,
    caption: str = "",
) -> str:
    """Lines over a shared x axis. Each series is {label, values, cls, end}, where
    end is the label written at the right-hand end of the line."""
    ph = height - top - bottom
    lo = min(min(s["values"]) for s in series)
    hi = max(max(s["values"]) for s in series)
    y_step = _nice_step((hi - lo) or 0.1, 5)
    lo, hi = math.floor(lo / y_step) * y_step, math.ceil(hi / y_step) * y_step

    def y(v):
        return top + (hi - v) / (hi - lo) * ph

    parts = [f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(aria)}">']
    for t in _ticks(lo, hi):
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        parts.append(f'<text class="axis-text" x="{left - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{_minus(f"{t:g}")}%</text>')
    parts.append(f'<line class="baseline" x1="{left}" x2="{width - right}" y1="{top + ph}" y2="{top + ph}"/>')
    parts.append(f'<g transform="translate(0,{top + ph})">' + "".join(x_axis) + "</g>")
    for s in series:
        path = "M" + "L".join(f"{x_of_index(i):.1f},{y(v):.1f}" for i, v in enumerate(s["values"]))
        parts.append(f'<path class="line {s["cls"]}" d="{path}"/>')
    ends = sorted(((y(s["values"][-1]), s) for s in series), key=lambda e: e[0])
    placed: list[float] = []
    xe = x_of_index(n - 1)
    for ypos, s in ends:
        if not s.get("end"):
            continue
        dot_y = ypos
        if placed and ypos - placed[-1] < 14:
            ypos = placed[-1] + 14
        placed.append(ypos)
        parts.append(f'<circle class="dot {s["cls"]}" cx="{xe:.1f}" cy="{dot_y:.1f}" r="3.5"/>')
        if abs(ypos - dot_y) > 2:
            parts.append(f'<path class="leader" d="M{xe + 6:.1f},{dot_y:.1f}L{xe + 14:.1f},{ypos:.1f}"/>')
        parts.append(f'<text class="end-label" x="{xe + 17:.1f}" y="{ypos + 4:.1f}">{_esc(s["end"])}</text>')
    parts.append(f'<line class="xhair" x1="0" x2="0" y1="{top}" y2="{top + ph}" visibility="hidden"/>')
    pw = width - left - right
    parts.append(f'<rect class="overlay" x="{left}" y="{top}" width="{pw}" height="{ph}" tabindex="0" aria-label="Use the arrow keys to step through the chart"/>')
    parts.append("</svg>")
    data = {
        "kind": "line",
        "x0": x0,
        "step": step,
        "labels": labels,
        "series": [{"label": s["label"], "values": s["tips"]} for s in series],
    }
    legend = _legend([(s["label"], s["cls"]) for s in series])
    return _interactive("".join(parts), legend, data, caption)


def _maturity_axis(x) -> list[str]:
    parts = []
    for m in range(0, 31, 5):
        parts.append(f'<line class="tick" x1="{x(m):.1f}" x2="{x(m):.1f}" y1="0" y2="4"/>')
        parts.append(f'<text class="axis-text" x="{x(m):.1f}" y="18" text-anchor="middle">{m}</text>')
    return parts


def curve_chart(analysis: Analysis) -> str:
    """Zero and forward rates on the valuation date."""
    m = CHART_MATURITIES
    width, left, right = 720, 48, 118
    pw = width - left - right

    def xm(t):
        return left + t / 30 * pw

    zero = analysis.curve.zero(m)
    fwd = analysis.curve.forward(m)
    series = [
        {"label": "Zero rate", "values": list(zero), "cls": "s1", "end": f"Zero {rate(zero[-1], 2)}", "tips": [rate(v) for v in zero]},
        {"label": "Instantaneous forward", "values": list(fwd), "cls": "s3", "end": f"Forward {rate(fwd[-1], 2)}", "tips": [rate(v) for v in fwd]},
    ]
    step = 0.25 / 30 * pw
    chart = _line_chart(
        series,
        len(m),
        lambda i: xm(m[i]),
        _maturity_axis(xm),
        [tenor_long(t) for t in m],
        aria=f"Zero and forward rates by maturity on {fmt_date(analysis.date)}",
        step=step,
        x0=xm(m[0]),
        caption="Years to maturity",
    )
    return chart


def comparison_dates(analysis: Analysis) -> list[tuple[str, dt.date]]:
    """The valuation date and the curves one month, three months and a year before it, where the history has them."""
    history, day = analysis.inputs.history, analysis.date
    out = []
    for label, target in (("A year ago", add_years(day, -1)), ("3 months ago", add_months(day, -3)), ("A month ago", add_months(day, -1))):
        if target >= history.dates[0]:
            out.append((label, history.on_or_before(target)))
    out.append(("Now", day))
    return out


def moves_chart(analysis: Analysis) -> str:
    """The zero curve on the valuation date against earlier ones."""
    history = analysis.inputs.history
    m = CHART_MATURITIES
    width, left, right = 720, 48, 24
    pw = width - left - right

    def xm(t):
        return left + t / 30 * pw

    dates = comparison_dates(analysis)
    ramp = ["a0", "a1", "a2", "a3"][-len(dates):]
    series = []
    for (label, day), cls in zip(dates, ramp):
        z = zero_rate(m, history.params_on(day))
        series.append({"label": f"{fmt_date(day)}", "values": list(z), "cls": cls, "end": "", "tips": [rate(v) for v in z]})
    chart = _line_chart(
        series,
        len(m),
        lambda i: xm(m[i]),
        _maturity_axis(xm),
        [tenor_long(t) for t in m],
        aria="The zero curve now and at earlier dates",
        step=0.25 / 30 * pw,
        x0=xm(m[0]),
        right=right,
        caption="Years to maturity",
    )
    return chart


def rate_history(analysis: Analysis, tenors=HISTORY_TENORS) -> pd.DataFrame:
    history = analysis.inputs.history
    days = [d for d in history.dates if d <= analysis.date]
    rows = []
    for day in days:
        z = zero_rate(list(tenors), history.params_on(day))
        rows.append({"date": day, **{tenor(t): float(v) for t, v in zip(tenors, z)}})
    return pd.DataFrame(rows)


def history_chart(analysis: Analysis) -> str:
    frame = rate_history(analysis)
    dates = list(frame["date"])
    n = len(dates)
    width, left, right = 720, 48, 118
    pw = width - left - right

    def x(i):
        return left + i * pw / (n - 1)

    series = []
    for t, cls in zip(HISTORY_TENORS, ["s1", "s2", "s3"]):
        v = list(frame[tenor(t)])
        series.append({"label": f"{t}-year", "values": v, "cls": cls, "end": f"{t}-year {rate(v[-1], 2)}", "tips": [rate(z) for z in v]})
    return _line_chart(
        series,
        n,
        x,
        _month_ticks(dates, x),
        [fmt_date(d) for d in dates],
        aria="2, 10 and 30-year zero rates through the year",
        step=pw / (n - 1),
        x0=left,
    )


def key_rate_chart(analysis: Analysis) -> str:
    values = list(analysis.kr_dv01)
    tenors = analysis.tenors
    total = analysis.dv01
    width, height, left, right, top, bottom = 720, 230, 64, 16, 22, 28
    pw, ph = width - left - right, height - top - bottom
    top_value = max(max(values), 1.0) * 1.12
    step_size = _nice_step(top_value, 5)
    hi = math.ceil(top_value / step_size) * step_size
    lo = min(0.0, min(values))
    n = len(values)
    step = pw / n
    bar = step * 0.62

    def xc(i):
        return left + (i + 0.5) * step

    def y(v):
        return top + (hi - v) / (hi - lo) * ph

    parts = [f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="DV01 by key tenor">']
    for t in _ticks(lo, hi, 5):
        label = f"{money(t / 1000)}k" if t else "0"
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        parts.append(f'<text class="axis-text" x="{left - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{label}</text>')
    for i, v in enumerate(values):
        y0, y1 = y(max(v, 0)), y(min(v, 0))
        parts.append(f'<rect class="bar s1" x="{xc(i) - bar / 2:.1f}" y="{y0:.1f}" width="{bar:.1f}" height="{max(y1 - y0, 0.5):.1f}" rx="2"/>')
        parts.append(f'<text class="bar-label" x="{xc(i):.1f}" y="{y0 - 6:.1f}" text-anchor="middle">{money(v / 1000, 1)}k</text>')
    parts.append(f'<line class="baseline" x1="{left}" x2="{width - right}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    axis = []
    for i, t in enumerate(tenors):
        axis.append(f'<text class="axis-text" x="{xc(i):.1f}" y="18" text-anchor="middle">{tenor(t)}</text>')
    parts.append(f'<g transform="translate(0,{top + ph})">' + "".join(axis) + "</g>")
    parts.append(f'<line class="xhair" x1="0" x2="0" y1="{top}" y2="{top + ph}" visibility="hidden"/>')
    parts.append(f'<rect class="overlay" x="{left}" y="{top}" width="{pw}" height="{ph}" tabindex="0" aria-label="Use the arrow keys to step through the tenors"/>')
    parts.append("</svg>")
    data = {
        "kind": "bars",
        "x0": left,
        "step": step,
        "labels": [tenor_long(t) for t in tenors],
        "series": [
            {"label": "per basis point", "values": [eur(v) for v in values]},
            {"label": "of the DV01", "values": [pct(v / total, 1) for v in values]},
        ],
    }
    return _interactive("".join(parts), "", data)


def pnl_chart(analysis: Analysis) -> str:
    """Full revaluation P&L for each scenario as a bar, with the key-rate estimate as a tick."""
    scenarios = analysis.scenarios
    width, row, left, right, top = 720, 38, 136, 86, 8
    height = top + row * len(scenarios) + 26
    span = max(max(abs(s.pnl), abs(s.estimate)) for s in scenarios) * 1.08
    pw = width - left - right

    def x(v):
        return left + (v + span) / (2 * span) * pw

    parts = [f'<svg class="chart-svg pnl-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Change in market value under each rate shock">']
    for t in _ticks(-span, span, 6):
        parts.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{top + row * len(scenarios)}"/>')
        parts.append(f'<text class="axis-text" x="{x(t):.1f}" y="{top + row * len(scenarios) + 18}" text-anchor="middle">{_minus(f"{t / 1e6:g}")}m</text>')
    for k, s in enumerate(scenarios):
        yc = top + row * k + row / 2
        x0, x1 = sorted((x(0), x(s.pnl)))
        cls = "gain" if s.pnl >= 0 else "loss"
        parts.append(f'<text class="row-label" x="{left - 12}" y="{yc + 4:.1f}" text-anchor="end">{_esc(SCENARIO_LABELS[s.name])}</text>')
        parts.append(f'<rect class="bar {cls}" x="{x0:.1f}" y="{yc - 10:.1f}" width="{max(x1 - x0, 0.5):.1f}" height="20" rx="2"/>')
        parts.append(f'<line class="estimate" x1="{x(s.estimate):.1f}" x2="{x(s.estimate):.1f}" y1="{yc - 14:.1f}" y2="{yc + 14:.1f}"/>')
        parts.append(f'<text class="bar-label" x="{width - 6}" y="{yc + 4:.1f}" text-anchor="end">{signed_millions(s.pnl)}</text>')
    parts.append(f'<line class="baseline" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{top - 4}" y2="{top + row * len(scenarios) + 4}"/>')
    parts.append("</svg>")
    legend = (
        '<div class="legend"><span><i class="key loss"></i>Loss</span><span><i class="key gain"></i>Gain</span>'
        '<span><i class="key-tick"></i>Key-rate estimate</span></div>'
    )
    return f'<div class="chart">{legend}<div class="chart-scroll">{"".join(parts)}</div></div>'


def sparkline(analysis: Analysis, name: str) -> str:
    sizes = analysis.inputs.settings.shock_sizes
    m = np.linspace(0, 30, 61)
    v = analysis.shock(name)(m)
    width, height, pad = 92, 26, 2
    limit = max(sizes.parallel, sizes.short, sizes.long)

    def x(t):
        return pad + t / 30 * (width - 2 * pad)

    def y(b):
        return height / 2 - b / limit * (height / 2 - pad)

    path = "M" + "L".join(f"{x(t):.1f},{y(b):.1f}" for t, b in zip(m, v))
    return (
        f'<svg class="spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-hidden="true">'
        f'<line class="spark-zero" x1="0" x2="{width}" y1="{height / 2}" y2="{height / 2}"/><path class="spark-line" d="{path}"/></svg>'
    )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _curve_table(analysis: Analysis) -> str:
    history = analysis.inputs.history
    params = analysis.curve.params
    earlier = [(label, d) for label, d in comparison_dates(analysis) if label != "Now"][::-1]
    t = np.array(TABLE_TENORS, dtype=float)
    zero = zero_rate(t, params)
    fwd = analysis.curve.forward(t)
    before = {label: zero_rate(t, history.params_on(d)) for label, d in earlier}
    rows = []
    for i, ti in enumerate(t):
        par = rate(par_yield(int(ti), params)) if ti >= 1 and ti == int(ti) else ""
        rows.append(
            [tenor_long(ti), rate(zero[i]), par, rate(fwd[i])]
            + [bp((zero[i] - before[label][i]) * 100) for label, _ in earlier]
        )
    short = {"A month ago": "1M", "3 months ago": "3M", "A year ago": "1Y"}
    headers = ["Maturity", "Zero rate", "Par yield", "Forward"] + [f"{short[label]} change (bp)" for label, _ in earlier]
    return table(headers, rows, numeric=set(range(1, len(headers))), cls="nowrap-first")


def _curve_note(analysis: Analysis) -> str:
    earlier = [(label, d) for label, d in comparison_dates(analysis) if label != "Now"][::-1]
    if not earlier:
        return ""
    dates = [fmt_date(d) for _, d in earlier]
    listed = dates[0] if len(dates) == 1 else ", ".join(dates[:-1]) + " and " + dates[-1]
    return f'<p class="table-note">Changes are measured from the curves of {listed}.</p>'


def _bond_table(analysis: Analysis) -> str:
    rows = []
    for b in analysis.bonds:
        bond = b.position.bond
        rows.append(
            [
                _esc(bond.bond_id),
                f"{bond.coupon:.2f}%",
                fmt_date(bond.maturity),
                millions(b.position.nominal),
                f"{b.clean:,.3f}",
                f"{b.accrued:,.3f}",
                millions(b.market_value),
                pct(b.ytm, 3),
                f"{b.modified:.2f}",
                f"{b.convexity:,.1f}",
                money(b.dv01),
            ]
        )
    foot = [
        "Portfolio",
        "",
        "",
        millions(analysis.nominal),
        "",
        "",
        millions(analysis.market_value),
        pct(analysis.weighted("ytm"), 3),
        f"{analysis.weighted('modified'):.2f}",
        f"{analysis.weighted('convexity'):,.1f}",
        money(analysis.dv01),
    ]
    headers = ["Bond", "Coupon", "Maturity", "Nominal", "Clean price", "Accrued", "Market value", "Yield", "Modified duration", "Convexity", "DV01 (EUR)"]
    return table(headers, rows, numeric={1, *range(3, 11)}, foot=foot, cls="nowrap")


def _key_rate_table(analysis: Analysis) -> str:
    total, mv = analysis.dv01, analysis.market_value
    rows = [
        [tenor_long(t), money(v), pct(v / total, 1), f"{v / (mv * 1e-4):.2f}"]
        for t, v in zip(analysis.tenors, analysis.kr_dv01)
    ]
    foot = ["All tenors", money(analysis.kr_dv01.sum()), pct(analysis.kr_dv01.sum() / total, 1), f"{analysis.kr_dv01.sum() / (mv * 1e-4):.2f}"]
    return table(["Key tenor", "DV01 (EUR)", "Share", "Key-rate duration"], rows, numeric={1, 2, 3}, foot=foot, cls="narrow")


def _key_rate_detail(analysis: Analysis) -> str:
    headers = ["Bond"] + [tenor(t) for t in analysis.tenors] + ["Total"]
    rows = [
        [_esc(b.position.bond.bond_id)] + [money(v) if abs(v) >= 0.5 else "" for v in b.kr_dv01] + [money(b.kr_dv01.sum())]
        for b in analysis.bonds
    ]
    return table(headers, rows, numeric=set(range(1, len(headers))))


def _scenario_table(analysis: Analysis) -> str:
    show = (2, 10, 30)
    rows = []
    for s in analysis.scenarios:
        shock = analysis.shock(s.name)(np.array(show, dtype=float))
        rows.append(
            [
                _esc(SCENARIO_LABELS[s.name]),
                sparkline(analysis, s.name),
                *[bp(v) for v in shock],
                signed_millions(s.pnl),
                pct(s.pnl / analysis.market_value, 1, sign=True),
                signed_millions(s.estimate),
            ]
        )
    headers = [
        "Scenario",
        "Shape, 0 to 30 years",
        "2Y (bp)",
        "10Y (bp)",
        "30Y (bp)",
        "Change in value (EUR)",
        "Of market value",
        "Key-rate estimate (EUR)",
    ]
    return table(headers, rows, numeric={2, 3, 4, 5, 6, 7}, cls="nowrap-first")


def _checks_table(checks: list[Check]) -> str:
    labels = {"ok": "OK", "warning": "Review", "fail": "Fails"}
    rows = [[chip(c.status, labels[c.status]), _esc(c.name), _esc(c.detail)] for c in checks]
    return table(["", "Check", "Detail"], rows)


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

CSS = """
:root{color-scheme:light;--page:#f5f3ed;--surface:#fcfbf8;--ink:#0b1f33;--ink-2:#4a5563;--muted:#6b7280;
--grid:#e6e3da;--axis:#c9c5b9;--rule:#c5a363;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;
--a0:#86b6ef;--a1:#3987e5;--a2:#1c5cab;--a3:#0d366b;--gain:#2a78d6;--loss:#eb6834;
--ok:#1c6350;--warning:#8a6414;--fail:#a1352a;--border:rgba(11,31,51,.10)}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#111417;--surface:#1b1e22;
--ink:#eef0f2;--ink-2:#b9c0c8;--muted:#8a929b;--grid:#2b3036;--axis:#3a4048;--s1:#3987e5;--s2:#d95926;--s3:#199e70;
--a0:#184f95;--a1:#256abf;--a2:#5598e7;--a3:#9ec5f4;--gain:#3987e5;--loss:#d95926;
--ok:#3fae8a;--warning:#e0b04a;--fail:#e06a5c;--border:rgba(255,255,255,.10)}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1000px;margin:0 auto;padding:32px 16px 48px}
.kicker{margin:0;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-2)}
h1{margin:4px 0 6px;font-size:28px;line-height:1.2}
h2{margin:0 0 4px;font-size:19px}
h3{margin:18px 0 4px;font-size:15px}
.meta{margin:0;color:var(--ink-2);font-size:14px}
header{padding-bottom:18px;border-bottom:2px solid var(--rule);margin-bottom:20px}
.banner{background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--rule);padding:10px 14px;border-radius:6px;margin:0 0 20px;font-size:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:24px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile-label{margin:0 0 6px;font-size:13px;color:var(--ink-2)}
.tile-value{margin:0 0 6px;font-size:26px;font-weight:600;line-height:1.15;font-variant-numeric:tabular-nums}
.tile-sub{margin:4px 0 0;font-size:13px;color:var(--ink-2)}
section.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 18px 14px;margin-bottom:16px}
.section-intro{margin:0 0 12px;color:var(--ink-2);font-size:14px;max-width:72ch}
.chip{display:inline-flex;align-items:center;gap:6px;white-space:nowrap;font-size:13px;font-weight:600}
.icon{flex:none}.icon-ok{color:var(--ok)}.icon-warning{color:var(--warning)}.icon-fail{color:var(--fail)}
.table-wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
table.narrow{width:auto;min-width:420px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid);vertical-align:middle}
thead th{font-size:12px;font-weight:600;color:var(--ink-2);border-bottom:1px solid var(--axis);vertical-align:bottom}
tfoot td{font-weight:600;border-top:1px solid var(--axis);border-bottom:none}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
th.num{white-space:normal}
details{margin-top:12px}summary{cursor:pointer;color:var(--ink-2);font-size:14px}
.chart{position:relative;margin:10px 0 8px}
.chart-scroll{overflow-x:auto}
.chart-svg{display:block;width:100%;min-width:620px;height:auto;overflow:visible}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink-2);margin:0 0 6px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.key{display:inline-block;width:12px;height:12px;border-radius:3px}
.key.loss{background:var(--loss)}.key.gain{background:var(--gain)}
.key-line{display:inline-block;width:16px;height:0;border-top:2px solid}
.key-line.s1{border-color:var(--s1)}.key-line.s2{border-color:var(--s2)}.key-line.s3{border-color:var(--s3)}
.key-line.a0{border-color:var(--a0)}.key-line.a1{border-color:var(--a1)}.key-line.a2{border-color:var(--a2)}.key-line.a3{border-color:var(--a3)}
.key-tick{display:inline-block;width:0;height:14px;border-left:2px solid var(--ink)}
.grid{stroke:var(--grid);stroke-width:1}
.baseline,.tick{stroke:var(--axis);stroke-width:1}
.axis-text{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.end-label,.bar-label{fill:var(--ink);font-size:12px;font-variant-numeric:tabular-nums}
.row-label{fill:var(--ink);font-size:13px}
.leader{fill:none;stroke:var(--muted);stroke-width:1}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.line.s1{stroke:var(--s1)}.line.s2{stroke:var(--s2)}.line.s3{stroke:var(--s3)}
.line.a0{stroke:var(--a0)}.line.a1{stroke:var(--a1)}.line.a2{stroke:var(--a2)}.line.a3{stroke:var(--a3)}
.dot{stroke:var(--surface);stroke-width:2}.dot.s1{fill:var(--s1)}.dot.s2{fill:var(--s2)}.dot.s3{fill:var(--s3)}
.dot.a0{fill:var(--a0)}.dot.a1{fill:var(--a1)}.dot.a2{fill:var(--a2)}.dot.a3{fill:var(--a3)}
.bar.s1{fill:var(--s1)}.bar.gain{fill:var(--gain)}.bar.loss{fill:var(--loss)}
.estimate{stroke:var(--ink);stroke-width:2}
.spark{display:block}.spark-zero{stroke:var(--axis);stroke-width:1;stroke-dasharray:2 2}.spark-line{fill:none;stroke:var(--ink-2);stroke-width:1.6}
.xhair{stroke:var(--ink-2);stroke-width:1}
.overlay{fill:transparent;outline:none}.overlay:focus{stroke:var(--axis);stroke-width:1}
.axis-caption{margin:2px 0 0;text-align:center;color:var(--muted);font-size:12px}
.table-note{margin:8px 0 0;color:var(--ink-2);font-size:13px}
.tip{position:absolute;z-index:2;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;
padding:8px 10px;box-shadow:0 4px 14px rgba(0,0,0,.12);font-size:13px;min-width:150px}
.tip-row{display:flex;gap:8px;align-items:baseline}.tip-row strong{font-variant-numeric:tabular-nums}.tip-row span{color:var(--ink-2)}
.tip-head{color:var(--ink-2);margin-bottom:2px}
table.nowrap td{white-space:nowrap}
table.nowrap-first td:first-child{white-space:nowrap}
footer{color:var(--ink-2);font-size:13px;margin-top:22px}footer p{margin:0 0 8px}
@media print{body{background:#fff}.tip{display:none}section.card,.tile{break-inside:avoid}}
@media (max-width:560px){h1{font-size:23px}.tile-value{font-size:22px}th,td{padding:6px 7px}}
"""

JS = """
(function(){
  document.querySelectorAll('.chart[data-interactive]').forEach(function(chart){
    var data = JSON.parse(chart.querySelector('.chart-data').textContent);
    var svg = chart.querySelector('svg'), overlay = chart.querySelector('.overlay'), xhair = chart.querySelector('.xhair');
    var tip = chart.querySelector('.tip'), n = data.labels.length, current = n - 1;
    function xAt(i){ return data.x0 + (data.kind === 'bars' ? i + 0.5 : i) * data.step; }
    function hide(){ xhair.setAttribute('visibility', 'hidden'); tip.hidden = true; }
    function place(i){
      current = Math.max(0, Math.min(n - 1, i));
      var x = xAt(current);
      xhair.setAttribute('x1', x); xhair.setAttribute('x2', x); xhair.setAttribute('visibility', 'visible');
      tip.replaceChildren();
      var head = document.createElement('div'); head.className = 'tip-head'; head.textContent = data.labels[current]; tip.append(head);
      data.series.forEach(function(s){
        var row = document.createElement('div'); row.className = 'tip-row';
        var v = document.createElement('strong'); v.textContent = s.values[current];
        var l = document.createElement('span'); l.textContent = s.label;
        row.append(v, l); tip.append(row);
      });
      tip.hidden = false;
      var pt = svg.createSVGPoint(); pt.x = x; pt.y = 0;
      var s = pt.matrixTransform(svg.getScreenCTM()), b = chart.getBoundingClientRect();
      var left = s.x - b.left + 14;
      if (left + tip.offsetWidth > b.width) left = s.x - b.left - tip.offsetWidth - 14;
      tip.style.left = Math.max(left, 0) + 'px'; tip.style.top = '28px';
    }
    overlay.addEventListener('pointermove', function(e){
      var pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      var p = pt.matrixTransform(svg.getScreenCTM().inverse());
      var i = data.kind === 'bars' ? Math.floor((p.x - data.x0) / data.step) : Math.round((p.x - data.x0) / data.step);
      place(i);
    });
    overlay.addEventListener('pointerleave', hide);
    overlay.addEventListener('focus', function(){ place(current); });
    overlay.addEventListener('blur', hide);
    overlay.addEventListener('keydown', function(e){
      if (e.key === 'ArrowLeft') { place(current - 1); e.preventDefault(); }
      if (e.key === 'ArrowRight') { place(current + 1); e.preventDefault(); }
    });
  });
})();
"""


def _ten_year_tile(analysis: Analysis) -> str:
    history, day = analysis.inputs.history, analysis.date
    now = float(zero_rate(10, analysis.curve.params)[0])
    subs = []
    for label, target in (("a month", add_months(day, -1)), ("a year", add_years(day, -1))):
        if target >= history.dates[0]:
            then = float(zero_rate(10, history.params_on(history.on_or_before(target)))[0])
            subs.append(f"{bp((now - then) * 100)}bp in {label}")
    return (
        '<div class="tile"><p class="tile-label">10-year zero rate</p>'
        f'<p class="tile-value">{rate(now, 2)}</p><p class="tile-sub">{", ".join(subs)}</p></div>'
    )


def render_html(
    analysis: Analysis, checks: list[Check], *, sample: bool = False, sample_curve: bool | None = None, run_at: dt.datetime | None = None
) -> str:
    """The report page. sample marks the made-up bonds, and sample_curve the
    made-up curves, which go with them unless another curve file was used."""
    settings = analysis.inputs.settings
    sizes = settings.shock_sizes
    run_at = run_at or dt.datetime.now()
    worst = analysis.worst
    failing = [c for c in checks if c.status == "fail"]
    looking = [c for c in checks if c.status == "warning"]

    tiles = [
        '<div class="tile"><p class="tile-label">Market value</p>'
        f'<p class="tile-value">{eur_m(analysis.market_value)}</p>'
        f'<p class="tile-sub">{plural(len(analysis.bonds), "bond", "bonds")}, EUR {millions(analysis.nominal)} nominal</p></div>',
        '<div class="tile"><p class="tile-label">DV01</p>'
        f'<p class="tile-value">{eur(analysis.dv01)}</p>'
        f'<p class="tile-sub">Per basis point, a duration of {analysis.curve_duration:.2f} years on the zero curve</p></div>',
        _ten_year_tile(analysis),
        '<div class="tile"><p class="tile-label">Worst rate shock</p>'
        f'<p class="tile-value">{eur_m(worst.pnl)}</p>'
        f'<p class="tile-sub">{_esc(SCENARIO_LABELS[worst.name])}, {pct(worst.pnl / analysis.market_value, 1, sign=True)} of market value</p></div>',
    ]
    sample_curve = sample if sample_curve is None else sample_curve
    if sample and sample_curve:
        banner = (
            '<p class="banner"><strong>Synthetic sample data.</strong> The curves are made up in the ECB\'s format and the '
            "bonds are made up too, so none of these numbers describe a real market or portfolio.</p>"
        )
    elif sample:
        banner = '<p class="banner"><strong>Sample bonds.</strong> The bonds are made up. The curves come from the file given.</p>'
    else:
        banner = ""
    check_note = (
        f"{plural(len(failing), 'check fails', 'checks fail')}" if failing else "All checks pass" if not looking else plural(len(looking), "check to review", "checks to review")
    )
    meta = " · ".join(
        [
            f"Valuation date {fmt_date(analysis.date)}",
            settings.curve_name,
            f"Run on {fmt_date(run_at.date())} at {run_at:%H:%M}",
        ]
    )
    start = analysis.inputs.history.dates[0]
    body = (
        f'<main><header><p class="kicker">Rate risk report</p><h1>{_esc(settings.name)}</h1><p class="meta">{_esc(meta)}</p></header>{banner}'
        f'<div class="tiles">{"".join(tiles)}</div>'
        '<section class="card"><h2>The curve</h2>'
        f'<p class="section-intro">Zero and instantaneous forward rates on {fmt_date(analysis.date)}, continuously compounded. '
        "Par yields are annual coupons that would price a new bond at 100.</p>"
        f"{curve_chart(analysis)}{_curve_table(analysis)}{_curve_note(analysis)}</section>"
        '<section class="card"><h2>How the curve moved</h2>'
        f'<p class="section-intro">The zero curve now against a month, three months and a year ago, and the 2, 10 and 30-year '
        f"zero rates on every day with a curve since {fmt_date(start)}.</p>"
        f"{moves_chart(analysis)}<h3>Zero rates through the year</h3>{history_chart(analysis)}</section>"
        '<section class="card"><h2>The bonds</h2>'
        '<p class="section-intro">Each bond is priced off the curve for settlement on the valuation date. The yield is the '
        "annual yield to maturity that gives the same price. Portfolio yield, duration and convexity are averages weighted "
        "by market value. DV01 is the gain for a 1 basis point fall in every zero rate.</p>"
        f"{_bond_table(analysis)}</section>"
        '<section class="card"><h2>Where the rate risk sits</h2>'
        f'<p class="section-intro">The DV01 split across {len(analysis.tenors)} key tenors. Each tenor\'s figure is the gain '
        "for a 1 basis point fall in a tent-shaped bump centered on it, and the tents add up to a parallel move.</p>"
        f"{key_rate_chart(analysis)}{_key_rate_table(analysis)}"
        f"<details><summary>By bond</summary>{_key_rate_detail(analysis)}</details></section>"
        '<section class="card"><h2>Rate shocks</h2>'
        f'<p class="section-intro">The six standard shocks from the Basel rules for interest rate risk in the banking book, '
        f"with {_esc(settings.shock_sizes_note)}: {sizes.parallel:g}bp parallel, {sizes.short:g}bp for short rates and "
        f"{sizes.long:g}bp for long rates. The shocks move the zero curve and every bond is repriced in full. The key-rate "
        "estimate is the first-order figure from the DV01s above. It leaves out convexity, and between key tenors it "
        "can only follow a shock's shape in straight lines.</p>"
        f"{pnl_chart(analysis)}{_scenario_table(analysis)}</section>"
        f'<section class="card"><h2>Checks</h2><p class="section-intro">{_esc(check_note)}.</p>{_checks_table(checks)}</section>'
        "<footer><p>Zero and forward rates are continuously compounded, in percent, on an actual/365 time axis. Bonds pay "
        "annual coupons, accrue on an actual/actual (ICMA) basis and are redeemed at 100. Prices are per 100 nominal and "
        "market values include accrued interest.</p>"
        "<p>A portfolio project, not a production risk system.</p></footer></main>"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(settings.name)}: rate risk, {fmt_date(analysis.date)}</title><style>{CSS}</style></head>"
        f"<body>{body}<script>{JS}</script></body></html>"
    )


def curve_frame(analysis: Analysis) -> pd.DataFrame:
    m = CHART_MATURITIES
    zero = analysis.curve.zero(m)
    return pd.DataFrame(
        {
            "maturity": m,
            "zero_rate": zero,
            "forward_rate": analysis.curve.forward(m),
            "discount_factor": analysis.curve.discount(m),
            "par_yield": [par_yield(int(t), analysis.curve.params) if t == int(t) else np.nan for t in m],
        }
    )


def write_outputs(
    analysis: Analysis, checks: list[Check], out_dir: pathlib.Path | str, *, sample: bool = False, sample_curve: bool | None = None
) -> dict[str, pathlib.Path]:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = analysis.date.isoformat()
    paths = {
        "report": out / f"rate_risk_{stamp}.html",
        "bonds": out / "bonds.csv",
        "key_rates": out / "key_rate_dv01.csv",
        "scenarios": out / "scenarios.csv",
        "curve": out / f"curve_{stamp}.csv",
        "history": out / "zero_rate_history.csv",
        "checks": out / "checks.csv",
    }
    paths["report"].write_text(render_html(analysis, checks, sample=sample, sample_curve=sample_curve), encoding="utf-8")
    analysis.bond_table().to_csv(paths["bonds"], index=False)
    analysis.key_rate_table().to_csv(paths["key_rates"], index=False)
    analysis.scenario_table().to_csv(paths["scenarios"], index=False)
    curve_frame(analysis).to_csv(paths["curve"], index=False)
    rate_history(analysis, (0.25, 1, 2, 5, 10, 20, 30)).to_csv(paths["history"], index=False)
    pd.DataFrame([vars(c) for c in checks]).to_csv(paths["checks"], index=False)
    return paths
