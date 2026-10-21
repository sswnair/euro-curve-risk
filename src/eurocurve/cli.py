"""Command line interface.

    eurocurve run --sample                       the sample portfolio on the sample curves
    eurocurve run --data path/to/folder          a data folder of your own
    eurocurve download --start 2025-09-30 --end 2026-09-30 --out curve.csv
    eurocurve run --sample --curve curve.csv     the sample bonds on the ECB's own curve

Run any command with --help for its options.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import os
import pathlib
import sys

from . import __version__
from .checks import run_checks
from .data import SAMPLE_DIR, load_inputs
from .ecb import CURVES, download_curve
from .portfolio import analyze
from .report import SCENARIO_LABELS, eur, fmt_date, millions, pct, plural, rate, write_outputs
from .svensson import zero_rate

EXIT_OK, EXIT_ERROR, EXIT_CHECKS = 0, 1, 2


def _date(text: str) -> dt.date:
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"dates look like 2026-09-30, got {text!r}") from None


def cmd_run(args) -> int:
    if not args.sample and not args.data:
        print("Choose the inputs: --sample, or --data with the path to a data folder.", file=sys.stderr)
        return EXIT_ERROR
    inputs = load_inputs(SAMPLE_DIR if args.sample else args.data, args.curve)
    if args.curve:
        inputs.settings = dataclasses.replace(inputs.settings, curve_name=f"Curve from {args.curve.name}")
    day = args.date
    if day is not None and day not in inputs.history.frame.index:
        day = inputs.history.on_or_before(day)
        print(f"No curve for {args.date}, using {day}.")
    analysis = analyze(inputs, day)
    checks = run_checks(analysis)

    print(f"\n{inputs.settings.name} · {fmt_date(analysis.date)} · {plural(len(analysis.bonds), 'bond', 'bonds')}")
    z = zero_rate([2, 10, 30], analysis.curve.params)
    print(f"  Zero rates   2Y {rate(z[0])}   10Y {rate(z[1])}   30Y {rate(z[2])}")
    print(f"  Market value EUR {millions(analysis.market_value)}   yield {pct(analysis.weighted('ytm'), 3)}")
    print(f"  DV01 {eur(analysis.dv01)} per bp   duration {analysis.curve_duration:.2f} years on the zero curve")
    print("  Rate shocks")
    for s in analysis.scenarios:
        sign = "+" if s.pnl > 0 else ""
        print(f"    {SCENARIO_LABELS[s.name]:<17} {sign + eur(s.pnl):>17}  {pct(s.pnl / analysis.market_value, 1, sign=True):>7}")
    flagged = [c for c in checks if c.status != "ok"]
    print("  Checks: all pass" if not flagged else f"  Checks: {len(checks) - len(flagged)} pass, {len(flagged)} to review")
    for c in flagged:
        print(f"    {c.status:<8} {c.name}. {c.detail}")

    paths = write_outputs(analysis, checks, args.out, sample=args.sample, sample_curve=args.sample and not args.curve)
    print(f"\nReport: {paths['report']}")
    print(f"CSV files in {pathlib.Path(args.out)}")
    if args.fail_on_checks and any(c.status == "fail" for c in checks):
        return EXIT_CHECKS
    return EXIT_OK


def cmd_download(args) -> int:
    if args.end < args.start:
        print("The end date is before the start date.", file=sys.stderr)
        return EXIT_ERROR
    frame = download_curve(args.start, args.end, args.rating)
    frame.to_csv(args.out, index=False, float_format="%.6f")
    print(f"{len(frame)} curves from {fmt_date(frame['date'].iloc[0])} to {fmt_date(frame['date'].iloc[-1])} written to {args.out}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eurocurve", description="The euro government yield curve, bond pricing and interest rate risk.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("run", help="price the portfolio off the curve and measure its rate risk")
    p.add_argument("--sample", action="store_true", help="the synthetic sample in data/sample")
    p.add_argument("--data", type=pathlib.Path, help="a folder with curve.csv, portfolio.csv and settings.json")
    p.add_argument("--curve", type=pathlib.Path, help="a curve file to use instead of the folder's curve.csv")
    p.add_argument("--date", type=_date, help="valuation date (default: the last date with a curve)")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("output"), help="folder for the outputs (default: output)")
    p.add_argument("--fail-on-checks", action="store_true", help=f"exit with code {EXIT_CHECKS} if a check fails")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("download", help="download the ECB's curve parameters and zero rates")
    p.add_argument("--start", type=_date, required=True)
    p.add_argument("--end", type=_date, required=True)
    p.add_argument("--rating", choices=sorted(CURVES), default="aaa", help="the curve for AAA-rated bonds (the default) or for all government bonds")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("curve.csv"))
    p.set_defaults(func=cmd_download)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return EXIT_OK
    except OSError as exc:
        if type(exc).__module__.startswith("requests") or isinstance(exc, (ConnectionError, TimeoutError)):
            print(f"\nCould not download the curve: {exc}", file=sys.stderr)
        else:
            print(f"\nError: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
