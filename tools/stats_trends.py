"""stats_trends — thin CLI for Mann-Kendall trends and rank correlations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from core.analysis.stats_trends import correlation_summary, mann_kendall
from tools._cli import write_or_die


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trend and rank-correlation statistics over a CSV file")
    sub = p.add_subparsers(dest="test", required=True)
    t = sub.add_parser("trend", help="Mann-Kendall trend test")
    t.add_argument("input", type=Path)
    t.add_argument("output", type=Path)
    t.add_argument("--date-col", required=True)
    t.add_argument("--value-col", required=True)
    r = sub.add_parser("rankcorr", help="Spearman/Kendall correlation summary")
    r.add_argument("input", type=Path)
    r.add_argument("output", type=Path)
    r.add_argument("--col-x", required=True)
    r.add_argument("--col-y", required=True)
    r.add_argument("--method", choices=("spearman", "kendall"), default="spearman")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        frame = pd.read_csv(args.input, encoding="utf-8")
    except Exception as exc:
        print(f"could not read {args.input}: {exc}", file=sys.stderr)
        return 2

    from core.io.writer import OutputWriter

    try:
        writer = OutputWriter(
            args.output, tool=f"stats_{args.test}", params=vars(args), inputs=(args.input,), corpus=None
        )
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1

    if args.test == "trend":
        result = mann_kendall(frame, args.date_col, args.value_col)
        if not result.ok:
            for d in result.diagnostics:
                print(d, file=sys.stderr)
            return 1
        value = result.unwrap()
        write_or_die(
            writer,
            writer.write_table(
                value.summary, "mann_kendall_summary.csv", kind="table", description="Mann-Kendall summary"
            ),
        )
        write_or_die(
            writer,
            writer.write_table(
                value.trend, "mann_kendall_trend.csv", kind="table", description="Values with trend line"
            ),
        )
    else:
        result = correlation_summary(frame, args.col_x, args.col_y, args.method)
        if not result.ok:
            for d in result.diagnostics:
                print(d, file=sys.stderr)
            return 1
        write_or_die(
            writer,
            writer.write_table(
                result.unwrap(), "rank_correlation.csv", kind="table", description="Rank correlation summary"
            ),
        )
    for diag in result.diagnostics:
        print(diag, file=sys.stderr)
    writer.add_diagnostics(*result.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {args.test} results to {writer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
