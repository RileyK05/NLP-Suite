"""compare — thin CLI for the FR-1.5 golden comparison harness.

Exit codes: 0 the goldens match, 1 they differ (diff printed), 2 the
harness itself failed (unreadable file, bad config, missing envelope).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.compare import CompareConfig, compare_csv_files, compare_runs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare golden vs actual tables or run directories")
    p.add_argument("expected", type=Path, help="expected (golden) CSV file or run directory")
    p.add_argument("actual", type=Path, help="actual CSV file or run directory")
    p.add_argument("--csv", action="store_true", help="compare two CSV files instead of run directories")
    p.add_argument("--keys", default="", help="comma-separated key columns for row alignment")
    p.add_argument("--order-matters", action="store_true", help="compare rows in stored order")
    p.add_argument("--rtol", type=float, default=1e-6, help="relative numeric tolerance")
    p.add_argument("--atol", type=float, default=1e-9, help="absolute numeric tolerance")
    p.add_argument("--empty-as-missing", action="store_true", help="treat empty string as missing")
    p.add_argument("--ignore", default="", help="comma-separated columns to ignore")
    p.add_argument("--ignore-params", default="", help="run mode: envelope params to ignore")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = CompareConfig(
            key_columns=tuple(k for k in (s.strip() for s in args.keys.split(",")) if k),
            rtol=args.rtol,
            atol=args.atol,
            order_matters=args.order_matters,
            empty_as_missing=args.empty_as_missing,
            ignore_columns=tuple(k for k in (s.strip() for s in args.ignore.split(",")) if k),
        )
    except ValueError as exc:
        print(f"compare config error: {exc}")
        return 2

    if args.csv:
        result = compare_csv_files(args.expected, args.actual, config)
    else:
        ignore_params = tuple(k for k in (s.strip() for s in args.ignore_params.split(",")) if k)
        result = compare_runs(args.expected, args.actual, config, ignore_params=ignore_params)

    if result.value is None:
        for diag in result.diagnostics:
            print(diag)
        return 2
    report = result.unwrap()
    for diag in result.diagnostics:
        print(diag)
    if report.passed:
        print(f"MATCH: {args.expected} == {args.actual}")
        return 0
    print(f"DIFF ({len(report.diff)} row(s), showing up to 20):")
    print(report.diff.head(20).to_string(index=False))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
