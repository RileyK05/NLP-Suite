"""data_manipulation — thin CLI for concat/merge/pivot/melt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data manipulation")
    p.add_argument("output", type=Path, help="output root")
    p.add_argument("--concat", nargs="+", type=Path, default=None, help="CSV files to concat vertically")
    p.add_argument("--merge-left", type=Path, default=None)
    p.add_argument("--merge-right", type=Path, default=None)
    p.add_argument("--merge-on", default=None)
    p.add_argument("--merge-how", default="inner", choices=["inner", "left", "right", "outer", "cross"])
    p.add_argument("--pivot", type=Path, default=None, help="CSV to pivot")
    p.add_argument("--pivot-index", default=None)
    p.add_argument("--pivot-columns", default=None)
    p.add_argument("--pivot-values", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from core.data.manipulation import concat, merge, pivot
    from core.io.writer import OutputWriter

    from tools._cli import write_or_die

    try:
        writer = OutputWriter(args.output, tool="data_manipulation", params=vars(args), corpus=None)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1

    wrote = False

    if args.concat:
        frames: list[pd.DataFrame] = []
        for p in args.concat:
            try:
                frames.append(pd.read_csv(p, encoding="utf-8"))
            except Exception as exc:
                print(f"could not read {p}: {exc}", file=sys.stderr)
                return 1
        res = concat(frames, axis=0)
        if not res.ok:
            for d in res.diagnostics:
                print(d, file=sys.stderr)
            return 1
        write_or_die(writer, writer.write_table(res.unwrap(), "concat.csv", kind="table", description="concatenated"))
        wrote = True

    if args.merge_left and args.merge_right and args.merge_on:
        try:
            left = pd.read_csv(args.merge_left, encoding="utf-8")
            right = pd.read_csv(args.merge_right, encoding="utf-8")
        except Exception as exc:
            print(f"read failed: {exc}", file=sys.stderr)
            return 1
        res = merge(left, right, on=args.merge_on, how=args.merge_how)
        if not res.ok:
            for d in res.diagnostics:
                print(d, file=sys.stderr)
            return 1
        write_or_die(writer, writer.write_table(res.unwrap(), "merge.csv", kind="table", description="merged"))
        wrote = True

    if args.pivot and args.pivot_index and args.pivot_columns and args.pivot_values:
        try:
            df = pd.read_csv(args.pivot, encoding="utf-8")
        except Exception as exc:
            print(f"could not read {args.pivot}: {exc}", file=sys.stderr)
            return 1
        res = pivot(df, index=args.pivot_index, columns=args.pivot_columns, values=args.pivot_values)
        if not res.ok:
            for d in res.diagnostics:
                print(d, file=sys.stderr)
            return 1
        write_or_die(writer, writer.write_table(res.unwrap(), "pivot.csv", kind="table", description="pivot"))
        wrote = True

    if not wrote:
        # No op -> write empty marker
        write_or_die(writer, writer.write_text("no operation requested", "README.txt", description="no-op"))

    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote to {writer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
