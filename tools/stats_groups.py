"""stats_groups — thin CLI for Mann-Whitney and Kruskal-Wallis (+Dunn)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from core.analysis.stats_groups import kruskal_wallis, mann_whitney
from core.result import Diagnostic
from tools._cli import write_or_die


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nonparametric group comparisons over a CSV file")
    sub = p.add_subparsers(dest="test", required=True)
    m = sub.add_parser("mw", help="Mann-Whitney U (2 groups)")
    m.add_argument("input", type=Path)
    m.add_argument("output", type=Path)
    m.add_argument("--value-col", required=True)
    m.add_argument("--group-col", required=True)
    m.add_argument("--alpha", type=float, default=0.05)
    k = sub.add_parser("kw", help="Kruskal-Wallis (+Dunn post-hoc)")
    k.add_argument("input", type=Path)
    k.add_argument("output", type=Path)
    k.add_argument("--value-col", required=True)
    k.add_argument("--group-col", required=True)
    k.add_argument("--alpha", type=float, default=0.05)
    k.add_argument("--posthoc-method", choices=("bonferroni", "holm"), default="bonferroni")
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

    if args.test == "mw":
        result = mann_whitney(frame, args.value_col, args.group_col, alpha=args.alpha)
        if not result.ok:
            for d in result.diagnostics:
                print(d, file=sys.stderr)
            writer.abandon()
            return 1
        diags = list(result.diagnostics)
        value = result.unwrap()
        write_or_die(
            writer,
            writer.write_table(
                value.summary, "mann_whitney_summary.csv", kind="table", description="Mann-Whitney summary"
            ),
        )
        write_or_die(
            writer,
            writer.write_table(value.medians, "mann_whitney_medians.csv", kind="table", description="Group medians"),
        )
    else:
        kw_result = kruskal_wallis(
            frame, args.value_col, args.group_col, alpha=args.alpha, posthoc_method=args.posthoc_method
        )
        if not kw_result.ok:
            for d in kw_result.diagnostics:
                print(d, file=sys.stderr)
            writer.abandon()
            return 1
        kw_value = kw_result.unwrap()
        diags = list(kw_result.diagnostics)
        write_or_die(
            writer,
            writer.write_table(
                kw_value.summary, "kruskal_wallis_summary.csv", kind="table", description="Kruskal-Wallis summary"
            ),
        )
        if not kw_value.posthoc.empty:
            write_or_die(
                writer,
                writer.write_table(
                    kw_value.posthoc, "kruskal_wallis_posthoc.csv", kind="table", description="Dunn post-hoc"
                ),
            )
        write_or_die(
            writer,
            writer.write_table(
                kw_value.medians, "kruskal_wallis_medians.csv", kind="table", description="Group medians"
            ),
        )
    for diag in diags:
        print(diag, file=sys.stderr)
    writer.add_diagnostics(*diags)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {args.test} results to {writer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
