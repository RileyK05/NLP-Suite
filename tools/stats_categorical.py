"""stats_categorical — thin CLI for crosstabs, chi-square, and keyness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from core.analysis.stats_categorical import chi_square, crosstab, log_likelihood
from core.result import Diagnostic
from tools._cli import write_or_die


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Association statistics over a CSV file")
    sub = p.add_subparsers(dest="test", required=True)
    c1 = sub.add_parser("chi2", help="chi-square test of independence")
    c1.add_argument("input", type=Path)
    c1.add_argument("output", type=Path)
    c1.add_argument("--col1", required=True)
    c1.add_argument("--col2", required=True)
    c1.add_argument("--alpha", type=float, default=0.05)
    c2 = sub.add_parser("crosstab", help="counts with margins + row percentages")
    c2.add_argument("input", type=Path)
    c2.add_argument("output", type=Path)
    c2.add_argument("--col1", required=True)
    c2.add_argument("--col2", required=True)
    c3 = sub.add_parser("keyness", help="G2 log-likelihood keyness")
    c3.add_argument("input", type=Path)
    c3.add_argument("output", type=Path)
    c3.add_argument("--word-col", required=True)
    c3.add_argument("--freq1", required=True)
    c3.add_argument("--freq2", default=None)
    c3.add_argument("--corpus-col", default=None)
    return p.parse_args(argv)


def _read_input(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception as exc:
        print(f"could not read {path}: {exc}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    frame = _read_input(args.input)
    if frame is None:
        return 2

    from core.io.writer import OutputWriter

    try:
        writer = OutputWriter(
            args.output, tool=f"stats_{args.test}", params=vars(args), inputs=(args.input,), corpus=None
        )
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1

    diags: list[Diagnostic] = []
    if args.test == "chi2":
        chi2_result = chi_square(frame, args.col1, args.col2, alpha=args.alpha)
        if not chi2_result.ok:
            for d in chi2_result.diagnostics:
                print(d, file=sys.stderr)
            writer.abandon()
            return 1
        value = chi2_result.unwrap()
        diags = list(chi2_result.diagnostics)
        write_or_die(
            writer,
            writer.write_table(value.summary, "chi2_summary.csv", kind="table", description="Chi-square summary"),
        )
        write_or_die(
            writer,
            writer.write_table(
                value.residuals, "chi2_residuals.csv", kind="table", description="Standardized residuals"
            ),
        )
        write_or_die(
            writer,
            writer.write_table(value.obs_exp, "chi2_obs_vs_exp.csv", kind="table", description="Observed vs expected"),
        )
    elif args.test == "crosstab":
        cross_result = crosstab(frame, args.col1, args.col2)
        if not cross_result.ok:
            for d in cross_result.diagnostics:
                print(d, file=sys.stderr)
            writer.abandon()
            return 1
        cross_value = cross_result.unwrap()
        diags = list(cross_result.diagnostics)
        write_or_die(
            writer,
            writer.write_table(cross_value.counts, "crosstab_counts.csv", kind="table", description="Crosstab counts"),
        )
        write_or_die(
            writer,
            writer.write_table(
                cross_value.row_pct, "crosstab_row_pct.csv", kind="table", description="Crosstab row percentages"
            ),
        )
    else:
        key_result = log_likelihood(frame, args.word_col, args.freq1, args.freq2, corpus_col=args.corpus_col)
        if not key_result.ok:
            for d in key_result.diagnostics:
                print(d, file=sys.stderr)
            writer.abandon()
            return 1
        key_value = key_result.unwrap()
        diags = list(key_result.diagnostics)
        write_or_die(
            writer, writer.write_table(key_value.to_frame(), "keyness.csv", kind="table", description="Keyness (G2)")
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
