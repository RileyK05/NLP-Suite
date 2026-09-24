"""csv_stats — thin CLI for CSV statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from core.analysis.csv_stats import correlation, correlation_matrix, describe
from core.io.reader import read_corpus
from tools._cli import handle_result_states, get_pipeline, resolve_config, write_or_die


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Reuse common parser but replace corpus with generic input
    p = argparse.ArgumentParser(description="CSV statistics (describe + correlation)")
    p.add_argument("input", type=Path, help="input: CSV file or corpus directory of .txt")
    p.add_argument("output", type=Path, help="output root directory")
    p.add_argument("--group-by", default=None, help="column to group by")
    p.add_argument("--corr-x", default=None, help="first column for correlation")
    p.add_argument("--corr-y", default=None, help="second column for correlation")
    p.add_argument(
        "--corr-method",
        choices=["pearson", "spearman", "kendall"],
        default="pearson",
        help="correlation used for the pairwise coefficient and the full matrix",
    )
    p.add_argument("--parser", choices=["spacy", "stanza"], default="stanza")
    p.add_argument("--language", default="en")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Resolve input to DataFrame
    input_path = args.input
    parse_state = "ok"
    parse_diagnostics = ()
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(input_path, encoding="utf-8")
        except Exception as exc:
            print(f"could not read {input_path}: {exc}", file=sys.stderr)
            return 1
        corpus_for_writer = None
    else:
        # Treat as corpus directory
        try:
            config = resolve_config(args)
        except ValueError as exc:
            print(f"config error: {exc}", file=sys.stderr)
            return 2
        corpus_result = read_corpus(input_path)
        if corpus_result.value is None:
            for d in corpus_result.diagnostics:
                print(d, file=sys.stderr)
            return 1
        corpus_for_writer = corpus_result.unwrap()
        from core.pipelines.cache import PipelineCache

        cache = PipelineCache()
        pipeline = get_pipeline(cache, config, args)
        table_result = pipeline.parse(corpus_for_writer)
        parse_state = handle_result_states(table_result)
        parse_diagnostics = table_result.diagnostics
        df = table_result.unwrap()

    result = describe(df, group_by=args.group_by)
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1

    from core.io.writer import OutputWriter

    writer_params = {**vars(args), "input": str(input_path)}
    raw_inputs = [input_path] if corpus_for_writer is None else []
    try:
        writer = OutputWriter(
            args.output,
            tool="csv_stats",
            params=writer_params,
            inputs=tuple(raw_inputs),
            corpus=corpus_for_writer,
        )
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1

    frame = result.unwrap().to_frame()
    written = writer.write_table(frame, "csv_stats.csv", kind="table", description="CSV descriptive stats")
    if not written.ok:
        for d in written.diagnostics:
            print(d, file=sys.stderr)
        writer.abandon()
        return 1

    if args.corr_x and args.corr_y:
        corr_result = correlation(df, args.corr_x, args.corr_y)
        if corr_result.ok:
            corr_df = pd.DataFrame(
                [{"Column X": args.corr_x, "Column Y": args.corr_y, "Pearson r": corr_result.unwrap()}]
            )
            write_or_die(
                writer, writer.write_table(corr_df, "correlation.csv", kind="table", description="correlation")
            )
        else:
            for d in corr_result.diagnostics:
                print(d, file=sys.stderr)
            writer.abandon()
            return 1

    # The full matrix, whenever the table has two numeric columns to relate:
    # this is what the correlation heatmap reads. A table with no pair to
    # correlate is not an error -- csv_stats still wrote its describe.
    matrix_result = correlation_matrix(df, method=args.corr_method)
    if matrix_result.ok:
        matrix_written = writer.write_table(
            matrix_result.unwrap(), "correlation_matrix.csv", kind="table", description="all pairwise correlations"
        )
        if not matrix_written.ok:
            for d in matrix_written.diagnostics:
                print(d, file=sys.stderr)
            writer.abandon()
            return 1
    else:
        for d in matrix_result.diagnostics:
            if d.severity.value == "error":
                print(d, file=sys.stderr)
                writer.abandon()
                return 1

    writer.add_diagnostics(*parse_diagnostics, *result.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(frame)} stat row(s) to {writer.run_dir / 'csv_stats.csv'}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
