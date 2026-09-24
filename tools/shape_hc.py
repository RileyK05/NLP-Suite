"""shape_hc — thin CLI for hierarchical clustering of story shapes."""

from __future__ import annotations

import argparse
import sys

from core.analysis.shape_reduction import build_shape_matrix, hierarchical_cluster, matrix_frame
from core.io.reader import read_corpus
from core.result import Diagnostic
from core.viz.shapes import story_shape
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Story shapes: hierarchical clustering with a dendrogram")
    p.add_argument("--method", choices=["ward", "average", "complete", "single"], default="ward")
    p.add_argument("--n-clusters", type=int, default=2)
    p.add_argument("--resample", type=int, default=32)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = resolve_config(args)
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    corpus_result = read_corpus(args.corpus)
    if corpus_result.value is None:
        for d in corpus_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    for d in corpus_result.diagnostics:
        print(d, file=sys.stderr)
    corpus = corpus_result.unwrap()
    from core.pipelines.cache import PipelineCache

    pipeline = get_pipeline(PipelineCache(), config, args)
    table_result = pipeline.parse(corpus)
    parse_state = handle_result_states(table_result)
    table = table_result.unwrap()
    shaped = story_shape(table)
    if shaped.value is None:
        for d in shaped.diagnostics:
            print(d, file=sys.stderr)
        return 1
    matrix = build_shape_matrix(shaped.unwrap(), resample=args.resample)
    if matrix.value is None:
        writer = make_writer(args, corpus, tool="shape_hc", params=vars(args))
        writer.add_diagnostics(*shaped.diagnostics, *matrix.diagnostics)
        writer.finalize()
        for d in matrix.diagnostics:
            print(d, file=sys.stderr)
        return 1
    shape = matrix.unwrap()
    clustered = hierarchical_cluster(shape, method=args.method, n_clusters=args.n_clusters)
    writer = make_writer(args, corpus, tool="shape_hc", params=vars(args))
    if clustered.value is None:
        writer.add_diagnostics(*shaped.diagnostics, *matrix.diagnostics, *clustered.diagnostics)
        writer.finalize()
        for d in clustered.diagnostics:
            print(d, file=sys.stderr)
        return 1
    result = clustered.unwrap()
    for frame, name, description in (
        (matrix_frame(shape), "shape_matrix.csv", "resampled story-shape trajectories"),
        (result.assignments, "shape_hc_assignments.csv", "hierarchical cluster assignment per document"),
        (result.merges, "shape_hc_merges.csv", "the linkage merge table behind the tree"),
    ):
        written = writer.write_table(frame, name, kind="table", description=description)
        if not written.ok:
            for d in written.diagnostics:
                print(d, file=sys.stderr)
            return 1
    from core.viz.shape_reduction import dendrogram_html

    chart = dendrogram_html(result, title="Story shapes: hierarchical clustering")
    writer.add_diagnostics(*chart.diagnostics)
    if chart.value is not None:
        written = writer.write_html(chart.unwrap(), "shape_hc_dendrogram.html", kind="chart", description="dendrogram")
        if not written.ok:
            for d in written.diagnostics:
                print(d, file=sys.stderr)
            return 1
    writer.add_diagnostics(
        *corpus_result.diagnostics,
        *table_result.diagnostics,
        *shaped.diagnostics,
        *matrix.diagnostics,
        *clustered.diagnostics,
        Diagnostic.info("SHAPE_HC_METHOD", f"{args.method} linkage, {args.n_clusters} clusters", method=args.method),
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(result.assignments)} document assignment(s) to {writer.run_dir}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
