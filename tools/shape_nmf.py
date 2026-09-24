"""shape_nmf — thin CLI for NMF parts of story shapes."""

from __future__ import annotations

import argparse
import sys

from core.analysis.shape_reduction import build_shape_matrix, matrix_frame, nmf_reduce
from core.io.reader import read_corpus
from core.result import Diagnostic
from core.viz.shapes import story_shape
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Story shapes: NMF non-negative shape parts")
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resample", type=int, default=32)
    p.add_argument("--max-iter", type=int, default=200)
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
    writer = make_writer(args, corpus, tool="shape_nmf", params=vars(args))
    shaped = story_shape(table)
    if shaped.value is None:
        writer.add_diagnostics(*shaped.diagnostics)
        writer.finalize()
        for d in shaped.diagnostics:
            print(d, file=sys.stderr)
        return 1
    matrix = build_shape_matrix(shaped.unwrap(), resample=args.resample)
    if matrix.value is None:
        writer.add_diagnostics(*shaped.diagnostics, *matrix.diagnostics)
        writer.finalize()
        for d in matrix.diagnostics:
            print(d, file=sys.stderr)
        return 1
    shape = matrix.unwrap()
    reduced = nmf_reduce(shape, n_components=args.n_components, seed=args.seed, max_iter=args.max_iter)
    if reduced.value is None:
        writer.add_diagnostics(*shaped.diagnostics, *matrix.diagnostics, *reduced.diagnostics)
        writer.finalize()
        for d in reduced.diagnostics:
            print(d, file=sys.stderr)
        return 1
    result = reduced.unwrap()
    for frame, name, description in (
        (matrix_frame(shape), "shape_matrix.csv", "resampled story-shape trajectories"),
        (result.scores, "shape_nmf_scores.csv", "documents on the non-negative parts"),
        (result.loadings, "shape_nmf_loadings.csv", "how each part rides the story path"),
        (result.explained, "shape_nmf_explained.csv", "mass share per part"),
    ):
        written = writer.write_table(frame, name, kind="table", description=description)
        if not written.ok:
            for d in written.diagnostics:
                print(d, file=sys.stderr)
            return 1
    from core.viz.shape_reduction import components_html

    chart = components_html(result, title="Story shapes: NMF parts")
    writer.add_diagnostics(*chart.diagnostics)
    if chart.value is not None:
        written = writer.write_html(
            chart.unwrap(), "shape_nmf_components.html", kind="chart", description="part profiles"
        )
        if not written.ok:
            for d in written.diagnostics:
                print(d, file=sys.stderr)
            return 1
    writer.add_diagnostics(
        *corpus_result.diagnostics,
        *table_result.diagnostics,
        *shaped.diagnostics,
        *matrix.diagnostics,
        *reduced.diagnostics,
        Diagnostic.info("SHAPE_NMF_SEED", f"NMF seed {args.seed}", seed=args.seed),
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(result.scores)} document score(s) to {writer.run_dir}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
