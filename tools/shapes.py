"""shapes — thin CLI for story shape and Sankey."""

from __future__ import annotations

import argparse
import sys

from core.conll.schema import Col
from core.io.reader import read_corpus
from core.viz.shapes import sankey_html, story_shape
from tools._cli import (
    handle_result_states,
    build_common_parser,
    get_pipeline,
    make_writer,
    resolve_config,
    write_or_die,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Shapes and networks")
    p.add_argument("--sankey-source", default=None)
    p.add_argument("--sankey-target", default=None)
    p.add_argument("--sankey-value", default="")
    p.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="cluster count for story-shape KMeans (default: the legacy PCA variance heuristic)",
    )
    p.add_argument("--no-cluster", action="store_true", help="skip story-shape clustering")
    p.add_argument("--seed", type=int, default=42, help="KMeans seed")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = resolve_config(args)
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    corpus_result = read_corpus(args.corpus)
    # A bad document is a diagnostic, not an aborted run: the reader returns
    # the good documents with ERROR diagnostics; only a total failure (value
    # is None) stops the tool. Partial corpora flow into the four-state
    # writer policy, matching tools/_cli.load_corpus.
    if corpus_result.value is None:
        for d in corpus_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    if corpus_result.diagnostics:
        for d in corpus_result.diagnostics:
            print(d, file=sys.stderr)
    corpus = corpus_result.unwrap()
    from core.pipelines.cache import PipelineCache

    cache = PipelineCache()
    pipeline = get_pipeline(cache, config, args)
    table_result = pipeline.parse(corpus)
    parse_state = handle_result_states(table_result)
    table = table_result.unwrap()
    shape = story_shape(table)
    if not shape.ok:
        for d in shape.diagnostics:
            print(d, file=sys.stderr)
        return 1
    writer = make_writer(args, corpus, tool="shapes", params=vars(args))
    s_frame = shape.unwrap()
    w1 = writer.write_table(s_frame, "story_shape.csv", kind="table", description="story shape per sentence")
    if not w1.ok:
        for d in w1.diagnostics:
            print(d, file=sys.stderr)
        return 1
    # Optional Sankey from the shape table
    if args.sankey_source and args.sankey_target:
        sank = sankey_html(
            s_frame,
            source=args.sankey_source,
            target=args.sankey_target,
            value=args.sankey_value,
            title="Sankey",
        )
        if sank.ok:
            write_or_die(writer, writer.write_html(sank.unwrap(), "sankey.html", description="sankey"))
        else:
            for d in sank.diagnostics:
                print(d, file=sys.stderr)
            return 1
    # Story-shape clustering (CAP-VIZ-08): KMeans over resampled shape
    # trajectories, k from the legacy PCA variance heuristic unless given.
    if not args.no_cluster:
        from core.viz.shape_clusters import cluster_shapes

        clustered = cluster_shapes(s_frame, n_clusters=args.clusters, seed=args.seed)
        writer.add_diagnostics(*clustered.diagnostics)
        if clustered.value is not None:
            assignments, effective_k = clustered.unwrap()
            wc = writer.write_table(
                assignments, "shape_clusters.csv", kind="table", description=f"story-shape clusters (k={effective_k})"
            )
            if not wc.ok:
                for d in wc.diagnostics:
                    print(d, file=sys.stderr)
                return 1
        else:
            for d in clustered.diagnostics:
                if d.severity.name == "ERROR":
                    print(d, file=sys.stderr)
                    return 1
    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *shape.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(s_frame)} shape row(s) to {writer.run_dir}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
