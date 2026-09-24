"""lda_stability — thin CLI for topic stability across seeds.

One refit per seed, every seed's topics matched back to the reference by
top-word overlap (Jaccard). The summary is the table the stability panel
draws; the matches are the raw comparisons behind it. A topic is only worth
naming ("the economy topic") if it comes back when the model is refit from
a different random start — this tool is how the suite answers that.
"""

from __future__ import annotations

import argparse
import sys

from core.analysis.lda import tokens_from_frame
from core.analysis.topic_stability import topic_stability
from core.conll.schema import Col
from core.io.reader import read_corpus
from core.result import Diagnostic
from tools._cli import build_common_parser, get_pipeline, handle_result_states, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Topic stability across seeds (seeded Gensim LDA)")
    p.add_argument("--topics", type=int, default=3)
    p.add_argument(
        "--seeds",
        default="100,101,102",
        help="comma-separated random seeds; the first is the reference the others are matched against",
    )
    p.add_argument("--top-n", type=int, default=10, help="top words used to match topics")
    p.add_argument("--passes", type=int, default=10, help="training passes per refit")
    p.add_argument(
        "--stable-at",
        dest="stable_at",
        type=float,
        default=0.5,
        help="worst-seed overlap a topic needs to count as Stable",
    )
    p.add_argument("--field", choices=["form", "lemma"], default="lemma")
    p.add_argument("--nouns-only", action="store_true", help="model noun lemmas only (the shared rule)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    field = Col.FORM if args.field == "form" else Col.LEMMA
    try:
        seeds = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
    except ValueError as exc:
        print(f"config error: --seeds must be comma-separated whole numbers: {exc}", file=sys.stderr)
        return 2
    # Comparing seeds is the point: one refit has nothing to compare against.
    _MINIMUM_SEEDS = 2
    if len(seeds) < _MINIMUM_SEEDS:
        print(
            "config error: topic stability compares refits; give at least two seeds "
            "(--seeds 100,101,102; the first is the reference)",
            file=sys.stderr,
        )
        return 2
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
    doc_tokens = tokens_from_frame(table, field=field, nouns_only=args.nouns_only, remove_stopwords=True)
    if not doc_tokens:
        print("[WARNING] TOPIC_NO_DOCUMENTS: no usable tokens; nothing to model", file=sys.stderr)
        return 1
    result = topic_stability(
        doc_tokens,
        n_topics=args.topics,
        seeds=seeds,
        top_n=args.top_n,
        passes=args.passes,
        stable_at=args.stable_at,
    )
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    stability = result.unwrap()
    writer = make_writer(args, corpus, tool="lda_stability", params=vars(args))
    for artifact, name, description in (
        (stability.summary, "summary.csv", "Topic stability summary (one row per reference topic)"),
        (stability.matches, "matches.csv", "Per-seed topic matches against the reference"),
    ):
        written = writer.write_table(artifact, name, kind="table", description=description)
        if not written.ok:
            for d in written.diagnostics:
                print(d, file=sys.stderr)
            return 1
    stable_count = int(stability.summary["Stable"].sum())
    writer.add_diagnostics(
        *corpus_result.diagnostics,
        *table_result.diagnostics,
        *result.diagnostics,
        Diagnostic.info(
            "STABILITY_REFERENCE",
            f"reference seed {stability.reference_seed}; {stable_count} of {len(stability.summary)} topic(s) "
            f"Stable at {stability.stable_at}",
            reference_seed=stability.reference_seed,
            stable=stable_count,
            stable_at=stability.stable_at,
        ),
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {stable_count}/{len(stability.summary)} stable topic(s) to {writer.run_dir / 'summary.csv'}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
