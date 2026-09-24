"""k_sentences — thin CLI for K-sentence windows."""

from __future__ import annotations

import argparse
import sys

from core.analysis.k_sentences import run
from core.io.reader import read_corpus
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("K-sentence windows (first/last K per document)")
    p.add_argument("--k-first", type=int, default=2, help="K for first sentences")
    p.add_argument("--k-last", type=int, default=2, help="K for last sentences")
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

    result = run(table, k_first=args.k_first, k_last=args.k_last)
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1

    kres = result.unwrap()
    writer = make_writer(args, corpus, tool="k_sentences", params=vars(args))

    w1 = writer.write_table(kres.counts, "k_counts.csv", kind="table", description="K-sentence counts")
    if not w1.ok:
        for d in w1.diagnostics:
            print(d, file=sys.stderr)
        return 1
    w2 = writer.write_table(kres.repetitions, "k_repetitions.csv", kind="table", description="bookend repetitions")
    if not w2.ok:
        for d in w2.diagnostics:
            print(d, file=sys.stderr)
        return 1

    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *result.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote counts {len(kres.counts)} row(s) and {len(kres.repetitions)} repetition(s) to {writer.run_dir}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
