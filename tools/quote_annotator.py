"""quote_annotator — thin CLI for quote/dialogue extraction + speaker sieve.

The sieve is a lightweight stand-in for quote-attribution research models:
stage 1 reporting-verb cues, stage 2 nearest named person, and a Cue column
saying which one decided. "Is there dialogue?" is the question; the
attribution column is an auditable guess, not a fact.
"""

from __future__ import annotations

import argparse
import sys

from core.analysis.quote_annotator import annotate_quotes, summarize_quotes
from core.io.reader import read_corpus
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Quote/dialogue extraction with a two-stage speaker sieve")
    p.add_argument("--min-length", type=int, default=1, help="fewest words for a span to count as a quote")
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
    result = annotate_quotes(table, min_length=args.min_length)
    if result.value is None:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    annotated = result.unwrap()
    summary_result = summarize_quotes(annotated)
    if summary_result.value is None:
        for d in summary_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    summary = summary_result.unwrap()
    writer = make_writer(args, corpus, tool="quote_annotator", params=vars(args))
    for frame, name, description in (
        (annotated, "quotes.csv", "quote spans with attributed speakers"),
        (summary, "quotes_summary.csv", "dialogue tallies per document"),
    ):
        written = writer.write_table(frame, name, kind="table", description=description)
        if not written.ok:
            for d in written.diagnostics:
                print(d, file=sys.stderr)
            return 1
    writer.add_diagnostics(
        *corpus_result.diagnostics,
        *table_result.diagnostics,
        *result.diagnostics,
        *summary_result.diagnostics,
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(annotated)} quote row(s) to {writer.run_dir / 'quotes.csv'}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
