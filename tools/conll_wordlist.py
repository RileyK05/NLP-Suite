"""conll_wordlist — thin CLI for the parameterized word-frequency tool."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from core.analysis.conll_wordlist import run
from core.conll.schema import Col
from core.io.reader import read_corpus
from core.pipelines.cache import PipelineCache
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_common_parser("ConLL word-frequency analysis")
    parser.add_argument("--field", choices=["form", "lemma"], default="form", help="column to count")
    parser.add_argument(
        "--category",
        choices=["all", "noun", "verb", "adjective", "adverb", "function", "noun-verb"],
        default="all",
        help="POS filter",
    )
    parser.add_argument("--top-n", type=int, default=20, help="how many words to keep (0 for all)")
    parser.add_argument("--case-sensitive", action="store_true", help="do not lower-case words")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    top_n = None if args.top_n == 0 else args.top_n
    field = Col.FORM if args.field == "form" else Col.LEMMA

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

    cache = PipelineCache()
    pipeline = get_pipeline(cache, config, args)

    table_result = pipeline.parse(corpus)
    parse_state = handle_result_states(table_result)
    table = table_result.unwrap()

    result = run(table, field=field, category=args.category, top_n=top_n, case_sensitive=args.case_sensitive)
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1

    frame: pd.DataFrame = result.unwrap().to_frame()

    writer = make_writer(args, corpus, tool="conll_wordlist", params=vars(args))

    table_written = writer.write_table(frame, "wordlist.csv", kind="table", description="word frequencies")
    if not table_written.ok:
        for d in table_written.diagnostics:
            print(d, file=sys.stderr)
        return 1

    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *result.diagnostics)
    envelope_result = writer.finalize()
    if not envelope_result.ok:
        for diag in envelope_result.diagnostics:
            print(diag, file=sys.stderr)
        return 1

    envelope = envelope_result.unwrap()
    print(f"Wrote {len(frame)} word(s) to {writer.run_dir / 'wordlist.csv'}")
    if envelope.diagnostics:
        print(f"{len(envelope.diagnostics)} diagnostic(s) recorded:", file=sys.stderr)
        for diag in envelope.diagnostics:
            print(f"  {diag}", file=sys.stderr)
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
