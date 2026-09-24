"""table_search — thin CLI for CoNLL table search."""

from __future__ import annotations

import argparse
import sys

from core.analysis.table_search import SearchFilter, search
from core.io.reader import read_corpus
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Search a CoNLL table by column predicates")
    p.add_argument("--field", required=True, help="CoNLL column to search (e.g. Form, Lemma, POS)")
    p.add_argument("--value", required=True, help="value to match")
    p.add_argument("--op", choices=["eq", "contains", "starts_with", "ends_with", "regex"], default="eq")
    p.add_argument(
        "--logic",
        choices=["AND", "OR"],
        default="AND",
        help="how to combine multiple --filter (single filter: ignored)",
    )
    p.add_argument("--case-sensitive", action="store_true")
    p.add_argument("--negate", action="store_true", help="negate the predicate")
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

    cache = __import__("core.pipelines.cache", fromlist=["PipelineCache"]).PipelineCache()
    pipeline = get_pipeline(cache, config, args)

    table_result = pipeline.parse(corpus)
    parse_state = handle_result_states(table_result)
    table = table_result.unwrap()

    filt = SearchFilter(
        field=args.field, op=args.op, value=args.value, case_sensitive=args.case_sensitive, negate=args.negate
    )
    result = search(table, [filt], logic=args.logic)
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1

    frame = result.unwrap().frame
    writer = make_writer(args, corpus, tool="table_search", params=vars(args))
    written = writer.write_table(frame, "search_results.csv", kind="table", description="table search results")
    if not written.ok:
        for d in written.diagnostics:
            print(d, file=sys.stderr)
        return 1

    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *result.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(frame)} row(s) to {writer.run_dir / 'search_results.csv'}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
