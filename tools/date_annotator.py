"""date_annotator — thin CLI for normalized date extraction (SUTime-style).

Runs over raw document text: no parse, no pipeline. Numeric dates are read
US month/day order (see core.analysis.date_annotator for the ambiguity) and
years outside --min-year..--max-year are skipped and counted, not guessed at.
"""

from __future__ import annotations

import argparse
import sys

from core.analysis.date_annotator import annotate_corpus, summarize_dates
from core.io.reader import display_names, read_corpus
from tools._cli import build_common_parser, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Normalized date extraction (SUTime-style)")
    p.add_argument("--min-year", type=int, default=1000, help="earliest plausible year")
    p.add_argument("--max-year", type=int, default=2100, help="latest plausible year")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = resolve_config(args)
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    _ = config  # the common parser always carries --parser/--language; this tool parses nothing
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
    names = display_names(corpus.docs)
    docs = [(names[doc.doc_id], str(doc.doc_id), doc.text) for doc in corpus.docs]
    result = annotate_corpus(docs, min_year=args.min_year, max_year=args.max_year)
    if result.value is None:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    annotated = result.unwrap()
    summary_result = summarize_dates(annotated)
    if summary_result.value is None:
        for d in summary_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    summary = summary_result.unwrap()
    writer = make_writer(args, corpus, tool="date_annotator", params=vars(args))
    for frame, name, description in (
        (annotated, "dates.csv", "normalized date expressions"),
        (summary, "dates_summary.csv", "date counts and range per document"),
    ):
        written = writer.write_table(frame, name, kind="table", description=description)
        if not written.ok:
            for d in written.diagnostics:
                print(d, file=sys.stderr)
            return 1
    writer.add_diagnostics(*corpus_result.diagnostics, *result.diagnostics, *summary_result.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(annotated)} date row(s) to {writer.run_dir / 'dates.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
