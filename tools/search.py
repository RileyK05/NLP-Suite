"""search — unified CLI for text, CSV-row, and CoNLL-row search with provenance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from tools._cli import handle_result_states, write_or_die


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search TXT corpus, CSV rows, or CoNLL rows")
    sub = p.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("text", help="search lines across a .txt corpus")
    t.add_argument("input", type=Path)
    t.add_argument("output", type=Path)
    t.add_argument("--query", required=True)
    c = sub.add_parser("csv", help="search cells across a CSV file")
    c.add_argument("input", type=Path)
    c.add_argument("output", type=Path)
    c.add_argument("--query", required=True)
    k = sub.add_parser("conll", help="search Form/Lemma rows of a parsed corpus")
    k.add_argument("input", type=Path, help="corpus directory")
    k.add_argument("output", type=Path)
    k.add_argument("--query", required=True)
    k.add_argument("--field", default="Form", help="CoNLL column (default Form)")
    k.add_argument("--parser", choices=["spacy", "stanza"], default="stanza")
    k.add_argument("--language", default="en")
    for s in (t, c, k):
        s.add_argument("--case-sensitive", action="store_true")
        s.add_argument("--regex", action="store_true")
    return p.parse_args(argv)


def _writer(output: Path, tool: str, args: argparse.Namespace, corpus=None):  # type: ignore[no-untyped-def]
    from core.io.writer import OutputWriter

    try:
        return OutputWriter(
            output,
            tool=tool,
            params={k: str(v) for k, v in vars(args).items()},
            inputs=(args.input,),
            corpus=corpus,
        )
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return None


def _text_mode(args: argparse.Namespace) -> int:
    from core.file_ops.search import search_in_text
    from core.io.reader import display_names, read_corpus

    if not args.input.is_dir():
        print(f"not a directory: {args.input}", file=sys.stderr)
        return 2
    corpus_result = read_corpus(args.input)
    if corpus_result.value is None:
        for d in corpus_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    corpus = corpus_result.unwrap()
    names = display_names(corpus.docs)
    rows = []
    for doc in corpus.docs:
        hits = search_in_text(doc.text, args.query, case_sensitive=args.case_sensitive, use_regex=args.regex)
        if not hits.ok:
            for d in hits.diagnostics:
                print(d, file=sys.stderr)
            return 1
        for lineno, line in hits.unwrap():
            rows.append({"Document": names[doc.doc_id], "Line": lineno, "Hit": line})
    writer = _writer(args.output, "search_text", args, corpus)
    if writer is None:
        return 1
    writer.add_diagnostics(*corpus_result.diagnostics)
    write_or_die(
        writer,
        writer.write_table(
            pd.DataFrame(rows, columns=["Document", "Line", "Hit"]),
            "text_hits.csv",
            kind="table",
            description="Text search hits",
        ),
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(rows)} hit(s) to {writer.run_dir}")
    return 0


def _csv_mode(args: argparse.Namespace) -> int:
    from core.file_ops.search import search_csv_frame

    try:
        frame = pd.read_csv(args.input, encoding="utf-8")
    except Exception as exc:
        print(f"could not read {args.input}: {exc}", file=sys.stderr)
        return 2
    result = search_csv_frame(frame, args.query, case_sensitive=args.case_sensitive, use_regex=args.regex)
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    writer = _writer(args.output, "search_csv", args)
    if writer is None:
        return 1
    write_or_die(
        writer,
        writer.write_table(result.unwrap().to_frame(), "csv_hits.csv", kind="table", description="CSV search hits"),
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(result.unwrap().to_frame())} hit(s) to {writer.run_dir}")
    return 0


def _conll_mode(args: argparse.Namespace) -> int:
    from core.analysis.table_search import SearchFilter, search
    from core.io.reader import read_corpus
    from tools._cli import get_pipeline, resolve_config

    try:
        config = resolve_config(args)
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    corpus_result = read_corpus(args.input)
    if corpus_result.value is None:
        for d in corpus_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    corpus = corpus_result.unwrap()
    from core.pipelines.cache import PipelineCache

    pipeline = get_pipeline(PipelineCache(), config)
    table_result = pipeline.parse(corpus)
    parse_state = handle_result_states(table_result)
    filt = SearchFilter(
        field=args.field, op="regex" if args.regex else "contains", value=args.query, case_sensitive=args.case_sensitive
    )
    result = search(table_result.unwrap(), [filt])
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    writer = _writer(args.output, "search_conll", args, corpus)
    if writer is None:
        return 1
    frame = result.unwrap().to_frame()
    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *result.diagnostics)
    write_or_die(writer, writer.write_table(frame, "conll_hits.csv", kind="table", description="CoNLL search hits"))
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(frame)} hit(s) to {writer.run_dir}")
    return 1 if parse_state == "partial" else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode == "text":
        return _text_mode(args)
    if args.mode == "csv":
        return _csv_mode(args)
    return _conll_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
