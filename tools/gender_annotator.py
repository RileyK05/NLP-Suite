"""gender_annotator — thin CLI for name-gender annotation (four dictionaries).

One tool per syllabus question: WHICH dictionary are we grading? Run the same
corpus with --dictionary census / carnegie_mellon / nltk / social_security
and compare the two tables. A missing dictionary fails loudly with its
install pointer; nothing is ever silently substituted for it.
"""

from __future__ import annotations

import argparse
import sys

from core.analysis.gender_annotator import annotate_names, summarize_names
from core.io.reader import read_corpus
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Name-gender annotation (male & female name dictionaries)")
    p.add_argument(
        "--dictionary",
        choices=["census", "carnegie_mellon", "nltk", "social_security"],
        default="nltk",
        help="name dictionary: census, carnegie_mellon, nltk, social_security",
    )
    p.add_argument("--names-dir", type=str, default=None, help="folder holding <dictionary>/male.txt and female.txt")
    p.add_argument(
        "--source",
        choices=["ner", "propn"],
        default="ner",
        help="where names come from: NER PERSON or PROPN tokens",
    )
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
    result = annotate_names(
        table,
        dictionary=args.dictionary,
        names_dir=args.names_dir,
        source=args.source,
    )
    if result.value is None:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    annotated = result.unwrap()
    summary_result = summarize_names(annotated)
    if summary_result.value is None:
        for d in summary_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    summary = summary_result.unwrap()
    writer = make_writer(args, corpus, tool="gender_annotator", params=vars(args))
    for frame, name, description in (
        (annotated, "gender_names.csv", "name-gender annotation per name"),
        (summary, "gender_summary.csv", "name-gender tallies per document"),
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
    print(f"Wrote {len(annotated)} name row(s) to {writer.run_dir / 'gender_names.csv'}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
