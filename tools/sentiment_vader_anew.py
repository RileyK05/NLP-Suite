"""sentiment_vader_anew — thin CLI for VADER + ANEW (full lexicons)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from core.analysis.sentiment_vader_anew import anew, load_vader_analyzer, vader, vader_sentences
from core.conll.schema import Col
from core.io.reader import read_corpus
from core.result import Result
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Sentiment I — VADER (package) + ANEW (full lexicon)")
    p.add_argument("--analysis", choices=["vader", "both"], default="both")
    p.add_argument("--field", choices=["form", "lemma"], default="form", help="column for VADER")
    p.add_argument("--anew-field", choices=["form", "lemma"], default="lemma", help="column for ANEW")
    p.add_argument("--vader-lexicon", type=Path, default=None, help="legacy-exact VADER lexicon file")
    p.add_argument("--anew-lexicon", type=Path, default=None, help="ANEW CSV (defaults to the registry asset)")
    return p.parse_args(argv)


def _analyzer(args: argparse.Namespace) -> object | None:
    if args.vader_lexicon is None:
        return None
    resolved = load_vader_analyzer(args.vader_lexicon)
    if resolved.value is None:
        for d in resolved.diagnostics:
            print(d, file=sys.stderr)
        raise SystemExit(1)
    return resolved.unwrap()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    field = Col.FORM if args.field == "form" else Col.LEMMA
    anew_field = Col.FORM if args.anew_field == "form" else Col.LEMMA

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
    analyzer = _analyzer(args)

    v_result = vader(table, field=field, analyzer=analyzer)
    if not v_result.ok:
        for d in v_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    s_result = vader_sentences(table, field=field, analyzer=analyzer)
    if not s_result.ok:
        for d in s_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    a_result = (
        anew(table, field=anew_field, lexicon=args.anew_lexicon)
        if args.analysis == "both"
        else Result.success(table.iloc[:0])
    )
    if not a_result.ok:
        for d in a_result.diagnostics:
            print(d, file=sys.stderr)
        return 1

    writer = make_writer(args, corpus, tool="sentiment_vader_anew", params=vars(args))
    v_frame = v_result.unwrap()
    s_frame = s_result.unwrap()
    a_frame = a_result.unwrap()

    w1 = writer.write_table(v_frame, "vader.csv", kind="table", description="VADER per-document")
    if not w1.ok:
        for d in w1.diagnostics:
            print(d, file=sys.stderr)
        return 1
    w1b = writer.write_table(s_frame, "vader_sentences.csv", kind="table", description="VADER per-sentence")
    if not w1b.ok:
        for d in w1b.diagnostics:
            print(d, file=sys.stderr)
        return 1
    if args.analysis == "both":
        w2 = writer.write_table(a_frame, "anew.csv", kind="table", description="ANEW per-document")
        if not w2.ok:
            for d in w2.diagnostics:
                print(d, file=sys.stderr)
            return 1

    writer.add_diagnostics(
        *corpus_result.diagnostics,
        *table_result.diagnostics,
        *v_result.diagnostics,
        *s_result.diagnostics,
        *a_result.diagnostics,
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(v_frame)} VADER + {len(a_frame)} ANEW rows to {writer.run_dir}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
