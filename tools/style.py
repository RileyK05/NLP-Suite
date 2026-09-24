"""style — thin CLI for concreteness + iconicity style analysis (FR-2.8)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from core.analysis.style import concreteness, iconic_words, iconicity
from core.conll.schema import Col
from core.io.reader import read_corpus
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Abstract/concrete + iconicity style analysis (user-supplied norm assets)")
    p.add_argument("--analysis", choices=["concreteness", "iconicity"], default="concreteness")
    p.add_argument("--field", choices=["form", "lemma"], default="lemma")
    p.add_argument("--concreteness-lexicon", type=Path, default=None, help="Brysbaert CSV (defaults to registry asset)")
    p.add_argument("--iconicity-lexicon", type=Path, default=None, help="Winter CSV (defaults to registry asset)")
    p.add_argument("--min-rating", type=float, default=5.0, help="most-iconic word threshold")
    p.add_argument("--max-rating-sd", type=float, default=2.0, help="most-iconic word SD ceiling")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
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
    from core.pipelines.cache import PipelineCache

    cache = PipelineCache()
    pipeline = get_pipeline(cache, config, args)
    table_result = pipeline.parse(corpus)
    parse_state = handle_result_states(table_result)
    table = table_result.unwrap()
    if args.analysis == "concreteness":
        result = concreteness(table, field=field, lexicon=args.concreteness_lexicon)
        if result.value is None:
            for d in result.diagnostics:
                print(d, file=sys.stderr)
            return 1
        frames = [(result.unwrap(), "style_concreteness.csv", "Per-sentence concreteness")]
        extra_diags = result.diagnostics
    else:
        scored = iconicity(table, field=field, lexicon=args.iconicity_lexicon)
        if scored.value is None:
            for d in scored.diagnostics:
                print(d, file=sys.stderr)
            return 1
        words = iconic_words(
            table,
            field=field,
            lexicon=args.iconicity_lexicon,
            min_rating=args.min_rating,
            max_rating_sd=args.max_rating_sd,
        )
        if words.value is None:
            for d in words.diagnostics:
                print(d, file=sys.stderr)
            return 1
        frames = [
            (scored.unwrap(), "style_iconicity.csv", "Per-sentence iconicity"),
            (words.unwrap(), "style_iconic_words.csv", "Most-iconic words"),
        ]
        extra_diags = (*scored.diagnostics, *words.diagnostics)
    writer = make_writer(args, corpus, tool="style", params=vars(args))
    for frame, name, description in frames:
        written = writer.write_table(frame, name, kind="table", description=description)
        if not written.ok:
            for d in written.diagnostics:
                print(d, file=sys.stderr)
            return 1
    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *extra_diags)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {args.analysis} style to {writer.run_dir / frames[0][1]}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
