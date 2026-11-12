"""valence_anew — thin CLI for ANEW word ratings (valence/arousal/dominance)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from core.analysis.valence_anew import anew
from core.conll.schema import Col
from core.io.reader import read_corpus
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Valence — ANEW word ratings")
    p.add_argument("--anew-field", choices=["form", "lemma"], default="lemma", help="column for ANEW")
    p.add_argument("--anew-lexicon", type=Path, default=None, help="ANEW CSV (defaults to the registry asset)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
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

    a_result = anew(table, field=anew_field, lexicon=args.anew_lexicon)
    if not a_result.ok:
        for d in a_result.diagnostics:
            print(d, file=sys.stderr)
        return 1

    writer = make_writer(args, corpus, tool="valence_anew", params=vars(args))
    a_frame = a_result.unwrap()

    w1 = writer.write_table(a_frame, "anew.csv", kind="table", description="ANEW per-document")
    if not w1.ok:
        for d in w1.diagnostics:
            print(d, file=sys.stderr)
        return 1

    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *a_result.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(a_frame)} ANEW rows to {writer.run_dir}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
