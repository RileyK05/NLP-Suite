"""nominalization — thin CLI for deverbal nominalization detection (FR-2.7)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from core.analysis.nominalization import detect_nominalizations, load_curated_words, sentence_frequency
from core.conll.schema import Col
from core.io.reader import read_corpus
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Deverbal nominalization detection (WordNet derivational morphology)")
    p.add_argument("--field", choices=["form", "lemma"], default="lemma")
    p.add_argument(
        "--check-ending",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply the nominal-suffix prefilter (use --no-check-ending for a pure WordNet pass)",
    )
    p.add_argument("--curated-list", type=Path, default=None, help="one-word-per-row CSV bypassing the suffix filter")
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
    curated: tuple[str, ...] = ()
    if args.curated_list is not None:
        loaded = load_curated_words(args.curated_list)
        if loaded.value is None:
            for d in loaded.diagnostics:
                print(d, file=sys.stderr)
            return 1
        curated = tuple(sorted(loaded.unwrap()))
    check_ending = bool(args.check_ending)
    noms = detect_nominalizations(table, field=field, check_ending=check_ending, curated_words=curated)
    if noms.value is None:
        for d in noms.diagnostics:
            print(d, file=sys.stderr)
        return 1
    sents = sentence_frequency(table, field=field, check_ending=check_ending, curated_words=curated)
    if sents.value is None:
        for d in sents.diagnostics:
            print(d, file=sys.stderr)
        return 1
    writer = make_writer(args, corpus, tool="nominalization", params=vars(args))
    for frame, name, description in (
        (noms.unwrap(), "nominalization.csv", "Nominalized nouns with base verbs"),
        (sents.unwrap(), "nominalization_by_sentence.csv", "Per-sentence nominalization counts"),
    ):
        written = writer.write_table(frame, name, kind="table", description=description)
        if not written.ok:
            for d in written.diagnostics:
                print(d, file=sys.stderr)
            return 1
    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *noms.diagnostics, *sents.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(noms.unwrap())} nominalization(s) to {writer.run_dir / 'nominalization.csv'}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
