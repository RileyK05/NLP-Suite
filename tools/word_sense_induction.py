"""word_sense_induction — thin CLI for contextual vectors + WSI."""

from __future__ import annotations

import argparse
import sys

from core.analysis.contextual import contextual_vectors, wsi_senses
from core.conll.schema import Col
from core.io.reader import read_corpus
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Contextual embeddings + WSI")
    p.add_argument("--field", choices=["form", "lemma"], default="form")
    p.add_argument("--model", default="bert-base-uncased", help="Hugging Face model (downloaded on first use)")
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
    cv = contextual_vectors(table, field=field, model=args.model)
    if not cv.ok:
        for d in cv.diagnostics:
            print(d, file=sys.stderr)
        return 1
    wsi = wsi_senses(table, field=Col.LEMMA if field == Col.FORM else Col.FORM, model=args.model)
    if not wsi.ok:
        for d in wsi.diagnostics:
            print(d, file=sys.stderr)
        return 1
    writer = make_writer(args, corpus, tool="word_sense_induction", params=vars(args))
    w1 = writer.write_table(cv.unwrap(), "contextual_vectors.csv", kind="table", description="contextual vectors")
    if not w1.ok:
        for d in w1.diagnostics:
            print(d, file=sys.stderr)
        return 1
    w2 = writer.write_table(wsi.unwrap(), "wsi.csv", kind="table", description="WSI senses")
    if not w2.ok:
        for d in w2.diagnostics:
            print(d, file=sys.stderr)
        return 1
    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *cv.diagnostics, *wsi.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(cv.unwrap())} vectors + {len(wsi.unwrap())} WSI rows to {writer.run_dir}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
