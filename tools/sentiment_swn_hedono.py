"""sentiment_swn_hedono — thin CLI for SentiWordNet + hedonometer (full asset)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from core.analysis.sentiment_swn_hedono import hedonometer, sentiwordnet
from core.conll.schema import Col
from core.io.reader import read_corpus
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Sentiment II — SentiWordNet + hedonometer")
    p.add_argument("--field", choices=["form", "lemma"], default="lemma")
    p.add_argument("--hedonometer-lexicon", type=Path, default=None, help="labMT JSON (defaults to registry asset)")
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

    swn = sentiwordnet(table, field=field)
    if not swn.ok:
        for d in swn.diagnostics:
            print(d, file=sys.stderr)
        return 1
    hed = hedonometer(table, field=field, lexicon=args.hedonometer_lexicon)
    if not hed.ok:
        for d in hed.diagnostics:
            print(d, file=sys.stderr)
        return 1

    writer = make_writer(args, corpus, tool="sentiment_swn_hedono", params=vars(args))
    s_frame = swn.unwrap()
    h_frame = hed.unwrap()

    w1 = writer.write_table(s_frame, "swn.csv", kind="table", description="SentiWordNet per-document")
    if not w1.ok:
        for d in w1.diagnostics:
            print(d, file=sys.stderr)
        return 1
    w2 = writer.write_table(h_frame, "hedonometer.csv", kind="table", description="hedonometer per-document")
    if not w2.ok:
        for d in w2.diagnostics:
            print(d, file=sys.stderr)
        return 1

    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *swn.diagnostics, *hed.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(s_frame)} SWN + {len(h_frame)} hedonometer rows to {writer.run_dir}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
