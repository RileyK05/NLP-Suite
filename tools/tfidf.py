"""tfidf — thin CLI for TF-IDF distinctive terms per document."""

from __future__ import annotations

import argparse
import sys

from core.analysis.tfidf import tfidf
from core.conll.schema import Col
from core.io.reader import read_corpus
from tools._cli import build_common_parser, get_pipeline, handle_result_states, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("TF-IDF distinctive terms per document")
    p.add_argument("--field", choices=["form", "lemma"], default="lemma")
    p.add_argument("--top-n", type=int, default=20, help="terms kept per document")
    p.add_argument("--min-df", type=int, default=1, help="drop terms in fewer documents than this")
    p.add_argument(
        "--max-df-ratio",
        type=float,
        default=1.0,
        help="drop terms appearing in more than this fraction of documents (1.0 keeps all)",
    )
    p.add_argument("--min-length", type=int, default=1, help="drop tokens shorter than this")
    p.add_argument("--sublinear-tf", action="store_true", help="use 1 + log(count) instead of the raw count")
    p.add_argument("--no-normalize", action="store_true", help="skip L2 normalization per document")
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
    # is None) stops the tool.
    if corpus_result.value is None:
        for d in corpus_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    for d in corpus_result.diagnostics:
        print(d, file=sys.stderr)
    corpus = corpus_result.unwrap()
    from core.pipelines.cache import PipelineCache

    cache = PipelineCache()
    pipeline = get_pipeline(cache, config, args)
    table_result = pipeline.parse(corpus)
    parse_state = handle_result_states(table_result)
    table = table_result.unwrap()
    result = tfidf(
        table,
        field=field,
        top_n=args.top_n,
        min_df=args.min_df,
        max_df_ratio=args.max_df_ratio,
        min_length=args.min_length,
        sublinear_tf=args.sublinear_tf,
        normalize=not args.no_normalize,
    )
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    frame = result.unwrap()
    writer = make_writer(args, corpus, tool="tfidf", params=vars(args))
    written = writer.write_table(frame, "tfidf.csv", kind="table", description="TF-IDF distinctive terms")
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
    print(f"Wrote {len(frame)} TF-IDF row(s) to {writer.run_dir / 'tfidf.csv'}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
