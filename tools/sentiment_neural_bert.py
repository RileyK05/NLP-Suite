"""sentiment_neural_bert — thin CLI for neural sentiment via a BERT classifier."""

from __future__ import annotations

import argparse
import sys

from core.analysis.sentiment_neural import bert_sentences, summarize_neural
from core.io.reader import read_corpus
from core.result import Diagnostic
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Neural sentiment (Hugging Face BERT classifier)")
    p.add_argument("--model", default="distilbert-base-uncased-finetuned-sst-2-english")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = resolve_config(args)
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    corpus_result = read_corpus(args.corpus)
    if corpus_result.value is None:
        for d in corpus_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    for d in corpus_result.diagnostics:
        print(d, file=sys.stderr)
    corpus = corpus_result.unwrap()
    from core.pipelines.cache import PipelineCache

    pipeline = get_pipeline(PipelineCache(), config, args)
    table_result = pipeline.parse(corpus)
    parse_state = handle_result_states(table_result)
    table = table_result.unwrap()
    result = bert_sentences(table, model=args.model)
    if result.value is None:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
            fix = d.context.get("fix") if isinstance(d.context, dict) else None
            if fix:
                print(f"fix: {fix}", file=sys.stderr)
        return 1
    sentences = result.unwrap()
    summary = summarize_neural(sentences)
    writer = make_writer(args, corpus, tool="sentiment_neural_bert", params=vars(args))
    for frame, name, description in (
        (sentences, "sentiment_sentences.csv", "neural sentiment per sentence"),
        (summary.unwrap(), "sentiment_documents.csv", "neural sentiment per document"),
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
        *summary.diagnostics,
        Diagnostic.info("SENTIMENT_NN_BACKEND", f"backend bert, model {args.model}", backend="bert", model=args.model),
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(sentences)} sentence score(s) to {writer.run_dir}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
