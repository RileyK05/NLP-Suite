"""bert_topics — thin CLI for embedding-clustered topics (CAP-TOPIC-04)."""

from __future__ import annotations

import argparse
import sys

from core.analysis.bert_topics import bert_topics
from core.io.reader import read_corpus
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("BERT topics: KMeans over document embeddings (model downloads on first use)")
    p.add_argument("--topics", type=int, default=3, help="number of topics (>= 2)")
    p.add_argument("--top-n", type=int, default=5, help="top terms per topic")
    p.add_argument("--seed", type=int, default=42, help="KMeans seed")
    p.add_argument("--model", default="bert-base-uncased", help="Hugging Face model")
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
    result = bert_topics(table, n_topics=args.topics, top_n=args.top_n, seed=args.seed, model=args.model)
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    topics, documents = result.unwrap()
    writer = make_writer(args, corpus, tool="bert_topics", params=vars(args))
    w1 = writer.write_table(topics, "bert_topics.csv", kind="table", description="embedding-clustered topics")
    if not w1.ok:
        for d in w1.diagnostics:
            print(d, file=sys.stderr)
        return 1
    w2 = writer.write_table(documents, "bert_topic_docs.csv", kind="table", description="per-document topic")
    if not w2.ok:
        for d in w2.diagnostics:
            print(d, file=sys.stderr)
        return 1
    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *result.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(topics)} topic row(s) over {len(documents)} document(s) to {writer.run_dir}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
