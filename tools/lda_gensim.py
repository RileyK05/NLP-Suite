"""lda_gensim — thin CLI for Gensim LDA topics and the Gensim GUI's views.

One of the two LDA workflows the syllabus compares (the other is
``lda_mallet``). Separated because HW2 grades them against each other: one
merged "topic model" tool cannot answer which backend reads a corpus better.

Emits the Gensim GUI's graded views as data: the topic tables, terms at the
requested relevance lambda, and the Intertopic Distance Map (coordinates as
a table, drawn as HTML when plotly is present).
"""

from __future__ import annotations

import argparse
import sys

from core.analysis.lda import fit_lda, tokens_from_frame
from core.conll.schema import Col
from core.io.reader import read_corpus
from core.result import Diagnostic
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Topic models (seeded Gensim LDA)")
    p.add_argument("--topics", type=int, default=3)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--field", choices=["form", "lemma"], default="lemma")
    p.add_argument("--seed", type=int, default=100, help="LDA random seed (recorded in the envelope)")
    p.add_argument("--keep-stopwords", action="store_true", help="do not filter English stopwords")
    p.add_argument("--nouns-only", action="store_true", help="model NOUN/PROPN lemmas only")
    p.add_argument(
        "--lambda",
        dest="lambda",
        type=float,
        default=0.6,
        help="term relevance lambda (1 = topic probability, 0 = distinctiveness)",
    )
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
    doc_tokens = tokens_from_frame(
        table, field=field, nouns_only=args.nouns_only, remove_stopwords=not args.keep_stopwords
    )
    if not doc_tokens:
        print("[WARNING] TOPIC_NO_DOCUMENTS: no usable tokens; nothing to model", file=sys.stderr)
        return 1
    result = fit_lda(
        doc_tokens,
        n_topics=args.topics,
        top_n=args.top_n,
        seed=args.seed,
        relevance_lambda=float(getattr(args, "lambda")),
    )
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    fitted = result.unwrap()
    writer = make_writer(args, corpus, tool="lda_gensim", params=vars(args))
    artifacts = [
        (fitted.topics, "topics.csv", "LDA per-topic top words"),
        (fitted.dominant, "topics_dominant.csv", "LDA dominant topic per document"),
    ]
    if fitted.relevance is not None:
        artifacts.append((fitted.relevance, "terms_by_relevance.csv", f"terms at lambda {args.lam}"))
    if fitted.intertopic is not None:
        artifacts.append((fitted.intertopic, "intertopic_distances.csv", "Intertopic Distance Map placement"))
    for artifact, name, description in artifacts:
        written = writer.write_table(artifact, name, kind="table", description=description)
        if not written.ok:
            for d in written.diagnostics:
                print(d, file=sys.stderr)
            return 1
    if fitted.intertopic is not None:
        from core.viz.embeddings import tsne_html

        renamed = fitted.intertopic.rename(columns={"Topic": "Word"})
        plot = tsne_html(renamed, title="Intertopic Distance Map")
        if plot.value is not None:
            html = writer.write_text(
                plot.unwrap(), "intertopic_map.html", kind="chart", description="Intertopic Distance Map"
            )
            if not html.ok:
                for d in html.diagnostics:
                    print(d, file=sys.stderr)
                return 1
    info = [
        Diagnostic.info("TOPIC_SEED", f"LDA seed {fitted.seed}", seed=fitted.seed),
        Diagnostic.info("TOPIC_PERPLEXITY", f"log-perplexity {fitted.perplexity:.4f}", perplexity=fitted.perplexity),
        Diagnostic.info(
            "TOPIC_LAMBDA", f"relevance lambda {getattr(args, 'lambda')}", lam=float(getattr(args, "lambda"))
        ),
    ]
    if fitted.coherence is not None:
        info.append(
            Diagnostic.info("TOPIC_COHERENCE", f"c_v coherence {fitted.coherence:.4f}", coherence=fitted.coherence)
        )
    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *result.diagnostics, *info)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(fitted.topics)} topic-word row(s) to {writer.run_dir / 'topics.csv'}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
