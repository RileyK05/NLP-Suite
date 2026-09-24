"""word2vec_gensim — thin CLI for Word2Vec vectors and neighbours."""

from __future__ import annotations

import argparse
import sys

from core.analysis.word_embeddings import distances, project_tsne, train, vectors
from core.conll.schema import Col
from core.io.reader import read_corpus
from core.result import Diagnostic
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Word embeddings (seeded Gensim Word2Vec)")
    p.add_argument("--field", choices=["form", "lemma"], default="lemma")
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--vector-size", type=int, default=100)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--sg", type=int, choices=[0, 1], default=1, help="0 = CBOW, 1 = skip-gram")
    p.add_argument("--seed", type=int, default=42, help="training seed (recorded in the envelope)")
    p.add_argument("--remove-stopwords", action="store_true")
    p.add_argument("--query", default=None, help="word to find neighbours for")
    p.add_argument("--top-n", type=int, default=5)
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
    train_args = {
        "field": field,
        "min_count": args.min_count,
        "vector_size": args.vector_size,
        "window": args.window,
        "seed": args.seed,
        "epochs": args.epochs,
        "sg": args.sg,
        "remove_stopwords": args.remove_stopwords,
    }
    t_result = train(table, **train_args)
    if t_result.value is None:
        for d in t_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    model = t_result.unwrap()
    writer = make_writer(args, corpus, tool="word2vec_gensim", params=vars(args))
    vec_result = vectors(table, **train_args)
    if not vec_result.ok:
        for d in vec_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    v_frame = vec_result.unwrap()
    w1 = writer.write_table(v_frame, "vectors.csv", kind="table", description="Word2Vec vectors")
    if not w1.ok:
        for d in w1.diagnostics:
            print(d, file=sys.stderr)
        return 1
    if args.query:
        d_result = distances(table, args.query, top_n=args.top_n, **train_args)
        if not d_result.ok:
            for d in d_result.diagnostics:
                print(d, file=sys.stderr)
            return 1
        n_frame = d_result.unwrap()
        w2 = writer.write_table(n_frame, "neighbours.csv", kind="table", description="nearest neighbours")
        if not w2.ok:
            for d in w2.diagnostics:
                print(d, file=sys.stderr)
            return 1
    tsne_result = project_tsne(model, seed=args.seed)
    if tsne_result.ok and not tsne_result.unwrap().empty:
        w3 = writer.write_table(tsne_result.unwrap(), "tsne.csv", kind="table", description="t-SNE coordinates")
        if not w3.ok:
            for d in w3.diagnostics:
                print(d, file=sys.stderr)
            return 1
        # Legacy parity (word2vec_tsne_plot_util.py): the interactive scatter
        # next to the coordinates. Knowable degradation without plotly.
        from core.viz.embeddings import tsne_html

        plot_result = tsne_html(tsne_result.unwrap(), title=f"t-SNE ({args.field} vectors, seed {args.seed})")
        writer.add_diagnostics(*plot_result.diagnostics)
        if plot_result.value is not None:
            w4 = writer.write_html(plot_result.unwrap(), "tsne.html", kind="chart", description="t-SNE scatter")
            if not w4.ok:
                for d in w4.diagnostics:
                    print(d, file=sys.stderr)
                return 1
    writer.add_diagnostics(
        *corpus_result.diagnostics,
        *table_result.diagnostics,
        *vec_result.diagnostics,
        *tsne_result.diagnostics,
        Diagnostic.info("W2V_SEED", f"Word2Vec seed {model.seed}", seed=model.seed),
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(v_frame)} vector(s) to {writer.run_dir}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
