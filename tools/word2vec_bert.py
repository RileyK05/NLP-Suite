"""word2vec_bert — thin CLI: Word2Vec via BERT vectors and neighbours.

The BERT half of the HW2 comparison (with ``word2vec_gensim``): the same
output tables (``vectors.csv``, ``neighbours.csv``, ``tsne.csv``,
``tsne.html``) over mean-pooled contextual type vectors from a pretrained
transformer. The model downloads on first use; the vectors are READ from your
corpus, not trained on it — which is exactly the question HW2 grades.
"""

from __future__ import annotations

import argparse
import sys

from core.result import Diagnostic
from tools._cli import (
    build_common_parser,
    get_pipeline,
    handle_result_states,
    make_writer,
    resolve_config,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Word2Vec via BERT (mean-pooled contextual type vectors)")
    p.add_argument("--field", choices=["form", "lemma"], default="lemma")
    p.add_argument("--model", default="bert-base-uncased", help="Hugging Face model (downloaded on first use)")
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--query", default="", help="word to find neighbours for")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42, help="t-SNE seed")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = resolve_config(args)
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    from core.io.reader import read_corpus

    corpus_result = read_corpus(args.corpus)
    # A bad document is a diagnostic, not an aborted run (tools/_cli policy).
    if corpus_result.value is None:
        for diag in corpus_result.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    for diag in corpus_result.diagnostics:
        print(diag, file=sys.stderr)
    corpus = corpus_result.unwrap()

    from core.pipelines.cache import PipelineCache

    pipeline = get_pipeline(PipelineCache(), config, args)
    table_result = pipeline.parse(corpus)
    parse_state = handle_result_states(table_result)
    table = table_result.unwrap()

    from core.analysis.word2vec_bert import distances, project_tsne, train_bert, vectors

    train_args = {
        "field": args.field,
        "model": args.model,
        "min_count": args.min_count,
    }
    t_result = train_bert(table, **train_args)
    if t_result.value is None:
        for diag in t_result.diagnostics:
            print(diag, file=sys.stderr)
            fix = diag.context.get("fix") if isinstance(diag.context, dict) else None
            if fix:
                print(f"fix: {fix}", file=sys.stderr)
        return 1
    model = t_result.unwrap()
    writer = make_writer(args, corpus, tool="word2vec_bert", params=vars(args))
    vec_result = vectors(model)
    if not vec_result.ok:
        for diag in vec_result.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    v_frame = vec_result.unwrap()
    w1 = writer.write_table(v_frame, "vectors.csv", kind="table", description="BERT type vectors")
    if not w1.ok:
        for diag in w1.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    if args.query:
        d_result = distances(table, args.query, top_n=args.top_n, **train_args)
        if not d_result.ok:
            for diag in d_result.diagnostics:
                print(diag, file=sys.stderr)
                fix = diag.context.get("fix") if isinstance(diag.context, dict) else None
                if fix:
                    print(f"fix: {fix}", file=sys.stderr)
            return 1
        n_frame = d_result.unwrap()
        w2 = writer.write_table(n_frame, "neighbours.csv", kind="table", description="nearest neighbours")
        if not w2.ok:
            for diag in w2.diagnostics:
                print(diag, file=sys.stderr)
            return 1
    tsne_result = project_tsne(model, seed=args.seed)
    if tsne_result.ok and not tsne_result.unwrap().empty:
        w3 = writer.write_table(tsne_result.unwrap(), "tsne.csv", kind="table", description="t-SNE coordinates")
        if not w3.ok:
            for diag in w3.diagnostics:
                print(diag, file=sys.stderr)
            return 1
        # Legacy parity (word2vec_tsne_plot_util.py): the interactive scatter
        # next to the coordinates. Knowable degradation without plotly.
        from core.viz.embeddings import tsne_html

        plot_result = tsne_html(tsne_result.unwrap(), title=f"t-SNE ({args.field} vectors, seed {args.seed})")
        writer.add_diagnostics(*plot_result.diagnostics)
        if plot_result.value is not None:
            w4 = writer.write_html(plot_result.unwrap(), "tsne.html", kind="chart", description="t-SNE scatter")
            if not w4.ok:
                for diag in w4.diagnostics:
                    print(diag, file=sys.stderr)
                return 1
    writer.add_diagnostics(
        *corpus_result.diagnostics,
        *table_result.diagnostics,
        *vec_result.diagnostics,
        *tsne_result.diagnostics,
        Diagnostic.info("W2V_BERT_MODEL", f"BERT type vectors from {model.model}", model=model.model, seed=args.seed),
    )
    env = writer.finalize()
    if not env.ok:
        for diag in env.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    print(f"Wrote {len(v_frame)} vector(s) to {writer.run_dir}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
