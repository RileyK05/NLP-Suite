"""doc_embeddings — thin CLI: documents (or sentences) by meaning, a map, and semantic search."""

from __future__ import annotations

import argparse
import sys

from core.analysis.doc_embeddings import DEFAULT_MODEL, embed_corpus, semantic_search
from core.io.reader import read_corpus
from core.result import Diagnostic
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Document embeddings: similarity by meaning, neighbours, a map, semantic search")
    p.add_argument("--model", default=DEFAULT_MODEL, help="sentence model (see the Models page)")
    p.add_argument("--unit", choices=["document", "sentence"], default="document")
    p.add_argument("--top-n", type=int, default=5, help="neighbours per document")
    p.add_argument("--query", default="", help="semantic search: sentences closest in meaning to this")
    p.add_argument("--seed", type=int, default=42, help="map and clustering seed")
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
    result = embed_corpus(table, model=args.model, unit=args.unit, top_n=args.top_n, seed=args.seed)
    if result.value is None:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
            fix = d.context.get("fix") if isinstance(d.context, dict) else None
            if fix:
                print(f"fix: {fix}", file=sys.stderr)
        return 1
    tables = result.unwrap()
    writer = make_writer(args, corpus, tool="doc_embeddings", params=vars(args))
    outputs = [
        (tables.vectors, "doc_vectors.csv", "embedding vectors"),
        (tables.map, "doc_map.csv", "2-D map with clusters"),
    ]
    if not tables.pairs.empty:
        outputs.append((tables.pairs, "doc_pairs.csv", "document similarity by meaning"))
        outputs.append((tables.neighbours, "doc_neighbours.csv", "nearest documents"))
    extra: list[Diagnostic] = []
    if args.query.strip():
        found = semantic_search(table, args.query, model=args.model)
        if found.value is None:
            for d in found.diagnostics:
                print(d, file=sys.stderr)
            return 1
        extra.extend(found.diagnostics)
        outputs.append((found.unwrap(), "search_results.csv", "semantic search results"))
    for frame, name, description in outputs:
        written = writer.write_table(frame, name, kind="table", description=description)
        if not written.ok:
            for d in written.diagnostics:
                print(d, file=sys.stderr)
            return 1
    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *result.diagnostics, *extra)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(tables.vectors)} vector(s) to {writer.run_dir}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
