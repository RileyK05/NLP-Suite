"""knowledge_graph — thin CLI for stub KG."""

from __future__ import annotations

import argparse
import sys

from core.analysis.knowledge_graph import build
from core.conll.schema import Col
from core.io.reader import read_corpus
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Knowledge graph (stub DBpedia/YAGO, or live Spotlight with --source dbpedia)")
    p.add_argument("--field", choices=["form", "lemma"], default="form")
    p.add_argument("--source", default="stub", help="stub (offline) or dbpedia (live Spotlight)")
    p.add_argument("--confidence", type=float, default=0.5, help="Spotlight confidence bound")
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
    if args.source == "dbpedia":
        import pandas as pd

        from core.kg.dbpedia import SpotlightClient

        try:
            client = SpotlightClient(confidence=args.confidence)
        except ValueError as exc:
            print(f"config error: {exc}", file=sys.stderr)
            return 2
        rows: list[dict[str, object]] = []
        for doc in corpus:
            annotated = client.annotate(doc.text)
            if annotated.value is None:
                for d in annotated.diagnostics:
                    print(d, file=sys.stderr)
                return 1
            for entity in annotated.unwrap():
                rows.append(
                    {
                        "Subject": entity["surface"],
                        "Predicate": "annotatedAs",
                        "Object": entity["uri"],
                        "Source": f"dbpedia@{args.confidence}",
                        "Document": doc.path.name,
                    }
                )
        frame = pd.DataFrame(rows, columns=["Subject", "Predicate", "Object", "Source", "Document"])
        writer = make_writer(args, corpus, tool="knowledge_graph", params=vars(args))
        written = writer.write_table(frame, "kg.csv", kind="table", description="knowledge triples")
        if not written.ok:
            for d in written.diagnostics:
                print(d, file=sys.stderr)
            return 1
        writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics)
        env = writer.finalize()
        if not env.ok:
            for d in env.diagnostics:
                print(d, file=sys.stderr)
            return 1
        print(f"Wrote {len(frame)} triple(s) to {writer.run_dir / 'kg.csv'}")
        return 1 if parse_state == "partial" else 0
    result = build(table, field=field, source=args.source)
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    frame = result.unwrap()
    writer = make_writer(args, corpus, tool="knowledge_graph", params=vars(args))
    written = writer.write_table(frame, "kg.csv", kind="table", description="knowledge triples")
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
    print(f"Wrote {len(frame)} triple(s) to {writer.run_dir / 'kg.csv'}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
