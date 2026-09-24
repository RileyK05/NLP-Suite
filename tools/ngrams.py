"""ngrams — thin CLI for n-grams and collocations."""

from __future__ import annotations

import argparse
import sys

from core.analysis.ngrams import collocation_network, collocations, ngrams
from core.conll.schema import Col
from core.io.reader import read_corpus
from core.result import Diagnostic
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("N-grams and collocations")
    p.add_argument("--n", type=int, default=2, help="n for n-grams 1..5")
    p.add_argument("--field", choices=["form", "lemma"], default="form")
    p.add_argument("--min-count", type=int, default=2, help="min count for collocations")
    p.add_argument("--network", action="store_true", help="also write collocation_network.gexf")
    p.add_argument("--min-pmi", type=float, default=0.0, help="minimum PMI for network edges")
    p.add_argument("--top-edges", type=int, default=200, help="strongest edges kept in the network")
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
    n_result = ngrams(table, n=args.n, field=field)
    if not n_result.ok:
        for d in n_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    c_result = collocations(table, field=field, min_count=args.min_count)
    if not c_result.ok:
        for d in c_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    writer = make_writer(args, corpus, tool="ngrams", params=vars(args))
    n_frame = n_result.unwrap()
    c_frame = c_result.unwrap()
    w1 = writer.write_table(n_frame, "ngrams.csv", kind="table", description=f"{args.n}-grams")
    if not w1.ok:
        for d in w1.diagnostics:
            print(d, file=sys.stderr)
        return 1
    w2 = writer.write_table(c_frame, "collocations.csv", kind="table", description="bigram PMI")
    if not w2.ok:
        for d in w2.diagnostics:
            print(d, file=sys.stderr)
        return 1
    net_diags: tuple[Diagnostic, ...] = ()
    if args.network:
        net = collocation_network(
            table,
            field=field,
            min_count=args.min_count,
            min_pmi=args.min_pmi,
            top_edges=args.top_edges,
        )
        if not net.ok:
            for d in net.diagnostics:
                print(d, file=sys.stderr)
            return 1
        from core.viz.wordcloud_gephi import gephi_gexf

        gex = gephi_gexf(net.unwrap(), source_col="Word 1", target_col="Word 2", weight_col="Weight")
        if not gex.ok:
            for d in gex.diagnostics:
                print(d, file=sys.stderr)
            return 1
        w3 = writer.write_text(
            gex.unwrap(), "collocation_network.gexf", kind="gexf", description="PMI collocation network"
        )
        if not w3.ok:
            for d in w3.diagnostics:
                print(d, file=sys.stderr)
            return 1
        w4 = writer.write_table(net.unwrap(), "collocation_edges.csv", kind="table", description="network edge list")
        if not w4.ok:
            for d in w4.diagnostics:
                print(d, file=sys.stderr)
            return 1
        net_diags = net.diagnostics
    writer.add_diagnostics(
        *corpus_result.diagnostics,
        *table_result.diagnostics,
        *n_result.diagnostics,
        *c_result.diagnostics,
        *net_diags,
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    printed = f"Wrote {len(n_frame)} n-gram(s) and {len(c_frame)} collocation(s) to {writer.run_dir}"
    if args.network:
        printed += " (+ collocation_network.gexf)"
    print(printed)
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
