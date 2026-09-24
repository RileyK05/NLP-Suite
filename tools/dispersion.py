"""dispersion — thin CLI for lexical dispersion measures."""

from __future__ import annotations

import argparse
import sys

from core.analysis.dispersion import PART_MODES, dispersion
from core.conll.schema import Col
from core.viz.dispersion_plot import MAX_TERMS, dispersion_plot
from core.io.reader import read_corpus
from tools._cli import build_common_parser, get_pipeline, handle_result_states, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Lexical dispersion: how evenly each word is spread across the corpus")
    p.add_argument("--field", choices=["form", "lemma"], default="lemma")
    p.add_argument(
        "--parts",
        choices=list(PART_MODES),
        default="document",
        help="split the corpus by document, or into equal chunks (use chunk for a single long text)",
    )
    p.add_argument("--chunks", type=int, default=10, help="number of chunks when --parts chunk")
    p.add_argument("--min-count", type=int, default=5, help="drop terms rarer than this")
    p.add_argument("--min-length", type=int, default=1, help="drop tokens shorter than this")
    p.add_argument("--top-n", type=int, default=500, help="rows kept after sorting by frequency (0 = all)")
    p.add_argument(
        "--plot",
        action="store_true",
        help="also write dispersion_plot.html showing where each word falls in the corpus",
    )
    p.add_argument(
        "--plot-terms",
        default="",
        help=f"comma-separated terms to plot (default: the {MAX_TERMS} most frequent surviving terms)",
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
    result = dispersion(
        table,
        field=field,
        parts=args.parts,
        chunks=args.chunks,
        min_count=args.min_count,
        min_length=args.min_length,
        top_n=args.top_n or None,
    )
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    frame = result.unwrap()
    writer = make_writer(args, corpus, tool="dispersion", params=vars(args))
    written = writer.write_table(frame, "dispersion.csv", kind="table", description="Lexical dispersion measures")
    if not written.ok:
        for d in written.diagnostics:
            print(d, file=sys.stderr)
        return 1
    if args.plot:
        requested = [term.strip() for term in args.plot_terms.split(",") if term.strip()]
        terms = requested or frame["Term"].head(MAX_TERMS).tolist()
        plotted = dispersion_plot(table, terms, field=field, title="Lexical dispersion")
        if plotted.value is None:
            # A failed plot must not discard the table that already succeeded.
            for d in plotted.diagnostics:
                print(d, file=sys.stderr)
            writer.add_diagnostics(*plotted.diagnostics)
        else:
            wrote_plot = writer.write_html(
                plotted.unwrap(),
                "dispersion_plot.html",
                kind="html",
                description="Lexical dispersion plot",
            )
            if not wrote_plot.ok:
                for d in wrote_plot.diagnostics:
                    print(d, file=sys.stderr)
                return 1
            writer.add_diagnostics(*plotted.diagnostics)
    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *result.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(frame)} dispersion row(s) to {writer.run_dir / 'dispersion.csv'}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
