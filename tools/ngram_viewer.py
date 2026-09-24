"""ngram_viewer — thin CLI for the culturomics frequency-over-time viewer (FR-1.6).

Counts queried 1-3 word phrases per year (years come from the filenames the
corpus reader parsed at intake) and writes the series plus a line chart. When
no document carries a date the run fails with the answer to that question: a time
series needs time, which is why the undated co-occurrence viewer runs and
this one cannot.
"""

from __future__ import annotations

import argparse
import sys

from core.analysis.ngram_viewer import frame_tokens, ngram_series
from core.io.reader import read_corpus
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("N-gram viewer: phrase frequency per year (culturomics)")
    p.add_argument("--queries", required=True, help="comma-separated words or phrases to track")
    p.add_argument("--case-sensitive", action="store_true", help="distinguish capitalized forms")
    p.add_argument("--smooth", type=int, default=1, help="moving-average window in years")
    return p.parse_args(argv)


def _queries(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


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
    doc_tokens = frame_tokens(table)
    dates = {str(doc.doc_id): (doc.date.year if doc.date is not None else None) for doc in corpus.docs}
    result = ngram_series(
        doc_tokens,
        dates,
        _queries(args.queries),
        case_sensitive=args.case_sensitive,
        smooth=args.smooth,
    )
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
            fix = d.context.get("fix") if isinstance(d.context, dict) else None
            if fix:
                print(f"fix: {fix}", file=sys.stderr)
        return 1
    series = result.unwrap()
    from core.viz.ngram_viewer import ngram_viewer_html

    chart = ngram_viewer_html(series)
    if chart.value is None:
        for d in chart.diagnostics:
            print(d, file=sys.stderr)
        return 1
    writer = make_writer(args, corpus, tool="ngram_viewer", params=vars(args))
    written = writer.write_table(series, "ngram_series.csv", kind="table", description="N-gram frequency per year")
    if not written.ok:
        for d in written.diagnostics:
            print(d, file=sys.stderr)
        return 1
    html = writer.write_html(chart.unwrap(), "ngram_viewer.html", kind="chart", description="N-gram viewer chart")
    if not html.ok:
        for d in html.diagnostics:
            print(d, file=sys.stderr)
        return 1
    writer.add_diagnostics(
        *corpus_result.diagnostics, *table_result.diagnostics, *result.diagnostics, *chart.diagnostics
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(series)} series row(s) to {writer.run_dir / 'ngram_series.csv'}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
