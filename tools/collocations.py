"""collocations — thin CLI for collocation association measures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from core.analysis.collocations import SPAN_MODES, collocations
from core.conll.schema import Col
from core.io.reader import read_corpus
from tools._cli import build_common_parser, get_pipeline, handle_result_states, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Collocation association measures (PMI, t-score, G2, Dice)")
    p.add_argument("--field", choices=["form", "lemma"], default="lemma")
    p.add_argument("--span", choices=list(SPAN_MODES), default="adjacent", help="adjacent bigrams or a token window")
    p.add_argument("--window", type=int, default=5, help="window size when --span window (tokens each side)")
    p.add_argument("--min-count", type=int, default=3, help="drop pairs seen fewer times than this")
    p.add_argument("--min-length", type=int, default=1, help="drop tokens shorter than this")
    p.add_argument("--top-n", type=int, default=200, help="rows kept after sorting by G2 (0 = all)")
    p.add_argument("--case-sensitive", action="store_true", help="do not casefold tokens")
    p.add_argument(
        "--stopwords",
        type=Path,
        default=None,
        help="optional file of stopwords to exclude, one per line",
    )
    return p.parse_args(argv)


def _load_stopwords(path: Path | None) -> frozenset[str] | None:
    """Explicit file or nothing -- the suite never bundles a hidden stoplist."""
    if path is None:
        return frozenset()
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"collocations: could not read stopwords {path}: {exc}", file=sys.stderr)
        return None
    return frozenset(line.strip() for line in text.splitlines() if line.strip())


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    field = Col.FORM if args.field == "form" else Col.LEMMA
    stopwords = _load_stopwords(args.stopwords)
    if stopwords is None:
        return 2
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
    result = collocations(
        table,
        field=field,
        span=args.span,
        window=args.window,
        min_count=args.min_count,
        min_length=args.min_length,
        top_n=args.top_n or None,
        stopwords=stopwords,
        case_sensitive=args.case_sensitive,
    )
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    frame = result.unwrap()
    writer = make_writer(args, corpus, tool="collocations", params=vars(args))
    written = writer.write_table(
        frame, "collocations.csv", kind="table", description="Collocation association measures"
    )
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
    print(f"Wrote {len(frame)} collocation row(s) to {writer.run_dir / 'collocations.csv'}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
