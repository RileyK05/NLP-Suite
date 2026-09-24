"""lexicon_series — thin CLI for counting named word groups along a corpus axis."""

from __future__ import annotations

import argparse
import sys

from core.analysis.lexicon_series import BY_CHOICES, facet_labels, lexicon_series, parse_lexicon
from core.conll.schema import Col
from core.io.reader import read_corpus
from tools._cli import build_common_parser, get_pipeline, handle_result_states, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Named word groups counted along a corpus axis, as rates per 1,000 tokens")
    p.add_argument(
        "--terms",
        required=True,
        help="word groups, e.g. 'Iraq: iraq, iraqi, saddam; Vietnam: vietnam, hanoi' (phrases allowed)",
    )
    p.add_argument("--by", choices=list(BY_CHOICES), default="year", help="the axis to count along")
    p.add_argument(
        "--group-pattern",
        default="",
        help="regex over document names when --by pattern; its first group names the axis value",
    )
    p.add_argument("--field", choices=["form", "lemma"], default="lemma")
    p.add_argument(
        "--within",
        default="",
        help="only count inside sentences matching these word groups, e.g. 'Iraq: iraq, saddam'",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    groups = parse_lexicon(args.terms)
    if groups.value is None:
        for d in groups.diagnostics:
            print(d, file=sys.stderr)
        return 2
    within = None
    if args.within.strip():
        restricted = parse_lexicon(args.within)
        if restricted.value is None:
            for d in restricted.diagnostics:
                print(d, file=sys.stderr)
            return 2
        within = restricted.unwrap()
    field = Col.FORM if args.field == "form" else Col.LEMMA
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
    labels = facet_labels([(doc.name, doc.date) for doc in corpus.docs], args.by, group_pattern=args.group_pattern)
    if labels.value is None:
        for d in labels.diagnostics:
            print(d, file=sys.stderr)
        return 1
    from core.pipelines.cache import PipelineCache

    cache = PipelineCache()
    pipeline = get_pipeline(cache, config, args)
    table_result = pipeline.parse(corpus)
    parse_state = handle_result_states(table_result)
    table = table_result.unwrap()
    result = lexicon_series(table, groups.unwrap(), labels.unwrap(), field=field, within=within)
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    frame = result.unwrap()
    writer = make_writer(args, corpus, tool="lexicon_series", params=vars(args))
    written = writer.write_table(
        frame, "lexicon_series.csv", kind="table", description="word-group rates along an axis"
    )
    if not written.ok:
        for d in written.diagnostics:
            print(d, file=sys.stderr)
        return 1
    writer.add_diagnostics(
        *corpus_result.diagnostics, *table_result.diagnostics, *labels.diagnostics, *result.diagnostics
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(frame)} row(s) to {writer.run_dir / 'lexicon_series.csv'}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
