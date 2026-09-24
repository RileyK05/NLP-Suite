"""verb_analysis — thin CLI for verb modality, tense and voice (FR-5.10)."""

from __future__ import annotations

import argparse
import sys

from core.analysis.verb_analysis import analyze_verbs, summarize_verbs
from core.io.reader import read_corpus
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Verb modality, tense and voice over the shared parse")
    p.add_argument(
        "--analysis",
        choices=["all", "modality", "tense", "voice"],
        default="all",
        help="facet to summarize: all, modality, tense, or voice",
    )
    return p.parse_args(argv)


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
    result = analyze_verbs(table, analysis=args.analysis)
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    frame = result.unwrap()
    summary = summarize_verbs(frame, analysis=args.analysis)
    if not summary.ok:
        for d in summary.diagnostics:
            print(d, file=sys.stderr)
        return 1
    writer = make_writer(args, corpus, tool="verb_analysis", params=vars(args))
    written = writer.write_table(frame, "verbs.csv", kind="table", description="Per-verb tense, modality and voice")
    if not written.ok:
        for d in written.diagnostics:
            print(d, file=sys.stderr)
        return 1
    tally = writer.write_table(
        summary.unwrap(), "verb_summary.csv", kind="table", description="Per-document verb counts"
    )
    if not tally.ok:
        for d in tally.diagnostics:
            print(d, file=sys.stderr)
        return 1
    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *result.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(frame)} verb row(s) to {writer.run_dir / 'verbs.csv'}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
