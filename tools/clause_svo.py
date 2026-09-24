"""clause_svo — thin CLI for clause frequencies + SVO extraction."""

from __future__ import annotations

import argparse
import sys

from core.analysis.clause_svo import clause_frequencies, extract_svo
from core.io.reader import read_corpus
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_common_parser("Clause tags and SVO extraction").parse_args(argv)


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

    clause_result = clause_frequencies(table)
    if not clause_result.ok:
        for d in clause_result.diagnostics:
            print(d, file=sys.stderr)
        return 1

    svo_result = extract_svo(table)
    if not svo_result.ok:
        for d in svo_result.diagnostics:
            print(d, file=sys.stderr)
        return 1

    from core.analysis.clause_svo import ClauseSvoResult

    combined = ClauseSvoResult(clauses=clause_result.unwrap(), svos=svo_result.unwrap())

    writer = make_writer(args, corpus, tool="clause_svo", params=vars(args))

    cframe = combined.clauses_frame()
    w1 = writer.write_table(cframe, "clauses.csv", kind="table", description="clause tag frequencies")
    if not w1.ok:
        for d in w1.diagnostics:
            print(d, file=sys.stderr)
        return 1

    sframe = combined.svo_frame()
    w2 = writer.write_table(sframe, "svo.csv", kind="table", description="SVO triples")
    if not w2.ok:
        for d in w2.diagnostics:
            print(d, file=sys.stderr)
        return 1

    writer.add_diagnostics(
        *corpus_result.diagnostics, *table_result.diagnostics, *clause_result.diagnostics, *svo_result.diagnostics
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(cframe)} clause tag(s) and {len(sframe)} SVO(s) to {writer.run_dir}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
