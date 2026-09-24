"""sentence_complexity — thin CLI for per-sentence dependency complexity."""

from __future__ import annotations

import sys

from core.analysis.sentence_complexity import run
from core.io.reader import read_corpus
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config
from tools._cli import write_or_die


def _parse_args(argv: list[str] | None = None):  # type: ignore[no-untyped-def]
    return build_common_parser("Sentence complexity (dependency distance, depth, subordination)").parse_args(argv)


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

    result = run(table_result.unwrap())
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1

    from core.io.writer import OutputWriter

    try:
        writer = OutputWriter(args.output, tool="sentence_complexity", params=vars(args), corpus=corpus)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1

    frame = result.unwrap().to_frame()
    write_or_die(
        writer,
        writer.write_table(frame, "sentence_complexity.csv", kind="table", description="Sentence complexity"),
    )
    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *result.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(frame)} complexity row(s) to {writer.run_dir / 'sentence_complexity.csv'}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
