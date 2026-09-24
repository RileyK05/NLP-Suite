"""readability — thin CLI for per-document readability scores.

Works on raw corpus text; no parser model needed (deterministic).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.analysis.readability import run
from core.io.reader import read_corpus
from tools._cli import write_or_die


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Readability scores (Flesch, Fog, SMOG, ...) per document")
    p.add_argument("corpus", type=Path, help="corpus directory of .txt files")
    p.add_argument("output", type=Path, help="output root directory")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

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
    for diag in corpus_result.diagnostics:
        print(diag, file=sys.stderr)

    result = run(corpus)
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    for diag in result.diagnostics:
        print(diag, file=sys.stderr)

    from core.io.writer import OutputWriter

    try:
        writer = OutputWriter(args.output, tool="readability", params={}, corpus=corpus)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1

    frame = result.unwrap().to_frame()
    write_or_die(
        writer,
        writer.write_table(frame, "readability.csv", kind="table", description="Readability scores per document"),
    )
    writer.add_diagnostics(*corpus_result.diagnostics, *result.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(frame)} readability row(s) to {writer.run_dir / 'readability.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
