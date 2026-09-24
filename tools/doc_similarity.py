"""doc_similarity — thin CLI for pairwise TF-IDF document similarity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.analysis.doc_similarity import find_duplicates, pairwise_similarity
from core.io.reader import read_corpus
from tools._cli import write_or_die


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise TF-IDF document similarity + duplicates")
    p.add_argument("corpus", type=Path, help="corpus directory of .txt files")
    p.add_argument("output", type=Path, help="output root directory")
    p.add_argument("--threshold", type=float, default=80.0, help="duplicate threshold 0-100")
    p.add_argument("--stopwords", type=Path, default=None, help="optional stopwords file (one per line)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from core.analysis.doc_similarity import _validate_threshold

    problem = _validate_threshold(args.threshold)
    if problem is not None:
        print(problem, file=sys.stderr)
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

    stopwords: list[str] | None = None
    if args.stopwords is not None:
        try:
            stopwords = [ln.strip() for ln in args.stopwords.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except OSError as exc:
            print(f"could not read stopwords {args.stopwords}: {exc}", file=sys.stderr)
            return 2

    from core.io.writer import OutputWriter

    try:
        writer = OutputWriter(
            args.output,
            tool="doc_similarity",
            params={"threshold": args.threshold},
            # The corpus goes in as a corpus so its directory keeps R3
            # protection; the stopword file is one named file, not a
            # folder being scanned.
            inputs=[args.stopwords] if args.stopwords else [],
            corpus=corpus,
        )
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1

    pairs_result = pairwise_similarity(corpus, stopwords)
    if not pairs_result.ok:
        for d in pairs_result.diagnostics:
            print(d, file=sys.stderr)
        writer.abandon()
        return 1
    dupes_result = find_duplicates(corpus, args.threshold, stopwords)
    if not dupes_result.ok:
        for d in dupes_result.diagnostics:
            print(d, file=sys.stderr)
        writer.abandon()
        return 1
    writer.add_diagnostics(*corpus_result.diagnostics, *pairs_result.diagnostics, *dupes_result.diagnostics)
    write_or_die(
        writer,
        writer.write_table(
            pairs_result.unwrap().to_frame(), "doc_pairs.csv", kind="table", description="Pairwise document similarity"
        ),
    )
    write_or_die(
        writer,
        writer.write_table(
            dupes_result.unwrap().to_frame(), "duplicates.csv", kind="table", description="Duplicate pairs at threshold"
        ),
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote similarity results to {writer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
