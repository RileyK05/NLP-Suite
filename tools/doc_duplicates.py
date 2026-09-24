"""doc_duplicates — thin CLI for exact/normalized/fuzzy duplicate matching."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.analysis.doc_duplicates import exact_groups, fuzzy_pairs, normalized_groups
from core.io.reader import read_corpus
from tools._cli import write_or_die


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exact, normalized, and fuzzy duplicate matching")
    p.add_argument("corpus", type=Path, help="corpus directory of .txt files")
    p.add_argument("output", type=Path, help="output root directory")
    p.add_argument("--threshold", type=float, default=80.0, help="fuzzy threshold 0-100")
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

    from core.io.writer import OutputWriter

    try:
        writer = OutputWriter(
            args.output,
            tool="doc_duplicates",
            params={"threshold": args.threshold},
            corpus=corpus,
        )
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1

    exact_result = exact_groups(corpus)
    normalized_result = normalized_groups(corpus)
    for _label, analysis in (("exact", exact_result), ("normalized", normalized_result)):
        if analysis.value is None:
            for d in analysis.diagnostics:
                print(d, file=sys.stderr)
            writer.abandon()
            return 1
    write_or_die(
        writer,
        writer.write_table(
            exact_result.unwrap().to_frame(),
            "exact_groups.csv",
            kind="table",
            description="Byte-identical groups",
        ),
    )
    write_or_die(
        writer,
        writer.write_table(
            normalized_result.unwrap().to_frame(),
            "normalized_groups.csv",
            kind="table",
            description="Normalized groups",
        ),
    )
    fuzzy = fuzzy_pairs(corpus, args.threshold)
    if not fuzzy.ok:
        for d in fuzzy.diagnostics:
            print(d, file=sys.stderr)
        writer.abandon()
        return 1
    writer.add_diagnostics(
        *corpus_result.diagnostics,
        *exact_result.diagnostics,
        *normalized_result.diagnostics,
        *fuzzy.diagnostics,
    )
    write_or_die(
        writer,
        writer.write_table(
            fuzzy.unwrap().to_frame(), "fuzzy_pairs.csv", kind="table", description="Fuzzy pairs at threshold"
        ),
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote duplicate results to {writer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
