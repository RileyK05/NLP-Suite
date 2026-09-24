"""wordnet — thin CLI for WordNet aggregation (FR-4.4).

``up`` aggregates a word-list CSV (first column) to top supersenses;
``down`` expands one keyword to all transitive hyponyms. Both need the
optional ``wordnet`` extra (NLTK) plus the WordNet 3.0 corpus; a missing
backend or corpus fails before any artifact is written, with the fix
command on stderr.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from core.analysis.wordnet import aggregate_up, category_counts, expand_down
from core.io.writer import OutputWriter
from core.result import Diagnostic
from tools._cli import exit_with_diagnostics

__all__ = ["main"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WordNet aggregation (UP) and hyponym expansion (DOWN)")
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="aggregate words UP to top supersenses")
    up.add_argument("words", type=Path, help="CSV file with one word per row (first column)")
    up.add_argument("output", type=Path, help="output root directory")
    up.add_argument("--pos", choices=["NOUN", "VERB"], default="NOUN")
    up.add_argument("--anchors", default="", help="comma-separated anchor terms or synset names")

    down = sub.add_parser("down", help="expand a keyword DOWN to all hyponyms")
    down.add_argument("output", type=Path, help="output root directory")
    down.add_argument("--keyword", required=True, help="anchor keyword, e.g. city")
    down.add_argument("--pos", choices=["NOUN", "VERB"], default="NOUN")
    return parser.parse_args(argv)


def _read_words(path: Path) -> list[str]:
    import pandas as pd

    try:
        frame = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except (OSError, ValueError) as exc:
        exit_with_diagnostics([Diagnostic.error("WORDNET_WORDS_UNREADABLE", f"cannot read words CSV {path}: {exc}")])
        raise SystemExit(1) from exc  # pragma: no cover — exit_with_diagnostics always raises
    if frame.shape[1] < 1:
        exit_with_diagnostics([Diagnostic.error("WORDNET_WORDS_EMPTY", f"words CSV {path} has no columns")])
        raise SystemExit(1)  # pragma: no cover — exit_with_diagnostics always raises
    return [str(value) for value in frame.iloc[:, 0].tolist()]


def _main_up(args: argparse.Namespace) -> int:
    words = _read_words(args.words)
    anchors = [term.strip() for term in str(args.anchors).split(",") if term.strip()]
    result = aggregate_up(words, pos=args.pos, anchors=anchors)
    if result.value is None:
        exit_with_diagnostics(result.diagnostics, code=1)
        return 1  # pragma: no cover — exit_with_diagnostics always raises
    frame = result.unwrap()
    counted = category_counts(frame)
    if counted.value is None:  # unreachable: frame just came from aggregate_up
        exit_with_diagnostics(counted.diagnostics, code=1)
        return 1  # pragma: no cover
    try:
        writer = OutputWriter(args.output, tool="wordnet-up", params=vars(args), inputs=(args.words,), corpus=None)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1
    for artifact, name, description in (
        (frame, "wordnet_up.csv", "WordNet UP aggregation"),
        (counted.unwrap(), "wordnet_up_frequency.csv", "WordNet UP category frequencies"),
    ):
        written = writer.write_table(artifact, name, kind="table", description=description)
        if not written.ok:
            writer.abandon()
            exit_with_diagnostics(written.diagnostics, code=1)
            return 1  # pragma: no cover
    writer.add_diagnostics(*result.diagnostics, *counted.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for diag in env.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    print(f"Wrote {len(frame)} word(s) to {writer.run_dir / 'wordnet_up.csv'}")
    return 0


def _main_down(args: argparse.Namespace) -> int:
    result = expand_down(args.keyword, pos=args.pos)
    if result.value is None:
        exit_with_diagnostics(result.diagnostics, code=1)
        return 1  # pragma: no cover — exit_with_diagnostics always raises
    frame = result.unwrap()
    try:
        writer = OutputWriter(args.output, tool="wordnet-down", params=vars(args), corpus=None)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1
    written = writer.write_table(frame, "wordnet_down.csv", kind="table", description="WordNet DOWN expansion")
    if not written.ok:
        writer.abandon()
        exit_with_diagnostics(written.diagnostics, code=1)
        return 1  # pragma: no cover
    writer.add_diagnostics(*result.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for diag in env.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    print(f"Wrote {len(frame)} term(s) to {writer.run_dir / 'wordnet_down.csv'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "up":
        return _main_up(args)
    return _main_down(args)


if __name__ == "__main__":
    raise SystemExit(main())
