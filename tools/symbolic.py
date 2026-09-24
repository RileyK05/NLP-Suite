"""symbolic — thin CLI for symbolic space + social actor typologies (FR-4.6)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from core.analysis.symbolic import aggregate_actors, aggregate_spaces, category_counts
from core.io.writer import OutputWriter
from core.result import Diagnostic
from tools._cli import exit_with_diagnostics


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symbolic space + social actor typologies (words CSV)")
    parser.add_argument("words", type=Path, help="CSV file with one word per row (first column)")
    parser.add_argument("output", type=Path, help="output root directory")
    parser.add_argument("--analysis", choices=["space", "actor"], default="space")
    parser.add_argument("--space-lexicon", type=Path, default=None, help="space typology CSV (defaults to asset)")
    parser.add_argument("--actor-lexicon", type=Path, default=None, help="actor typology CSV (defaults to asset)")
    return parser.parse_args(argv)


def _read_words(path: Path) -> list[str]:
    import pandas as pd

    try:
        frame = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except (OSError, ValueError) as exc:
        exit_with_diagnostics([Diagnostic.error("SYMBOLIC_WORDS_UNREADABLE", f"cannot read words CSV {path}: {exc}")])
        raise SystemExit(1) from exc  # pragma: no cover — exit_with_diagnostics always raises
    if frame.shape[1] < 1:
        exit_with_diagnostics([Diagnostic.error("SYMBOLIC_WORDS_EMPTY", f"words CSV {path} has no columns")])
        raise SystemExit(1)  # pragma: no cover — exit_with_diagnostics always raises
    return [str(value) for value in frame.iloc[:, 0].tolist()]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    words = _read_words(args.words)
    if args.analysis == "space":
        result = aggregate_spaces(words, lexicon=args.space_lexicon)
        column, stem = "Space Type", "symbolic_space"
    else:
        result = aggregate_actors(words, lexicon=args.actor_lexicon)
        column, stem = "Actor Type", "symbolic_actor"
    if result.value is None:
        exit_with_diagnostics(result.diagnostics, code=1)
        return 1  # pragma: no cover — exit_with_diagnostics always raises
    frame = result.unwrap()
    counted = category_counts(frame, column)
    if counted.value is None:  # unreachable: frame just came from the aggregation
        exit_with_diagnostics(counted.diagnostics, code=1)
        return 1  # pragma: no cover
    try:
        writer = OutputWriter(args.output, tool="symbolic", params=vars(args), inputs=(args.words,), corpus=None)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1
    for artifact, name, description in (
        (frame, f"{stem}.csv", f"Symbolic {args.analysis} classification"),
        (counted.unwrap(), f"{stem}_frequency.csv", f"Symbolic {args.analysis} frequencies"),
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
    print(f"Wrote {len(frame)} word(s) to {writer.run_dir / (stem + '.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
