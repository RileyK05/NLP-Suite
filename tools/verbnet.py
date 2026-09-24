"""verbnet — thin CLI for VerbNet class aggregation (FR-4.5).

Aggregates a verb-list CSV (first column) to first VerbNet classes.
Needs the optional ``wordnet`` extra (NLTK) plus the VerbNet corpus.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from core.analysis.verbnet import aggregate_verbs, category_counts
from core.io.writer import OutputWriter
from core.result import Diagnostic
from tools._cli import exit_with_diagnostics


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VerbNet class aggregation (verbs CSV)")
    parser.add_argument("words", type=Path, help="CSV file with one verb per row (first column)")
    parser.add_argument("output", type=Path, help="output root directory")
    return parser.parse_args(argv)


def _read_words(path: Path) -> list[str]:
    import pandas as pd

    try:
        frame = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except (OSError, ValueError) as exc:
        exit_with_diagnostics([Diagnostic.error("VERBNET_WORDS_UNREADABLE", f"cannot read words CSV {path}: {exc}")])
        raise SystemExit(1) from exc  # pragma: no cover — exit_with_diagnostics always raises
    if frame.shape[1] < 1:
        exit_with_diagnostics([Diagnostic.error("VERBNET_WORDS_EMPTY", f"words CSV {path} has no columns")])
        raise SystemExit(1)  # pragma: no cover — exit_with_diagnostics always raises
    return [str(value) for value in frame.iloc[:, 0].tolist()]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = aggregate_verbs(_read_words(args.words))
    if result.value is None:
        exit_with_diagnostics(result.diagnostics, code=1)
        return 1  # pragma: no cover — exit_with_diagnostics always raises
    frame = result.unwrap()
    counted = category_counts(frame)
    if counted.value is None:  # unreachable: frame just came from aggregate_verbs
        exit_with_diagnostics(counted.diagnostics, code=1)
        return 1  # pragma: no cover
    try:
        writer = OutputWriter(args.output, tool="verbnet", params=vars(args), inputs=(args.words,), corpus=None)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1
    for artifact, name, description in (
        (frame, "verbnet.csv", "VerbNet class aggregation"),
        (counted.unwrap(), "verbnet_frequency.csv", "VerbNet class frequencies"),
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
    print(f"Wrote {len(frame)} verb(s) to {writer.run_dir / 'verbnet.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
