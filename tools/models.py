"""models — parser model inventory (FR-9.3).

Lists every backend x language the suite configures, whether its model
is installed, and the exact fix when it is not. Read-only: nothing is
downloaded (downloads are explicit, so CI never pulls models by
accident). Wraps the same probes the doctor uses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from core.config import BACKEND_LANGUAGES
from core.io.writer import OutputWriter
from tools._cli import exit_with_diagnostics
from tools.doctor import _spacy_model_ok, _stanza_model_ok

__all__ = ["main", "model_table"]


def model_table() -> pd.DataFrame:
    """One row per configured backend x language with install state."""
    from core.pipelines.spacy_backend import spacy_model_name

    rows: list[dict[str, object]] = []
    for backend in ("stanza", "spacy"):
        for language in sorted(BACKEND_LANGUAGES.get(backend, ())):
            if backend == "stanza":
                ok, detail = _stanza_model_ok(language)
                fix = f'python -c "import stanza; stanza.download({language!r})"' if not ok else ""
            else:
                ok, detail = _spacy_model_ok(language)
                fix = f"python -m spacy download {spacy_model_name(language)}" if not ok else ""
            rows.append(
                {
                    "Backend": backend,
                    "Language": language,
                    "Model": spacy_model_name(language) if backend == "spacy" else f"stanza-{language}",
                    "Installed": ok,
                    "Detail": detail,
                    "Fix": fix,
                }
            )
    return pd.DataFrame(rows, columns=["Backend", "Language", "Model", "Installed", "Detail", "Fix"])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parser model inventory (read-only)")
    parser.add_argument("output", type=Path, help="output root directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    frame = model_table()
    try:
        writer = OutputWriter(args.output, tool="models", params=vars(args), inputs=(), corpus=None)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1
    written = writer.write_table(frame, "models.csv", kind="table", description="Parser model inventory")
    if not written.ok:
        writer.abandon()
        exit_with_diagnostics(written.diagnostics, code=1)
        return 1  # pragma: no cover — exit_with_diagnostics always raises
    env = writer.finalize()
    if not env.ok:
        for diag in env.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    missing = frame[~frame["Installed"].astype(bool)]
    for _, row in frame.iterrows():
        mark = "ok" if row["Installed"] else "MISSING"
        print(f"[{mark}] {row['Backend']} {row['Language']} ({row['Model']})")
        if not row["Installed"]:
            print(f"       fix: {row['Fix']}")
    print(f"Wrote {len(frame)} model(s) to {writer.run_dir / 'models.csv'}")
    return 0 if missing.empty else 1
