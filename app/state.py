"""Shared application state (FR-8.1) — pure, no Streamlit import.

Every page renders the same corpus/parser/language/output sidebar through
``app.session``; the state object, its validation, and the guided-setup
readiness check (FR-8.6) live here so tests cover them without a
Streamlit runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import pandas as pd

from core.io.reader import read_corpus
from core.result import Diagnostic, Result

__all__ = ["DEFAULT_STATE", "AppState", "ReadinessItem", "resolve_corpus", "setup_readiness", "summarize_validation"]


@dataclass(frozen=True, slots=True)
class AppState:
    """Values shared by all pages (persisted in ``st.session_state``)."""

    corpus_dir: str = "tests/fixtures/mini-corpus"
    output_root: str = "out"
    parser: str = "spacy"
    language: str = "en"


DEFAULT_STATE = AppState()


def resolve_corpus(raw: str) -> Result[Path]:
    """Resolve the sidebar corpus string to a directory, or fail loudly."""
    cleaned = (raw or "").strip()
    if not cleaned:
        return Result.failure(Diagnostic.error("APP_NO_CORPUS", "no corpus directory entered"))
    target = Path(cleaned)
    if not target.exists():
        return Result.failure(
            Diagnostic.error("APP_CORPUS_MISSING", f"corpus directory not found: {target}", path=str(target))
        )
    if not target.is_dir():
        return Result.failure(
            Diagnostic.error("APP_CORPUS_NOT_A_DIR", f"corpus path is not a directory: {target}", path=str(target))
        )
    return Result.success(target)


def summarize_validation(frame: pd.DataFrame) -> dict[str, int]:
    """Small status rollup over a ``core.data.validation`` report frame."""
    if frame.empty or "Status" not in frame.columns:
        return {"files": 0, "ok": 0, "issues": 0, "tokens": 0}
    ok = int((frame["Status"] == "OK").sum())
    tokens = int(pd.to_numeric(frame["Tokens"], errors="coerce").fillna(0).sum()) if "Tokens" in frame.columns else 0
    return {"files": len(frame), "ok": ok, "issues": int(len(frame) - ok), "tokens": tokens}


@dataclass(frozen=True, slots=True)
class ReadinessItem:
    """One guided-setup checklist row: what, whether it passes, and the fix."""

    label: str
    ok: bool
    fix: str = ""


def setup_readiness(corpus_dir: str, parser: str = "spacy") -> list[ReadinessItem]:
    """Fast first-run checklist (no model loads — `nlp-suite doctor` goes deeper).

    Python version, core imports, the chosen parser backend package, and
    the shared-sidebar corpus. Every failure names its fix.
    """
    items = [
        ReadinessItem(
            "Python 3.12+",
            sys.version_info >= (3, 12),
            "" if sys.version_info >= (3, 12) else "install Python 3.12+ (see .python-version)",
        )
    ]
    for module in ("pandas", "sklearn", "scipy"):
        present = importlib.util.find_spec(module) is not None
        items.append(
            ReadinessItem(
                f"core dependency: {module}",
                present,
                "" if present else f"pip install {module}",
            )
        )
    backend_present = importlib.util.find_spec(parser) is not None
    items.append(
        ReadinessItem(
            f"parser backend package ({parser})",
            backend_present,
            "" if backend_present else f'pip install "nlp-suite-ng[{parser}]"',
        )
    )
    resolved = resolve_corpus(corpus_dir)
    if resolved.value is None:
        items.append(ReadinessItem("corpus directory", False, resolved.diagnostics[0].message))
    else:
        loaded = read_corpus(resolved.unwrap())
        if loaded.value is None:
            items.append(ReadinessItem("corpus readable", False, "; ".join(d.message for d in loaded.diagnostics)))
        else:
            items.append(ReadinessItem(f"corpus readable ({len(loaded.unwrap())} documents)", True))
    return items
