"""Classifiers — date and simple NER-based."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from core.io.reader import date_from_filename
from core.result import Diagnostic, Result

__all__ = ["classify_by_date", "classify_by_ner_pattern"]


def classify_by_date(paths: list[Path]) -> Result[pd.DataFrame]:
    """Group *paths* by the date embedded in their filenames."""
    if not paths:
        return Result.failure(Diagnostic.error("CLASSIFY_EMPTY", "no paths to classify"))
    rows: list[dict[str, object]] = []
    for path in paths:
        path = Path(path)
        extracted = date_from_filename(path)
        rows.append(
            {"path": path.as_posix(), "filename": path.name, "date": extracted.isoformat() if extracted else ""}
        )
    frame = pd.DataFrame(rows)
    return Result.success(frame)


def classify_by_ner_pattern(text: str, pattern: str = r"\b[A-Z][a-z]+\b") -> Result[pd.DataFrame]:
    """Very small NER-like classifier: regex over *text*.

    The legacy NER classifier shelled out to a model; here we expose the
    interface with a deterministic pattern so it is testable offline. A real
    model can replace the regex later without changing the caller.
    """
    if not pattern:
        return Result.failure(Diagnostic.error("CLASSIFY_BAD_PATTERN", "pattern must be non-empty"))
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return Result.failure(Diagnostic.error("CLASSIFY_BAD_PATTERN", f"invalid pattern: {exc}"))
    matches = [(match.group(0), match.start()) for match in regex.finditer(text)]
    # Explicit columns so an empty result has the same schema as a non-empty one.
    frame = pd.DataFrame(matches, columns=["entity", "start"])
    return Result.success(frame)
