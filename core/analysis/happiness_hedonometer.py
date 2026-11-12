"""Happiness — the hedonometer (full labMT asset).

Hedonometer happiness is the mean over a full labMT asset
(``hedonometer.json``); the baked sample map is gone.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import pandas as pd

from core.analysis.lexicon_assets import resolve_asset_path
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = [
    "hedonometer",
    "load_hedonometer_lexicon",
]


def load_hedonometer_lexicon(path: str | Path) -> dict[str, float]:
    """Parse a labMT JSON asset ({"objects": [{"word", "happs"}]})."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    objects = payload.get("objects", []) if isinstance(payload, dict) else []
    lexicon: dict[str, float] = {}
    for entry in objects:
        try:
            lexicon[str(entry["word"]).strip().lower()] = float(entry["happs"])
        except (KeyError, TypeError, ValueError):
            continue
    return lexicon


def hedonometer(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    lexicon: Mapping[str, float] | str | Path | None = None,
) -> Result[pd.DataFrame]:
    """Per-document mean happiness over the full asset (5.0 when no hits)."""
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("HEDONO_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if field.value not in frame.columns or Col.DOCUMENT_ID.value not in frame.columns:
        return Result.failure(Diagnostic.error("HEDONO_MISSING_COLUMN", "missing field or Document ID"))
    if frame.empty:
        return Result.success(pd.DataFrame(columns=["Document ID", "Document", "Tokens", "Hits", "Happiness"]))
    table: Mapping[str, float] | None = lexicon if isinstance(lexicon, Mapping) else None
    diags: list[Diagnostic] = []
    if table is None:
        target, diags = resolve_asset_path("hedonometer", lexicon)
        if target is None:
            return Result.failure(*diags)
        table = load_hedonometer_lexicon(target)

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    rows: list[dict[str, object]] = []
    for doc_id, group in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        doc = str(group[doc_col].iloc[0]) if doc_col is not None else ""
        words = [str(x).lower() for x in group[field.value].tolist() if str(x).strip() and str(x) != "nan"]
        words = [w for w in words if w.isalpha()]
        vals = [table[w] for w in words if w in table]
        happy = round(sum(vals) / len(vals), 3) if vals else 5.0
        rows.append(
            {
                "Document ID": str(doc_id),
                "Document": doc,
                "Tokens": len(words),
                "Hits": len(vals),
                "Happiness": happy,
            }
        )
    df = pd.DataFrame(rows, columns=["Document ID", "Document", "Tokens", "Hits", "Happiness"])
    return Result.success(df, *diags)
