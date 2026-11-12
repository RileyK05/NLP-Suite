"""Valence — ANEW word ratings (valence/arousal/dominance means).

ANEW valence/arousal/dominance are means over a full lexicon file
(``EnglishShortenedANEW.csv`` format); the baked sample map is gone.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from core.analysis.lexicon_assets import resolve_asset_path
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = [
    "anew",
    "load_anew_lexicon",
]

_ANEW_COLUMNS = ["Document ID", "Document", "Tokens", "Hits", "Valence", "Arousal", "Dominance"]


def load_anew_lexicon(path: str | Path) -> dict[str, tuple[float, float, float]]:
    """Parse an ANEW CSV (Word, valence, arousal, dominance); skip bad rows."""
    lexicon: dict[str, tuple[float, float, float]] = {}
    with open(path, encoding="utf-8", errors="ignore", newline="") as fh:
        import csv

        for raw in csv.DictReader(fh):
            # Headers whatever their case: "Valence" files are as common as
            # "valence" ones, and a miss here silently empties the lexicon.
            row = {str(key).strip().lower(): value for key, value in raw.items() if key is not None}
            try:
                word = str(row.get("word", "")).strip().lower()
                vals = (float(str(row["valence"])), float(str(row["arousal"])), float(str(row["dominance"])))
            except (KeyError, TypeError, ValueError):
                continue
            if word:
                lexicon[word] = vals
    return lexicon


def anew(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    lexicon: Mapping[str, tuple[float, float, float]] | str | Path | None = None,
) -> Result[pd.DataFrame]:
    """Per-document ANEW means over the full lexicon (no baked sample)."""
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("ANEW_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if field.value not in frame.columns or Col.DOCUMENT_ID.value not in frame.columns:
        return Result.failure(Diagnostic.error("ANEW_MISSING_COLUMN", f"missing {field.value!r} or Document ID"))
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_ANEW_COLUMNS))
    table: Mapping[str, tuple[float, float, float]] | None = lexicon if isinstance(lexicon, Mapping) else None
    diags: list[Diagnostic] = []
    if table is None:
        target, diags = resolve_asset_path("anew-lexicon", lexicon)
        if target is None:
            return Result.failure(*diags)
        table = load_anew_lexicon(target)

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    rows: list[dict[str, object]] = []
    for doc_id, group in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        doc = str(group[doc_col].iloc[0]) if doc_col is not None else ""
        words = [str(x).lower() for x in group[field.value].tolist() if str(x).strip() and str(x) != "nan"]
        words = [w for w in words if w.isalpha()]
        hits = [table[w] for w in words if w in table]
        if hits:
            valence = round(sum(v for v, _a, _d in hits) / len(hits), 3)
            arousal = round(sum(a for _v, a, _d in hits) / len(hits), 3)
            dominance = round(sum(d for _v, _a, d in hits) / len(hits), 3)
        else:
            valence = arousal = dominance = 0.0
        rows.append(
            {
                "Document ID": str(doc_id),
                "Document": doc,
                "Tokens": len(words),
                "Hits": len(hits),
                "Valence": valence,
                "Arousal": arousal,
                "Dominance": dominance,
            }
        )
    return Result.success(pd.DataFrame(rows, columns=_ANEW_COLUMNS), *diags)
