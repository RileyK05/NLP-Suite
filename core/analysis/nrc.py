"""NRC emotion lexicon (FR-4.3) — word→emotion counting with Plutchik output.

Scores each document as the fraction of affect hits per category over ten
fixed columns (nrclex ``EMOTION_ORDER``): fear, anger, anticipation, trust,
surprise, positive, negative, sadness, disgust, joy. The lexicon is a plain
JSON word→[emotions] mapping: pass one explicitly, install it as the
registry ``nrc-lexicon`` asset, or fall back to the file bundled with the
``nrclex`` package. No TokenBlob, no charts, no radar — data out, viewer
renders.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import pandas as pd

from core.analysis.sentiment_vader_anew import _resolve_asset_path
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["EMOTION_ORDER", "default_nrc_lexicon", "emotions", "load_nrc_lexicon"]

EMOTION_ORDER: tuple[str, ...] = (
    "fear",
    "anger",
    "anticipation",
    "trust",
    "surprise",
    "positive",
    "negative",
    "sadness",
    "disgust",
    "joy",
)

# The eight Plutchik emotions, normalized over their own total like legacy
# (sentiment_analysis_NRC_util.py:62-67: the denominator excludes the two
# valence labels). Affect words usually carry a valence tag, so folding
# positive/negative into the denominator deflates every emotion column and
# breaks the sum-to-1 radar/intensity conventions. Valence is reported as
# its own separately-normalized pair.
EIGHT_EMOTIONS: tuple[str, ...] = (
    "fear",
    "anger",
    "anticipation",
    "trust",
    "surprise",
    "sadness",
    "disgust",
    "joy",
)
VALENCE_LABELS: tuple[str, ...] = ("positive", "negative")

_COLUMNS = ["Document ID", "Document", "Tokens", "Hits", *EMOTION_ORDER]


def load_nrc_lexicon(path: str | Path) -> dict[str, list[str]]:
    """Parse an NRC JSON mapping (word -> [emotions]); skip malformed rows."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        return {}
    lexicon: dict[str, list[str]] = {}
    for word, affects in payload.items():
        if isinstance(word, str) and isinstance(affects, list) and all(isinstance(a, str) for a in affects):
            lexicon[word.strip().lower()] = [a for a in affects if a in EMOTION_ORDER]
    return lexicon


def default_nrc_lexicon() -> dict[str, list[str]]:
    """The lexicon file bundled with the ``nrclex`` package, if installed."""
    try:
        from importlib import resources

        data_path = resources.files("nrclex.data").joinpath("nrc_en.json")
        with data_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (ImportError, FileNotFoundError, OSError) as exc:
        raise ImportError(
            "nrclex is not installed and no NRC lexicon was supplied; "
            "install nrclex or point the 'nrc-lexicon' asset at an NRC JSON file"
        ) from exc
    return {
        str(w).lower(): [str(a) for a in v if str(a) in EMOTION_ORDER]
        for w, v in payload.items()
        if isinstance(v, list)
    }


def _resolve_lexicon(
    lexicon: Mapping[str, list[str]] | str | Path | None,
) -> tuple[dict[str, list[str]] | None, list[Diagnostic]]:
    if isinstance(lexicon, Mapping):
        return dict(lexicon), []
    if lexicon is not None:
        target = Path(lexicon)
        if not target.is_file():
            return None, [Diagnostic.error("NRC_LEXICON_MISSING", f"NRC lexicon not found at {target}")]
        return load_nrc_lexicon(target), []
    from core.assets.registry import default_registry

    asset = default_registry().path("nrc-lexicon")
    if asset.value is not None:
        return load_nrc_lexicon(asset.unwrap()), list(asset.diagnostics)
    try:
        return default_nrc_lexicon(), []
    except ImportError:
        return None, [
            Diagnostic.error(
                "NRC_LEXICON_MISSING",
                "no NRC lexicon: the 'nrc-lexicon' asset is not installed and nrclex is absent",
                fix="pip install nrclex  OR  place an NRC JSON file at assets/sentiment/nrc_lexicon.json",
            )
        ]


def emotions(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    lexicon: Mapping[str, list[str]] | str | Path | None = None,
) -> Result[pd.DataFrame]:
    """Per-document NRC affect fractions (ten fixed emotion columns)."""
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("NRC_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if field.value not in frame.columns or Col.DOCUMENT_ID.value not in frame.columns:
        return Result.failure(Diagnostic.error("NRC_MISSING_COLUMN", "missing field or Document ID"))
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_COLUMNS))
    table, diags = _resolve_lexicon(lexicon)
    if table is None:
        return Result.failure(*diags)

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    rows: list[dict[str, object]] = []
    for doc_id, group in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        doc = str(group[doc_col].iloc[0]) if doc_col is not None else ""
        words = [str(x).lower() for x in group[field.value].tolist() if str(x).strip() and str(x) != "nan"]
        words = [w for w in words if w.isalpha()]
        hits: list[str] = []
        for word in words:
            hits.extend(table.get(word, []))
        total_eight = sum(1 for h in hits if h in EIGHT_EMOTIONS)
        total_valence = sum(1 for h in hits if h in VALENCE_LABELS)
        row: dict[str, object] = {
            "Document ID": str(doc_id),
            "Document": doc,
            "Tokens": len(words),
            "Hits": sum(1 for w in words if w in table),
        }
        for emotion in EIGHT_EMOTIONS:
            row[emotion] = round(hits.count(emotion) / total_eight, 4) if total_eight else 0.0
        for label in VALENCE_LABELS:
            row[label] = round(hits.count(label) / total_valence, 4) if total_valence else 0.0
        rows.append(row)
    return Result.success(pd.DataFrame(rows, columns=_COLUMNS), *diags)
