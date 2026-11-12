"""Polarity I — VADER (the package scorer).

VADER scoring delegates to the ``vaderSentiment`` package — the legacy called
``SentimentIntensityAnalyzer().polarity_scores`` directly, so package parity
is legacy parity by construction. The default analyzer uses the packaged
lexicon; pass ``lexicon_path`` (e.g. the registry ``vader-lexicon`` asset
copied from the legacy ``lib/sentimentLib``) for the legacy-exact lexicon —
both score the recorded probes identically.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = [
    "load_vader_analyzer",
    "load_vader_lexicon",
    "vader",
    "vader_documents",
    "vader_label",
    "vader_sentences",
]

_VADER_DOC_COLUMNS = ["Document ID", "Document", "Sentences", "Neg", "Neu", "Pos", "Compound"]
_VADER_SENT_COLUMNS = ["Document ID", "Document", "Sentence ID", "Sentence", "Neg", "Neu", "Pos", "Compound", "Label"]


def vader_label(compound: float) -> str:
    """Legacy sentence label thresholds (±0.05)."""
    if compound > 0.05:
        return "positive"
    if compound < -0.05:
        return "negative"
    return "neutral"


def load_vader_lexicon(path: str | Path) -> dict[str, float]:
    """Parse a VADER lexicon file (word, measure, std, ratings); skip bad lines."""
    lexicon: dict[str, float] = {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            try:
                lexicon[parts[0].strip().lower()] = float(parts[1])
            except ValueError:
                continue
    return lexicon


def load_vader_analyzer(lexicon_path: str | Path | None = None) -> Result[object]:
    """Build the package analyzer (lazy import; optional legacy-exact lexicon)."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return Result.failure(
            Diagnostic.error(
                "VADER_BACKEND_MISSING",
                "vaderSentiment is not installed; VADER needs the optional 'sentiment' extra",
                fix="python -m pip install nlp-suite-ng[sentiment]",
            )
        )
    if lexicon_path is None:
        return Result.success(SentimentIntensityAnalyzer())
    target = Path(lexicon_path)
    if not target.is_file():
        return Result.failure(
            Diagnostic.error("VADER_LEXICON_MISSING", f"VADER lexicon not found at {target}", path=str(target))
        )
    return Result.success(SentimentIntensityAnalyzer(lexicon_file=str(target)))


def _sentence_texts(frame: pd.DataFrame, field: Col) -> list[tuple[str, str, str, str]]:
    """(doc_id, document, sentence_id, text) with tokens joined in row order."""
    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    out: list[tuple[str, str, str, str]] = []
    for (doc_id, sent_id), group in frame.groupby([Col.DOCUMENT_ID.value, "Sentence ID"], sort=False):
        doc = str(group[doc_col].iloc[0]) if doc_col is not None else ""
        text = " ".join(str(x) for x in group[field.value].tolist() if str(x).strip() and str(x) != "nan")
        out.append((str(doc_id), doc, str(sent_id), text))
    return out


def _require_sentence_columns(frame: pd.DataFrame, field: Col) -> Diagnostic | None:
    if field not in (Col.FORM, Col.LEMMA):
        return Diagnostic.error("VADER_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}")
    for column in (field.value, Col.DOCUMENT_ID.value, "Sentence ID"):
        if column not in frame.columns:
            return Diagnostic.error("VADER_MISSING_COLUMN", f"missing {column!r}")
    return None


def vader_sentences(
    frame: pd.DataFrame,
    *,
    field: Col = Col.FORM,
    analyzer: object = None,
) -> Result[pd.DataFrame]:
    """Per-sentence package scores with legacy labels (the legacy surface)."""
    bad = _require_sentence_columns(frame, field)
    if bad is not None:
        return Result.failure(bad)
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if frame.empty:
        # The same columns as a table with rows, Hits included.
        return Result.success(pd.DataFrame(columns=[*_VADER_SENT_COLUMNS, "Hits"]))
    resolved = analyzer if analyzer is not None else load_vader_analyzer(None).unwrap_or(None)
    if resolved is None:
        failed = load_vader_analyzer(None)
        return Result.failure(*failed.diagnostics)
    lexicon = getattr(resolved, "lexicon", {})
    rows: list[dict[str, object]] = []
    for doc_id, doc, sent_id, text in _sentence_texts(frame, field):
        scores = resolved.polarity_scores(text)  # type: ignore[attr-defined]
        compound = round(float(scores["compound"]), 4)
        toks = [w.lower() for w in text.split() if w.strip()]
        hits = sum(1 for w in toks if w in lexicon)
        rows.append(
            {
                "Document ID": doc_id,
                "Document": doc,
                "Sentence ID": sent_id,
                "Sentence": text,
                "Neg": round(float(scores["neg"]), 4),
                "Neu": round(float(scores["neu"]), 4),
                "Pos": round(float(scores["pos"]), 4),
                "Compound": compound,
                "Label": vader_label(compound),
                "Hits": hits,
            }
        )
    return Result.success(pd.DataFrame(rows, columns=[*_VADER_SENT_COLUMNS, "Hits"]))


def vader_documents(sentences: pd.DataFrame) -> pd.DataFrame:
    """Per-document VADER from an already scored sentence table (see :func:`vader_sentences`)."""
    if sentences.empty:
        return pd.DataFrame(columns=_VADER_DOC_COLUMNS)
    rows: list[dict[str, object]] = []
    for (doc_id, doc), group in sentences.groupby(["Document ID", "Document"], sort=False):
        rows.append(
            {
                "Document ID": doc_id,
                "Document": doc,
                "Sentences": len(group),
                "Neg": round(float(group["Neg"].mean()), 4),
                "Neu": round(float(group["Neu"].mean()), 4),
                "Pos": round(float(group["Pos"].mean()), 4),
                "Compound": round(float(group["Compound"].mean()), 4),
            }
        )
    return pd.DataFrame(rows, columns=_VADER_DOC_COLUMNS)


def vader(
    frame: pd.DataFrame,
    *,
    field: Col = Col.FORM,
    analyzer: object = None,
) -> Result[pd.DataFrame]:
    """Per-document VADER: mean of the document's sentence components."""
    sent = vader_sentences(frame, field=field, analyzer=analyzer)
    if sent.value is None:
        return Result.failure(*sent.diagnostics)
    return Result.success(vader_documents(sent.unwrap()), *sent.diagnostics)
