"""Coreference — simple lemma-based clustering for nouns."""

from __future__ import annotations

import pandas as pd

from core.analysis.postags import is_noun_tag
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["run"]


def _is_noun(pos: str) -> bool:
    # Nouns plus PRON on purpose: a coreference chain runs through pronouns,
    # which is one tag wider than the analysis noun rule (postags.py).
    return is_noun_tag(pos) or pos in ("PRON",)


def run(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
) -> Result[pd.DataFrame]:
    """Cluster noun mentions by lower-cased lemma.

    The legacy used Stanford CoreNLP neural coref; this deterministic stub
    groups identical noun lemmas (the most common evaluation baseline) and
    is sufficient for the chunk contract without a model download.
    """
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("COREF_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    for need in [field.value, Col.POS.value, Col.SENTENCE_ID.value, Col.DOCUMENT_ID.value]:
        if need not in frame.columns:
            return Result.failure(Diagnostic.error("COREF_MISSING_COLUMN", f"missing {need!r}"))
    if frame.empty:
        return Result.success(
            pd.DataFrame(columns=["Cluster ID", "Mention", "Lemma", "Sentence ID", "Document ID", "Document"])
        )

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None

    # Collect noun mentions per lemma
    mentions: dict[str, list[tuple[str, object, object, str]]] = {}
    for _, row in frame.iterrows():
        pos = str(row[Col.POS.value])
        if not _is_noun(pos):
            continue
        lemma = str(row[field.value]).strip()
        if not lemma or lemma.lower() in ("nan", "none", ""):
            continue
        key = lemma.lower()
        form = str(row[Col.FORM.value])
        sid = row[Col.SENTENCE_ID.value]
        did = row[Col.DOCUMENT_ID.value]
        doc = str(row[doc_col]) if doc_col is not None else ""
        mentions.setdefault(key, []).append((form, sid, did, doc))

    rows: list[dict[str, object]] = []
    for cluster_id, (lemma, occs) in enumerate(sorted(mentions.items()), start=1):
        for form, sid, did, doc in occs:
            rows.append(
                {
                    "Cluster ID": cluster_id,
                    "Mention": form,
                    "Lemma": lemma,
                    "Sentence ID": sid,
                    "Document ID": str(did),
                    "Document": doc,
                }
            )

    if not rows:
        df = pd.DataFrame(columns=["Cluster ID", "Mention", "Lemma", "Sentence ID", "Document ID", "Document"])
    else:
        df = pd.DataFrame(
            rows,
            columns=["Cluster ID", "Mention", "Lemma", "Sentence ID", "Document ID", "Document"],
        )
        df = df.sort_values(["Cluster ID", "Document ID", "Sentence ID"]).reset_index(drop=True)
    return Result.success(df)
