"""Coreference — simple lemma-based clustering for nouns."""

from __future__ import annotations

import pandas as pd

from core.analysis.postags import NOUN_TAGS
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["run"]

_COLUMNS = ["Cluster ID", "Mention", "Lemma", "Sentence ID", "Document ID", "Document"]


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
        return Result.success(pd.DataFrame(columns=_COLUMNS))

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    from core.conll.frames import col_str_list

    # One columnar frame instead of one dict per mention: a 600k-token table
    # made hundreds of thousands of short-lived dicts, and the old noun test
    # called ``str``/``strip``/``upper`` on every token from Python.
    work = pd.DataFrame(
        {
            "_pos": col_str_list(frame, Col.POS),
            "_lemma": col_str_list(frame, field),
            "_form": col_str_list(frame, Col.FORM),
            "_sid": frame[Col.SENTENCE_ID.value].to_numpy(),
            "_did": frame[Col.DOCUMENT_ID.value].to_numpy(),
            "_doc": col_str_list(frame, doc_col) if doc_col is not None else [""] * len(frame),
        }
    )
    tags = work["_pos"].str.strip().str.upper()
    # Nouns plus PRON on purpose: a coreference chain runs through pronouns,
    # one tag wider than the analysis noun rule (postags.py).
    is_noun = tags.str.startswith("NN") | tags.isin(NOUN_TAGS) | tags.eq("PRON")
    nouns = work.loc[is_noun].copy()
    if nouns.empty:
        return Result.success(pd.DataFrame(columns=_COLUMNS))
    lemmas = nouns["_lemma"].str.strip()
    folded = lemmas.str.lower()
    keep = lemmas.ne("") & ~folded.isin(("nan", "none"))
    nouns = nouns.loc[keep].copy()
    if nouns.empty:
        return Result.success(pd.DataFrame(columns=_COLUMNS))
    nouns["_lemma"] = lemmas.loc[keep].str.lower()
    # ``sort=True`` numbers clusters in lexicographic lemma order, as the
    # sorted dict did; ngroup keeps the original row order within a cluster.
    cluster_ids = nouns.groupby("_lemma", sort=True).ngroup() + 1
    out = pd.DataFrame(
        {
            "Cluster ID": cluster_ids.to_numpy(),
            "Mention": nouns["_form"].to_numpy(),
            "Lemma": nouns["_lemma"].to_numpy(),
            "Sentence ID": nouns["_sid"].to_numpy(),
            "Document ID": nouns["_did"].astype(str).to_numpy(),
            "Document": nouns["_doc"].to_numpy(),
        },
        columns=_COLUMNS,
    )
    return Result.success(out.sort_values(["Cluster ID", "Document ID", "Sentence ID"]).reset_index(drop=True))
