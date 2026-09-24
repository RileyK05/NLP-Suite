"""K-sentence windows — first/last K sentences per document."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.analysis.postags import NOUN_TAGS
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["KSentencesResult", "RepeatedWordRow", "run"]


@dataclass(frozen=True, slots=True)
class RepeatedWordRow:
    lemma: str
    form: str
    document_id: str
    doc_frequency: int
    sentence_id_first: int | str
    sentence_id_last: int | str


@dataclass(frozen=True, slots=True)
class KSentencesResult:
    counts: pd.DataFrame
    repetitions: pd.DataFrame
    k_first: int
    k_last: int

    def counts_frame(self) -> pd.DataFrame:
        return self.counts.copy()

    def repetitions_frame(self) -> pd.DataFrame:
        return self.repetitions.copy()


def _is_content_pos(pos: str) -> bool:
    # Content words: nouns (the shared rule, NOUN_TAGS holds its tag names),
    # verbs and adjectives, under either tagset.
    return (
        pos.startswith("NN")
        or pos.startswith("VB")
        or pos.startswith("JJ")
        or pos in ("VERB", "AUX", "ADJ")
        or pos in NOUN_TAGS
    )


def run(
    frame: pd.DataFrame,
    *,
    k_first: int,
    k_last: int,
) -> Result[KSentencesResult]:
    """For each document, count the first K and last K sentences and find bookend repetition."""
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[KSentencesResult](None, checked.diagnostics)

    required = [Col.FORM.value, Col.LEMMA.value, Col.POS.value, Col.SENTENCE_ID.value, Col.DOCUMENT_ID.value]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        return Result.failure(
            Diagnostic.error("KSENT_MISSING_COLUMN", f"missing column(s): {missing}", missing=missing),
        )

    if k_first < 0 or k_last < 0:
        return Result.failure(
            Diagnostic.error("KSENT_BAD_K", f"k_first and k_last must be >=0, got {k_first}, {k_last}"),
        )
    if k_first == 0 and k_last == 0:
        return Result.failure(
            Diagnostic.error("KSENT_ZERO_K", "at least one of k_first or k_last must be >0"),
        )

    if frame.empty:
        counts = pd.DataFrame(
            columns=[
                "Document ID",
                "Section",
                "K",
                "Sentences",
                "Tokens",
                "Content Tokens",
                "Document",
            ],
        )
        reps = pd.DataFrame(
            columns=["Lemma", "Form", "Document ID", "Doc Frequency", "Sentence ID First", "Sentence ID Last"],
        )
        return Result.success(KSentencesResult(counts=counts, repetitions=reps, k_first=k_first, k_last=k_last))

    # Corpus doc frequency of content lemmas (how many docs contain the lemma).
    lemma_doc_freq: dict[str, int] = {}
    for doc_id, g in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        lemmas = {
            str(lem).lower()
            for lem, pos, form in zip(g[Col.LEMMA.value], g[Col.POS.value], g[Col.FORM.value], strict=False)
            if _is_content_pos(str(pos)) and str(form).isalpha()
        }
        for lem in lemmas:
            if lem:
                lemma_doc_freq[lem] = lemma_doc_freq.get(lem, 0) + 1

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None

    counts_rows: list[dict[str, object]] = []
    rep_rows: list[dict[str, object]] = []

    for doc_id, doc_frame in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        doc = str(doc_frame[doc_col].iloc[0]) if doc_col is not None else ""
        sent_ids = sorted(doc_frame[Col.SENTENCE_ID.value].unique().tolist())
        if not sent_ids:
            continue

        first_ids = sent_ids[:k_first] if k_first > 0 else []
        # last K excluding any overlap with first K — same as legacy.
        last_ids_raw = sent_ids[-k_last:] if k_last > 0 else []
        last_ids = [s for s in last_ids_raw if s not in first_ids]

        def _content_lemmas(ids: list[object]) -> list[tuple[str, str, object]]:
            sub = doc_frame[doc_frame[Col.SENTENCE_ID.value].isin(ids)]
            items: list[tuple[str, str, object]] = []
            for form, lemma, pos, sid in zip(
                sub[Col.FORM.value], sub[Col.LEMMA.value], sub[Col.POS.value], sub[Col.SENTENCE_ID.value], strict=False
            ):
                if _is_content_pos(str(pos)) and str(form).isalpha():
                    lem = str(lemma).lower()
                    if lem:
                        items.append((lem, str(form), sid))
            return items

        def _push_counts(section: str, k: int, ids: list[object]) -> None:
            sub = doc_frame[doc_frame[Col.SENTENCE_ID.value].isin(ids)] if ids else doc_frame.iloc[0:0]
            tokens = len(sub)
            content = sum(1 for p in sub[Col.POS.value].astype(str) if _is_content_pos(p))
            counts_rows.append(
                {
                    "Document ID": str(doc_id),
                    "Section": section,
                    "K": k,
                    "Sentences": len(ids),
                    "Tokens": tokens,
                    "Content Tokens": content,
                    "Document": doc,
                }
            )

        _push_counts("first", k_first, first_ids)
        _push_counts("last", k_last, last_ids)

        first_content = _content_lemmas(first_ids)
        last_content = _content_lemmas(last_ids)
        first_lemmas = {lem for lem, _, _ in first_content}
        last_lemmas = {lem for lem, _, _ in last_content}
        repeated = first_lemmas & last_lemmas
        if repeated:
            # Emit one row per repeated lemma (deduped), with exemplar form from the first section.
            seen: set[str] = set()
            for lem, form, sid_first in first_content:
                if lem in repeated and lem not in seen:
                    seen.add(lem)
                    # find a last-side sentence for this lemma
                    sid_last = next((sid for l, _f, sid in last_content if l == lem), "")
                    rep_rows.append(
                        {
                            "Lemma": lem,
                            "Form": form,
                            "Document ID": str(doc_id),
                            "Doc Frequency": lemma_doc_freq.get(lem, 0),
                            "Sentence ID First": sid_first,
                            "Sentence ID Last": sid_last,
                        }
                    )

    counts_df = pd.DataFrame(counts_rows)
    if rep_rows:
        reps_df = pd.DataFrame(rep_rows).sort_values("Doc Frequency", kind="stable").reset_index(drop=True)
    else:
        reps_df = pd.DataFrame(
            columns=["Lemma", "Form", "Document ID", "Doc Frequency", "Sentence ID First", "Sentence ID Last"]
        )

    # Stable column order even when empty.
    if counts_df.empty:
        counts_df = pd.DataFrame(
            columns=["Document ID", "Section", "K", "Sentences", "Tokens", "Content Tokens", "Document"]
        )
    else:
        counts_df = counts_df[["Document ID", "Section", "K", "Sentences", "Tokens", "Content Tokens", "Document"]]

    return Result.success(KSentencesResult(counts=counts_df, repetitions=reps_df, k_first=k_first, k_last=k_last))
