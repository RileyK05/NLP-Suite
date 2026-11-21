"""Columnar builder for canonical CoNLL tables (R9)."""

from __future__ import annotations

import pandas as pd

from core.conll.schema import Col, canonical_columns

__all__ = ["ConllTableBuilder"]

_CANONICAL = canonical_columns()


class ConllTableBuilder:
    """Accumulates token fields in flat column lists without per-token dict allocations."""

    __slots__ = (
        "_deprels",
        "_document_ids",
        "_documents",
        "_forms",
        "_heads",
        "_ids",
        "_lemmas",
        "_ner",
        "_pos",
        "_sentence_ids",
    )

    def __init__(self) -> None:
        self._ids: list[int] = []
        self._forms: list[str] = []
        self._lemmas: list[str] = []
        self._pos: list[str] = []
        self._ner: list[str] = []
        self._heads: list[int] = []
        self._deprels: list[str] = []
        self._sentence_ids: list[int] = []
        self._document_ids: list[str] = []
        self._documents: list[str] = []

    def append(  # noqa: PLR0913 -- maps directly to the 10 per-token CoNLL columns
        self,
        *,
        token_id: int,
        form: str,
        lemma: str,
        pos: str,
        ner: str,
        head: int,
        deprel: str,
        sentence_id: int,
        document_id: str,
        document: str,
    ) -> None:
        self._ids.append(token_id)
        self._forms.append(form)
        self._lemmas.append(lemma)
        self._pos.append(pos)
        self._ner.append(ner)
        self._heads.append(head)
        self._deprels.append(deprel)
        self._sentence_ids.append(sentence_id)
        self._document_ids.append(document_id)
        self._documents.append(document)

    def __len__(self) -> int:
        return len(self._ids)

    def to_frame(self) -> pd.DataFrame:
        count = len(self._ids)
        if count == 0:
            return pd.DataFrame(columns=list(_CANONICAL))
        return pd.DataFrame(
            {
                Col.ID.value: self._ids,
                Col.FORM.value: self._forms,
                Col.LEMMA.value: self._lemmas,
                Col.POS.value: self._pos,
                Col.NER.value: self._ner,
                Col.HEAD.value: self._heads,
                Col.DEPREL.value: self._deprels,
                Col.DEPS.value: ["_"] * count,
                Col.CLAUSE_TAG.value: [""] * count,
                Col.RECORD_ID.value: range(1, count + 1),
                Col.SENTENCE_ID.value: self._sentence_ids,
                Col.DOCUMENT_ID.value: self._document_ids,
                Col.DOCUMENT.value: self._documents,
            },
            columns=list(_CANONICAL),
        )
