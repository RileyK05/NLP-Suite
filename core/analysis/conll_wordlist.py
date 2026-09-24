"""Parameterized CoNLL word analysis — one engine for six legacy clones.

The legacy suite had six near-identical analyzers (noun, verb, adjective,
adverb, ratio, function-words) differing only in the POS filter. This module
is the single replacement: the caller selects a category and the same code
path runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from core.analysis.postags import is_noun_tag
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["Category", "WordFrequencyResult", "WordRow", "run"]


Category = Literal["all", "noun", "verb", "adjective", "adverb", "function", "noun-verb"]


def _is_noun(pos: object) -> bool:
    return is_noun_tag(pos)


def _is_verb(pos: object) -> bool:
    text = str(pos)
    return text.startswith("VB") or text in ("VERB", "AUX")


def _is_adjective(pos: object) -> bool:
    text = str(pos)
    return text.startswith("JJ") or text == "ADJ"


def _is_adverb(pos: object) -> bool:
    text = str(pos)
    return text.startswith("RB") or text == "ADV"


def _is_function(pos: object) -> bool:
    text = str(pos)
    return (
        text
        in (
            "IN",
            "DT",
            "CC",
            "PRP",
            "PRP$",
            "WP",
            "WP$",
            "WDT",
            "MD",
            "TO",
            "RP",
            "PDT",
            "POS",
            "EX",
            "ADP",
            "DET",
            "PRON",
            "AUX",
            "CCONJ",
            "SCONJ",
            "PART",
        )
        or text.startswith("DT")
        or text.startswith("PRP")
        or text.startswith("IN")
    )


_CATEGORY_PREDICATES: dict[str, Callable[[object], bool]] = {
    "all": lambda pos: True,
    "noun": _is_noun,
    "verb": _is_verb,
    "adjective": _is_adjective,
    "adverb": _is_adverb,
    "function": _is_function,
    "noun-verb": lambda pos: _is_noun(pos) or _is_verb(pos),
}


@dataclass(frozen=True, slots=True)
class WordRow:
    word: str
    count: int
    percentage: float


@dataclass(frozen=True, slots=True)
class WordFrequencyResult:
    rows: tuple[WordRow, ...]
    total_tokens: int
    filtered_tokens: int

    def to_frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=["Word", "Count", "Percentage"])
        return pd.DataFrame([{"Word": row.word, "Count": row.count, "Percentage": row.percentage} for row in self.rows])


def run(
    frame: pd.DataFrame,
    *,
    field: Col = Col.FORM,
    category: Category = "all",
    top_n: int | None = 20,
    case_sensitive: bool = False,
) -> Result[WordFrequencyResult]:
    """Count word frequencies in a CoNLL table.

    Parameters
    ----------
    frame:
        A canonical CoNLL table.
    field:
        Which column to count (``Form`` or ``Lemma``).
    category:
        POS-based filter. ``all`` keeps every token.
    top_n:
        How many of the most frequent words to return. ``None`` means all.
    case_sensitive:
        When ``False`` (the default) the *Word* values are lower-cased before
        counting.
    """
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(
            Diagnostic.error(
                "WORDLIST_BAD_FIELD",
                f"field must be Form or Lemma, got {field.value!r}",
                field=field.value,
            )
        )

    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[WordFrequencyResult](None, checked.diagnostics)

    required = [field.value, Col.POS.value]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        return Result.failure(
            Diagnostic.error("CONLL_MISSING_COLUMN", f"missing column(s): {missing}", missing=missing)
        )

    if category not in _CATEGORY_PREDICATES:
        return Result.failure(
            Diagnostic.error(
                "WORDLIST_BAD_CATEGORY",
                f"unknown category {category!r}; expected one of {sorted(_CATEGORY_PREDICATES)}",
                category=category,
            )
        )

    if top_n is not None and top_n <= 0:
        return Result.failure(Diagnostic.error("WORDLIST_BAD_TOP_N", f"top_n must be positive or None, got {top_n}"))

    if frame.empty:
        return Result.success(WordFrequencyResult(rows=(), total_tokens=0, filtered_tokens=0))

    predicate = _CATEGORY_PREDICATES[category]

    pos_series = frame[Col.POS.value].tolist()
    word_series = frame[field.value].tolist()

    filtered_words: list[str] = []
    for pos, word in zip(pos_series, word_series, strict=True):
        if predicate(pos):
            text = str(word)
            if not case_sensitive:
                text = text.lower()
            filtered_words.append(text)

    total_tokens = len(frame)
    filtered_tokens = len(filtered_words)

    if not filtered_words:
        return Result.success(WordFrequencyResult(rows=(), total_tokens=total_tokens, filtered_tokens=0))

    counts: dict[str, int] = {}
    for word in filtered_words:
        counts[word] = counts.get(word, 0) + 1

    sorted_words = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if top_n is not None:
        sorted_words = sorted_words[:top_n]

    rows = tuple(
        WordRow(word=word, count=count, percentage=round(count / filtered_tokens * 100, 6))
        for word, count in sorted_words
    )

    return Result.success(WordFrequencyResult(rows=rows, total_tokens=total_tokens, filtered_tokens=filtered_tokens))
