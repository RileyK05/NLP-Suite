"""Sentence division over a CoNLL table.

Replaces the legacy ``CoNLL_record_division`` / ``sentence_division``, which
carried two suites-wide bugs:

* **The dropped final sentence.** Both functions accumulated tokens and flushed
  only when the sentence or document changed, so the last sentence of the table
  was never emitted. The old code was later patched with an extra append; this
  module removes the failure mode instead by *grouping by column* — every group
  pandas produces is a sentence, including the last one. There is no flush
  step to forget.
* **The phantom leading sentence.** The legacy loop seeded its "previous"
  sentinel with ``Sentence_ID=1`` / ``Document_ID='1.0'``. A table whose first
  document is ``'1'`` failed that comparison on row 0 and flushed an empty
  accumulator, producing a spurious empty sentence that callers then had to
  skip by hand. Grouping has no sentinel and so no phantom.

The legacy functions also indexed columns positionally and wrapped everything
in a bare ``except:`` that returned ``None``. Here columns are reached by name
and every outcome is a ``Result``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.conll.schema import Col
from core.result import Diagnostic, Result

__all__ = ["Sentence", "sentence_records", "sentence_tokens", "sentences_text"]


@dataclass(frozen=True, slots=True)
class Sentence:
    """One sentence: a stable identity plus the rows that belong to it."""

    document_id: str
    sentence_id: str
    rows: tuple[int, ...]

    @property
    def token_count(self) -> int:
        return len(self.rows)


def _require_columns(frame: pd.DataFrame, *columns: Col) -> tuple[Diagnostic, ...]:
    missing = [c.value for c in columns if c.value not in [str(x) for x in frame.columns]]
    if not missing:
        return ()
    return (
        Diagnostic.error(
            "CONLL_MISSING_COLUMN",
            f"table is missing required CoNLL column(s): {missing}",
            missing=missing,
        ),
    )


def sentence_records(frame: pd.DataFrame) -> Result[tuple[Sentence, ...]]:
    """Group a CoNLL table into sentences by ``(Document ID, Sentence ID)``.

    Groups appear in first-appearance order and every row lands in exactly one
    sentence. An empty table yields zero sentences rather than one empty one.
    """
    problems = _require_columns(frame, Col.DOCUMENT_ID, Col.SENTENCE_ID)
    if problems:
        return Result.failure(*problems)

    grouped = frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False, dropna=False)
    sentences = tuple(
        Sentence(
            document_id=str(key[0]),
            sentence_id=str(key[1]),
            rows=tuple(int(i) for i in positions),
        )
        for key, positions in grouped.indices.items()
    )
    return Result.success(sentences)


def sentence_tokens(
    frame: pd.DataFrame,
    field: Col = Col.FORM,
) -> Result[tuple[tuple[str, ...], ...]]:
    """The tokens of each sentence, in sentence order, for one CoNLL field."""
    problems = _require_columns(frame, Col.DOCUMENT_ID, Col.SENTENCE_ID, field)
    if problems:
        return Result.failure(*problems)

    records = sentence_records(frame)
    if records.value is None:
        return Result[tuple[tuple[str, ...], ...]](None, records.diagnostics)

    values = [str(v) for v in frame[field.value].tolist()]
    tokenized = tuple(tuple(values[i] for i in sentence.rows) for sentence in records.value)
    return Result.success(tokenized, *records.diagnostics)


def sentences_text(
    frame: pd.DataFrame,
    field: Col = Col.FORM,
    separator: str = " ",
) -> Result[tuple[str, ...]]:
    """The text of each sentence, in sentence order.

    Replaces the legacy ``tokenize_stanza_text`` accumulator, whose ``=``
    instead of ``+=`` returned only the final sentence for the whole document.
    """
    tokenized = sentence_tokens(frame, field)
    if tokenized.value is None:
        return Result[tuple[str, ...]](None, tokenized.diagnostics)

    joined = tuple(separator.join(tokens) for tokens in tokenized.value)
    return Result.success(joined, *tokenized.diagnostics)
