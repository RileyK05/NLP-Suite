"""Topic flow: how a speech moves between topics, paragraph by paragraph.

A topic model scores each *document* as a mixture, and the dominant-topic
table reduces that to one topic per speech. Both hide the thing a reader of a
single speech wants to see: its structure. Does it open on the economy, turn
to war, and close on a conclusion? Does it stay on one subject for pages, or
change every paragraph? That is a question about the order of paragraphs, and
nothing downstream of a document-level mixture can answer it.

This module supplies the half that is not the model: cutting each document
into segments and handing over each segment's tokens, so that the fitted model
can score them while it is still in hand (``fit_lda(..., segments=...)``).

**Segmentation is a choice, and the choice is reported.** Three rules, tried in
order for each document, and the rule that applied is returned with it:

* ``blank-line`` -- paragraphs separated by an empty line, the usual shape of
  prose;
* ``line`` -- one paragraph per line. The State of the Union files this suite
  is built around have no blank lines at all: each speech is a single block of
  about thirty lines, one paragraph per line;
* ``sentences`` -- fixed windows of consecutive sentences, when a document has
  no line structure to use or its tokens cannot be placed in its text.

The figure changes with the rule, so a panel that did not say which rule drew
it would be asking the reader to trust an unstated decision.

**A token is placed in a paragraph only if its position is proved.** Tokens
are aligned against the document's own text with the same strict aligner the
phrase explorer uses (``core.research.phrase._align_tokens``): one mismatch and
the whole document is treated as unaligned, rather than guessing which
occurrence of a repeated word a token was. An unaligned document falls back to
sentence windows and is named, never silently mis-paragraphed.

Short segments are kept, with their token count. A one-line paragraph such as
"Mr. President, Mr. Speaker" leaves almost nothing after stopwords, and its
topic is a guess. Merging it into its neighbour would hide that decision; the
count lets the panel say it instead.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING

import pandas as pd

from core.conll.schema import Col

# One implementation of "prove where a token sits in its text": the phrase
# explorer's strict aligner. Imported rather than restated (R10) -- two copies
# of an alignment rule would disagree about exactly the hard cases, such as the
# whitespace tokens spaCy emits.
from core.research.phrase import _align_tokens

if TYPE_CHECKING:
    from core.io.reader import Corpus

__all__ = [
    "SEGMENT_RULES",
    "SegmentPlan",
    "paragraph_spans",
    "plan_segments",
]

SEGMENT_RULES: tuple[str, ...] = ("blank-line", "line", "sentences")

_BLANK_LINE = re.compile(r"\n[ \t\r\f\v]*\n")
_LINE = re.compile(r"\n")


def paragraph_spans(text: str) -> tuple[str, list[tuple[int, int]]]:
    """The document's paragraphs as character spans, and the rule that found them.

    Returns ``("sentences", [])`` when the text has fewer than two paragraphs
    under either rule, which tells the caller to fall back to sentence
    windows. Spans cover content only: leading and trailing whitespace of each
    paragraph is trimmed, and empty paragraphs are dropped.
    """
    for rule, pattern in (("blank-line", _BLANK_LINE), ("line", _LINE)):
        spans = _split(text, pattern)
        if len(spans) >= 2:  # one paragraph is no structure to follow
            return rule, spans
    return "sentences", []


def _split(text: str, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in pattern.finditer(text):
        spans.extend(_trimmed(text, start, match.start()))
        start = match.end()
    spans.extend(_trimmed(text, start, len(text)))
    return spans


def _trimmed(text: str, start: int, end: int) -> list[tuple[int, int]]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return [(start, end)] if end > start else []


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """Where each document's segments are, before any model has scored them.

    ``labels`` maps a document name to one segment number per row of that
    document in the table (1-based, row order), which is how a caller turns
    kept tokens into per-segment token lists. ``spans`` gives each segment's
    character span in the document text, for reading it back as a passage;
    ``None`` where the span cannot be proved. ``rules`` records which rule cut
    each document, and ``unaligned`` names the documents that fell back to
    sentence windows because their tokens could not be placed in their text.
    """

    labels: dict[str, pd.Series] = field(default_factory=dict)
    spans: dict[str, list[tuple[int, int] | None]] = field(default_factory=dict)
    rules: dict[str, str] = field(default_factory=dict)
    unaligned: tuple[str, ...] = ()


def plan_segments(frame: pd.DataFrame, corpus: Corpus, *, window: int = 5) -> SegmentPlan:
    """Cut every document in *corpus* into segments over its rows in *frame*.

    *window* is the sentence-window size for documents that fall back to it.
    A document with no rows in the frame is skipped; the topic model never
    saw it either.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1 sentence, got {window}")
    labels: dict[str, pd.Series] = {}
    spans: dict[str, list[tuple[int, int] | None]] = {}
    rules: dict[str, str] = {}
    unaligned: list[str] = []
    by_id = {str(doc_id): rows for doc_id, rows in frame.groupby(Col.DOCUMENT_ID.value, sort=False)}
    for doc in corpus.docs:
        rows = by_id.get(str(doc.doc_id))
        if rows is None or rows.empty:
            continue
        name = str(rows[Col.DOCUMENT.value].iloc[0])
        forms = rows[Col.FORM.value].astype(str).tolist()
        token_spans = _align_tokens(doc.text, forms)
        aligned = bool(token_spans) and token_spans[0] is not None
        rule, paragraphs = paragraph_spans(doc.text) if aligned else ("sentences", [])
        if not aligned:
            unaligned.append(name)
        if rule == "sentences":
            segment, segment_spans = _sentence_windows(rows, token_spans if aligned else None, window)
        else:
            segment, segment_spans = _paragraphs(token_spans, paragraphs)
        labels[name] = pd.Series(segment, index=rows.index)
        spans[name] = segment_spans
        rules[name] = rule
    return SegmentPlan(labels=labels, spans=spans, rules=rules, unaligned=tuple(unaligned))


def _paragraphs(
    token_spans: list[tuple[int, int] | None], paragraphs: list[tuple[int, int]]
) -> tuple[list[int], list[tuple[int, int] | None]]:
    """Each token's paragraph, by where its proved span starts.

    A token that begins between two paragraphs (whitespace tokens do) belongs
    to the paragraph before it; one before the first paragraph belongs to the
    first. Paragraph numbers are 1-based and follow the text's own order.
    """
    starts = [start for start, _ in paragraphs]
    segment: list[int] = []
    for span in token_spans:
        position = span[0] if span is not None else 0
        segment.append(max(1, bisect_right(starts, position)))
    return segment, [(start, end) for start, end in paragraphs]


def _sentence_windows(
    rows: pd.DataFrame, token_spans: list[tuple[int, int] | None] | None, window: int
) -> tuple[list[int], list[tuple[int, int] | None]]:
    """Fixed windows of *window* consecutive sentences, in the order they occur."""
    order: dict[str, int] = {}
    for sentence in rows[Col.SENTENCE_ID.value].astype(str):
        order.setdefault(sentence, len(order))
    segment = [order[str(sentence)] // window + 1 for sentence in rows[Col.SENTENCE_ID.value]]
    count = max(segment) if segment else 0
    spans: list[tuple[int, int] | None] = [None] * count
    if token_spans is not None:
        for number, span in zip(segment, token_spans, strict=True):
            if span is None:
                continue
            current = spans[number - 1]
            spans[number - 1] = span if current is None else (min(current[0], span[0]), max(current[1], span[1]))
    return segment, spans
