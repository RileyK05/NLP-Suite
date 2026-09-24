"""Quote/dialogue extraction with a two-stage speaker sieve (FR-4.8).

CoreNLP's quote annotator answers "is there dialogue?" by finding quotation
spans and attributing speakers. This port keeps the SHAPE of Muzny et al.'s
two-stage quote attribution -- a high-precision reporting-verb stage, then a
fallback -- but it is a LIGHTWEIGHT SIEVE, not their model: no features, no
classifier, no training data. Stage 1 is a cue regex (``X said``, ``said X``,
``said to X`` and the same reporting verbs) in the quote's sentences or an
adjacent one; stage 2 is the nearest NER PERSON token in the quote's
sentences or the one before; else no speaker. The ``Cue`` column records
which stage fired (``said``, ``said-to``, ``nearest-person``, ``none``), so
every attribution can be audited.

Intentional limitations (documented, not silent):

- quotes are double-quote spans: straight ``"..."`` toggles per document,
  typographic pairs by shape; single quotes are not dialogue;
- nested quotes and quotation marks used for inches or scare quotes can split
  spans; an unclosed opening quote is dropped with a WARNING;
- quote text is rebuilt from the token stream (punctuation rejoined where it
  is obviously attached), so spacing is token spacing;
- stage 1's X is the adjacent word -- "he said" can name "he" -- and stage 2
  contributes only its nearest token, so "Harry Potter" speaks as "Potter".
"""

from __future__ import annotations

import re

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["annotate_quotes", "summarize_quotes"]

_QUOTE_COLUMNS = ["Document", "Document ID", "Quote ID", "Quote", "Speaker", "Cue", "Sentence ID"]
_SUMMARY_COLUMNS = ["Document", "Document ID", "Quotes", "Attributed", "Unattributed", "Words"]

#: The stage-1 reporting verbs (Muzny et al.'s paralinguistic mentions, minus
#: the model around them).
CUE_VERBS: frozenset[str] = frozenset(
    {"said", "asked", "replied", "answered", "whispered", "shouted", "exclaimed", "continued", "muttered"}
)

_DOUBLE = '"'
_LEFT_DOUBLE = "\u201c"
_RIGHT_DOUBLE = "\u201d"


def _detokenize(tokens: list[str]) -> str:
    """Rejoin a token span the way a reader expects it printed.

    A parser gives punctuation its own token, so "Hello there ." would leak
    into every Quote cell; only the obvious attachments are reversed.
    """
    text = " ".join(tok for tok in tokens if tok.strip())
    text = re.sub(r"\s+([.,!?;:%)\]}>])", r"\1", text)
    text = re.sub(r"([(\[{<])\s+", r"\1", text)
    return text.strip()


def _word_count(text: str) -> int:
    """Words in a quote (alphanumeric-bearing tokens), for min-length + totals."""
    return sum(1 for tok in text.split() if any(ch.isalnum() for ch in tok))


def _is_person(tag: object) -> bool:
    text = str(tag).strip()
    if "-" in text:
        text = text.split("-", 1)[1]
    return text.upper() == "PERSON"


def _name_like(form: str) -> bool:
    """Candidate speaker word: has letters/digits and is not punctuation."""
    return any(ch.isalnum() for ch in form)


class _Sentence:
    """One sentence's inclusive token index range within its document."""

    __slots__ = ("end", "sid", "start")

    def __init__(self, sid: object, start: int, end: int) -> None:
        self.sid = sid
        self.start = start
        self.end = end  # inclusive


def _sentences(sids: list[object]) -> list[_Sentence]:
    out: list[_Sentence] = []
    start = 0
    for i in range(1, len(sids) + 1):
        if i == len(sids) or str(sids[i]) != str(sids[start]):
            out.append(_Sentence(sids[start], start, i - 1))
            start = i
    return out


def _cue_in_sentence(forms: list[str], sentence: _Sentence) -> tuple[str, str] | None:
    """Stage 1: a reporting verb with its adjacent name word (X said / said X / said to X)."""
    for i in range(sentence.start, sentence.end + 1):
        if forms[i].lower() not in CUE_VERBS:
            continue
        if i - 1 >= sentence.start and _name_like(forms[i - 1]):
            return (forms[i - 1], "said")
        if i + 2 <= sentence.end and forms[i + 1].lower() == "to" and _name_like(forms[i + 2]):
            return (forms[i + 2], "said-to")
        if i + 1 <= sentence.end and forms[i + 1].lower() != "to" and _name_like(forms[i + 1]):
            return (forms[i + 1], "said")
    return None


def _touched(sentences: list[_Sentence], open_idx: int, close_idx: int) -> list[int]:
    """Sentence positions the quote span runs through (a quote may cross sentences)."""
    return [pos for pos, s in enumerate(sentences) if not (s.end < open_idx or s.start > close_idx)]


def _nearest_person(
    forms: list[str],
    ners: list[object],
    sentences: list[_Sentence],
    touched: list[int],
    *,
    open_idx: int,
    close_idx: int,
) -> str | None:
    """Stage 2: nearest PERSON token outside the span, in the quote's sentences else the one before."""
    candidates = [
        i
        for pos in touched
        for i in range(sentences[pos].start, sentences[pos].end + 1)
        if _is_person(ners[i]) and not (open_idx <= i <= close_idx)
    ]
    if candidates:
        # Ties prefer the token AFTER the quote: "..." said Harry is the norm.
        best = min(candidates, key=lambda i: (min(abs(i - open_idx), abs(i - close_idx)), i <= close_idx))
        return forms[best]
    if touched and touched[0] > 0:
        previous = sentences[touched[0] - 1]
        behind = [i for i in range(previous.start, previous.end + 1) if _is_person(ners[i])]
        if behind:
            return forms[behind[-1]]
    return None


def _attribute(
    forms: list[str], ners: list[object], sentences: list[_Sentence], open_idx: int, close_idx: int
) -> tuple[str, str]:
    """(Speaker, Cue) through the two-stage sieve; ("", "none") when nothing fires."""
    touched = _touched(sentences, open_idx, close_idx)
    if not touched:
        return ("", "none")
    scope = list(touched)
    if touched[0] > 0:
        scope.append(touched[0] - 1)
    if touched[-1] < len(sentences) - 1:
        scope.append(touched[-1] + 1)
    for pos in scope:
        hit = _cue_in_sentence(forms, sentences[pos])
        if hit is not None:
            return hit
    person = _nearest_person(forms, ners, sentences, touched, open_idx=open_idx, close_idx=close_idx)
    if person is not None:
        return (person, "nearest-person")
    return ("", "none")


def _quote_spans(forms: list[str]) -> tuple[list[tuple[int, int]], int]:
    """(open, close) index pairs plus the count of unclosed opening quotes.

    Straight double quotes toggle per document (odd opens, even closes);
    typographic quotes pair by shape. A stray close is ignored; an unclosed
    open is dropped and counted -- a half-quote is data damage, not dialogue.
    """
    spans: list[tuple[int, int]] = []
    open_idx: int | None = None
    unclosed = 0
    for i, form in enumerate(forms):
        token = form.strip()
        if token == _LEFT_DOUBLE:
            if open_idx is None:
                open_idx = i
        elif token == _RIGHT_DOUBLE:
            if open_idx is not None:
                spans.append((open_idx, i))
                open_idx = None
        elif token == _DOUBLE:
            if open_idx is None:
                open_idx = i
            else:
                spans.append((open_idx, i))
                open_idx = None
    if open_idx is not None:
        unclosed += 1
    return spans, unclosed


def annotate_quotes(frame: pd.DataFrame, *, min_length: int = 1) -> Result[pd.DataFrame]:
    """Double-quote spans per document with sieve-attributed speakers.

    Rows: Document, Document ID, Quote ID (per document), Quote (inside text),
    Speaker, Cue (said|said-to|nearest-person|none), Sentence ID (the quote's
    opening sentence). Spans with fewer than ``min_length`` words are not
    quotes; empty input yields the empty frame with its columns.
    """
    if min_length < 1:
        return Result.failure(
            Diagnostic.error("QUOTE_BAD_MIN_LENGTH", f"min_length must be >=1, got {min_length}", min_length=min_length)
        )
    for need in (Col.FORM.value, Col.NER.value, Col.SENTENCE_ID.value, Col.DOCUMENT_ID.value):
        if need not in frame.columns:
            return Result.failure(Diagnostic.error("QUOTE_MISSING_COLUMN", f"missing {need!r}", missing=need))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_QUOTE_COLUMNS))

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    rows: list[dict[str, object]] = []
    diags: list[Diagnostic] = []
    for doc_id, group in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        doc = str(group[doc_col].iloc[0]) if doc_col is not None else ""
        forms = [str(x) for x in group[Col.FORM.value].tolist()]
        ners = list(group[Col.NER.value].tolist())
        sids = list(group[Col.SENTENCE_ID.value].tolist())
        sentences = _sentences(sids)
        spans, unclosed = _quote_spans(forms)
        if unclosed:
            diags.append(
                Diagnostic.warning(
                    "QUOTE_UNCLOSED",
                    f"{doc or doc_id}: {unclosed} unclosed opening quote(s) dropped",
                    document=doc,
                    document_id=str(doc_id),
                    unclosed=unclosed,
                )
            )
        quote_id = 0
        for open_idx, close_idx in spans:
            quote = _detokenize(forms[open_idx + 1 : close_idx])
            if _word_count(quote) < min_length:
                continue
            quote_id += 1
            speaker, cue = _attribute(forms, ners, sentences, open_idx, close_idx)
            rows.append(
                {
                    "Document": doc,
                    "Document ID": str(doc_id),
                    "Quote ID": quote_id,
                    "Quote": quote,
                    "Speaker": speaker,
                    "Cue": cue,
                    "Sentence ID": sids[open_idx],
                }
            )
    return Result.success(pd.DataFrame(rows, columns=_QUOTE_COLUMNS), *diags)


def summarize_quotes(annotated: pd.DataFrame) -> Result[pd.DataFrame]:
    """Per-document dialogue tally: Quotes, Attributed, Unattributed, Words.

    Words is the total quoted word count (the same measure as ``min_length``).
    """
    for need in ("Document", "Document ID", "Quote", "Speaker"):
        if need not in annotated.columns:
            return Result.failure(Diagnostic.error("QUOTE_MISSING_COLUMN", f"missing {need!r}", missing=need))
    if annotated.empty:
        return Result.success(pd.DataFrame(columns=_SUMMARY_COLUMNS))
    rows: list[dict[str, object]] = []
    for (doc_id, doc), group in annotated.groupby(["Document ID", "Document"], sort=False):
        speakers = [str(s) for s in group["Speaker"]]
        attributed = sum(1 for s in speakers if s.strip())
        rows.append(
            {
                "Document": doc,
                "Document ID": doc_id,
                "Quotes": len(group),
                "Attributed": attributed,
                "Unattributed": len(group) - attributed,
                "Words": sum(_word_count(str(q)) for q in group["Quote"]),
            }
        )
    return Result.success(pd.DataFrame(rows, columns=_SUMMARY_COLUMNS))
