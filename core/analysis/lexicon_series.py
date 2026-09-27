"""Named word lists counted across a corpus facet, as rates.

This is dictionary-based content analysis, the method most of the hand-written
research scripts in ``scripts/`` reimplement from scratch: name a category,
list the words that stand for it, count them per document, divide by how much
was said, and look at the result over time.

Three things make it a tool rather than a ``Counter``:

*A category is not a word.* "Civil rights" is five phrases, "Immigration" is
fifteen words. Tracing a single term answers a smaller question than anyone
actually asks, and phrases have to match as phrases -- ``civil rights`` is not
``civil`` near ``rights``.

*A count is not an answer.* Speeches differ in length by a factor of five, so a
raw count of "poverty" over time is largely a chart of how long each speech was.
Every rate here is per 1,000 tokens, and the count it came from stays in the
table beside it so the arithmetic is checkable.

*Zero is a measurement.* A facet where a category never appears is a row with
zero, not a missing row. Dropping it would make a gap in a series look like an
absence of data rather than an absence of the subject.

The optional ``within`` lexicon is the conditional question: restrict counting
to the sentences that match it, and the table answers "when they talked about
X, what vocabulary came with it" instead of "how often did they say Y".
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.corpus_axis import Axis
from core.io.document_fields import FIELD_VALUE_PREFIX
from core.result import Diagnostic, Result

__all__ = [
    "BY_CHOICES",
    "BY_FIELD_PREFIX",
    "RAW_WORD",
    "facet_labels",
    "lexicon_series",
    "parse_lexicon",
    "raw_sentences",
    "raw_text_series",
    "term_hits",
]

#: What counts are grouped along. ``order`` is each chapter or session value;
#: ``period`` is the axis cut into blocks ("Chapters 1-15"); ``field:<name>``
#: (any name, see :data:`BY_FIELD_PREFIX`) is one document detail's values.
BY_CHOICES: tuple[str, ...] = ("year", "decade", "document", "pattern", "order", "period")

#: A ``by`` of ``field:Party`` groups by the Party detail. The prefix keeps a
#: detail named "pattern" from being read as one of :data:`BY_CHOICES`.
BY_FIELD_PREFIX = FIELD_VALUE_PREFIX

# Categories are separated by a semicolon or a newline so the whole lexicon fits
# on one line of a form and also survives being pasted in from a file.
_CATEGORY_SPLIT = re.compile(r"[;\n]")
_MAX_CATEGORIES = 64
_MAX_TERMS = 2000


def parse_lexicon(text: str) -> Result[dict[str, tuple[str, ...]]]:
    """Read ``Name: term, term; Other: term`` into categories and their terms.

    Plain terms, deliberately, not regexes. The word lists in the research
    scripts are plain words, a mistyped regex fails in ways that look like a
    finding rather than an error, and a phrase like ``civil rights`` reads the
    same to everyone. Matching is case-insensitive, so the terms are lowered
    here once rather than at every comparison.
    """
    lexicon: dict[str, tuple[str, ...]] = {}
    for chunk in _CATEGORY_SPLIT.split(text):
        entry = chunk.strip()
        if not entry:
            continue
        if ":" not in entry:
            return Result.failure(
                Diagnostic.error(
                    "LEXICON_NO_CATEGORY",
                    f"every group needs a name: write 'Name: word, word', not {entry!r}",
                )
            )
        raw_name, raw_terms = entry.split(":", 1)
        name = raw_name.strip()
        if not name:
            return Result.failure(Diagnostic.error("LEXICON_EMPTY_NAME", "a group name cannot be blank"))
        terms = tuple(dict.fromkeys(t.strip().lower() for t in raw_terms.split(",") if t.strip()))
        if not terms:
            return Result.failure(Diagnostic.error("LEXICON_EMPTY_GROUP", f"{name!r} has no words after the colon"))
        if name in lexicon:
            return Result.failure(Diagnostic.error("LEXICON_DUPLICATE", f"{name!r} is named twice"))
        lexicon[name] = terms
    if not lexicon:
        return Result.failure(Diagnostic.error("LEXICON_EMPTY", "no word groups given; write 'Name: word, word'"))
    if len(lexicon) > _MAX_CATEGORIES:
        return Result.failure(
            Diagnostic.error("LEXICON_TOO_MANY", f"at most {_MAX_CATEGORIES} groups, got {len(lexicon)}")
        )
    total = sum(len(terms) for terms in lexicon.values())
    if total > _MAX_TERMS:
        return Result.failure(Diagnostic.error("LEXICON_TOO_MANY_TERMS", f"at most {_MAX_TERMS} words, got {total}"))
    return Result.success(lexicon)


def facet_labels(
    documents: Sequence[tuple[str, date | None]],
    by: str,
    *,
    group_pattern: str = "",
    details: Mapping[str, Mapping[str, str]] | None = None,
    positions: Mapping[str, float] | None = None,
    noun: str = "",
) -> Result[dict[str, str]]:
    """Label each document with the group it belongs to on this axis.

    Documents that cannot be labelled are left out and reported, never given a
    fallback label. A speech with no readable date silently filed under "0000"
    or "unknown" would sit in the chart as if it were evidence.

    ``details`` is document name -> its document details, for ``field:<name>``
    (``by="field:Party"``); ``positions`` is document name -> its Order value,
    for ``order`` and ``period``; ``noun`` names one order step ("Chapter"),
    so a period reads "Chapters 1-15".
    """
    field_name = by[len(BY_FIELD_PREFIX) :].strip() if by.startswith(BY_FIELD_PREFIX) else ""
    if by not in BY_CHOICES and not field_name:
        return Result.failure(
            Diagnostic.error(
                "LEXICON_BAD_BY", f"by must be one of {BY_CHOICES} or '{BY_FIELD_PREFIX}<detail name>', got {by!r}"
            )
        )
    if by in ("order", "period") and not positions:
        return Result.failure(Diagnostic.error("LEXICON_NO_ORDER", f"by={by!r} needs each document's Order detail"))
    if field_name and not details:
        return Result.failure(Diagnostic.error("LEXICON_NO_DETAILS", f"by={by!r} needs the documents' details"))
    pattern: re.Pattern[str] | None = None
    if by == "pattern":
        if not group_pattern.strip():
            return Result.failure(
                Diagnostic.error("LEXICON_NO_PATTERN", "by='pattern' needs group-pattern, a regex over document names")
            )
        try:
            pattern = re.compile(group_pattern)
        except re.error as exc:
            return Result.failure(Diagnostic.error("LEXICON_BAD_REGEX", f"invalid group-pattern regex: {exc}"))
    periods = _periods(positions or {}, documents, noun) if by == "period" else {}

    labels: dict[str, str] = {}
    undated: list[str] = []
    unmatched: list[str] = []
    for name, when in documents:
        if by == "document":
            labels[name] = name
        elif by == "pattern":
            assert pattern is not None  # noqa: S101 - guarded above
            found = pattern.search(name)
            if found is None:
                unmatched.append(name)
            else:
                # The first capture group when there is one, so a pattern can
                # name the group; otherwise whatever matched.
                labels[name] = (found.group(1) if found.groups() else found.group(0)).strip()
        elif by == "order":
            at = (positions or {}).get(name)
            if at is None:
                unmatched.append(name)
            else:
                labels[name] = f"{noun} {_step(at)}" if noun else _step(at)
        elif by == "period":
            bucket = periods.get(name)
            if bucket is None:
                unmatched.append(name)
            else:
                labels[name] = bucket
        elif field_name:
            detail_value = (details or {}).get(name, {}).get(field_name)
            if detail_value:
                labels[name] = detail_value
            else:
                unmatched.append(name)
        elif when is None:
            undated.append(name)
        elif by == "year":
            labels[name] = f"{when.year:04d}"
        else:
            labels[name] = f"{when.year // 10 * 10:04d}s"

    notes: list[Diagnostic] = []
    if undated:
        notes.append(
            Diagnostic.warning(
                "LEXICON_UNDATED",
                f"{len(undated)} document(s) have no readable date and are left out: {', '.join(undated[:3])}"
                + ("…" if len(undated) > 3 else ""),
            )
        )
    if unmatched:
        why = "did not match group-pattern" if by == "pattern" else f"have no value to group by ({by})"
        notes.append(
            Diagnostic.warning(
                "LEXICON_UNMATCHED",
                f"{len(unmatched)} document name(s) {why} and are left out: "
                + ", ".join(unmatched[:3])
                + ("…" if len(unmatched) > 3 else ""),
            )
        )
    if not labels:
        return Result[dict[str, str]](
            None,
            (*notes, Diagnostic.error("LEXICON_NO_FACET", "no document could be placed on this axis")),
        )
    return Result.success(labels, *notes)


def _step(value: float) -> str:
    """An order value as a label: "3", never "3.0"."""
    return f"{int(value)}" if float(value).is_integer() else f"{value:g}"


def _periods(positions: Mapping[str, float], documents: Sequence[tuple[str, date | None]], noun: str) -> dict[str, str]:
    """The axis's period labels ("Chapters 1-15"), by document name."""
    if positions:
        return Axis("order", noun.strip() or "Document", dict(positions)).periods()
    years = {name: float(when.year) for name, when in documents if when is not None}
    return Axis("time", "Year", years).periods()


def _index(lexicon: Mapping[str, Sequence[str]]) -> dict[str, list[tuple[tuple[str, ...], str]]]:
    """Terms keyed by their first word.

    Scanning every term against every token is a third of a billion string
    comparisons on a corpus this size. Keyed on the first word, almost every
    token costs one failed dict lookup and the phrases are only assembled where
    they could possibly match.
    """
    index: dict[str, list[tuple[tuple[str, ...], str]]] = {}
    for category, terms in lexicon.items():
        for term in terms:
            words = tuple(term.lower().split())
            if words:
                index.setdefault(words[0], []).append((words, category))
    return index


# Hyphens a parser splits into their own tokens. "Viet-Nam" becomes three
# tokens, so a phrase written "viet nam" would miss every 1960s mention of it
# and report near-silence on the subject of the decade. Only separators, never
# words: skipping arbitrary tokens would let "civil" and "rights" three words
# apart count as "civil rights".
# U+2010..U+2014 are hyphen, non-breaking hyphen, figure dash, en dash and
# em dash. Written as code points because the characters are
# indistinguishable from "-" on screen, which is the whole reason a parser
# treats them alike and a reader cannot see which one a corpus used.
_JOINERS = frozenset({"-", "/", *(chr(point) for point in range(0x2010, 0x2015))})


def _spans(tokens: list[str], start: int, words: tuple[str, ...]) -> bool:
    """Whether *words* run from *start*, allowing a hyphen between them."""
    position = start
    for offset, word in enumerate(words):
        if offset:
            while position < len(tokens) and tokens[position] in _JOINERS:
                position += 1
        if position >= len(tokens) or tokens[position] != word:
            return False
        position += 1
    return True


def _hits(tokens: list[str], index: dict[str, list[tuple[tuple[str, ...], str]]]) -> dict[str, int]:
    """How many times each category's terms occur in one sentence."""
    counts: dict[str, int] = {}
    for position, token in enumerate(tokens):
        for words, category in index.get(token, ()):
            if len(words) == 1 or _spans(tokens, position, words):
                counts[category] = counts.get(category, 0) + 1
    return counts


def lexicon_series(
    frame: pd.DataFrame,
    lexicon: Mapping[str, Sequence[str]],
    labels: Mapping[str, str],
    *,
    field: Col = Col.LEMMA,
    within: Mapping[str, Sequence[str]] | None = None,
) -> Result[pd.DataFrame]:
    """Count each category per facet and turn the counts into rates."""
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(
            Diagnostic.error("LEXICON_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}")
        )
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if frame.empty:
        return Result.failure(Diagnostic.error("LEXICON_EMPTY_FRAME", "frame is empty"))
    for needed in (Col.DOCUMENT, Col.SENTENCE_ID, field):
        if needed.value not in frame.columns:
            return Result.failure(Diagnostic.error("LEXICON_MISSING_COLUMN", f"frame needs the {needed.value} column"))
    if not lexicon:
        return Result.failure(Diagnostic.error("LEXICON_EMPTY", "no word groups to count"))

    index = _index(lexicon)
    anchor = _index(within) if within else None

    documents = frame[Col.DOCUMENT.value].astype(str).to_numpy()
    sentences = frame[Col.SENTENCE_ID.value].astype(str).to_numpy()
    tokens = frame[field.value].fillna("").astype(str).str.lower().to_numpy()

    categories = list(lexicon)
    facets = sorted(set(labels.values()))
    occurrences: dict[tuple[str, str], int] = {(f, c): 0 for f in facets for c in categories}
    matching: dict[tuple[str, str], int] = dict.fromkeys(occurrences, 0)
    counted_tokens: dict[str, int] = dict.fromkeys(facets, 0)
    counted_sentences: dict[str, int] = dict.fromkeys(facets, 0)
    seen_documents: dict[str, set[str]] = {facet: set() for facet in facets}

    def close(document: str, sentence: list[str]) -> None:
        """Fold one finished sentence into the totals."""
        facet = labels.get(document)
        if facet is None or not sentence:
            return
        if anchor is not None and not _hits(sentence, anchor):
            return
        counted_tokens[facet] += len(sentence)
        counted_sentences[facet] += 1
        seen_documents[facet].add(document)
        for category, count in _hits(sentence, index).items():
            occurrences[(facet, category)] += count
            matching[(facet, category)] += 1

    current: list[str] = []
    key: tuple[str, str] | None = None
    for position in range(len(tokens)):
        here = (documents[position], sentences[position])
        if here != key:
            if key is not None:
                close(key[0], current)
            key, current = here, []
        current.append(tokens[position])
    if key is not None:
        close(key[0], current)

    rows = [
        {
            "Facet": facet,
            "Category": category,
            "Documents": len(seen_documents[facet]),
            "Sentences": counted_sentences[facet],
            "Tokens": counted_tokens[facet],
            "Occurrences": occurrences[(facet, category)],
            "Per 1000": round(1000 * occurrences[(facet, category)] / counted_tokens[facet], 6)
            if counted_tokens[facet]
            else 0.0,
            "Matching Sentences": matching[(facet, category)],
            "Sentence Percent": round(100 * matching[(facet, category)] / counted_sentences[facet], 6)
            if counted_sentences[facet]
            else 0.0,
        }
        # Sorted by facet then category, so the same question always writes the
        # same file whatever order the documents arrived in.
        for facet in facets
        for category in categories
    ]
    table = pd.DataFrame(rows)

    notes: list[Diagnostic] = []
    empty = [facet for facet in facets if not counted_sentences[facet]]
    if empty and anchor is not None:
        notes.append(
            Diagnostic.warning(
                "LEXICON_NO_ANCHOR",
                f"{len(empty)} group(s) have no sentence matching the 'within' words: {', '.join(empty[:5])}",
            )
        )
    silent = [c for c in categories if not any(occurrences[(f, c)] for f in facets)]
    if silent:
        notes.append(
            Diagnostic.warning(
                "LEXICON_NEVER_FOUND",
                f"{len(silent)} group(s) never occur in this corpus: {', '.join(silent[:5])}. "
                "Check the field -- lemmas are dictionary forms, so 'immigrants' is 'immigrant'.",
            )
        )
    return Result.success(table, *notes)


def term_hits(words: Sequence[str], lexicon: Mapping[str, Sequence[str]]) -> dict[str, int]:
    """How many times each group's terms occur in one sentence's lowercased *words*.

    The matcher both counts use, exposed for callers that need hits per
    sentence (the script library's passages) rather than totals, so a passage
    is found by exactly the rule that counted it.
    """
    return _hits(list(words), _index(lexicon))


#: What the raw-text count calls a word: a run of ASCII letters, lowercased.
#: The rule the hand-written research scripts used (``re.findall(r"[A-Za-z]+",
#: text.lower())``), kept so their published numbers can be reproduced exactly.
#: "don't" is two words here, "covid-19" is one, and numbers are not words.
RAW_WORD = re.compile(r"[a-z]+")
_RAW_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def raw_sentences(text: str) -> list[str]:
    """*text* cut into sentences the raw way: after ``.``, ``!`` or ``?`` and a space.

    Crude on purpose (titles like "Mr." end a sentence here), because it is the
    rule the raw count uses; a passage has to be cut where the count cut it.
    """
    return [part.strip() for part in _RAW_SENTENCE_END.split(text) if part.strip()]


def raw_text_series(texts: Mapping[str, str], lexicon: Mapping[str, Sequence[str]]) -> Result[pd.DataFrame]:
    """The same table as :func:`lexicon_series`, counted in raw text, one row per document.

    Parsed counting (``lexicon_series``) reads the parser's tokens and, by
    default, their lemmas: "immigrants" is counted as "immigrant". Raw counting
    reads the text as written, lowercased, split into runs of letters, with no
    parser at all. The two give different numbers for the same word list, and
    neither is wrong -- a paper has to say which it used. This one exists so a
    script written the raw way (and figures already published from one) can be
    reproduced to the last digit.

    *texts* maps a document name to its text. Phrases match consecutive words
    within a sentence (sentences end at ``.``, ``!`` or ``?`` followed by
    space); single words match anywhere.
    """
    if not lexicon:
        return Result.failure(Diagnostic.error("LEXICON_EMPTY", "no word groups to count"))
    if not texts:
        return Result.failure(Diagnostic.error("LEXICON_EMPTY_FRAME", "no documents to count"))
    index = _index(lexicon)
    categories = list(lexicon)
    notes: list[Diagnostic] = []
    unmatchable = sorted(
        {term for terms in lexicon.values() for term in terms if not all(RAW_WORD.fullmatch(w) for w in term.split())}
    )
    if unmatchable:
        notes.append(
            Diagnostic.warning(
                "LEXICON_RAW_UNMATCHABLE",
                "raw counting only sees runs of letters, so these terms can never match: "
                + ", ".join(unmatchable[:5])
                + ("…" if len(unmatchable) > 5 else ""),
            )
        )
    rows: list[dict[str, object]] = []
    for name, text in texts.items():
        sentences = [RAW_WORD.findall(part.lower()) for part in raw_sentences(text)]
        sentences = [words for words in sentences if words]
        tokens = sum(len(words) for words in sentences)
        occurrences = dict.fromkeys(categories, 0)
        matching = dict.fromkeys(categories, 0)
        for words in sentences:
            for category, count in _hits(words, index).items():
                occurrences[category] += count
                matching[category] += 1
        for category in categories:
            rows.append(
                {
                    "Facet": name,
                    "Category": category,
                    "Documents": 1,
                    "Sentences": len(sentences),
                    "Tokens": tokens,
                    "Occurrences": occurrences[category],
                    "Per 1000": round(1000 * occurrences[category] / tokens, 6) if tokens else 0.0,
                    "Matching Sentences": matching[category],
                    "Sentence Percent": round(100 * matching[category] / len(sentences), 6) if sentences else 0.0,
                }
            )
    return Result.success(pd.DataFrame(rows), *notes)
