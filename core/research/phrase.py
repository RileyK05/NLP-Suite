"""Occurrence-preserving phrase research over a canonical token table.

The legacy n-gram tools answer which sequences are frequent.  This module
answers a different question: where a requested sequence occurs, how its use
changes over time, and where it falls inside each document.  Aggregates are
derived from the complete occurrence set; pagination applies only to the
evidence returned to the interface.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import re
from typing import TYPE_CHECKING, Any
import unicodedata

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from core.io.reader import Corpus, Document

_REQUIRED = {"Form", "Sentence ID", "Document ID"}
_WORD = re.compile(r"[^\W_]", re.UNICODE)
_QUERY_PART = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)?|[^\w\s]", re.UNICODE)
_CLITICS = (
    "n't",
    "'re",
    "'ve",
    "'ll",
    "'d",
    "'m",
    "'s",
    "\u2019re",
    "\u2019ve",
    "\u2019ll",
    "\u2019d",
    "\u2019m",
    "\u2019s",
)
#: Bumped whenever what counts as a match changes. Version 1 tokenized queries
#: with the regex below regardless of which parser produced the table, so
#: ``U.S.``, ``COVID-19``, ``3.5`` and ``AT&T`` could not be found at all.
#: Version 2 asks the snapshot's own tokenizer. Version 3 added the three
#: widenings below, each off by default: a version 2 question and a version 3
#: question with nothing ticked are the same question, and re-asking one under
#: the other is not a migration. A saved question records the version it was
#: resolved under, so reopening it can say whether today's answer is the same
#: question or a migrated one.
MATCHING_PROFILE_VERSION = 3

#: Characters that mean the same thing and are typed differently. A query
#: pasted from a word processor carries curly quotes and en dashes; the
#: document it is asked about may carry straight ones, or the other way round,
#: and the two never meet. Folding both sides is what makes ``COVID\u201119``
#: and ``COVID-19`` one phrase rather than two absences.
_SAME_CHARACTER = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u02bc": "'",
    "\u00b4": "'",
    "\u0060": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    "\u00a0": " ",
    "\u2026": "...",
}
#: Nouns are checked for a base verb only when they end like a nominalization.
#: The same prefilter the nominalization analysis uses, and the reason this
#: option costs a lookup for a few thousand words rather than for every one.
_NOMINAL_SUFFIXES_3 = ("ent", "ing", "ion")
_NOMINAL_SUFFIXES_4 = ("ance", "ence")

#: Splits a phrase into the tokens the corpus was parsed into. Supplied by the
#: snapshot; see :meth:`core.pipelines.protocol.Pipeline.tokenize`.
Tokenizer = Callable[[str], Sequence[str]]

MAX_PHRASE_CHARACTERS = 300
MAX_PHRASE_TOKENS = 32
MIN_POSITION_BINS = 5
MAX_POSITION_BINS = 50
MAX_EVIDENCE_ROWS = 250
#: Positions carried on one document's track. A pathological document (a
#: thousand repetitions of the phrase) would otherwise ship its whole
#: occurrence list twice -- once as evidence, once as positions -- so the
#: track keeps the first N in document order and says it stopped.
MAX_TRACK_POSITIONS = 2000
#: How far a source-reader request may reach from one occurrence. The reader
#: fetches a window of the imported text; without a bound, "expand the
#: context" on a long speech would ship the whole document for every step.
MAX_READER_RADIUS = 20_000
#: The last code point a browser stores in one UTF-16 code unit. Above it a
#: character is a surrogate pair, and Python's index and the browser's part ways.
LAST_SINGLE_UNIT_CODE_POINT = 0xFFFF
SOURCE_CONTEXT_CHARACTERS = 100


@dataclass(frozen=True, slots=True)
class PhraseQuery:
    """The explicit, versioned interpretation of a literal phrase question.

    The three widenings are independent and additive: each *adds* forms that
    count, and none removes one. With all three off this is exactly the
    literal question it has always been, which is why they default to off and
    why a version 2 question re-asked here is not a migration.

    They are separate flags rather than one mode because they answer different
    questions and are wanted in different combinations. "Did they say
    ``COVID-19``, however it was typed?" is normalization alone. "How often do
    they promise anything?" is lemma alone. "Do they talk about governing or
    about government?" wants nominalization, and usually wants lemma with it.
    """

    text: str
    case_sensitive: bool = False
    position_bins: int = 10
    #: Fold away the differences that are typography rather than wording:
    #: curly against straight quotes, the several dashes, accents, and the
    #: non-breaking space. Applies to both sides of every comparison.
    normalize: bool = False
    #: Count every inflection of the word. ``promise`` then also counts
    #: ``promised``, ``promises`` and ``promising``, because the corpus's own
    #: parse says they share a lemma.
    match_lemma: bool = False
    #: Count the noun a verb turns into, and the verb a noun came from.
    #: ``govern`` then also counts ``government`` and ``governance``. Needs
    #: WordNet and a POS column; without either it is reported as unavailable
    #: rather than silently doing nothing.
    match_nominalization: bool = False

    @property
    def widened(self) -> bool:
        """Whether anything but the literal phrase counts."""
        return self.normalize or self.match_lemma or self.match_nominalization

    def tokens(self, tokenize: Tokenizer | None = None) -> tuple[str, ...]:
        """The query as tokens, split the way the corpus was split.

        *tokenize* comes from the snapshot that will be searched. Without it
        this falls back to :func:`_approximate_tokens`, which is a guess at
        one particular parser's rules and is wrong for abbreviations,
        hyphenated terms, decimals and symbols. Callers that have a snapshot
        must pass its tokenizer; the fallback exists for tables that arrived
        without one, and says so through :meth:`profile`.
        """
        text = self.text.strip()
        if not text:
            raise ValueError("Enter a phrase to track.")
        if len(text) > MAX_PHRASE_CHARACTERS:
            raise ValueError("A tracked phrase may be at most 300 characters.")
        if tokenize is not None:
            tokens = [str(token) for token in tokenize(text) if str(token).strip()]
        else:
            tokens = _approximate_tokens(text)
        if not tokens:
            raise ValueError("That phrase contains no trackable tokens.")
        if len(tokens) > MAX_PHRASE_TOKENS:
            raise ValueError("Track at most 32 tokens in one phrase.")
        return tuple(tokens)

    def resolved(self, tokens: Sequence[str]) -> tuple[str, ...]:
        """The tokens an earlier resolution of this phrase produced.

        A saved question records what its phrase was split into, and asking it
        again uses that record rather than splitting the text afresh. Without
        it, a parser upgrade -- or publishing on a machine whose model differs
        -- would quietly answer a different question under the saved question's
        name. Re-splitting is a migration, and a migration should be visible.
        """
        cleaned = tuple(str(token) for token in tokens if str(token).strip())
        if not cleaned:
            raise ValueError("That saved question records no trackable tokens.")
        if len(cleaned) > MAX_PHRASE_TOKENS:
            raise ValueError("Track at most 32 tokens in one phrase.")
        return cleaned

    def validate(self, tokenize: Tokenizer | None = None) -> None:
        self.tokens(tokenize)
        self.validate_shape()

    def validate_shape(self) -> None:
        """Everything about the question that does not depend on tokenizing."""
        if not MIN_POSITION_BINS <= self.position_bins <= MAX_POSITION_BINS:
            raise ValueError("Choose between 5 and 50 position bins.")


def _approximate_tokens(text: str) -> list[str]:
    """A parser-free guess at token boundaries.

    Only for callers with no snapshot. It approximates spaCy's English rules
    for words and clitics and gets abbreviations, hyphenation, decimals and
    symbols wrong, which is exactly why a query resolved against a real corpus
    does not use it.
    """
    tokens: list[str] = []
    for token in _QUERY_PART.findall(text):
        folded = token.casefold()
        clitic = next((suffix for suffix in _CLITICS if folded.endswith(suffix) and len(token) > len(suffix)), None)
        if clitic is None:
            tokens.append(token)
            continue
        cut = len(clitic)
        stem, suffix = token[:-cut], token[-cut:]
        # spaCy's English tokenizer uses "ca"/"wo" before n't.
        if suffix.casefold() == "n't" and stem.casefold() in {"can", "won"}:
            stem = stem[:-1]
        tokens.extend((stem, suffix))
    return tokens


@dataclass(frozen=True, slots=True)
class Tokenization:
    """A tokenizer and the name an answer records it under.

    One value because they are one fact. Passing the function and the label
    separately allows an answer to say it was matched by a tokenizer that did
    not match it, and that is the one part of the record a reader has no way
    to check.
    """

    split: Tokenizer
    name: str = ""


@dataclass(frozen=True, slots=True)
class Accepted:
    """What counts as a match, resolved against one corpus.

    One frozenset of comparison keys per token of the query, plus the record
    of how they were arrived at. The keys are already in comparison form --
    folded and cased exactly as :func:`_normal` will render a corpus token --
    so the scan stays a set membership test per token and costs what the
    equality test it replaced cost.

    Resolving against the corpus rather than against a dictionary is the
    whole idea. "Which words here are inflections of *promise*?" is answered
    by the parse that is already in memory and already right about this text;
    a general-purpose morphology would be a second opinion about the corpus,
    and a second opinion is not a smaller error than a wrong answer.
    """

    #: Per query token, every corpus form that counts as it.
    keys: tuple[frozenset[str], ...]
    #: The surface forms those keys stand for, for each token, in corpus
    #: frequency order. This is what the interface shows: a reader who ticks
    #: "count every inflection" has a right to see which words are now being
    #: counted, rather than to watch a number go up.
    forms: tuple[tuple[str, ...], ...] = ()
    #: Why an option that was asked for did nothing, in words. Empty when
    #: every option asked for was applied.
    unavailable: tuple[str, ...] = ()


def _literal(tokens: Sequence[str], query: PhraseQuery) -> Accepted:
    """The phrase and nothing else -- what this has always done."""
    keys = tuple(frozenset({_normal(token, query.case_sensitive, query.normalize)}) for token in tokens)
    return Accepted(keys=keys, forms=tuple((token,) for token in tokens))


def _nominal_candidate(word: str) -> bool:
    """The cheap prefilter before a WordNet lookup, as the analysis has it."""
    return word[-3:] in _NOMINAL_SUFFIXES_3 or word[-4:] in _NOMINAL_SUFFIXES_4


def _base_verbs(words: Sequence[str]) -> tuple[dict[str, str], str]:
    """Each word mapped to the verb it was derived from, where there is one.

    Returns the mapping and, when the lookup could not be made at all, the
    reason -- so an option that did nothing says so instead of quietly
    behaving like an option that was never ticked.

    ``default_wordnet_module`` hands back a ``Result``, not a module. Passing
    the Result itself to the backend is not a crash: ``base_verb`` catches
    everything and answers None, so every word came back underived and the
    option appeared to work while doing nothing at all. Unwrapping it here,
    and carrying its diagnostic as the reason, is what makes the difference
    visible.
    """
    try:
        # Deferred: NLTK is the optional 'wordnet' extra, and this module has
        # to import -- and every literal question has to answer -- on an
        # install that does not have it.
        from core.analysis.nominalization import (  # noqa: PLC0415
            NltkNominalization,
            default_wordnet_module,
        )
    except ImportError:  # pragma: no cover - the package is an optional extra
        return {}, "Matching nominalizations needs NLTK, which is not installed."
    resolved = default_wordnet_module()
    if resolved.value is None:
        # Carry the diagnostic's own fix. A reader told only that something is
        # unavailable has to go and find out what to do about it, and the
        # thing that knows already said so.
        first = next(iter(resolved.diagnostics), None)
        detail = first.message if first else "the WordNet data is not available"
        fix = str(first.context.get("fix", "")) if first else ""
        return {}, f"Nominalizations were not matched: {detail}." + (f" Fix: {fix}" if fix else "")
    backend = NltkNominalization(resolved.unwrap())
    found = {word: base for word in words if (base := backend.base_verb(word))}
    if not found and words:
        return {}, "WordNet derived no verbs from the nouns in these documents, so nothing was added."
    return found, ""


def _column(table: pd.DataFrame, name: str, query: PhraseQuery) -> pd.Series | None:
    """One column of the table in comparison form, or None if it is absent."""
    if name not in table.columns:
        return None
    values = table[name].fillna("").astype(str)
    if query.normalize:
        values = values.map(_fold)
    return values if query.case_sensitive else values.str.casefold()


def _accepted(table: pd.DataFrame, tokens: Sequence[str], query: PhraseQuery) -> Accepted:
    """Resolve the query's tokens against this corpus, once.

    Every widening is expressed the same way -- as more surface forms that
    count as this token -- because that is what they all reduce to, and
    because it keeps the per-token scan a set lookup however many options are
    on. It also makes the answer explainable: the forms behind each token are
    exactly the words that will be counted, and they are returned.
    """
    if not query.widened or table.empty:
        return _literal(tokens, query)

    forms = _column(table, "Form", query)
    if forms is None:  # pragma: no cover - Form is required upstream
        return _literal(tokens, query)
    lemmas = _column(table, "Lemma", query)
    unavailable: list[str] = []

    if query.match_lemma and lemmas is None:
        unavailable.append("Matching every inflection needs a Lemma column, which this parse does not have.")
    nominal: dict[str, str] = {}
    if query.match_nominalization:
        if lemmas is None:
            unavailable.append("Matching nominalizations needs a Lemma column, which this parse does not have.")
        elif "POS" not in table.columns:
            unavailable.append("Matching nominalizations needs a POS column, which this parse does not have.")
        else:
            nouns = table["POS"].fillna("").astype(str).map(_is_noun_tag)
            candidates = sorted({word for word in lemmas[nouns].unique() if word and _nominal_candidate(word)})
            nominal, reason = _base_verbs(candidates)
            if reason:
                unavailable.append(reason)

    counts = forms.value_counts()
    by_lemma: dict[str, set[str]] = {}
    if lemmas is not None:
        for form, lemma in zip(forms, lemmas, strict=True):
            by_lemma.setdefault(lemma, set()).add(form)
    # Every lemma that shares a base verb, so "govern" reaches "government"
    # and "government" reaches back to "govern".
    by_base: dict[str, set[str]] = {}
    for lemma, base in nominal.items():
        by_base.setdefault(base, set()).add(lemma)

    widen = _Widening(query, by_lemma, by_base, nominal)
    keys: list[frozenset[str]] = []
    shown: list[tuple[str, ...]] = []
    for token in tokens:
        accepted = widen.forms_for(_normal(token, query.case_sensitive, query.normalize))
        present = [form for form in accepted if form in counts.index]
        present.sort(key=lambda form: (-int(counts[form]), form))
        keys.append(frozenset(accepted))
        shown.append(tuple(present) or (token,))
    return Accepted(keys=tuple(keys), forms=tuple(shown), unavailable=tuple(unavailable))


@dataclass(frozen=True, slots=True)
class _Widening:
    """Everything one query token needs to know to widen itself.

    A small object rather than a long parameter list, because the four things
    it holds are all indexes over the same table and are only ever used
    together. Holding them lets the per-token rule read as the two sentences
    it is.
    """

    query: PhraseQuery
    #: Forms sharing each lemma. Empty when this parse has no Lemma column,
    #: which makes both lemma-based widenings add nothing -- and which
    #: :func:`_accepted` has already reported as unavailable, so an empty
    #: index here is a quiet no-op rather than a silent one.
    by_lemma: dict[str, set[str]]
    #: Lemmas sharing each base verb.
    by_base: dict[str, set[str]]
    #: Each noun lemma's base verb.
    nominal: dict[str, str]

    def forms_for(self, wanted: str) -> set[str]:
        """Every form in this corpus that counts as *wanted*."""
        accepted = {wanted}
        if self.query.match_lemma:
            accepted |= self._inflections(wanted)
        if self.query.match_nominalization:
            accepted |= self._derivations(wanted)
        return accepted

    def _inflections(self, wanted: str) -> set[str]:
        """The lemmas this corpus gives the typed word, then their forms.

        The word may itself be a lemma that never appears as a surface form
        ("be" in a corpus that only ever writes "is" and "are"), so it counts
        as one of its own.
        """
        lemmas = {wanted} | {lemma for lemma, group in self.by_lemma.items() if wanted in group}
        return {form for lemma in lemmas for form in self.by_lemma.get(lemma, set())}

    def _derivations(self, wanted: str) -> set[str]:
        """The other side of a derivation, as this corpus writes it.

        A derivation relates two *lemmas*, so what it adds is the other lemma
        and its forms -- deliberately not the typed word's own inflections.
        "govern" with nominalization ticked and inflections unticked should
        reach "government", not "governed"; adding those here is what made
        this option look like it worked while WordNet was returning nothing.
        """
        if not self.by_lemma:
            return set()
        related: set[str] = set(self.by_base.get(wanted, set()))
        base = self.nominal.get(wanted, "")
        if base:
            related.add(base)
            related |= self.by_base.get(base, set())
        related.discard(wanted)
        return {form for lemma in related for form in (self.by_lemma.get(lemma) or {lemma})}


def _is_noun_tag(pos: str) -> bool:
    """Penn ``NN*`` or Universal ``NOUN``/``PROPN``, as the analysis has it."""
    from core.analysis.postags import is_noun_tag  # noqa: PLC0415

    return is_noun_tag(pos)


@dataclass(frozen=True, slots=True)
class EvidenceRequest:
    """Paging and drill-down apply to evidence, never to the aggregates."""

    offset: int = 0
    limit: int = 100
    year: int | None = None
    document_id: str | None = None
    #: The position range a brush selected, in the document's own relative
    #: coordinates (0.0 to 1.0). None means unfiltered. An occurrence counts
    #: when its *start* falls in the range -- the same rule that placed it in
    #: a bin -- so a phrase crossing an edge is counted once, and brushing
    #: bin 3 selects exactly the occurrences that bin counts.
    position_start: float | None = None
    position_end: float | None = None
    all_rows: bool = False

    def validate(self) -> None:
        if self.offset < 0:
            raise ValueError("Evidence offset cannot be negative.")
        if not self.all_rows and not 1 <= self.limit <= MAX_EVIDENCE_ROWS:
            raise ValueError("Request between 1 and 250 evidence rows.")
        for name in ("position_start", "position_end"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError("A position filter runs from 0.0 to 1.0 of a document.")
        if (
            self.position_start is not None
            and self.position_end is not None
            and self.position_start >= self.position_end
        ):
            raise ValueError("A position filter needs an end after its start.")


def _fold(value: str) -> str:
    """One spelling for characters that are one character.

    Compatibility-decomposed first so that a precomposed ``\u00e9`` and a
    typed ``e`` + combining accent fold together, then the combining marks are
    dropped, then the punctuation nobody distinguishes when reading is
    unified. Deliberately not a stemmer: nothing here changes a word, only how
    it is spelled.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return "".join(_SAME_CHARACTER.get(ch, ch) for ch in stripped)


def _normal(value: str, case_sensitive: bool, normalize: bool = False) -> str:
    """One token as it is compared. Both sides always go through here."""
    folded = _fold(value) if normalize else value
    return folded if case_sensitive else folded.casefold()


def _is_word(value: str) -> bool:
    return _WORD.search(value) is not None


def _join(tokens: list[str]) -> str:
    """Readable token context without claiming source-character fidelity."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+(['\u2019](?:s|re|ve|ll|d|m|t))\b", r"\1", text, flags=re.IGNORECASE)
    return text


def _align_tokens(text: str, forms: list[str]) -> list[tuple[int, int] | None]:
    """Prove token spans against the imported text, or return no spans.

    Searching ahead for a token after a mismatch is tempting and wrong: a
    repeated word later in the document could be mistaken for the parser token
    we failed to align.  One mismatch therefore discards the whole document's
    spans, which makes whitespace the delicate part.

    Canonical token tables usually omit whitespace, so the scan steps over it
    in the source -- but a parser may also emit whitespace *as* a token, and
    spaCy does: 10,017 of them across the 87 speeches this was developed
    against, including the newline that begins every one of those files.
    Skipping the source's whitespace before trying to match a whitespace token
    failed on the first token of all 87, so no source passage in the entire
    corpus could be pointed back at the text it came from, and the interface
    said the parser could not be aligned to the text -- which sounded like a
    property of the documents rather than of this loop.
    """
    spans: list[tuple[int, int] | None] = []
    cursor = 0
    for form in forms:
        if not form:
            return [None] * len(forms)
        if form.isspace():
            # Located inside the current run of whitespace and never past it,
            # so this cannot skip over content to find a later match.
            run = cursor
            while run < len(text) and text[run].isspace():
                run += 1
            found = text.find(form, cursor, run)
            if found < 0:
                return [None] * len(forms)
            spans.append((found, found + len(form)))
            cursor = found + len(form)
            continue
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        end = cursor + len(form)
        if text[cursor:end] != form:
            return [None] * len(forms)
        spans.append((cursor, end))
        cursor = end
    return spans


def _astral_positions(text: str) -> list[int]:
    """Where the characters live that the browser counts twice.

    Python indexes by code point and a browser by UTF-16 code unit, and the two
    agree everywhere except above U+FFFF, where one character is a surrogate
    pair. Finding those once per document turns each offset conversion into a
    binary search, instead of re-encoding the whole prefix of the document for
    every occurrence -- twice per occurrence, which on a long speech with
    hundreds of matches is most of the work of answering the question.
    """
    return [index for index, character in enumerate(text) if ord(character) > LAST_SINGLE_UNIT_CODE_POINT]


def _utf16_offset(astral: list[int], codepoint_offset: int) -> int:
    """Python's Unicode index as the browser's UTF-16 index."""
    return codepoint_offset + bisect_right(astral, codepoint_offset - 1)


def _source_evidence(
    text: str,
    astral: list[int],
    alignment: list[tuple[int, int] | None],
    token_start: int,
    token_end: int,
) -> dict[str, Any]:
    selected = alignment[token_start:token_end]
    if not selected or any(span is None for span in selected):
        return {
            "character_start": None,
            "character_end": None,
            "browser_character_start": None,
            "browser_character_end": None,
            "character_offset_unit": None,
            "left_source": None,
            "match_source": None,
            "right_source": None,
            "exact_source_highlight": False,
        }
    verified = [span for span in selected if span is not None]
    start, end = verified[0][0], verified[-1][1]
    context_start = max(0, start - SOURCE_CONTEXT_CHARACTERS)
    context_end = min(len(text), end + SOURCE_CONTEXT_CHARACTERS)
    return {
        "character_start": start,
        "character_end": end,
        "browser_character_start": _utf16_offset(astral, start),
        "browser_character_end": _utf16_offset(astral, end),
        "character_offset_unit": "unicode_code_point",
        "left_source": text[context_start:start],
        "match_source": text[start:end],
        "right_source": text[end:context_end],
        "exact_source_highlight": True,
    }


def _doc_name(doc: Document) -> str:
    return doc.label or doc.path.name


def _doc_source_id(doc: Document) -> str:
    return doc.source_id or doc.sha256 or str(doc.doc_id)


def _occurrence_id(snapshot: str, doc: Document, location: tuple[str, int, int], phrase: str) -> str:
    sentence, start, end = location
    payload = "\x00".join((snapshot, _doc_source_id(doc), sentence, str(start), str(end), phrase))
    return sha256(payload.encode("utf-8")).hexdigest()[:24]


def _rows_by_document(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The table split by document, once.

    The per-document lookup converted and compared the whole Document ID
    column every time, so the cost of finding each document's rows grew with
    the size of the corpus times the number of documents in it.
    """
    ids = table["Document ID"].astype(str)
    return {str(key): group for key, group in table.groupby(ids, sort=False)}


def _position_bin(word_start: int, word_tokens: int, bins: int) -> int:
    if word_tokens <= 0:
        return 0
    return min(int((word_start / word_tokens) * bins), bins - 1)


def _time_rows(corpus: Corpus, doc_words: dict[str, int], doc_occurrences: dict[str, int]) -> list[dict[str, Any]]:
    dated_years = sorted({doc.date.year for doc in corpus.docs if doc.date})
    if not dated_years:
        return []
    result: list[dict[str, Any]] = []
    for year in range(dated_years[0], dated_years[-1] + 1):
        year_docs = [doc for doc in corpus.docs if doc.date and doc.date.year == year]
        words = sum(doc_words[_doc_source_id(doc)] for doc in year_docs)
        count = sum(doc_occurrences[_doc_source_id(doc)] for doc in year_docs)
        matching = sum(doc_occurrences[_doc_source_id(doc)] > 0 for doc in year_docs)
        result.append(
            {
                "year": year,
                "observed": bool(year_docs),
                "occurrences": count if year_docs else None,
                "word_tokens": words if year_docs else None,
                "eligible_documents": len(year_docs),
                "matching_documents": matching,
                "occurrences_per_10000": (count / words * 10_000) if words else None,
                "document_prevalence_percent": (matching / len(year_docs) * 100) if year_docs else None,
            }
        )
    return result


def _position_rows(
    occurrences: list[dict[str, Any]],
    exposure: list[int],
    contributing: list[set[str]],
) -> list[dict[str, Any]]:
    bins = len(exposure)
    counts = [0] * bins
    for occurrence in occurrences:
        counts[int(occurrence["position_bin"]) - 1] += 1
    return [
        {
            "bin": index + 1,
            "start_percent": index / bins * 100,
            "end_percent": (index + 1) / bins * 100,
            "occurrences": counts[index],
            "word_tokens": tokens,
            "contributing_documents": len(contributing[index]),
            "occurrences_per_10000": (counts[index] / tokens * 10_000) if tokens else None,
        }
        for index, tokens in enumerate(exposure)
    ]


def _in_position_range(occurrence: dict[str, Any], start: float | None, end: float | None) -> bool:
    """Whether one occurrence's *start* position falls in the brushed range.

    Start-assignment is the counting rule the bins already use: an occurrence
    belongs to the bin its start falls in even if the phrase crosses the bin's
    edge. A brush that selects a range therefore selects the same occurrences
    the corresponding bins count, which is what keeps "brush two bins" and
    "click one bin" the same question.
    """
    relative = occurrence["relative_position"]
    if relative is None:
        return start is None and end is None
    if start is not None and relative < start:
        return False
    return not (end is not None and relative >= end)


def _evidence_page(occurrences: list[dict[str, Any]], request: EvidenceRequest) -> dict[str, Any]:
    filtered = occurrences
    if request.year is not None:
        filtered = [item for item in filtered if item["year"] == request.year]
    if request.document_id is not None:
        filtered = [item for item in filtered if item["document_id"] == request.document_id]
    if request.position_start is not None or request.position_end is not None:
        filtered = [item for item in filtered if _in_position_range(item, request.position_start, request.position_end)]
    page = filtered if request.all_rows else filtered[request.offset : request.offset + request.limit]
    verified = sum(bool(item["exact_source_highlight"]) for item in occurrences)
    status = "verified" if occurrences and verified == len(occurrences) else "partial" if verified else "unavailable"
    # Reported for the filtered set as well as the whole corpus. The overall
    # status alone said "partial" over a filtered set whose every span is
    # verified, which reads as a warning about the highlights on screen.
    shown = sum(bool(item["exact_source_highlight"]) for item in filtered)
    shown_status = "verified" if filtered and shown == len(filtered) else "partial" if shown else "unavailable"
    return {
        "rows": page,
        "total": len(occurrences),
        "filtered_total": len(filtered),
        "offset": request.offset,
        "limit": request.limit,
        "truncated": False if request.all_rows else request.offset + len(page) < len(filtered),
        "filter": {
            "year": request.year,
            "document_id": request.document_id,
            "position_start": request.position_start,
            "position_end": request.position_end,
        },
        "source_offsets": {
            "status": status,
            "unit": "unicode_code_point",
            "browser_unit": "utf16_code_unit",
            "verified": verified,
            "total": len(occurrences),
            "filtered_status": shown_status,
            "filtered_verified": shown,
            "filtered_total": len(filtered),
        },
    }


def _document_occurrences(
    doc: Document,
    rows: pd.DataFrame,
    query: PhraseQuery,
    wanted: tuple[frozenset[str], ...],
    snapshot_id: str,
) -> tuple[list[dict[str, Any]], int]:
    """Every occurrence in one document, and how many words it has.

    Matching runs sentence by sentence because a phrase does not cross a
    sentence boundary, but both coordinates an occurrence reports -- its token
    offset and its word position -- belong to the document. Keeping the two
    document-level running totals here, next to the sentence loop that advances
    them, is what stops one of them being reset with the other.

    *wanted* is one set of acceptable forms per token of the query, resolved
    against the whole corpus once by :func:`_accepted`. A literal question
    gives sets of one and this is the equality test it was; a widened question
    gives larger sets and costs the same per token.
    """
    forms = [str(value) for value in rows["Form"].fillna("").tolist()]
    alignment = _align_tokens(doc.text, forms)
    astral = _astral_positions(doc.text) if any(alignment) else []
    total_words = sum(_is_word(form) for form in forms)
    source_id = _doc_source_id(doc)
    found: list[dict[str, Any]] = []

    document_token_offset = 0
    # Words counted from the start of the document, not of the sentence.
    # Restarting this at each sentence put every match in the first bin of its
    # own sentence, so a phrase ten words into the second sentence of a speech
    # read as 0% of the way through the speech.
    document_word_offset = 0
    width = len(wanted)
    for sentence_id, sentence in rows.groupby("Sentence ID", sort=False, dropna=False):
        sentence_forms = [str(value) for value in sentence["Form"].fillna("").tolist()]
        normalized = [_normal(value, query.case_sensitive, query.normalize) for value in sentence_forms]
        word_before = [0]
        for form in sentence_forms:
            word_before.append(word_before[-1] + int(_is_word(form)))
        for start in range(max(0, len(normalized) - width + 1)):
            if any(normalized[start + offset] not in accept for offset, accept in enumerate(wanted)):
                continue
            end = start + width
            word_start = document_word_offset + word_before[start]
            token_start = document_token_offset + start
            token_end = document_token_offset + end
            found.append(
                {
                    "id": _occurrence_id(snapshot_id, doc, (str(sentence_id), token_start, token_end), query.text),
                    "document_id": source_id,
                    "document": _doc_name(doc),
                    "content_sha256": doc.sha256,
                    "date": doc.date.isoformat() if doc.date else None,
                    "year": doc.date.year if doc.date else None,
                    "sentence_id": str(sentence_id),
                    "token_start": token_start,
                    "token_end": token_end,
                    "source_token_ids": [str(value) for value in sentence.iloc[start:end]["ID"].tolist()]
                    if "ID" in sentence.columns
                    else [],
                    "word_start": word_start,
                    "relative_position": (word_start / total_words) if total_words else None,
                    "position_bin": _position_bin(word_start, total_words, query.position_bins) + 1,
                    "left": _join(sentence_forms[max(0, start - 8) : start]),
                    # The words actually found, which a widened question does
                    # not guarantee are the words typed: tracking "promise"
                    # with inflections on finds "promised", and an evidence
                    # row that echoed the query back would hide that.
                    "match": _join(sentence_forms[start:end]),
                    "matched_forms": [str(value) for value in sentence_forms[start:end]],
                    "right": _join(sentence_forms[end : end + 8]),
                    "sentence": _join(sentence_forms),
                    **_source_evidence(doc.text, astral, alignment, token_start, token_end),
                }
            )
        document_token_offset += len(sentence_forms)
        document_word_offset += word_before[-1]
    return found, total_words


def _document_track(
    doc: Document,
    occurrences: list[dict[str, Any]],
    total_words: int,
    position_bins: int,
) -> dict[str, Any]:
    """One document's occurrences laid out along its own length.

    The per-document row answers the question the pooled bins cannot: does
    this phrase recur throughout this text, cluster at its end, or appear only
    once in one long speech? Positions come from the occurrences already
    computed -- no second pass over the token table -- and the pooled-bin rule
    is reused so the two views agree on what an occurrence's position is.
    """
    count = len(occurrences)
    return {
        "document_id": _doc_source_id(doc),
        "document": _doc_name(doc),
        "content_sha256": doc.sha256,
        "date": doc.date.isoformat() if doc.date else None,
        "year": doc.date.year if doc.date else None,
        "word_tokens": total_words,
        "occurrences": count,
        # Zero stays available as a real zero: a document without the phrase
        # belongs on the track, or a recurring pattern reads as absent.
        "positions": [
            {
                "word_start": item["word_start"],
                "relative_position": item["relative_position"],
                "position_bin": item["position_bin"],
                "sentence_id": item["sentence_id"],
                "occurrence_id": item["id"],
                # Carried here so clicking a point on the track opens the
                # exact passage without paging the evidence first: a reader
                # following one document should not have to find the page an
                # occurrence happens to sit on to read it.
                "character_start": item["character_start"],
                "character_end": item["character_end"],
                "browser_character_start": item["browser_character_start"],
                "browser_character_end": item["browser_character_end"],
                "exact_source_highlight": item["exact_source_highlight"],
            }
            for item in occurrences[:MAX_TRACK_POSITIONS]
        ],
        "positions_truncated": count > MAX_TRACK_POSITIONS,
        "position_bins": position_bins,
    }


def source_passage(
    corpus: Corpus,
    document_id: str,
    character_start: int,
    character_end: int | None = None,
    radius: int = SOURCE_CONTEXT_CHARACTERS,
) -> dict[str, Any]:
    """A window of one imported document's text around a verified offset.

    The evidence rows already carry ±100 characters of context. This is the
    reader's way to expand past that: the same document, the same offsets, a
    wider window -- returned as the exact characters between the bounds, not
    a line-snapped rendering of them, because a passage a reader follows has
    to begin where the evidence said it begins. The window is read from the
    loaded snapshot's own text, so a document re-imported since the answer
    was computed cannot masquerade as the one the offsets point into.
    """
    if radius < 0 or radius > MAX_READER_RADIUS:
        raise ValueError(f"Ask for up to {MAX_READER_RADIUS} characters of context.")
    if character_start < 0:
        raise ValueError("Character offsets cannot be negative.")
    end = character_start if character_end is None else character_end
    if end < character_start:
        raise ValueError("A passage cannot end before it starts.")
    for doc in corpus.docs:
        if _doc_source_id(doc) != document_id:
            continue
        if not doc.text or character_start >= len(doc.text):
            break
        window_start = max(0, character_start - radius)
        window_end = min(len(doc.text), end + radius)
        astral = _astral_positions(doc.text)
        return {
            "document_id": document_id,
            "document": _doc_name(doc),
            "content_sha256": doc.sha256,
            "character_start": window_start,
            "character_end": window_end,
            "browser_character_start": _utf16_offset(astral, window_start),
            "browser_character_end": _utf16_offset(astral, window_end),
            "character_offset_unit": "unicode_code_point",
            "text": doc.text[window_start:window_end],
        }
    raise ValueError("That passage's document is not part of this answer's corpus.")


def track_phrase(  # noqa: PLR0913 - the evidence, the question, and the rules that
    # decide what a word is; folding any of them together would hide one of them
    table: pd.DataFrame,
    corpus: Corpus,
    query: PhraseQuery,
    *,
    snapshot_id: str,
    evidence: EvidenceRequest | None = None,
    tokenize: Tokenizer | None = None,
    tokenizer_name: str = "",
    resolved_tokens: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Resolve a phrase question and retain the evidence behind every count.

    *tokenize* is the snapshot's own tokenizer and *tokenizer_name* identifies
    it in the record. Both should be given: a query split by different rules
    from the table it searches reports absence rather than disagreement, which
    is the one failure a reader cannot detect from the answer.

    *resolved_tokens* are the tokens a saved question was resolved into when it
    was first asked, and they take precedence over both. Publishing a saved
    question then asks the question that was saved, rather than whatever
    today's tokenizer makes of the same characters.
    """
    query.validate_shape()
    evidence = evidence or EvidenceRequest()
    evidence.validate()
    missing = sorted(_REQUIRED - set(table.columns))
    if missing:
        raise ValueError(f"The parsed corpus is missing required columns: {', '.join(missing)}")

    query_tokens = query.resolved(resolved_tokens) if resolved_tokens else query.tokens(tokenize)
    by_document = _rows_by_document(table)
    # A document the parse produced no rows for still belongs in the
    # denominator, so it needs an empty frame rather than to be skipped.
    empty = table.iloc[0:0]
    # Resolved against the whole table once, not per document: which words
    # count as "promise" is a fact about the corpus, and asking it 87 times
    # would give 87 different answers on 87 different documents.
    accepted = _accepted(table, query_tokens, query)
    wanted = accepted.keys
    occurrences: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    tracks: list[dict[str, Any]] = []
    doc_occurrences: dict[str, int] = {}
    doc_words: dict[str, int] = {}
    position_exposure = [0] * query.position_bins
    position_documents: list[set[str]] = [set() for _ in range(query.position_bins)]

    for doc in corpus.docs:
        source_id = _doc_source_id(doc)
        found, total_words = _document_occurrences(
            doc, by_document.get(str(doc.doc_id), empty), query, wanted, snapshot_id
        )
        occurrences.extend(found)
        doc_words[source_id] = total_words
        tracks.append(_document_track(doc, found, total_words, query.position_bins))
        for word_index in range(total_words):
            bin_index = _position_bin(word_index, total_words, query.position_bins)
            position_exposure[bin_index] += 1
            position_documents[bin_index].add(source_id)

        count = len(found)
        doc_occurrences[source_id] = count
        documents.append(
            {
                "document_id": source_id,
                "document": _doc_name(doc),
                "content_sha256": doc.sha256,
                "date": doc.date.isoformat() if doc.date else None,
                "year": doc.date.year if doc.date else None,
                "word_tokens": total_words,
                "occurrences": count,
                "occurrences_per_10000": (count / total_words * 10_000) if total_words else None,
                "contains_phrase": count > 0,
            }
        )

    total_words = sum(doc_words.values())
    matching_documents = sum(value > 0 for value in doc_occurrences.values())
    return {
        "schema_version": 1,
        "kind": "phrase_distribution",
        "snapshot_id": snapshot_id,
        "question": {
            "subject": {
                "type": "token_sequence",
                "text": query.text.strip(),
                "tokens": list(query_tokens),
                "field": "surface",
                "case_sensitive": query.case_sensitive,
                "boundary": "sentence",
                "punctuation": "preserve",
                "overlapping_matches": True,
                # What each typed token was taken to mean in this corpus. A
                # literal question lists the token itself; a widened one lists
                # every form being counted under it, commonest first, so the
                # reader can see what the tick box did rather than only that a
                # number went up.
                "counted_forms": [list(group) for group in accepted.forms],
            },
            # What "a match" meant when this answer was produced. Saved with
            # the question so reopening it can tell an identical question from
            # one the rules have moved under.
            "matching_profile": {
                "version": MATCHING_PROFILE_VERSION,
                "tokenizer": tokenizer_name or ("snapshot" if tokenize else "approximate"),
                "resolved_by_snapshot": tokenize is not None,
                # Which of the three rules produced the tokens above. "saved"
                # means the question carried them; "snapshot" means the loaded
                # parser split them; "approximate" means neither was available
                # and the guess in _approximate_tokens was used.
                "source": "saved" if resolved_tokens else "snapshot" if tokenize else "approximate",
                # The three widenings, as asked for. Recorded even when off,
                # because "this question did not count inflections" is a fact
                # about the answer and not the absence of one.
                "normalize": query.normalize,
                "match_lemma": query.match_lemma,
                "match_nominalization": query.match_nominalization,
                # An option that was asked for and could not be applied, in
                # words. An empty list means every option asked for was used.
                "unavailable": list(accepted.unavailable),
            },
            "position": {"coordinate": "relative_word_position", "bins": query.position_bins},
            "time": {"unit": "year"},
        },
        "summary": {
            "occurrences": len(occurrences),
            "word_tokens": total_words,
            "occurrences_per_10000": (len(occurrences) / total_words * 10_000) if total_words else None,
            "eligible_documents": len(corpus.docs),
            "matching_documents": matching_documents,
            "document_prevalence_percent": (matching_documents / len(corpus.docs) * 100) if corpus.docs else None,
            "dated_documents": sum(doc.date is not None for doc in corpus.docs),
            "undated_documents": sum(doc.date is None for doc in corpus.docs),
        },
        "time": _time_rows(corpus, doc_words, doc_occurrences),
        "position": _position_rows(occurrences, position_exposure, position_documents),
        "documents": documents,
        # One row per document, in corpus order, including documents with no
        # match: the track is the denominator's picture, not a top-N list.
        "tracks": tracks,
        "evidence": _evidence_page(occurrences, evidence),
    }


__all__ = [
    "MATCHING_PROFILE_VERSION",
    "Accepted",
    "EvidenceRequest",
    "PhraseQuery",
    "Tokenization",
    "Tokenizer",
    "source_passage",
    "track_phrase",
]
