"""CAP-ANNO-12 — quote/dialogue extraction + two-stage speaker sieve.

Frames are canonical CoNLL built from Col.*.value columns with quote marks
as their own tokens. Each sieve stage is pinned by hand-built dialogue:
reporting-verb cues (said / said-to), nearest-person fallback, and honest
"none". The docstring's promise -- a sieve, not Muzny et al.'s model -- is
what these tests hold it to.
"""

from __future__ import annotations

import pandas as pd

from core.analysis.quote_annotator import annotate_quotes, summarize_quotes
from core.conll.schema import Col

LEFT = "\u201c"
RIGHT = "\u201d"


def _conll(rows: list[tuple[str, str, str, int, int, str]]) -> pd.DataFrame:
    """(form, pos, ner, sentence_id, document_id, document) -> canonical CoNLL frame."""
    out: list[dict[str, object]] = []
    for i, (form, pos, ner, sent, doc_id, doc) in enumerate(rows, start=1):
        out.append(
            {
                Col.ID.value: i,
                Col.FORM.value: form,
                Col.LEMMA.value: form.lower(),
                Col.POS.value: pos,
                Col.NER.value: ner,
                Col.HEAD.value: 0,
                Col.DEPREL.value: "dep",
                Col.DEPS.value: "",
                Col.CLAUSE_TAG.value: "",
                Col.RECORD_ID.value: i,
                Col.SENTENCE_ID.value: sent,
                Col.DOCUMENT_ID.value: doc_id,
                Col.DOCUMENT.value: doc,
            }
        )
    return pd.DataFrame(out)


def _tokens(sents: list[list[tuple[str, str]]], *, doc_id: int = 1, doc: str = "a.txt") -> pd.DataFrame:
    """Sentence token lists of (form, ner) -> one document's frame."""
    rows: list[tuple[str, str, str, int, int, str]] = []
    for sent_id, sentence in enumerate(sents, start=1):
        for form, ner in sentence:
            pos = "PROPN" if ner == "PERSON" else "X"
            rows.append((form, pos, ner, sent_id, doc_id, doc))
    return _conll(rows)


class TestAttribution:
    def test_name_after_the_verb(self) -> None:
        frame = _tokens([[('"', "O"), ("Hello", "O"), ('"', "O"), ("said", "O"), ("John", "PERSON"), (".", "O")]])
        result = annotate_quotes(frame)
        assert result.ok, result.diagnostics
        row = result.unwrap().iloc[0]
        assert row["Quote"] == "Hello"
        assert row["Speaker"] == "John"
        assert row["Cue"] == "said"

    def test_name_before_the_verb(self) -> None:
        frame = _tokens([[("John", "PERSON"), ("said", "O"), ('"', "O"), ("Hello", "O"), ('"', "O"), (".", "O")]])
        result = annotate_quotes(frame)
        row = result.unwrap().iloc[0]
        assert row["Speaker"] == "John"
        assert row["Cue"] == "said"

    def test_said_to_x_has_its_own_cue(self) -> None:
        frame = _tokens([[('"', "O"), ("Hello", "O"), ('"', "O"), ("said", "O"), ("to", "O"), ("Mary", "PERSON")]])
        result = annotate_quotes(frame)
        row = result.unwrap().iloc[0]
        assert row["Speaker"] == "Mary"
        assert row["Cue"] == "said-to"

    def test_nearest_person_in_the_quote_sentence(self) -> None:
        frame = _tokens([[('"', "O"), ("Hello", "O"), ('"', "O"), ("Harry", "PERSON"), ("left", "O"), (".", "O")]])
        result = annotate_quotes(frame)
        row = result.unwrap().iloc[0]
        assert row["Speaker"] == "Harry"
        assert row["Cue"] == "nearest-person"

    def test_nearest_person_falls_back_to_the_sentence_before(self) -> None:
        frame = _tokens(
            [
                [("Mary", "PERSON"), ("left", "O"), (".", "O")],
                [('"', "O"), ("Hello", "O"), ('"', "O"), (".", "O")],
            ]
        )
        result = annotate_quotes(frame)
        row = result.unwrap().iloc[0]
        assert row["Speaker"] == "Mary"
        assert row["Cue"] == "nearest-person"
        assert row["Sentence ID"] == 2  # the quote's own sentence

    def test_no_cue_and_no_person_is_honest_none(self) -> None:
        frame = _tokens([[('"', "O"), ("Hello", "O"), ('"', "O"), ("a", "O"), ("storm", "O"), ("came", "O")]])
        result = annotate_quotes(frame)
        row = result.unwrap().iloc[0]
        assert row["Speaker"] == ""
        assert row["Cue"] == "none"

    def test_typographic_quotes_pair_by_shape(self) -> None:
        frame = _tokens([[(LEFT, "O"), ("Hello", "O"), (RIGHT, "O"), ("said", "O"), ("John", "PERSON")]])
        result = annotate_quotes(frame)
        row = result.unwrap().iloc[0]
        assert row["Quote"] == "Hello"
        assert row["Speaker"] == "John"

    def test_cue_search_reaches_an_adjacent_sentence(self) -> None:
        frame = _tokens(
            [
                [('"', "O"), ("Hello", "O"), ('"', "O"), (".", "O")],
                [("Harry", "PERSON"), ("said", "O"), ("later", "O"), (".", "O")],
            ]
        )
        result = annotate_quotes(frame)
        row = result.unwrap().iloc[0]
        assert row["Speaker"] == "Harry"
        assert row["Cue"] == "said"

    def test_quote_id_is_per_document_and_reading_order(self) -> None:
        frame = _tokens(
            [
                [('"', "O"), ("One", "O"), ('"', "O"), ("said", "O"), ("John", "PERSON"), (".", "O")],
                [('"', "O"), ("Two words", "O"), ('"', "O"), (".", "O")],
            ]
        )
        result = annotate_quotes(frame)
        assert list(result.unwrap()["Quote ID"]) == [1, 2]

    def test_multi_sentence_quote_keeps_its_opening_sentence(self) -> None:
        frame = _tokens(
            [
                [('"', "O"), ("One", "O"), (".", "O")],
                [("Two", "O"), (".", "O")],
                [('"', "O"), ("said", "O"), ("John", "PERSON"), (".", "O")],
            ]
        )
        result = annotate_quotes(frame)
        row = result.unwrap().iloc[0]
        assert row["Sentence ID"] == 1
        assert row["Speaker"] == "John"


class TestFiltersAndWarnings:
    def test_min_length_drops_short_spans(self) -> None:
        frame = _tokens([[('"', "O"), ("Hi", "O"), ('"', "O"), (".", "O")]])
        result = annotate_quotes(frame, min_length=2)
        assert result.ok
        assert result.unwrap().empty

    def test_unclosed_quote_is_dropped_with_a_warning(self) -> None:
        frame = _tokens([[('"', "O"), ("Hello there", "O"), (".", "O")]])
        result = annotate_quotes(frame)
        assert result.ok
        assert result.unwrap().empty
        diag = next(d for d in result.diagnostics if d.code == "QUOTE_UNCLOSED")
        assert diag.severity.value == "WARNING"

    def test_stray_typographic_close_is_ignored(self) -> None:
        # A lone straight quote would toggle open; a stray shape-close is inert.
        frame = _tokens([[("He", "O"), (RIGHT, "O"), ("left", "O"), (".", "O")]])
        result = annotate_quotes(frame)
        assert result.ok
        assert result.unwrap().empty
        assert not any(d.code == "QUOTE_UNCLOSED" for d in result.diagnostics)


class TestEmptyContract:
    def test_empty_frame_yields_the_declared_columns(self) -> None:
        empty = _tokens([[("Hello", "O")]]).iloc[:0]
        result = annotate_quotes(empty)
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert list(frame.columns) == ["Document", "Document ID", "Quote ID", "Quote", "Speaker", "Cue", "Sentence ID"]
        assert len(frame) == 0

    def test_empty_summary_yields_the_declared_columns(self) -> None:
        empty = pd.DataFrame(columns=["Document", "Document ID", "Quote ID", "Quote", "Speaker", "Cue", "Sentence ID"])
        result = summarize_quotes(empty)
        assert result.ok
        frame = result.unwrap()
        assert list(frame.columns) == ["Document", "Document ID", "Quotes", "Attributed", "Unattributed", "Words"]
        assert len(frame) == 0


class TestLoudFailures:
    def test_bad_min_length_is_loud(self) -> None:
        frame = _tokens([[('"', "O"), ("Hello", "O"), ('"', "O")]])
        result = annotate_quotes(frame, min_length=0)
        assert not result.ok
        assert result.diagnostics[0].code == "QUOTE_BAD_MIN_LENGTH"

    def test_missing_tool_column_is_loud(self) -> None:
        frame = _tokens([[('"', "O"), ("Hello", "O"), ('"', "O")]]).drop(columns=[Col.NER.value])
        result = annotate_quotes(frame)
        assert not result.ok
        assert result.diagnostics[0].code == "QUOTE_MISSING_COLUMN"

    def test_noncanonical_frame_is_loud(self) -> None:
        frame = pd.DataFrame(
            {
                Col.FORM.value: ['"'],
                Col.NER.value: ["O"],
                Col.SENTENCE_ID.value: [1],
                Col.DOCUMENT_ID.value: [1],
            }
        )
        result = annotate_quotes(frame)
        assert not result.ok
        assert result.diagnostics[0].code == "CONLL_MISSING_COLUMN"

    def test_missing_summary_column_is_loud(self) -> None:
        result = summarize_quotes(pd.DataFrame({"Document": ["a.txt"]}))
        assert not result.ok
        assert result.diagnostics[0].code == "QUOTE_MISSING_COLUMN"


class TestSummary:
    def test_tallies_and_quoted_word_counts(self) -> None:
        # The second quote sits past the adjacent-sentence reach of the cue,
        # so the sieve honestly leaves it unattributed.
        frame = _tokens(
            [
                [('"', "O"), ("Hello there", "O"), ('"', "O"), ("said", "O"), ("John", "PERSON"), (".", "O")],
                [("It", "O"), ("rained", "O"), (".", "O")],
                [('"', "O"), ("Bye", "O"), ('"', "O"), (".", "O")],
            ]
        )
        result = annotate_quotes(frame)
        summary = summarize_quotes(result.unwrap())
        row = summary.unwrap().iloc[0]
        assert row["Quotes"] == 2
        assert row["Attributed"] == 1
        assert row["Unattributed"] == 1
        assert row["Words"] == 3  # "Hello there" + "Bye"
