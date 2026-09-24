"""Sentence division.

Two legacy bugs were suite-wide, and these tests exist specifically so they
cannot come back. Both are pinned against the smallest input that reproduces
them, and both would fail against the legacy
``CoNLL_record_division`` / ``sentence_division``.
"""

from __future__ import annotations

import pandas as pd

from core.conll.division import sentence_records, sentence_tokens, sentences_text
from core.conll.schema import Col

REQUIRED = [
    "ID",
    "Form",
    "Lemma",
    "POS",
    "NER",
    "Head",
    "DepRel",
    "Record ID",
    "Sentence ID",
    "Document ID",
    "Document",
]


def single_sentence_frame() -> pd.DataFrame:
    """One document, one sentence. The legacy sentinel bug's minimal trigger."""
    rows: list[list[object]] = [
        [1, "Hello", "hello", "UH", "O", 0, "root", 1, 1, "1", "a.txt"],
        [2, "world", "world", "NN", "O", 1, "vocative", 2, 1, "1", "a.txt"],
        [3, ".", ".", ".", "O", 1, "punct", 3, 1, "1", "a.txt"],
    ]
    return pd.DataFrame(rows, columns=REQUIRED)


class TestFinalSentence:
    def test_last_sentence_is_not_dropped(self, conll_frame: pd.DataFrame) -> None:
        """Legacy: accumulated tokens and flushed only on change, so the final
        sentence of the table was never emitted."""
        result = sentences_text(conll_frame)
        assert result.ok
        assert result.unwrap()[-1] == "Ladies and gentlemen ."

    def test_all_three_sentences_are_present(self, conll_frame: pd.DataFrame) -> None:
        result = sentences_text(conll_frame)
        assert result.unwrap() == (
            "The president went to Italy .",
            "He stayed .",
            "Ladies and gentlemen .",
        )

    def test_single_sentence_table_yields_exactly_one_sentence(self) -> None:
        result = sentences_text(single_sentence_frame())
        assert result.unwrap() == ("Hello world .",)


class TestNoPhantomSentence:
    def test_no_empty_sentences(self, conll_frame: pd.DataFrame) -> None:
        """Legacy seeded Document_ID_prev='1.0'; a table whose first document is
        '1' flushed an empty accumulator on row 0, producing a phantom empty
        sentence that callers had to skip by hand."""
        result = sentence_records(conll_frame)
        assert result.ok
        assert all(sentence.token_count > 0 for sentence in result.unwrap())

    def test_single_sentence_table_has_no_leading_empty(self) -> None:
        result = sentence_records(single_sentence_frame())
        assert len(result.unwrap()) == 1
        assert result.unwrap()[0].token_count == 3


class TestRowConservation:
    def test_every_row_belongs_to_exactly_one_sentence(self, conll_frame: pd.DataFrame) -> None:
        """Property: division is a partition of the table's rows."""
        result = sentence_records(conll_frame)
        assert result.ok

        assigned = [row for sentence in result.unwrap() for row in sentence.rows]
        assert sorted(assigned) == list(range(len(conll_frame)))
        assert len(assigned) == len(set(assigned))

    def test_token_count_sum_equals_row_count(self, conll_frame: pd.DataFrame) -> None:
        result = sentence_records(conll_frame)
        assert sum(s.token_count for s in result.unwrap()) == len(conll_frame)


class TestGrouping:
    def test_groups_across_documents(self, conll_frame: pd.DataFrame) -> None:
        result = sentence_records(conll_frame)
        identities = [(s.document_id, s.sentence_id) for s in result.unwrap()]
        assert identities == [("1", "1"), ("1", "2"), ("2", "1")]

    def test_order_is_first_appearance_not_sorted(self, conll_frame: pd.DataFrame) -> None:
        """Grouping must not reorder the table."""
        result = sentence_records(conll_frame)
        first_rows = [s.rows[0] for s in result.unwrap()]
        assert first_rows == sorted(first_rows)
        assert first_rows == [0, 6, 9]


class TestFields:
    def test_lemma_field(self, conll_frame: pd.DataFrame) -> None:
        result = sentences_text(conll_frame, field=Col.LEMMA)
        assert result.unwrap()[0] == "the president go to Italy ."

    def test_tokens_are_returned_per_sentence(self, conll_frame: pd.DataFrame) -> None:
        result = sentence_tokens(conll_frame)
        assert result.ok
        assert result.unwrap()[1] == ("He", "stayed", ".")

    def test_custom_separator(self, conll_frame: pd.DataFrame) -> None:
        result = sentences_text(conll_frame, separator="")
        assert result.unwrap()[1] == "Hestayed."


class TestFailureShape:
    def test_empty_table_yields_zero_sentences(self) -> None:
        empty = pd.DataFrame(columns=REQUIRED)
        result = sentences_text(empty)
        assert result.ok
        assert result.unwrap() == ()

    def test_missing_grouping_column_fails_with_a_result(self, conll_frame: pd.DataFrame) -> None:
        """Legacy wrapped everything in a bare `except:` and returned None."""
        broken = conll_frame.drop(columns=[Col.SENTENCE_ID.value])
        result = sentences_text(broken)
        assert not result.ok
        assert result.value is None
        assert any(d.code == "CONLL_MISSING_COLUMN" for d in result.errors)

    def test_missing_field_column_fails_with_a_result(self, conll_frame: pd.DataFrame) -> None:
        broken = conll_frame.drop(columns=[Col.LEMMA.value])
        result = sentences_text(broken, field=Col.LEMMA)
        assert not result.ok
        assert any(d.code == "CONLL_MISSING_COLUMN" for d in result.errors)

    def test_no_silent_none_on_success(self, conll_frame: pd.DataFrame) -> None:
        """R7: a successful division returns a value, never None-as-empty."""
        result = sentences_text(single_sentence_frame())
        assert result.value is not None
