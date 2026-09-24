"""Parameterized CoNLL word frequency (the 6→1 consolidation).

The legacy suite had six copies of this logic (noun/verb/adjective/adverb/
ratio/function) differing only in the predicate. Tests cover the engine
rather than the old outputs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import CANONICAL_MINIMAL, build_frame
from core.analysis.conll_wordlist import run
from core.conll.schema import Col


def _frame() -> pd.DataFrame:
    rows: list[list[object]] = [
        [1, "The", "the", "DT", "O", 2, "det", 1, 1, "1", "doc_a.txt"],
        [2, "cats", "cat", "NNS", "O", 3, "nsubj", 2, 1, "1", "doc_a.txt"],
        [3, "ran", "run", "VBD", "O", 0, "root", 3, 1, "1", "doc_a.txt"],
        [4, "quickly", "quickly", "RB", "O", 3, "advmod", 4, 1, "1", "doc_a.txt"],
        [5, "The", "the", "DT", "O", 2, "det", 5, 2, "1", "doc_a.txt"],
        [6, "dogs", "dog", "NNS", "O", 3, "nsubj", 6, 2, "1", "doc_a.txt"],
        [7, "ran", "run", "VBD", "O", 0, "root", 7, 2, "1", "doc_a.txt"],
        [8, "slowly", "slowly", "RB", "O", 7, "advmod", 8, 2, "1", "doc_a.txt"],
    ]
    return build_frame(rows, CANONICAL_MINIMAL)


class TestBasic:
    def test_counts_are_case_insensitive_by_default(self) -> None:
        result = run(_frame(), field=Col.FORM, category="all", top_n=None)
        assert result.ok
        words = {row.word: row.count for row in result.unwrap().rows}
        assert words["the"] == 2
        assert words["ran"] == 2
        assert "The" not in words

    def test_case_sensitive(self) -> None:
        result = run(_frame(), field=Col.FORM, category="all", top_n=None, case_sensitive=True)
        words = {row.word: row.count for row in result.unwrap().rows}
        assert words["The"] == 2
        assert "the" not in words

    def test_lemma_field(self) -> None:
        result = run(_frame(), field=Col.LEMMA, category="all", top_n=None)
        words = {row.word: row.count for row in result.unwrap().rows}
        assert words["cat"] == 1
        assert words["dog"] == 1
        assert words["run"] == 2

    def test_top_n(self) -> None:
        result = run(_frame(), field=Col.FORM, top_n=2)
        assert len(result.unwrap().rows) == 2

    def test_percentages_sum_to_100(self) -> None:
        result = run(_frame(), field=Col.FORM, category="all", top_n=None)
        total = sum(row.percentage for row in result.unwrap().rows)
        assert total == pytest.approx(100.0, abs=1e-5)

    def test_to_frame(self) -> None:
        result = run(_frame(), top_n=2)
        frame = result.unwrap().to_frame()
        assert list(frame.columns) == ["Word", "Count", "Percentage"]
        assert len(frame) == 2


class TestCategories:
    def test_noun_filter(self) -> None:
        result = run(_frame(), field=Col.FORM, category="noun", top_n=None)
        words = {row.word for row in result.unwrap().rows}
        assert words == {"cats", "dogs"}

    def test_verb_filter(self) -> None:
        result = run(_frame(), field=Col.FORM, category="verb", top_n=None)
        words = {row.word for row in result.unwrap().rows}
        assert words == {"ran"}

    def test_adverb_filter(self) -> None:
        result = run(_frame(), field=Col.FORM, category="adverb", top_n=None)
        words = {row.word for row in result.unwrap().rows}
        assert words == {"quickly", "slowly"}

    def test_function_filter(self) -> None:
        result = run(_frame(), field=Col.FORM, category="function", top_n=None)
        words = {row.word for row in result.unwrap().rows}
        assert "the" in words
        assert "cats" not in words


class TestFailures:
    def test_empty_frame(self) -> None:
        empty = pd.DataFrame(columns=CANONICAL_MINIMAL)
        result = run(empty)
        assert result.ok
        assert result.unwrap().rows == ()
        assert result.unwrap().total_tokens == 0

    def test_missing_column(self) -> None:
        frame = _frame().drop(columns=[Col.POS.value])
        result = run(frame)
        assert not result.ok
        assert any(d.code == "CONLL_MISSING_COLUMN" for d in result.errors)

    def test_unknown_category(self) -> None:
        result = run(_frame(), category="unknown")  # type: ignore[arg-type]
        assert not result.ok
        assert result.errors[0].code == "WORDLIST_BAD_CATEGORY"

    def test_bad_field(self) -> None:
        result = run(_frame(), field=Col.NER)
        assert not result.ok
        assert result.errors[0].code == "WORDLIST_BAD_FIELD"

    def test_bad_top_n(self) -> None:
        result = run(_frame(), top_n=0)
        assert not result.ok
        assert result.errors[0].code == "WORDLIST_BAD_TOP_N"
