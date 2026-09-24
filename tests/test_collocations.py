"""CAP-NGRAM-04 — collocation association measures.

The measures are checked against hand-computed values and against scipy where
an independent implementation exists, not against whatever the code happens to
emit.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from core.analysis.collocations import SPAN_MODES, _g2, collocations
from core.conll.schema import Col


def _table(sentences: list[list[str]], *, document: str = "a.txt") -> pd.DataFrame:
    """Minimal canonical CoNLL table: one row per token."""
    rows: list[dict[str, object]] = []
    for sentence_id, tokens in enumerate(sentences, start=1):
        for token_id, token in enumerate(tokens, start=1):
            rows.append(
                {
                    Col.ID.value: token_id,
                    Col.FORM.value: token,
                    Col.LEMMA.value: token.lower(),
                    Col.POS.value: "NOUN",
                    Col.NER.value: "",
                    Col.HEAD.value: 0,
                    Col.DEPREL.value: "root",
                    Col.DEPS.value: "",
                    Col.CLAUSE_TAG.value: "",
                    Col.RECORD_ID.value: len(rows) + 1,
                    Col.SENTENCE_ID.value: sentence_id,
                    Col.DOCUMENT_ID.value: 1,
                    Col.DOCUMENT.value: document,
                }
            )
    return pd.DataFrame(rows)


class TestMeasures:
    def test_hand_computed_row(self) -> None:
        """One pair, arithmetic done by hand from the contingency table."""
        # "alpha bravo" twice; each word occurs exactly twice; N = 4 tokens.
        frame = _table([["alpha", "bravo"], ["alpha", "bravo"]])
        result = collocations(frame, min_count=2)
        assert result.ok
        row = result.unwrap().iloc[0]

        observed, f1, f2, total = 2, 2, 2, 4
        expected = f1 * f2 / total  # 1.0
        # The engine rounds display columns to 4 decimals, so round the
        # hand-computed value the same way rather than loosening the tolerance.
        assert row["Co-occurrences"] == observed
        assert row["Expected"] == pytest.approx(expected)
        assert row["PMI"] == pytest.approx(round(math.log2(observed / expected), 4))  # 1.0
        assert row["T-Score"] == pytest.approx(round((observed - expected) / math.sqrt(observed), 4))
        assert row["Z-Score"] == pytest.approx(round((observed - expected) / math.sqrt(expected), 4))
        assert row["Dice"] == pytest.approx(2 * observed / (f1 + f2))  # 1.0
        assert row["Log Dice"] == pytest.approx(14.0)  # log2(1) == 0

    def test_ppmi_clamps_negative_association_at_zero(self) -> None:
        # "alpha" is common and "zulu" rare; make them co-occur less than chance
        # by giving alpha many other neighbours.
        sentences = [["alpha", "bravo"]] * 10 + [["alpha", "zulu"], ["zulu", "charlie"]] * 2
        result = collocations(_table(sentences), min_count=1)
        assert result.ok
        frame = result.unwrap()
        assert (frame["PPMI"] >= 0).all()
        negative = frame[frame["PMI"] < 0]
        if not negative.empty:
            assert (negative["PPMI"] == 0).all()

    def test_g2_matches_scipy(self) -> None:
        """Dunning's statistic, not the legacy single-cell approximation."""
        scipy_stats = pytest.importorskip("scipy.stats")
        for observed, f1, f2, total in ((5, 20, 30, 1000), (2, 3, 4, 50), (17, 40, 60, 500)):
            o12, o21, o22 = f1 - observed, f2 - observed, total - f1 - f2 + observed
            reference = scipy_stats.chi2_contingency(
                [[observed, o12], [o21, o22]], correction=False, lambda_="log-likelihood"
            )[0]
            assert _g2(observed, f1, f2, total) == pytest.approx(reference)

    def test_g2_is_not_the_legacy_single_cell_form(self) -> None:
        """Guard the documented departure from the legacy module.

        The legacy util computed ``2 * O * ln(O/E)`` alone. That is a different
        number, so a test asserting equality with it would lock in the bug.
        """
        observed, f1, f2, total = 5, 20, 30, 1000
        legacy = 2 * observed * math.log(observed / (f1 * f2 / total))
        assert _g2(observed, f1, f2, total) != pytest.approx(legacy)


class TestSpanModes:
    def test_adjacent_preserves_order(self) -> None:
        result = collocations(_table([["new", "york"], ["new", "york"]]), min_count=2)
        assert result.ok
        assert result.unwrap()["Pair"].tolist() == ["new york"]

    def test_window_pairs_are_unordered(self) -> None:
        """``a ... b`` and ``b ... a`` accumulate into one pair."""
        frame = _table([["alpha", "x", "bravo"], ["bravo", "x", "alpha"]])
        result = collocations(frame, span="window", window=5, min_count=2)
        assert result.ok
        pairs = set(result.unwrap()["Pair"])
        assert "alpha bravo" in pairs
        assert "bravo alpha" not in pairs

    def test_window_reaches_past_an_intervening_token(self) -> None:
        frame = _table([["verb", "the", "object"]] * 3)
        adjacent = collocations(frame, span="adjacent", min_count=3).unwrap()
        windowed = collocations(frame, span="window", window=3, min_count=3).unwrap()
        assert "verb object" not in set(adjacent["Pair"])
        assert "object verb" in set(windowed["Pair"]) or "verb object" in set(windowed["Pair"])

    def test_pairs_never_cross_a_sentence_boundary(self) -> None:
        frame = _table([["alpha"], ["bravo"]])
        result = collocations(frame, min_count=1)
        assert result.ok
        assert result.unwrap().empty


class TestFiltering:
    def test_min_count_drops_rare_pairs(self) -> None:
        frame = _table([["alpha", "bravo"], ["charlie", "delta"], ["alpha", "bravo"]])
        result = collocations(frame, min_count=2)
        assert result.ok
        assert result.unwrap()["Pair"].tolist() == ["alpha bravo"]

    def test_stopwords_are_removed_before_counting(self) -> None:
        frame = _table([["alpha", "the", "bravo"]] * 3)
        without = collocations(frame, span="adjacent", min_count=3).unwrap()
        with_stops = collocations(frame, span="adjacent", min_count=3, stopwords=frozenset({"the"})).unwrap()
        assert "alpha the" in set(without["Pair"])
        assert "alpha bravo" in set(with_stops["Pair"])

    def test_min_length_drops_short_tokens(self) -> None:
        frame = _table([["a", "alpha", "bravo"]] * 3)
        result = collocations(frame, min_count=3, min_length=2)
        assert result.ok
        assert "a alpha" not in set(result.unwrap()["Pair"])

    def test_non_alphabetic_tokens_are_skipped(self) -> None:
        frame = _table([["alpha", "123", "bravo"]] * 3)
        result = collocations(frame, min_count=3)
        assert result.ok
        assert not any("123" in pair for pair in result.unwrap()["Pair"])

    def test_top_n_truncates_and_reports(self) -> None:
        # Alphabetic only: a token like "w0" fails str.isalpha() and is
        # filtered before counting, which would empty the frame.
        words = [f"word{chr(ord('a') + i)}" for i in range(21)]
        sentences = [[words[i], words[i + 1]] for i in range(20)] * 3
        result = collocations(_table(sentences), min_count=3, top_n=5)
        assert result.ok
        assert len(result.unwrap()) == 5
        assert any(d.code == "COLLOC_TRUNCATED" for d in result.diagnostics)


class TestDeterminism:
    def test_repeated_runs_are_byte_identical(self) -> None:
        """R6: ties must not reorder between runs (dict/set iteration order)."""
        sentences = [["alpha", "bravo"], ["charlie", "delta"], ["echo", "foxtrot"]] * 4
        frame = _table(sentences)
        first = collocations(frame, min_count=2).unwrap()
        for _ in range(5):
            assert collocations(frame, min_count=2).unwrap().equals(first)

    def test_sorted_by_g2_then_pair(self) -> None:
        sentences = [["alpha", "bravo"], ["charlie", "delta"], ["echo", "foxtrot"]] * 4
        frame = collocations(_table(sentences), min_count=2).unwrap()
        expected = frame.sort_values(
            ["G2 (log-likelihood)", "Pair"], ascending=[False, True], kind="stable"
        ).reset_index(drop=True)
        assert frame.equals(expected)


class TestContracts:
    @pytest.mark.parametrize(
        ("kwargs", "code"),
        [
            ({"span": "sideways"}, "COLLOC_BAD_SPAN"),
            ({"window": 0}, "COLLOC_BAD_WINDOW"),
            ({"window": 999}, "COLLOC_BAD_WINDOW"),
            ({"min_count": 0}, "COLLOC_BAD_MIN_COUNT"),
            ({"min_length": 0}, "COLLOC_BAD_MIN_LENGTH"),
            ({"top_n": 0}, "COLLOC_BAD_TOP_N"),
            ({"field": Col.POS}, "COLLOC_BAD_FIELD"),
        ],
    )
    def test_bad_arguments_fail_with_a_named_diagnostic(self, kwargs: dict[str, object], code: str) -> None:
        result = collocations(_table([["alpha", "bravo"]]), **kwargs)  # type: ignore[arg-type]
        assert not result.ok
        assert [d.code for d in result.errors] == [code]

    def test_empty_frame_returns_the_full_schema(self) -> None:
        empty = _table([["alpha", "bravo"]]).iloc[0:0]
        result = collocations(empty)
        assert result.ok
        assert "G2 (log-likelihood)" in result.unwrap().columns

    def test_no_surviving_pair_warns_rather_than_failing(self) -> None:
        result = collocations(_table([["alpha", "bravo"]]), min_count=50)
        assert result.ok
        assert result.unwrap().empty
        assert any(d.code == "COLLOC_NO_PAIRS" for d in result.diagnostics)

    def test_every_span_mode_is_reachable(self) -> None:
        frame = _table([["alpha", "bravo"]] * 3)
        for span in SPAN_MODES:
            assert collocations(frame, span=span, min_count=3).ok


class TestRegistryWiring:
    def test_registered_outputs_match_the_cli(self) -> None:
        from core.profiler.registry import get_tool

        spec = get_tool("collocations")
        assert spec is not None
        assert spec.outputs == ("collocations.csv",)
        assert spec.capability_ids == ("CAP-NGRAM-04",)

    def test_adapter_is_wired_for_the_profiler(self) -> None:
        from core.profiler.executor import ADAPTERS

        assert "collocations" in ADAPTERS


class TestMemoryGuards:
    """The pair table is what makes this tool fall over on a real corpus."""

    @staticmethod
    def _zipfian(tokens: int, types: int, seed: int = 5) -> list[list[str]]:
        import random
        import string

        def word(index: int) -> str:
            out = ""
            index += 1
            while index:
                index, rest = divmod(index - 1, 26)
                out = string.ascii_lowercase[rest] + out
            return out

        # Seeded for reproducibility; this is test data, not cryptography.
        rng = random.Random(seed)  # noqa: S311
        vocabulary = [word(i) for i in range(types)]
        weights = [1.0 / (i + 1) for i in range(types)]
        drawn = rng.choices(vocabulary, weights=weights, k=tokens)
        return [drawn[i : i + 20] for i in range(0, len(drawn), 20)]

    @pytest.mark.parametrize("span,window", [("adjacent", 5), ("window", 2), ("window", 5), ("window", 10)])
    @pytest.mark.parametrize("min_count", [2, 3, 5, 10])
    def test_filter_is_exact_across_settings(self, span: str, window: int, min_count: int) -> None:
        """Pre-filtering must never change a surviving count.

        The bound differs by span and both versions of getting it wrong are
        silent: filtering on ``min(f1, f2)`` in window mode drops real pairs,
        and so does forgetting that a token pairs with both sides of its
        window. This compares against an unfiltered reference.
        """
        sentences = self._zipfian(12_000, 400)
        frame = _table(sentences)
        reference = collocations(frame, span=span, window=window, min_count=1, top_n=None).unwrap()
        reference = reference[reference["Co-occurrences"] >= min_count].reset_index(drop=True)
        filtered = collocations(frame, span=span, window=window, min_count=min_count, top_n=None).unwrap()
        assert reference.equals(filtered)

    def test_exceeding_the_pair_ceiling_fails_with_advice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A named refusal, not an out-of-memory kill of the worker."""
        import core.analysis.collocations as module

        monkeypatch.setattr(module, "MAX_DISTINCT_PAIRS", 50)
        frame = _table(self._zipfian(4_000, 2_000))
        result = module.collocations(frame, span="window", window=25, min_count=1)
        assert not result.ok
        assert [d.code for d in result.errors] == ["COLLOC_TOO_MANY_PAIRS"]
        message = result.errors[0].message
        assert "--min-count" in message
        assert "--span adjacent" in message

    def test_the_ceiling_does_not_fire_on_ordinary_input(self) -> None:
        frame = _table(self._zipfian(4_000, 200))
        assert collocations(frame, span="window", window=10, min_count=3).ok
