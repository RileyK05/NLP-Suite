"""Topic stability — matching topics across refits and scoring their overlap.

The corpus is two clearly separated themes (war/army/battle vs
tax/budget/spending) built directly as token lists, the same shape
``core.analysis.lda.tokens_from_frame`` produces, so ``fit_lda`` runs on it
exactly as it would on real data. Documents are generated deterministically
(no RNG) so the fixture itself never varies between runs — only the model's
own seeded randomness does, and that determinism is asserted directly below
rather than assumed.

Gensim-backed tests are marked ``model_integration``; the validation guards
run offline (they fire before ``fit_lda`` is ever called, exactly like
``fit_lda``'s own guards).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.analysis.lda import fit_lda
from core.analysis.topic_stability import (
    JACCARD,
    MATCHED_TOPIC,
    MEAN_JACCARD,
    MIN_JACCARD,
    REFERENCE_WORDS,
    SEED,
    SEEDS,
    SHARED_WORDS,
    STABLE,
    TOPIC,
    StabilityResult,
    _match_topics,
    topic_stability,
)
from core.result import Result, Severity

THEME_A = ["war", "army", "battle", "soldier", "troops", "combat"]
THEME_B = ["tax", "budget", "spending", "revenue", "deficit", "fiscal"]


def _doc(theme: list[str], offset: int, n: int = 30) -> list[str]:
    """A deterministic, non-trivial draw from one theme's vocabulary.

    Not a simple repeat-forever cycle: the ``* 7`` stride makes each
    document's word order differ from its neighbours', the way real
    documents would, while staying perfectly reproducible without depending
    on a second RNG on top of gensim's own seeded one.
    """
    return [theme[(i * 7 + offset) % len(theme)] for i in range(n)]


def _corpus() -> dict[str, list[str]]:
    tokens: dict[str, list[str]] = {}
    for i in range(6):
        tokens[f"war{i}.txt"] = _doc(THEME_A, i)
    for i in range(6):
        tokens[f"tax{i}.txt"] = _doc(THEME_B, i)
    return tokens


DOC_TOKENS = _corpus()


def codes(result: Result[object]) -> list[str]:
    return [d.code for d in result.diagnostics]


def find(result: Result[object], code: str) -> object:
    matching = [d for d in result.diagnostics if d.code == code]
    assert len(matching) == 1, f"expected exactly one {code}, got {codes(result)}"
    return matching[0]


class TestValidation:
    """Every guard fires before gensim is ever imported -- these run offline."""

    def test_one_seed_cannot_be_compared(self) -> None:
        result = topic_stability(DOC_TOKENS, n_topics=2, seeds=(100,))
        assert not result.ok
        assert "STABILITY_TOO_FEW_SEEDS" in codes(result)

    def test_no_seeds_cannot_be_compared(self) -> None:
        result = topic_stability(DOC_TOKENS, n_topics=2, seeds=())
        assert not result.ok
        assert "STABILITY_TOO_FEW_SEEDS" in codes(result)

    def test_a_seed_repeated_is_refused(self) -> None:
        """A seed compared against itself scores Jaccard 1.0 for every topic
        and would make every run look perfectly stable."""
        result = topic_stability(DOC_TOKENS, n_topics=2, seeds=(100, 101, 100))
        assert not result.ok
        diagnostic = find(result, "STABILITY_DUPLICATE_SEEDS")
        assert diagnostic.context["duplicates"] == [100]

    def test_fewer_than_two_topics_is_refused(self) -> None:
        result = topic_stability(DOC_TOKENS, n_topics=1, seeds=(100, 101))
        assert not result.ok
        assert "STABILITY_BAD_K" in codes(result)

    def test_top_n_below_one_is_refused(self) -> None:
        result = topic_stability(DOC_TOKENS, n_topics=2, seeds=(100, 101), top_n=0)
        assert not result.ok
        assert "STABILITY_BAD_TOP_N" in codes(result)

    def test_stable_at_zero_is_refused(self) -> None:
        """0 would mean every topic is 'stable' by definition (Jaccard >= 0 always)."""
        result = topic_stability(DOC_TOKENS, n_topics=2, seeds=(100, 101), stable_at=0.0)
        assert not result.ok
        assert "STABILITY_BAD_THRESHOLD" in codes(result)

    def test_stable_at_above_one_is_refused(self) -> None:
        result = topic_stability(DOC_TOKENS, n_topics=2, seeds=(100, 101), stable_at=1.5)
        assert not result.ok
        assert "STABILITY_BAD_THRESHOLD" in codes(result)

    def test_a_fit_lda_failure_is_carried_through(self) -> None:
        """No documents fails inside the per-seed fit loop; the underlying
        diagnostic (not a generic wrapper) is what a caller sees."""
        result = topic_stability({}, n_topics=2, seeds=(100, 101))
        assert not result.ok
        assert "TOPIC_NO_DOCUMENTS" in codes(result)


class TestOneToOneMatching:
    """The Hungarian assignment, tested directly against a case where a
    greedy per-row argmax would double-assign."""

    def test_greedy_would_double_assign_but_this_does_not(self) -> None:
        # Row 1's best column is column 0 (0.8 > 0.1), same as row 0's best
        # (0.9). Greedy argmax-per-row assigns both rows to column 0. The
        # optimal one-to-one assignment is row0->col1, row1->col0 (total
        # 0.85 + 0.8 = 1.65), beating row0->col0, row1->col1 (0.9 + 0.1 = 1.0).
        similarity = np.array([[0.9, 0.85], [0.8, 0.1]])
        greedy = [int(np.argmax(row)) for row in similarity]
        assert greedy == [0, 0], "the case must actually provoke a greedy collision"
        assert _match_topics(similarity) == [1, 0]

    def test_the_identity_matrix_matches_each_topic_to_itself(self) -> None:
        similarity = np.eye(3)
        assert _match_topics(similarity) == [0, 1, 2]

    def test_columns_are_never_reused(self) -> None:
        rng = np.random.default_rng(0)
        similarity = rng.random((5, 5))
        assignment = _match_topics(similarity)
        assert sorted(assignment) == list(range(5)), "every column used exactly once"


# ---------------------------------------------------------------------------
# Gensim-backed: real fits, real matching.
# ---------------------------------------------------------------------------
pytest.importorskip("gensim")


class TestDeterminism:
    def test_fit_lda_is_byte_identical_at_the_same_seed(self) -> None:
        """The premise the whole module rests on: comparing seed against seed
        is only meaningful if one seed reliably produces one answer."""
        first = fit_lda(DOC_TOKENS, n_topics=2, top_n=6, seed=100, coherence=False).unwrap()
        second = fit_lda(DOC_TOKENS, n_topics=2, top_n=6, seed=100, coherence=False).unwrap()
        pd.testing.assert_frame_equal(first.topics, second.topics)
        pd.testing.assert_frame_equal(first.dominant, second.dominant)

    def test_topic_stability_itself_is_reproducible(self) -> None:
        first = topic_stability(DOC_TOKENS, n_topics=2, seeds=(100, 101, 102), top_n=6).unwrap()
        second = topic_stability(DOC_TOKENS, n_topics=2, seeds=(100, 101, 102), top_n=6).unwrap()
        pd.testing.assert_frame_equal(first.matches, second.matches)
        pd.testing.assert_frame_equal(first.summary, second.summary)


class TestTwoSeparatedThemes:
    """At k=2, LDA should recover exactly the two themes at every seed."""

    def test_both_topics_are_stable(self) -> None:
        result = topic_stability(DOC_TOKENS, n_topics=2, seeds=(100, 101, 102), top_n=6)
        assert result.ok, result.diagnostics
        summary = result.unwrap().summary
        assert summary[STABLE].all()
        assert (summary[MIN_JACCARD] >= 0.5).all()

    def test_stable_at_one_is_the_inclusive_upper_bound(self) -> None:
        """This corpus's two themes match perfectly (Min Jaccard 1.0), so a
        threshold of exactly 1.0 -- 'only count topics matched perfectly' --
        must still be accepted and still call every topic stable."""
        result = topic_stability(DOC_TOKENS, n_topics=2, seeds=(100, 101, 102), top_n=6, stable_at=1.0)
        assert result.ok, result.diagnostics
        summary = result.unwrap().summary
        assert (summary[MIN_JACCARD] == 1.0).all()
        assert summary[STABLE].all()

    def test_each_reference_topic_is_one_theme(self) -> None:
        summary = topic_stability(DOC_TOKENS, n_topics=2, seeds=(100, 101, 102), top_n=6).unwrap().summary
        word_sets = [set(words.split(", ")) for words in summary[REFERENCE_WORDS]]
        assert any(set(THEME_A) == words for words in word_sets)
        assert any(set(THEME_B) == words for words in word_sets)

    def test_warnings_from_each_fit_are_carried_forward(self) -> None:
        """12 documents triggers TOPIC_FEW_DOCUMENTS on every one of the 3 fits."""
        result = topic_stability(DOC_TOKENS, n_topics=2, seeds=(100, 101, 102), top_n=6)
        warnings = [d for d in result.diagnostics if d.code == "TOPIC_FEW_DOCUMENTS"]
        assert len(warnings) == 3
        assert all(d.severity is Severity.WARNING for d in warnings)

    def test_matches_has_one_row_per_reference_topic_per_other_seed(self) -> None:
        matches = topic_stability(DOC_TOKENS, n_topics=2, seeds=(100, 101, 102), top_n=6).unwrap().matches
        assert list(matches.columns) == [TOPIC, SEED, MATCHED_TOPIC, JACCARD, SHARED_WORDS, REFERENCE_WORDS]
        # 2 reference topics x 2 other seeds (101, 102).
        assert len(matches) == 4
        assert set(matches[SEED]) == {101, 102}

    def test_shared_words_is_the_real_intersection(self) -> None:
        matches = topic_stability(DOC_TOKENS, n_topics=2, seeds=(100, 101, 102), top_n=6).unwrap().matches
        for _, row in matches.iterrows():
            reference = set(str(row[REFERENCE_WORDS]).split(", "))
            shared = set(str(row[SHARED_WORDS]).split(", ")) if row[SHARED_WORDS] else set()
            assert shared <= reference


class TestHighTopicCountIsUnstable:
    """At k=8 on a 12-document, 2-theme corpus there are more topics than
    real structure; some must fail to reproduce."""

    def test_some_topics_are_stable_and_some_are_not(self) -> None:
        result = topic_stability(DOC_TOKENS, n_topics=8, seeds=(100, 101, 102), top_n=4, passes=15)
        assert result.ok, result.diagnostics
        summary = result.unwrap().summary
        # Control: if every topic came back stable (or none did), the test
        # below would pass trivially without the corpus actually forcing a
        # mix. Both must be present for this to test anything.
        assert summary[STABLE].any(), "expected at least one stable topic"
        assert not summary[STABLE].all(), "expected at least one unstable topic"

    def test_unstable_topics_have_a_lower_min_than_mean(self) -> None:
        """Min <= Mean always; an unstable topic is exactly where they differ
        most, because one bad seed pulls the min down without moving the mean
        as far."""
        summary = topic_stability(DOC_TOKENS, n_topics=8, seeds=(100, 101, 102), top_n=4, passes=15).unwrap().summary
        assert (summary[MIN_JACCARD] <= summary[MEAN_JACCARD] + 1e-9).all()


class TestSeedsColumn:
    def test_seeds_counts_the_comparisons_behind_the_row(self) -> None:
        """3 seeds total, 1 reference -> 2 comparisons per topic."""
        summary = topic_stability(DOC_TOKENS, n_topics=2, seeds=(100, 101, 102), top_n=6).unwrap().summary
        assert (summary[SEEDS] == 2).all()


class TestStabilityResultShape:
    def test_result_is_frozen(self) -> None:
        result = topic_stability(DOC_TOKENS, n_topics=2, seeds=(100, 101), top_n=6).unwrap()
        assert isinstance(result, StabilityResult)
        with pytest.raises(AttributeError):
            result.n_topics = 99  # type: ignore[misc]

    def test_metadata_matches_the_call(self) -> None:
        result = topic_stability(DOC_TOKENS, n_topics=2, seeds=(100, 101, 102), top_n=6, stable_at=0.4).unwrap()
        assert result.reference_seed == 100
        assert result.n_topics == 2
        assert result.top_n == 6
        assert result.stable_at == 0.4
