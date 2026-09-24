"""The two Gensim views HW2 grades: lambda relevance and the Intertopic Map.

The syllabus asks what varying lambda *does* and what the "ideal" distribution
of topics in the Intertopic Distance Map looks like. These are the numbers
behind the Gensim GUI's two panels, computed natively (no pyLDAvis dependency)
from the fitted model and the corpus. Verification is definitional: small
matrices whose rankings and distances can be checked by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.analysis.lda import intertopic_map, relevance_terms

VOCAB = ["cat", "dog"]
TOKENS = {
    # cat-heavy and dog-heavy documents, so p(w) is not uniform.
    "d1": ["cat", "cat", "cat", "dog"],
    "d2": ["dog", "dog", "dog", "cat"],
}
# Topic 0 is almost all cat; topic 1 is almost all dog.
PHI = np.array([[0.9, 0.1], [0.2, 0.8]])


class TestRelevanceLambda:
    def test_at_lambda_one_the_ranking_is_the_topics_own_probability(self) -> None:
        # lam=1 drops the corpus term entirely: relevance is log p(w|t), so
        # each topic's own most probable word leads.
        table = relevance_terms(PHI, VOCAB, TOKENS, lam=1.0, top_n=2)
        topic0 = table[table["Topic"] == 0].sort_values("Relevance", ascending=False)
        assert list(topic0["Word"]) == ["cat", "dog"]
        topic1 = table[table["Topic"] == 1].sort_values("Relevance", ascending=False)
        assert list(topic1["Word"]) == ["dog", "cat"]

    def test_at_lambda_zero_the_ranking_is_how_rare_the_word_is(self) -> None:
        # lam=0 drops the topic term: relevance is log p(w), and here "cat"
        # and "dog" are equally frequent, so the two rank by their tiny
        # numerical difference only -- the point is the formula's shape:
        # relevance at lam=0 does not look at the topic at all.
        table = relevance_terms(PHI, VOCAB, TOKENS, lam=0.0, top_n=2)
        topic0 = table[table["Topic"] == 0]
        # Identical corpus frequencies: equal relevance for both words.
        assert topic0["Relevance"].nunique() == 1

    def test_the_relevance_column_is_the_documented_formula(self) -> None:
        lam = 0.6
        table = relevance_terms(PHI, VOCAB, TOKENS, lam=lam, top_n=2)
        row = table[(table["Topic"] == 0) & (table["Word"] == "cat")].iloc[0]
        # p(cat) = 4/8; p(cat|topic0) = 0.9.
        expected = lam * np.log(0.9) + (1 - lam) * np.log(4 / 8)
        assert row["Relevance"] == pytest.approx(round(float(expected), 4), abs=1e-3)

    def test_saliency_is_carried_beside_relevance(self) -> None:
        # HW2's panel shows both readings of a term; one column cannot answer
        # "salient or relevant". The two counts ride along for drawing.
        table = relevance_terms(PHI, VOCAB, TOKENS, lam=0.6, top_n=2)
        assert set(table.columns) == {"Topic", "Word", "Relevance", "Saliency", "Corpus frequency", "Topic frequency"}

    def test_saliency_is_the_chuang_kl_divergence(self) -> None:
        """The corrected definition (Chuang et al. 2010 / pyLDAvis):
        s(w) = sum_t p(t|w) * log(p(t|w) / p(t)) -- a divergence, in bits.

        Hand-checked on PHI: prevalence is uniform (each topic margin sums to
        1), so p(t) = 0.5. For "cat": p(t0|cat) = 0.9/1.1 = 0.8182,
        p(t1|cat) = 0.2/1.1 = 0.1818.
        s = 0.8182*ln(0.8182/0.5) + 0.1818*ln(0.1818/0.5) = 0.1990.
        """
        table = relevance_terms(PHI, VOCAB, TOKENS, lam=0.6, top_n=2)
        saliency = table.groupby("Word")["Saliency"].first()
        expected_cat = 0.8181818 * np.log(0.8181818 / 0.5) + 0.1818182 * np.log(0.1818182 / 0.5)
        assert saliency["cat"] == pytest.approx(round(float(expected_cat), 4), abs=1e-3)

    def test_saliency_is_never_negative(self) -> None:
        """A KL divergence cannot be negative. The engine's old definition
        was inverted and always negative on real data, which is why the
        relevance panel used to order by magnitude."""
        table = relevance_terms(PHI, VOCAB, TOKENS, lam=0.6, top_n=2)
        assert (table["Saliency"] >= 0).all()

    def test_saliency_is_zero_for_a_word_that_says_nothing_about_topics(self) -> None:
        # A word split evenly across topics whose prevalence is uniform has
        # p(t|w) = p(t) for every t, so every log ratio is 0. (numpy check,
        # not `== pytest.approx`: that comparison against a Series is not
        # elementwise and silently returns all-False.)
        flat = np.array([[0.5, 0.5], [0.5, 0.5]])
        table = relevance_terms(flat, ["cat", "dog"], TOKENS, lam=0.6, top_n=2)
        assert np.allclose(table["Saliency"], 0.0, atol=1e-6)

    def test_a_word_that_concentrates_in_one_topic_is_more_salient(self) -> None:
        # "dog" leans harder into its topic (0.8 vs 0.9's counterpart 0.1/0.2
        # split): its p(t|w) diverges further from the marginal.
        table = relevance_terms(PHI, VOCAB, TOKENS, lam=0.6, top_n=2)
        saliency = table.groupby("Word")["Saliency"].first()
        assert saliency["dog"] > saliency["cat"]

    def test_relevance_is_negative_so_it_cannot_be_a_bar_length(self) -> None:
        # The reason the counts exist. A sum of logs of probabilities is never
        # positive, and the best-ranked term has the smallest magnitude, so a
        # bar drawn from zero would make the top term the shortest bar.
        table = relevance_terms(PHI, VOCAB, TOKENS, lam=0.6, top_n=2)
        assert (table["Relevance"] < 0).all()

    def test_corpus_frequency_is_the_raw_count(self) -> None:
        # cat: 3 in d1 + 1 in d2; dog: 1 + 3.
        table = relevance_terms(PHI, VOCAB, TOKENS, lam=0.6, top_n=2)
        counts = dict(zip(table["Word"], table["Corpus frequency"], strict=False))
        assert counts == {"cat": 4, "dog": 4}

    def test_topic_frequency_is_probability_times_topic_mass(self) -> None:
        # p(cat|t0) = 0.9 and N_0 = 5 tokens -> 4.5; p(dog|t1) = 0.8 and N_1 = 3 -> 2.4.
        table = relevance_terms(PHI, VOCAB, TOKENS, lam=0.6, top_n=2, topic_tokens=[5.0, 3.0])
        at = table.set_index(["Topic", "Word"])["Topic frequency"]
        assert at[(0, "cat")] == pytest.approx(4.5)
        assert at[(1, "dog")] == pytest.approx(2.4)

    def test_topic_frequency_without_a_topic_mass_is_missing_not_guessed(self) -> None:
        table = relevance_terms(PHI, VOCAB, TOKENS, lam=0.6, top_n=2)
        assert table["Topic frequency"].isna().all(), "no N_t means no estimate, and blank is not zero"

    def test_an_out_of_range_lambda_is_refused_loudly(self) -> None:
        table = relevance_terms(PHI, VOCAB, TOKENS, lam=1.5, top_n=2)
        assert table.empty, "a lambda outside 0..1 must not invent a ranking"


class TestIntertopicDistanceMap:
    def test_identical_topics_sit_on_the_same_point(self) -> None:
        phi = np.array([[0.5, 0.5], [0.5, 0.5]])
        layout = intertopic_map(phi, [1.0, 1.0], seed=1)
        assert layout["X"].iloc[0] == pytest.approx(layout["X"].iloc[1], abs=1e-6)
        assert layout["Y"].iloc[0] == pytest.approx(layout["Y"].iloc[1], abs=1e-6)

    def test_opposite_topics_sit_far_apart(self) -> None:
        # (cat-only, dog-only) is the maximum JS distance here; nearly
        # identical topics are the minimum worth drawing. The map must show
        # the first pair further apart than the second.
        far = intertopic_map(np.array([[1.0, 0.0], [0.0, 1.0]]), [1.0, 1.0], seed=1)
        near = intertopic_map(np.array([[0.5, 0.5], [0.51, 0.49]]), [1.0, 1.0], seed=1)
        gap_far = float(np.hypot(far["X"].iloc[0] - far["X"].iloc[1], far["Y"].iloc[0] - far["Y"].iloc[1]))
        gap_near = float(np.hypot(near["X"].iloc[0] - near["X"].iloc[1], near["Y"].iloc[0] - near["Y"].iloc[1]))
        assert gap_far > gap_near + 0.5

    def test_prevalence_is_relative_to_the_largest_topic(self) -> None:
        # The map's circle areas are relative: the biggest topic is 1.0.
        layout = intertopic_map(PHI, [3.0, 6.0], seed=1)
        by_topic = {int(row["Topic"]): row for _, row in layout.iterrows()}
        assert by_topic[1]["Prevalence"] == pytest.approx(1.0)
        assert by_topic[0]["Prevalence"] == pytest.approx(0.5)

    def test_the_placement_is_seeded_so_the_map_can_be_regenerated(self) -> None:
        first = intertopic_map(PHI, [1.0, 2.0], seed=7)
        again = intertopic_map(PHI, [1.0, 2.0], seed=7)
        assert first.equals(again), "the same seed must redraw the same map"

    def test_one_topic_has_no_map(self) -> None:
        # MDS needs something to place; one circle is not a distribution.
        assert intertopic_map(np.array([[1.0]]), [1.0]).empty
