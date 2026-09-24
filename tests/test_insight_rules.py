"""Per-tool readouts: what each rule says, and the trap it exists to catch.

Each fixture here is shaped exactly like the first table its tool publishes,
because that is the table :meth:`core.io.writer.OutputWriter.write_table`
hands the readout. The column names are pinned separately, against the
analysis modules, in ``test_insight_columns.py``.

Every rule is tested twice: once on a result where the trap has been sprung,
where the caution must appear, and once on a clean result, where it must not.
A decoder that warns about everything is the same as one that warns about
nothing.
"""

from __future__ import annotations

import pandas as pd

from core.insight.readout import readout
from core.insight.recommend import recommend_charts


def _text(frame: pd.DataFrame, tool: str) -> str:
    return readout(frame, tool=tool).to_text()


def _cautions(frame: pd.DataFrame, tool: str) -> str:
    return " ".join(readout(frame, tool=tool).cautions)


class TestNer:
    @staticmethod
    def _frame(counts: list[int], tags: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Entity": [f"Entity{i}" for i in range(len(counts))],
                "NER Tag": tags,
                "Count": counts,
                "First Sentence": list(range(len(counts))),
                "Last Sentence": list(range(len(counts))),
                "Document ID": ["d1"] * len(counts),
                "Document": ["a.txt"] * len(counts),
            }
        )

    def test_names_the_most_mentioned_entity(self) -> None:
        frame = self._frame([9, 4, 2, 7], ["PERSON", "ORG", "GPE", "PERSON"])
        assert "'Entity0'" in _text(frame, "ner")
        assert "9 mentions" in _text(frame, "ner")

    def test_one_mention_is_singular(self) -> None:
        frame = self._frame([1, 1], ["PERSON", "ORG"])
        assert "1 mention)" in _text(frame, "ner")

    def test_a_name_tagged_two_ways_is_flagged(self) -> None:
        frame = self._frame([5, 5, 3], ["PERSON", "GPE", "ORG"])
        frame.loc[1, "Entity"] = "Entity0"  # the same name, a second tag
        assert "tagged more than one way" in _cautions(frame, "ner")

    def test_consistent_tagging_is_not_flagged(self) -> None:
        frame = self._frame([5, 4, 3, 6], ["PERSON", "GPE", "ORG", "PERSON"])
        assert "tagged more than one way" not in _cautions(frame, "ner")

    def test_a_tail_of_single_mentions_is_flagged(self) -> None:
        frame = self._frame([1, 1, 1, 4], ["PERSON"] * 4)
        assert "seen exactly once" in _cautions(frame, "ner")


class TestTopicModel:
    @staticmethod
    def _frame(topics: dict[int, list[tuple[str, float]]]) -> pd.DataFrame:
        rows = [
            {"Topic": topic, "Word": word, "Weight": weight}
            for topic, pairs in topics.items()
            for word, weight in pairs
        ]
        return pd.DataFrame(rows, columns=["Topic", "Word", "Weight"])

    def test_reads_the_first_topic_back_as_words(self) -> None:
        frame = self._frame(
            {
                0: [("harvest", 0.4), ("soil", 0.2), ("crop", 0.1)],
                1: [("market", 0.5), ("price", 0.2), ("trade", 0.1)],
            }
        )
        text = _text(frame, "topic_model")
        assert "2 topic(s) fitted" in text
        assert "harvest, soil, crop" in text

    def test_topics_sharing_their_top_words_are_flagged(self) -> None:
        frame = self._frame(
            {
                0: [("said", 0.4), ("harvest", 0.2)],
                1: [("said", 0.4), ("market", 0.2)],
                2: [("said", 0.4), ("soil", 0.2)],
            }
        )
        cautions = _cautions(frame, "topic_model")
        assert "at least half the topics" in cautions
        assert "'said'" in cautions

    def test_separated_topics_are_not_flagged(self) -> None:
        frame = self._frame(
            {
                0: [("harvest", 0.6), ("soil", 0.1)],
                1: [("market", 0.6), ("price", 0.1)],
                2: [("river", 0.6), ("flood", 0.1)],
            }
        )
        assert "at least half the topics" not in _cautions(frame, "topic_model")

    def test_a_topic_with_no_peak_is_flagged(self) -> None:
        frame = self._frame({0: [("a", 0.21), ("b", 0.20), ("c", 0.195)]})
        assert "nearly equal weight" in _cautions(frame, "topic_model")


class TestSentiment:
    @staticmethod
    def _sentences(compounds: list[float], hits: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Document ID": ["d1"] * len(compounds),
                "Document": ["a.txt"] * len(compounds),
                "Sentence ID": list(range(len(compounds))),
                "Sentence": ["text"] * len(compounds),
                "Neg": [0.0] * len(compounds),
                "Neu": [1.0] * len(compounds),
                "Pos": [0.0] * len(compounds),
                "Compound": compounds,
                "Label": ["neutral"] * len(compounds),
                "Hits": hits,
            }
        )

    def test_counts_the_three_bands(self) -> None:
        frame = self._sentences([0.8, -0.6, 0.0, 0.4], [3, 2, 1, 2])
        text = _text(frame, "sentiment_vader_anew")
        assert "2 positive, 1 negative, 1 neutral" in text
        assert "sentence(s) scored" in text

    def test_a_result_that_is_mostly_neutral_is_flagged(self) -> None:
        frame = self._sentences([0.0, 0.01, -0.02, 0.9], [1, 1, 1, 4])
        assert "inside the neutral band" in _cautions(frame, "sentiment_vader_anew")

    def test_sentences_with_no_matched_word_are_flagged(self) -> None:
        frame = self._sentences([0.5, 0.0, -0.4, 0.0], [2, 0, 3, 0])
        cautions = _cautions(frame, "sentiment_vader_anew")
        assert "contained no lexicon word at all" in cautions
        assert "2 of 4 rows" in cautions

    def test_full_coverage_is_not_flagged(self) -> None:
        frame = self._sentences([0.5, 0.6, -0.4, 0.7], [2, 1, 3, 1])
        assert "no lexicon word" not in _cautions(frame, "sentiment_vader_anew")

    def test_documents_rather_than_sentences_are_named_as_such(self) -> None:
        frame = pd.DataFrame(
            {
                "Document ID": ["d1", "d2"],
                "Document": ["a.txt", "b.txt"],
                "Sentences": [10, 12],
                "Neg": [0.1, 0.2],
                "Neu": [0.7, 0.6],
                "Pos": [0.2, 0.2],
                "Compound": [0.4, -0.3],
            }
        )
        assert "document(s) scored" in _text(frame, "sentiment_vader_anew")

    def test_anew_reports_valence_and_its_coverage(self) -> None:
        frame = pd.DataFrame(
            {
                "Document ID": ["d1", "d2"],
                "Document": ["a.txt", "b.txt"],
                "Tokens": [1000, 1200],
                "Hits": [10, 12],
                "Valence": [5.4, 6.1],
                "Arousal": [4.2, 4.8],
                "Dominance": [5.0, 5.2],
            }
        )
        assert "ANEW valence runs" in _text(frame, "sentiment_vader_anew")
        assert "appear in the ANEW lexicon" in _cautions(frame, "sentiment_vader_anew")


class TestDocSimilarity:
    @staticmethod
    def _frame(similarities: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Document ID A": [f"a{i}" for i in range(len(similarities))],
                "Document A": [f"a{i}.txt" for i in range(len(similarities))],
                "Document ID B": [f"b{i}" for i in range(len(similarities))],
                "Document B": [f"b{i}.txt" for i in range(len(similarities))],
                "Similarity": similarities,
                "Band": ["50%"] * len(similarities),
            }
        )

    def test_names_the_closest_pair(self) -> None:
        frame = self._frame([12.0, 44.5, 31.0])
        text = _text(frame, "doc_similarity")
        assert "a1.txt and b1.txt" in text
        assert "3 document pair(s) compared" in text

    def test_near_duplicates_are_flagged(self) -> None:
        frame = self._frame([12.0, 97.5, 31.0])
        assert "the same text is in the corpus twice" in _cautions(frame, "doc_similarity")

    def test_ordinary_similarities_are_not_flagged(self) -> None:
        frame = self._frame([12.0, 44.5, 31.0])
        assert "twice" not in _cautions(frame, "doc_similarity")

    def test_a_high_floor_is_read_as_shared_function_words(self) -> None:
        frame = self._frame([61.0, 64.5, 72.0])
        assert "shared function words rather than shared content" in _cautions(frame, "doc_similarity")


class TestNgrams:
    @staticmethod
    def _frame(rows: list[tuple[str, int, int, str]]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "N-gram": gram,
                    "Frequency in Document": in_doc,
                    "Frequency in Corpus": in_corpus,
                    "Document ID": doc,
                    "Document": f"{doc}.txt",
                }
                for gram, in_doc, in_corpus, doc in rows
            ]
        )

    def test_names_the_most_repeated_phrase(self) -> None:
        frame = self._frame([("of the", 4, 9, "d1"), ("in a", 2, 3, "d1"), ("to be", 1, 2, "d1")])
        assert "'of the'" in _text(frame, "ngrams")
        assert "9 times" in _text(frame, "ngrams")

    def test_a_corpus_of_hapax_phrases_is_flagged(self) -> None:
        frame = self._frame([(f"phrase {i}", 1, 1, "d1") for i in range(9)] + [("of the", 4, 4, "d1")])
        assert "the n is too long for" in _cautions(frame, "ngrams")

    def test_the_double_counting_trap_is_named_when_it_applies(self) -> None:
        frame = self._frame([("of the", 4, 9, "d1"), ("of the", 5, 9, "d2"), ("in a", 3, 3, "d1")])
        assert "Summing that column double-counts" in _cautions(frame, "ngrams")

    def test_a_single_document_table_is_not_warned_about_double_counting(self) -> None:
        frame = self._frame([("of the", 4, 4, "d1"), ("in a", 3, 3, "d1")])
        assert "double-counts" not in _cautions(frame, "ngrams")


class TestKwic:
    def test_reports_lines_documents_and_forms(self) -> None:
        frame = pd.DataFrame(
            {
                "Document ID": ["d1", "d1", "d2"],
                "Document": ["a.txt", "a.txt", "b.txt"],
                "Sentence ID": [1, 1, 4],
                "Hit": ["river", "river", "Rivers"],
                "Left Context": ["down the", "beside the", "two"],
                "Right Context": ["bank", "mouth", "met"],
            }
        )
        text = _text(frame, "kwic")
        assert "3 concordance line(s) over 2 document(s)" in text
        assert "'river'" in text
        assert "evidence to read, not a count to total" in _cautions(frame, "kwic")


class TestLengthDependence:
    """The shared trap: a measure that is really just document length."""

    @staticmethod
    def _corpus(tokens: list[int], ttr: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Document ID": [f"d{i}" for i in range(len(tokens))],
                "Document": [f"d{i}.txt" for i in range(len(tokens))],
                "Tokens": tokens,
                "Types": [int(t * r) for t, r in zip(tokens, ttr, strict=True)],
                "TTR": ttr,
                "Root TTR": ttr,
                "Log TTR": ttr,
                "Herdan": ttr,
                "Yule K": [50.0 + i for i in range(len(tokens))],
            }
        )

    def test_ttr_falling_with_length_is_named(self) -> None:
        frame = self._corpus([100, 200, 400, 800, 1600, 3200], [0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        cautions = _cautions(frame, "corpus_statistics")
        assert "largely a length ranking" in cautions
        assert "TTR falls with Tokens" in cautions

    def test_a_length_independent_column_is_not_flagged(self) -> None:
        frame = self._corpus([100, 200, 400, 800, 1600, 3200], [0.42, 0.55, 0.39, 0.61, 0.44, 0.58])
        assert "length ranking" not in _cautions(frame, "corpus_statistics")

    def test_too_few_documents_makes_no_claim(self) -> None:
        """Three points can correlate perfectly and mean nothing."""
        frame = self._corpus([100, 200, 400], [0.7, 0.5, 0.3])
        assert "length ranking" not in _cautions(frame, "corpus_statistics")

    def test_yule_k_is_read_in_the_right_direction(self) -> None:
        frame = self._corpus([100, 200, 400, 800, 1600, 3200], [0.42, 0.55, 0.39, 0.61, 0.44, 0.58])
        assert "opposite direction to TTR" in _text(frame, "corpus_statistics")


class TestTextStatistics:
    @staticmethod
    def _frame(tokens: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Document ID": [f"d{i}" for i in range(len(tokens))],
                "Document": [f"d{i}.txt" for i in range(len(tokens))],
                "Sentences": [max(1, t // 20) for t in tokens],
                "Tokens": tokens,
                "Types": [max(1, t // 3) for t in tokens],
                "Avg Sentence Length": [18.0 + (i % 4) for i in range(len(tokens))],
                "Avg Word Length": [4.5] * len(tokens),
                "Syllables": [t * 2 for t in tokens],
                "Avg Syllables per Word": [1.6] * len(tokens),
            }
        )

    def test_reports_corpus_totals(self) -> None:
        frame = self._frame([500, 900, 1400])
        assert "2,800 tokens in total" in _text(frame, "text_statistics")

    def test_short_documents_are_flagged(self) -> None:
        frame = self._frame([40, 900, 1400])
        assert "fewer than 100 tokens" in _cautions(frame, "text_statistics")

    def test_ordinary_documents_are_not_flagged(self) -> None:
        frame = self._frame([500, 900, 1400])
        assert "fewer than 100 tokens" not in _cautions(frame, "text_statistics")


class TestLexicalDiversity:
    @staticmethod
    def _frame(tokens: list[int], mtld: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Document ID": [f"d{i}" for i in range(len(tokens))],
                "Document": [f"d{i}.txt" for i in range(len(tokens))],
                "Total Tokens": tokens,
                "Unique Types": [t // 2 for t in tokens],
                "TTR": [0.5] * len(tokens),
                "Root TTR (Guiraud)": [7.0] * len(tokens),
                "Log TTR (Herdan)": [0.9] * len(tokens),
                "MTLD": mtld,
                "vocd-D": [70.0] * len(tokens),
            }
        )

    def test_reports_the_mtld_range(self) -> None:
        frame = self._frame([500, 900], [62.5, 88.1])
        assert "MTLD runs 62.5 to 88.1" in _text(frame, "lexical_diversity")

    def test_documents_too_short_for_mtld_are_flagged(self) -> None:
        frame = self._frame([40, 900], [62.5, 88.1])
        assert "shorter than 100 tokens" in _cautions(frame, "lexical_diversity")

    def test_long_enough_documents_are_not_flagged(self) -> None:
        frame = self._frame([500, 900], [62.5, 88.1])
        assert "shorter than 100 tokens" not in _cautions(frame, "lexical_diversity")


class TestSentenceComplexity:
    @staticmethod
    def _frame(tokens: list[int], mdd: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Sentence ID": list(range(len(tokens))),
                "Document ID": ["d1"] * len(tokens),
                "Document": ["a.txt"] * len(tokens),
                "Sentence": ["words here"] * len(tokens),
                "Tokens": tokens,
                "Mean Dependency Distance": mdd,
                "Max Dependency Distance": [int(m * 2) for m in mdd],
                "Depth": [3 + (i % 3) for i in range(len(tokens))],
                "Subordinate Clauses": [i % 2 for i in range(len(tokens))],
            }
        )

    def test_reports_the_range_and_deepest_tree(self) -> None:
        frame = self._frame([8, 14, 22, 31, 40, 55], [1.8, 2.1, 2.6, 3.0, 3.4, 4.1])
        text = _text(frame, "sentence_complexity")
        assert "mean dependency distance runs 1.8 to 4.1" in text
        assert "tree depth reaches 5" in text

    def test_complexity_that_is_only_length_is_flagged(self) -> None:
        frame = self._frame([8, 14, 22, 31, 40, 55], [1.8, 2.1, 2.6, 3.0, 3.4, 4.1])
        assert "short ones that still score high" in _cautions(frame, "sentence_complexity")

    def test_complexity_independent_of_length_is_not_flagged(self) -> None:
        frame = self._frame([8, 14, 22, 31, 40, 55], [3.9, 2.0, 4.1, 2.2, 3.5, 2.1])
        assert "length ranking" not in _cautions(frame, "sentence_complexity")


class TestDocDuplicates:
    def test_counts_the_copies_and_says_what_they_cost(self) -> None:
        frame = pd.DataFrame(
            {
                "Group": [0, 1],
                "Size": [3, 2],
                "Members": ["a.txt, a_copy.txt, a_2.txt", "b.txt, b_copy.txt"],
            }
        )
        text = _text(frame, "doc_duplicates")
        assert "2 duplicate group(s) covering 5 file(s)" in text
        assert "3 file(s) are counted more than once" in _cautions(frame, "doc_duplicates")


class TestSignedColumnsAreNotTreatedAsMagnitudes:
    """A share of a total is meaningless when the total is a cancellation.

    Found while reading a real sentiment readout, which announced that one
    row held "417% of all Compound" -- the positives and negatives had very
    nearly cancelled in the denominator.
    """

    def test_a_signed_score_gets_no_share_of_total_claim(self) -> None:
        frame = pd.DataFrame(
            {
                "Document": [f"d{i}.txt" for i in range(6)],
                "Compound": [0.9, -0.8, 0.7, -0.75, 0.05, -0.02],
            }
        )
        assert "of all Compound" not in _cautions(frame, "")

    def test_a_count_column_still_gets_one(self) -> None:
        frame = pd.DataFrame(
            {
                "Document": [f"d{i}.txt" for i in range(6)],
                "Hits": [900, 10, 20, 15, 5, 8],
            }
        )
        assert "of all Hits" in _cautions(frame, "")

    def test_no_share_is_ever_reported_above_one_hundred_percent(self) -> None:
        frame = pd.DataFrame(
            {
                "Document": [f"d{i}.txt" for i in range(8)],
                "Score": [5.0, -4.9, 3.0, -2.9, 1.0, -0.9, 0.4, -0.3],
                "Ratio": [2.0, -1.9, 1.5, -1.4, 0.8, -0.7, 0.2, -0.1],
            }
        )
        for caution in readout(frame, tool="").cautions:
            assert "% of all" not in caution or "100%" not in caution


class TestRecommendationsForTheNewTools:
    def test_a_tool_histogram_is_not_offered_twice(self) -> None:
        """The generic pass used to re-propose a histogram the tool already asked for."""
        frame = TestDocSimilarity._frame([12.0, 44.5, 31.0, 58.2, 22.0])
        charts = recommend_charts(frame, tool="doc_similarity")
        histograms = [c for c in charts if c.spec.kind == "histogram" and c.spec.x == "Similarity"]
        assert len(histograms) == 1

    def test_similarity_is_recommended_as_a_histogram_not_a_pairing(self) -> None:
        frame = TestDocSimilarity._frame([12.0, 44.5, 31.0, 58.2, 22.0])
        charts = recommend_charts(frame, tool="doc_similarity")
        assert charts[0].spec.kind == "histogram"
        assert charts[0].spec.x == "Similarity"

    def test_sentiment_is_recommended_against_its_own_coverage(self) -> None:
        frame = TestSentiment._sentences([0.5, 0.0, -0.4, 0.8, 0.2], [2, 0, 3, 5, 1])
        pairs = {(c.spec.x, c.spec.y) for c in recommend_charts(frame, tool="sentiment_vader_anew")}
        assert ("Hits", "Compound") in pairs

    def test_corpus_statistics_leads_with_the_length_check(self) -> None:
        frame = TestLengthDependence._corpus([100, 200, 400, 800, 1600, 3200], [0.42, 0.55, 0.39, 0.61, 0.44, 0.58])
        charts = recommend_charts(frame, tool="corpus_statistics")
        assert (charts[0].spec.x, charts[0].spec.y) == ("Tokens", "TTR")

    def test_a_collinear_pairing_is_still_refused_for_a_new_tool(self) -> None:
        """Tokens against TTR is only offered while it is capable of saying something."""
        frame = TestLengthDependence._corpus([100, 200, 400, 800, 1600, 3200], [0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        pairs = {(c.spec.x, c.spec.y) for c in recommend_charts(frame, tool="corpus_statistics")}
        assert ("Tokens", "TTR") not in pairs

    def test_lexical_diversity_recommends_the_column_that_exists(self) -> None:
        frame = TestLexicalDiversity._frame([500, 900, 1400, 220, 640], [62.5, 88.1, 70.2, 55.0, 91.4])
        pairs = {(c.spec.x, c.spec.y) for c in recommend_charts(frame, tool="lexical_diversity")}
        assert ("Total Tokens", "MTLD") in pairs


class TestStructuralRepetitionIsNotAFinding:
    """Repetition built into a table's shape is arithmetic, not an observation.

    A pairwise similarity table over n documents has n(n-1)/2 rows, and every
    document appears in exactly n-1 of them. On a four-document corpus that is
    50% of the rows, which tripped the generic dominance rule and produced
    "50% of rows share one Document A value" on every such run.
    """

    def test_a_pairwise_table_does_not_report_its_own_arithmetic(self) -> None:
        frame = pd.DataFrame(
            {
                "Document ID A": ["1", "1", "1", "2", "2", "3"],
                "Document A": ["a.txt", "a.txt", "a.txt", "b.txt", "b.txt", "c.txt"],
                "Document ID B": ["2", "3", "4", "3", "4", "4"],
                "Document B": ["b.txt", "c.txt", "d.txt", "c.txt", "d.txt", "d.txt"],
                "Similarity": [100.0, 38.2, 64.4, 41.0, 55.5, 47.1],
                "Band": ["90-100%", "30-40%", "60-70%", "40-50%", "50-60%", "40-50%"],
            }
        )
        said = _text(frame, "doc_similarity")
        assert "share one Document A value" not in said
        assert "share one Document B value" not in said
        # The real reading survives.
        assert "6 document pair(s) compared" in said

    def test_an_ngram_table_does_not_report_one_row_per_document(self) -> None:
        frame = TestNgrams._frame(
            [("of the", 4, 9, "d1"), ("of the", 5, 9, "d2"), ("in a", 3, 3, "d1"), ("in a", 2, 3, "d2")]
        )
        said = _text(frame, "ngrams")
        assert "share one N-gram value" not in said
        assert "share one Document value" not in said

    def test_a_genuinely_lopsided_category_is_still_reported(self) -> None:
        """Suppression is per-column, not a blanket switch."""
        frame = pd.DataFrame(
            {
                "Document ID A": [str(i) for i in range(6)],
                "Document A": [f"a{i}.txt" for i in range(6)],
                "Document ID B": [str(i + 10) for i in range(6)],
                "Document B": [f"b{i}.txt" for i in range(6)],
                "Similarity": [12.0, 14.0, 16.0, 18.0, 11.0, 13.0],
                "Band": ["10-20%"] * 5 + ["20-30%"],
            }
        )
        assert "share one Band value" in _text(frame, "doc_similarity")

    def test_a_table_with_no_tool_rule_is_unaffected(self) -> None:
        frame = pd.DataFrame(
            {
                "Category": ["x"] * 5 + ["y"],
                "Value": [1, 2, 3, 4, 5, 6],
            }
        )
        assert "share one Category value" in _text(frame, "")


class TestOnlyMagnitudesGetAShareOfTotal:
    """``core.insight.profile`` says of PROPORTION: "sums do not" [mean anything].

    The readout was ignoring its own model and announcing that one document
    held "70% of all TTR" -- 70% of the sum of every document's type-token
    ratio, which is not a fact about anything.
    """

    @staticmethod
    def _frame(column: str, values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"Document": [f"d{i}.txt" for i in range(len(values))], column: values})

    def test_a_ratio_column_gets_no_share_of_total(self) -> None:
        frame = self._frame("TTR", [0.9, 0.1, 0.12, 0.08, 0.05, 0.06])
        assert "of all TTR" not in _cautions(frame, "")

    def test_an_unbounded_score_gets_no_share_of_total(self) -> None:
        """Real MTLD values, which are not whole numbers.

        A non-negative column of whole numbers is classified as a count, and
        rightly so -- the distinction being tested is the role, not the size.
        """
        frame = self._frame("MTLD", [903.4, 10.7, 12.2, 8.9, 5.1, 6.3])
        assert "of all MTLD" not in _cautions(frame, "")

    def test_a_count_column_still_does(self) -> None:
        frame = pd.DataFrame(
            {
                "Document": [f"d{i}.txt" for i in range(6)],
                "Tokens": [9000, 100, 120, 80, 50, 60],
            }
        )
        assert "of all Tokens" in _cautions(frame, "")


class TestConcordanceNoise:
    """A concordance describes the query as much as the corpus."""

    @staticmethod
    def _frame(n: int = 8) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Document ID": ["d1"] * n,
                "Document": ["a.txt"] * n,
                "Sentence ID": list(range(n)),
                "Hit": ["harvest"] * n,  # one query term: constant by construction
                "Left Context": ["The"] * (n - 1) + ["a late"],
                "Right Context": ["was late that year"] * (n - 1) + ["season"],
            }
        )

    def test_the_single_query_term_is_not_reported_as_a_constant_column(self) -> None:
        said = _text(self._frame(), "kwic")
        assert "same value in every row" not in said

    def test_repeated_context_is_not_reported_as_dominance(self) -> None:
        said = _text(self._frame(), "kwic")
        assert "share one Left Context value" not in said
        assert "share one Right Context value" not in said

    def test_the_real_reading_survives(self) -> None:
        said = _text(self._frame(), "kwic")
        assert "8 concordance line(s) over 1 document(s)" in said
        assert "evidence to read, not a count to total" in said

    def test_a_constant_column_elsewhere_is_still_reported(self) -> None:
        """Suppression is scoped to the columns a rule declares, not global."""
        frame = pd.DataFrame({"Category": ["a", "b", "c"], "Parser": ["spacy"] * 3, "Count": [1, 2, 3]})
        assert "same value in every row" in _text(frame, "")


class TestTheReadingAlwaysSaysSomething:
    """A readout whose whole content is "240 row(s) and 5 column(s)".

    Four tools produced exactly that -- a word list, a co-occurrence table, a
    per-sentence emotion arc, a per-sentence shape table -- because no tool
    rule covered them and the generic checks all declined. Each had an obvious
    first sentence available.
    """

    def test_a_labelled_count_table_names_its_largest_row(self) -> None:
        frame = pd.DataFrame({"Word": ["the", "about", "harvest"], "Count": [81, 60, 12]})
        said = _text(frame, "conll_wordlist")
        assert "The largest Count is 81, at Word 'the'" in said
        assert "runs from 12 to 81" in said

    def test_an_unlabelled_measure_table_reports_its_range(self) -> None:
        frame = pd.DataFrame({"Document ID": [1, 1, 2], "Compound": [-0.61, 0.0, 0.2]})
        assert "Compound runs from -0.61 to 0.2 across 3 row(s)" in _text(frame, "narrative")

    def test_ties_are_broken_the_same_way_every_run(self) -> None:
        """R6: two rows with the same count must not swap between runs."""
        frame = pd.DataFrame({"Word": ["zebra", "apple", "mango"], "Count": [9, 9, 3]})
        assert "at Word 'apple'" in _text(frame, "conll_wordlist")

    def test_it_is_a_floor_not_a_default(self) -> None:
        """It must never crowd out a reading a tool rule already produced."""
        frame = pd.DataFrame(
            {
                "Term": ["alpha", "beta", "gamma"],
                "Gries DP": [0.9, 0.2, 0.1],
                "Frequency": [50, 30, 10],
                "Range": [1, 4, 4],
                "Parts": [4, 4, 4],
                "Adjusted Frequency": [5.0, 25.0, 9.0],
            }
        )
        said = _text(frame, "dispersion")
        assert "The largest" not in said
        assert "term(s) measured across 4 parts" in said

    def test_a_table_with_nothing_numeric_stays_quiet(self) -> None:
        """Silence is right when there is genuinely nothing to measure."""
        frame = pd.DataFrame({"Left": ["a", "b"], "Right": ["c", "d"]})
        assert readout(frame, tool="").observations == ()


class TestOneRowResults:
    """With a single row every column is constant and every value is 100%.

    A summary table was being told that "10 column(s) hold the same value in
    every row" — true of any one-row table, informative about none of them.
    """

    FRAME = pd.DataFrame({"Group": ["all"], "Column": ["Count"], "Count": [3], "Mean": [34.0], "Std": [40.730824]})

    def test_the_shape_observations_are_suppressed(self) -> None:
        said = _text(self.FRAME, "csv_stats")
        assert "same value in every row" not in said
        assert "share one" not in said

    def test_the_row_is_read_out_instead(self) -> None:
        said = _text(self.FRAME, "csv_stats")
        assert "A single row of results: Count 3, Mean 34, Std 40.73." in said

    def test_a_one_row_table_with_no_numbers_stays_quiet(self) -> None:
        frame = pd.DataFrame({"Group": ["all"], "Note": ["nothing to report"]})
        assert readout(frame, tool="").observations == ()

    def test_two_rows_still_get_the_full_reading(self) -> None:
        frame = pd.DataFrame({"Parser": ["spacy", "spacy"], "Count": [3, 9]})
        assert "same value in every row" in _text(frame, "")


class TestSvoCompare:
    """Jaccard calls two empty sets identical, which is not a similarity."""

    @staticmethod
    def _frame(rows: list[tuple[int, int, int, float]]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Doc A": a,
                    "Doc B": b,
                    "Common Triples": common,
                    "Subject Jaccard": jaccard,
                    "Verb Jaccard": jaccard,
                    "Object Jaccard": jaccard,
                    "Triple Jaccard": jaccard,
                }
                for a, b, common, jaccard in rows
            ]
        )

    def test_a_perfect_score_from_two_empty_sets_is_called_out(self) -> None:
        frame = self._frame([(1, 2, 0, 1.0), (1, 3, 0, 0.0), (2, 3, 0, 0.0)])
        cautions = _cautions(frame, "svo_compare")
        assert "Jaccard calls two empty sets identical" in cautions
        assert "sorting this column puts them on top" in cautions

    def test_it_is_not_reported_as_the_closest_pair(self) -> None:
        frame = self._frame([(1, 2, 0, 1.0), (1, 3, 0, 0.0)])
        said = _text(frame, "svo_compare")
        assert "closest pair" not in said
        assert "none of them share a complete subject-verb-object triple" in said

    def test_a_real_overlap_is_named(self) -> None:
        frame = self._frame([(1, 2, 0, 1.0), (1, 3, 4, 0.55), (2, 3, 1, 0.20)])
        said = _text(frame, "svo_compare")
        assert "closest pair that actually shares a triple is 1 and 3" in said
        assert "Jaccard calls two empty sets identical" not in _cautions(frame, "svo_compare")

    def test_document_ids_are_not_printed_as_floats(self) -> None:
        frame = self._frame([(1, 3, 4, 0.55), (2, 3, 1, 0.2)])
        said = _text(frame, "svo_compare")
        assert "1.0 and 3.0" not in said
        assert "is 1 and 3" in said

    def test_the_pairwise_columns_are_structural(self) -> None:
        frame = self._frame([(1, 2, 2, 0.5), (1, 3, 2, 0.4), (1, 4, 2, 0.3)])
        assert "share one Doc A value" not in _text(frame, "svo_compare")
