"""Tests for the "terms" panel family (core/viz/panels_terms.py).

Fixtures use the exact column names the real tools produce (see
core/analysis/{nrc,tfidf,nominalization,verbnet,framenet,symbolic,wordnet}.py
and core/profiler/executor.py's ``_adapt_ngrams``/``_adapt_conll_wordlist``),
not invented ones -- an impossible fixture would pass every test written
against it and tell us nothing about the real output.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from core.result import Result
from core.viz.panels_terms import (
    FRAMENET_RANKED_FRAMES,
    NGRAMS_RANKED_TERMS,
    NOMINALIZATION_RANKED_TERMS,
    NRC_EMOTION_HEATMAP,
    SEMANTIC_RANKED_LEMMAS,
    SYMBOLIC_RANKED_ACTORS,
    SYMBOLIC_RANKED_SPACES,
    TERMS_PANELS,
    TFIDF_HEATMAP,
    TFIDF_TOP_TERMS,
    VERBNET_RANKED_CLASSES,
    WORDLIST_RANKED_TERMS,
    WORDNET_RANKED_CATEGORIES,
)
from core.viz.panelspec import PreparedPanel, Provenance


def provenance(tool: str, panel: str, settings: dict[str, Any] | None = None) -> Provenance:
    return Provenance(tool=tool, panel=panel, source="f.csv", settings=settings or {})


def build(
    definition, frame: pd.DataFrame, params: dict[str, Any] | None = None, settings=None
) -> Result[PreparedPanel]:
    merged = definition.defaults() | (params or {})
    return definition.build(frame, merged, provenance(definition.tool, definition.name, settings))


def filtered_rows(prepared: PreparedPanel, filters: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    rows = prepared.data
    for column, value in filters:
        rows = rows[rows[column].astype(str) == value]
    return rows


# ------------------------------------------------------------------ ngrams --


def ngram_row(gram: str, doc_id: str, doc: str, freq_doc: int, freq_corpus: int) -> dict[str, Any]:
    return {
        "N-gram": gram,
        "Frequency in Document": freq_doc,
        "Frequency in Corpus": freq_corpus,
        "Document ID": doc_id,
        "Document": doc,
        "Date": "1934-01-03",
        "Year": 1934,
    }


def ngrams_frame() -> pd.DataFrame:
    # "of the" (pure function words) occurs in both documents; "united states"
    # (content bigram) occurs in one; "of congress" starts on a function word,
    # the fragment shape that topped the real corpus ("the world", "the
    # united") before the phrase-boundary rule.
    return pd.DataFrame(
        [
            ngram_row("of the", "1", "doc_a", 10, 18),
            ngram_row("of the", "2", "doc_b", 8, 18),
            ngram_row("united states", "1", "doc_a", 5, 5),
            ngram_row("of congress", "1", "doc_a", 3, 3),
            ngram_row("state of the union", "2", "doc_b", 2, 2),
        ]
    )


class TestNgrams:
    def test_registers_itself(self) -> None:
        assert NGRAMS_RANKED_TERMS.tool == "ngrams"
        assert NGRAMS_RANKED_TERMS.shape == "ranked_bars"
        assert set(NGRAMS_RANKED_TERMS.requires) <= set(ngrams_frame().columns)

    def test_default_hides_fragments_and_keeps_phrases(self) -> None:
        result = build(NGRAMS_RANKED_TERMS, ngrams_frame())
        assert result.value is not None
        labels = {mark.label for mark in result.unwrap().marks}
        assert "of the" not in labels
        assert "of congress" not in labels, "a phrase does not start on a function word"
        assert "united states" in labels
        assert "state of the union" in labels, "function words inside a phrase are fine"

    def test_punctuation_is_not_a_term(self) -> None:
        frame = pd.concat([ngrams_frame(), pd.DataFrame([ngram_row(",", "1", "doc_a", 40, 40)])], ignore_index=True)
        labels = {mark.label for mark in build(NGRAMS_RANKED_TERMS, frame).unwrap().marks}
        assert "," not in labels

    def test_showing_function_words_restores_of_the(self) -> None:
        result = build(NGRAMS_RANKED_TERMS, ngrams_frame(), {"hide-function-words": False})
        assert result.value is not None
        labels = {mark.label for mark in result.unwrap().marks}
        assert "of the" in labels

    def test_corpus_frequency_is_not_summed_across_document_rows(self) -> None:
        # "of the" has Frequency in Corpus = 18 on both its rows (repeated,
        # not additive); summing would wrongly report 36.
        result = build(NGRAMS_RANKED_TERMS, ngrams_frame(), {"hide-function-words": False})
        prepared = result.unwrap()
        mark = next(m for m in prepared.marks if m.label == "of the")
        assert mark.x == 18

    def test_coverage_is_distinct_documents_not_row_count(self) -> None:
        result = build(NGRAMS_RANKED_TERMS, ngrams_frame(), {"hide-function-words": False})
        prepared = result.unwrap()
        mark = next(m for m in prepared.marks if m.label == "of the")
        assert "in 2 of 2 documents" in mark.evidence.describe
        solo = next(m for m in prepared.marks if m.label == "united states")
        assert "in 1 of 2 documents" in solo.evidence.describe

    def test_evidence_filters_resolve_against_published_data(self) -> None:
        result = build(NGRAMS_RANKED_TERMS, ngrams_frame())
        prepared = result.unwrap()
        for mark in prepared.marks:
            rows = filtered_rows(prepared, mark.evidence.filters)
            assert len(rows) == 1, mark.key
        # Control: a filter value that should not exist selects nothing.
        assert filtered_rows(prepared, (("N-gram", "not a real gram"),)).empty

    def test_phrase_and_lemma_evidence(self) -> None:
        result = build(NGRAMS_RANKED_TERMS, ngrams_frame(), settings={"field": "lemma"})
        mark = next(m for m in result.unwrap().marks if m.label == "united states")
        assert mark.evidence.phrase == "united states"
        assert mark.evidence.lemma is True

    def test_n_filters_by_ngram_length(self) -> None:
        frame = pd.concat([ngrams_frame(), pd.DataFrame([ngram_row("war", "1", "doc_a", 4, 4)])], ignore_index=True)
        result = build(NGRAMS_RANKED_TERMS, frame, {"n": 1, "hide-function-words": False})
        labels = {mark.label for mark in result.unwrap().marks}
        assert labels == {"war"}

    def test_order_by_coverage_changes_rank_not_bar_length(self) -> None:
        # "of congress" appears in one doc with freq 3; "united states" also
        # one doc, freq 5. Both have coverage 1, so order=coverage ties them
        # on document count but the bar (x) must still be raw frequency.
        result = build(NGRAMS_RANKED_TERMS, ngrams_frame(), {"order": "coverage", "hide-function-words": False})
        prepared = result.unwrap()
        by_label = {m.label: m.x for m in prepared.marks}
        assert by_label["of the"] == 18
        assert by_label["united states"] == 5

    def test_empty_after_filtering_fails_with_reason(self) -> None:
        frame = pd.DataFrame([ngram_row("of the", "1", "doc_a", 10, 18)])
        result = build(NGRAMS_RANKED_TERMS, frame)
        assert result.value is None
        assert any(d.code == "PANEL_NO_DATA" for d in result.diagnostics)


# ---------------------------------------------------------- conll_wordlist --


class TestWordlist:
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Word": "the", "Count": 500, "Percentage": 5.0},
                {"Word": "congress", "Count": 40, "Percentage": 0.4},
                {"Word": "freedom", "Count": 25, "Percentage": 0.25},
            ]
        )

    def test_hides_function_words_by_default(self) -> None:
        result = build(WORDLIST_RANKED_TERMS, self.frame())
        labels = {mark.label for mark in result.unwrap().marks}
        assert labels == {"congress", "freedom"}

    def test_no_document_coverage_offered(self) -> None:
        result = build(WORDLIST_RANKED_TERMS, self.frame(), {"hide-function-words": False})
        prepared = result.unwrap()
        assert "Documents" not in prepared.data.columns
        assert all("documents" not in mark.evidence.describe for mark in prepared.marks)
        assert "no document association" in " ".join(prepared.notes)

    def test_evidence_resolves(self) -> None:
        result = build(WORDLIST_RANKED_TERMS, self.frame())
        prepared = result.unwrap()
        for mark in prepared.marks:
            assert len(filtered_rows(prepared, mark.evidence.filters)) == 1

    def test_a_list_of_function_words_says_how_to_fix_the_run(self) -> None:
        """The real run's top 20 was every one a function word or punctuation."""
        frame = pd.DataFrame(
            [{"Word": w, "Count": 100 - i, "Percentage": 1.0} for i, w in enumerate(["the", ",", "of", ".", "and"])]
        )
        result = build(WORDLIST_RANKED_TERMS, frame)
        # Drawn unfiltered rather than refused: an error box on the default
        # run shows nothing, the list shows what the run kept.
        assert result.value is not None
        assert any(d.code == "PANEL_FUNCTION_WORDS_SHOWN" for d in result.diagnostics)
        advice = [d for d in result.diagnostics if d.code == "PANEL_WORDLIST_TOO_SHORT"]
        assert advice and "category" in advice[0].message


# --------------------------------------------------------- nominalization --


class TestNominalization:
    def frame(self) -> pd.DataFrame:
        rows = []
        for doc in ("doc_a", "doc_b"):
            for _ in range(3):
                rows.append(
                    {
                        "Word": "decision",
                        "Base Verb": "decide",
                        "Document ID": doc,
                        "Document": doc,
                        "Date": "1934-01-03",
                        "Year": 1934,
                        "Sentence ID": 1,
                    }
                )
        rows.append(
            {
                "Word": "creation",
                "Base Verb": "create",
                "Document ID": "doc_a",
                "Document": "doc_a",
                "Date": "1934-01-03",
                "Year": 1934,
                "Sentence ID": 2,
            }
        )
        return pd.DataFrame(rows)

    def test_frequency_is_row_count_per_word(self) -> None:
        result = build(NOMINALIZATION_RANKED_TERMS, self.frame())
        prepared = result.unwrap()
        mark = next(m for m in prepared.marks if m.label == "decision")
        assert mark.x == 6
        assert "in 2 of 2 documents" in mark.evidence.describe

    def test_hover_names_the_base_verb(self) -> None:
        result = build(NOMINALIZATION_RANKED_TERMS, self.frame())
        prepared = result.unwrap()
        mark = next(m for m in prepared.marks if m.label == "decision")
        assert "Base Verb decide" in mark.evidence.describe

    def test_no_function_word_toggle_declared(self) -> None:
        assert "hide-function-words" not in {p.name for p in NOMINALIZATION_RANKED_TERMS.params}

    def test_evidence_resolves(self) -> None:
        result = build(NOMINALIZATION_RANKED_TERMS, self.frame())
        prepared = result.unwrap()
        for mark in prepared.marks:
            assert len(filtered_rows(prepared, mark.evidence.filters)) == 1


# -------------------------------------------------------------- semantic --


class TestSemantic:
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Lemma": "the", "WordNet": "", "VerbNet": "", "FrameNet": "", "Count": 900},
                {
                    "Lemma": "govern",
                    "WordNet": "govern.v.01",
                    "VerbNet": "govern-... class",
                    "FrameNet": "Government",
                    "Count": 12,
                },
            ]
        )

    def test_hides_function_words_by_default(self) -> None:
        result = build(SEMANTIC_RANKED_LEMMAS, self.frame())
        labels = {mark.label for mark in result.unwrap().marks}
        assert labels == {"govern"}

    def test_tags_appear_in_hover(self) -> None:
        result = build(SEMANTIC_RANKED_LEMMAS, self.frame())
        mark = next(m for m in result.unwrap().marks if m.label == "govern")
        assert "FrameNet Government" in mark.evidence.describe

    def test_evidence_resolves(self) -> None:
        result = build(SEMANTIC_RANKED_LEMMAS, self.frame(), {"hide-function-words": False})
        prepared = result.unwrap()
        for mark in prepared.marks:
            assert len(filtered_rows(prepared, mark.evidence.filters)) == 1


# ---------------------------------------------------- category lookups --


class TestVerbNet:
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Word": "run", "VerbNet Class": "run-51.3.2"},
                {"Word": "sprint", "VerbNet Class": "run-51.3.2"},
                {"Word": "give", "VerbNet Class": "give-13.1"},
                {"Word": "zzz", "VerbNet Class": "Not found"},
            ]
        )

    def test_not_found_is_excluded(self) -> None:
        result = build(VERBNET_RANKED_CLASSES, self.frame())
        labels = {mark.label for mark in result.unwrap().marks}
        assert "Not found" not in labels
        assert labels == {"run-51.3.2", "give-13.1"}

    def test_class_frequency_is_word_count(self) -> None:
        result = build(VERBNET_RANKED_CLASSES, self.frame())
        mark = next(m for m in result.unwrap().marks if m.label == "run-51.3.2")
        assert mark.x == 2

    def test_no_document_coverage_or_phrase(self) -> None:
        result = build(VERBNET_RANKED_CLASSES, self.frame())
        prepared = result.unwrap()
        assert "Documents" not in prepared.data.columns
        for mark in prepared.marks:
            assert mark.evidence.phrase == ""
            assert mark.evidence.scope == "rows"

    def test_evidence_resolves(self) -> None:
        result = build(VERBNET_RANKED_CLASSES, self.frame())
        prepared = result.unwrap()
        for mark in prepared.marks:
            assert len(filtered_rows(prepared, mark.evidence.filters)) == 1


class TestFrameNetSymbolicWordNet:
    def test_framenet_requires(self) -> None:
        assert set(FRAMENET_RANKED_FRAMES.requires) == {"Word", "FrameNet Frame"}

    def test_symbolic_spaces_excludes_unclassified(self) -> None:
        frame = pd.DataFrame(
            [
                {"Word": "castle", "Space Type": "royal_court"},
                {"Word": "hall", "Space Type": "royal_court"},
                {"Word": "thing", "Space Type": "unclassified"},
            ]
        )
        result = build(SYMBOLIC_RANKED_SPACES, frame)
        labels = {mark.label for mark in result.unwrap().marks}
        assert labels == {"royal_court"}

    def test_symbolic_actors_uses_actor_column(self) -> None:
        assert set(SYMBOLIC_RANKED_ACTORS.requires) == {"Actor", "Actor Type"}
        frame = pd.DataFrame(
            [
                {"Actor": "senator", "Actor Type": "authority_official"},
                {"Actor": "governor", "Actor Type": "authority_official"},
            ]
        )
        result = build(SYMBOLIC_RANKED_ACTORS, frame)
        mark = result.unwrap().marks[0]
        assert mark.label == "authority_official"
        assert mark.x == 2

    def test_wordnet_requires(self) -> None:
        assert set(WORDNET_RANKED_CATEGORIES.requires) == {"Word", "WordNet Category"}
        frame = pd.DataFrame(
            [
                {"Word": "dog", "WordNet Category": "animal"},
                {"Word": "cat", "WordNet Category": "animal"},
                {"Word": "xyzzy", "WordNet Category": "Not found"},
            ]
        )
        result = build(WORDNET_RANKED_CATEGORIES, frame)
        labels = {mark.label for mark in result.unwrap().marks}
        assert labels == {"animal"}


# ------------------------------------------------------------------- nrc --


class TestNrcHeatmap:
    def frame(self) -> pd.DataFrame:
        emotions = ["fear", "anger", "anticipation", "trust", "surprise", "sadness", "disgust", "joy"]
        row_a = {"Document ID": "1", "Document": "1934-01-03_franklin d roosevelt_sotu.txt", "Tokens": 2000, "Hits": 80}
        row_a.update({e: round(1 / 8, 4) for e in emotions})
        row_b = {"Document ID": "2", "Document": "1961-01-20_john f kennedy_sotu.txt", "Tokens": 1800, "Hits": 60}
        row_b.update({e: 0.5 if e == "joy" else 0.0 for e in emotions})
        row_undated = {"Document ID": "3", "Document": "addendum_notes.txt", "Tokens": 100, "Hits": 5}
        row_undated.update({e: 0.0 for e in emotions})
        row_undated["fear"] = 1.0
        return pd.DataFrame([row_b, row_a, row_undated])  # deliberately out of date order

    def test_shape_and_categories(self) -> None:
        result = build(NRC_EMOTION_HEATMAP, self.frame())
        prepared = result.unwrap()
        assert prepared.shape == "heatmap"
        assert prepared.x_categories == (
            "fear",
            "anger",
            "anticipation",
            "trust",
            "surprise",
            "sadness",
            "disgust",
            "joy",
        )
        assert len(prepared.y_categories) == 3
        assert len(prepared.marks) == 3 * 8

    def test_rows_are_ordered_by_parsed_date_with_undated_last(self) -> None:
        result = build(NRC_EMOTION_HEATMAP, self.frame())
        prepared = result.unwrap()
        # 1934 Roosevelt before 1961 Kennedy before the undated file.
        assert "1934" in prepared.y_categories[0]
        assert "1961" in prepared.y_categories[1]
        assert prepared.y_categories[2] not in {prepared.y_categories[0], prepared.y_categories[1]}

    def test_value_is_the_documents_own_share(self) -> None:
        result = build(NRC_EMOTION_HEATMAP, self.frame())
        prepared = result.unwrap()
        kennedy_joy = next(m for m in prepared.marks if m.key == "2:joy")
        assert kennedy_joy.value == 0.5

    def test_evidence_resolves_to_the_document_row(self) -> None:
        result = build(NRC_EMOTION_HEATMAP, self.frame())
        prepared = result.unwrap()
        for mark in prepared.marks:
            assert len(filtered_rows(prepared, mark.evidence.filters)) == 1

    def test_missing_emotion_column_fails_cleanly(self) -> None:
        frame = self.frame().drop(columns=["joy"])
        result = build(NRC_EMOTION_HEATMAP, frame)
        assert result.value is None
        assert any(d.code == "PANEL_MISSING_COLUMN" for d in result.diagnostics)


# ----------------------------------------------------------------- tfidf --


def tfidf_frame() -> pd.DataFrame:
    rows = []
    docs = [
        ("1", "1934-01-03_franklin d roosevelt_sotu.txt", "1934-01-03"),
        ("2", "1961-01-20_john f kennedy_sotu.txt", "1961-01-20"),
    ]
    # congress appears in both documents with a high TF-IDF in each, so its
    # summed weight beats a term that is extremely distinctive in only one
    # (vigor, 1.2). Ranks follow each document's own descending TF-IDF.
    terms = {
        "1": [("recovery", 12, 1, 0.9812), ("congress", 30, 2, 0.7), ("banking", 8, 1, 0.6)],
        "2": [("vigor", 6, 1, 1.2), ("congress", 25, 2, 0.65), ("freedom", 10, 2, 0.4)],
    }
    for doc_id, doc, date in docs:
        for rank, (term, count, doc_freq, tfidf) in enumerate(terms[doc_id], start=1):
            rows.append(
                {
                    "Document ID": doc_id,
                    "Document": doc,
                    "Date": date,
                    "Year": int(date[:4]),
                    "Term": term,
                    "Count": count,
                    "Document Frequency": doc_freq,
                    "TF": float(count),
                    "IDF": 1.0,
                    "TF-IDF": tfidf,
                    "Rank": rank,
                }
            )
    return pd.DataFrame(rows)


class TestTfidfTopTerms:
    def test_defaults_to_earliest_document_by_date(self) -> None:
        result = build(TFIDF_TOP_TERMS, tfidf_frame())
        prepared = result.unwrap()
        assert "1934" in prepared.subtitle

    def test_explicit_document_param(self) -> None:
        result = build(TFIDF_TOP_TERMS, tfidf_frame(), {"document": "1961-01-20_john f kennedy_sotu.txt"})
        prepared = result.unwrap()
        labels = {mark.label for mark in prepared.marks}
        assert labels == {"vigor", "congress", "freedom"}

    def test_ranked_by_the_tools_own_rank_column(self) -> None:
        result = build(TFIDF_TOP_TERMS, tfidf_frame())
        prepared = result.unwrap()
        # rank 1 for doc "1" is recovery: ranked_bars draws y = 0 as the top row.
        top = min(prepared.marks, key=lambda m: m.y)
        assert top.label == "recovery"
        assert top.y == 0.0

    def test_no_warning_when_the_terms_are_distinctive(self) -> None:
        assert not [d for d in build(TFIDF_TOP_TERMS, tfidf_frame()).diagnostics if d.code == "PANEL_CORPUS_WIDE_TERMS"]

    def test_corpus_wide_top_terms_say_how_to_fix_the_run(self) -> None:
        """The real 87-speech run at the default max-df-ratio 1.0: every
        speech's top terms were 'the, of, and', each found in every speech."""
        frame = tfidf_frame()
        frame["Document Frequency"] = 2
        warnings = [d for d in build(TFIDF_TOP_TERMS, frame).diagnostics if d.code == "PANEL_CORPUS_WIDE_TERMS"]
        assert len(warnings) == 1
        assert "max-df-ratio" in warnings[0].message

    def test_unknown_document_fails_with_reason(self) -> None:
        result = build(TFIDF_TOP_TERMS, tfidf_frame(), {"document": "not_a_real_document.txt"})
        assert result.value is None
        assert any(d.code == "PANEL_DOCUMENT_NOT_FOUND" for d in result.diagnostics)

    def test_evidence_resolves(self) -> None:
        result = build(TFIDF_TOP_TERMS, tfidf_frame())
        prepared = result.unwrap()
        for mark in prepared.marks:
            assert len(filtered_rows(prepared, mark.evidence.filters)) == 1


class TestTfidfHeatmap:
    def test_columns_are_top_k_by_summed_tfidf(self) -> None:
        result = build(TFIDF_HEATMAP, tfidf_frame(), {"top-k": 3})
        prepared = result.unwrap()
        assert len(prepared.x_categories) == 3
        assert "congress" in prepared.x_categories  # appears in both docs, high combined weight

    def test_rows_ordered_by_date(self) -> None:
        result = build(TFIDF_HEATMAP, tfidf_frame())
        prepared = result.unwrap()
        assert "1934" in prepared.y_categories[0]
        assert "1961" in prepared.y_categories[1]

    def test_a_term_absent_from_a_documents_top_list_has_no_mark(self) -> None:
        result = build(TFIDF_HEATMAP, tfidf_frame(), {"top-k": 5})
        prepared = result.unwrap()
        # "vigor" only appears in document 2's top terms; no mark should
        # exist for document 1 / vigor.
        assert not any(m.key == "1:vigor" for m in prepared.marks)
        assert any(m.key == "2:vigor" for m in prepared.marks)

    def test_evidence_resolves(self) -> None:
        result = build(TFIDF_HEATMAP, tfidf_frame(), {"top-k": 5})
        prepared = result.unwrap()
        for mark in prepared.marks:
            assert len(filtered_rows(prepared, mark.evidence.filters)) == 1


# --------------------------------------------------------------- registry --


class TestRegistration:
    def test_every_panel_has_a_unique_name(self) -> None:
        names = [p.name for p in TERMS_PANELS]
        assert len(names) == len(set(names))

    def test_every_panel_declares_question_and_notes(self) -> None:
        for panel in TERMS_PANELS:
            assert panel.question.strip(), panel.name
            assert panel.notes, panel.name

    def test_every_panel_names_a_real_tool(self) -> None:
        expected = {
            "ngrams",
            "conll_wordlist",
            "nominalization",
            "semantic",
            "verbnet",
            "framenet",
            "symbolic",
            "wordnet",
            "nrc",
            "tfidf",
        }
        assert {p.tool for p in TERMS_PANELS} <= expected

    def test_twelve_panels_registered(self) -> None:
        assert len(TERMS_PANELS) == 12


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
