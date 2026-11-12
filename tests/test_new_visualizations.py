"""Tests for newly added panels: spellcheck, k_sentences, bert_extract, table_search, emotion_nrc, keyness, and readability.

Verifies mark construction, shape, required columns, and evidence resolution.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.viz.panels import best_table, panels_for_tool, prepare_panel
from core.viz.panelspec import Source


def test_spellcheck_top_errors_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Document": ["doc1.txt", "doc1.txt", "doc2.txt"],
            "Word": ["teh", "recieve", "teh"],
            "Count": [5, 2, 3],
            "Suggestion": ["the", "receive", "the"],
            "Score": [0.85, 0.90, 0.85],
        }
    )
    res = prepare_panel("spellcheck_top_errors", df, {"top-n": 10}, source=Source("spellcheck.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "ranked_bars"
    assert len(panel.marks) == 2
    # teh (8) > recieve (2)
    assert panel.marks[0].key == "teh"
    assert panel.marks[0].x == 8.0
    assert panel.marks[0].evidence.phrase == "teh"


def test_spellcheck_document_counts_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Document": ["doc1.txt", "doc1.txt", "doc2.txt"],
            "Word": ["teh", "recieve", "teh"],
            "Count": [5, 2, 3],
            "Suggestion": ["the", "receive", "the"],
        }
    )
    res = prepare_panel("spellcheck_document_counts", df, {"top-n": 10}, source=Source("spellcheck.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "ranked_bars"
    assert len(panel.marks) == 2
    # doc1 (7) > doc2 (3)
    assert panel.marks[0].key == "doc1.txt"
    assert panel.marks[0].x == 7.0


def test_spellcheck_confidence_scatter_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Document": ["doc1.txt", "doc2.txt"],
            "Word": ["teh", "unobtanium"],
            "Count": [10, 1],
            "Suggestion": ["the", "uranium"],
            "Score": [0.90, 0.35],
        }
    )
    res = prepare_panel("spellcheck_confidence_scatter", df, {"label-top": 5}, source=Source("spellcheck.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "scatter_labelled"
    assert len(panel.marks) == 2
    assert len(panel.annotations) == 1


def test_k_sentences_top_repetitions_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Lemma": ["freedom", "nation", "peace"],
            "Form": ["freedom", "nation", "peace"],
            "Document ID": ["1", "1", "2"],
            "Doc Frequency": [12, 8, 3],
        }
    )
    res = prepare_panel("k_sentences_top_repetitions", df, {"top-n": 10}, source=Source("k_repetitions.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "ranked_bars"
    assert len(panel.marks) == 3
    assert panel.marks[0].key == "freedom"
    assert panel.marks[0].x == 12.0


def test_k_sentences_section_density_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Document ID": ["doc1", "doc1", "doc2", "doc2"],
            "Document": ["doc1.txt", "doc1.txt", "doc2.txt", "doc2.txt"],
            "Section": ["first", "last", "first", "last"],
            "Tokens": [100, 120, 80, 90],
            "Content Tokens": [55, 60, 42, 45],
        }
    )
    res = prepare_panel("k_sentences_section_density", df, {}, source=Source("k_counts.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "distribution"
    assert len(panel.marks) == 4
    assert len(panel.y_categories) == 2


def test_bert_extract_sentence_positions_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Document ID": ["1", "2"],
            "Document": ["speech_1933.txt", "speech_1934.txt"],
            "Summary": ["s1", "s2"],
            "Summary Sentences": [2, 2],
            "Sentence IDs": ["[1, 14]", "2, 8"],
        }
    )
    res = prepare_panel("bert_extract_sentence_positions", df, {}, source=Source("bert_extract.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "positions"
    assert len(panel.marks) == 4
    assert len(panel.y_categories) == 2


def test_table_search_hits_by_document_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Document ID": ["1", "1", "2"],
            "Document": ["docA.txt", "docA.txt", "docB.txt"],
            "Sentence ID": [1, 5, 2],
        }
    )
    res = prepare_panel("table_search_hits_by_document", df, {"top-n": 5}, source=Source("search_results.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "ranked_bars"
    assert len(panel.marks) == 2
    assert panel.marks[0].x == 2.0


def test_table_search_hit_positions_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Document ID": ["1", "1", "2"],
            "Document": ["docA.txt", "docA.txt", "docB.txt"],
            "Sentence ID": [1, 5, 2],
        }
    )
    res = prepare_panel("table_search_hit_positions", df, {}, source=Source("search_results.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "positions"
    assert len(panel.marks) == 3


def test_nrc_emotion_profile_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Document ID": ["1", "2"],
            "Document": ["doc1.txt", "doc2.txt"],
            "Tokens": [100, 200],
            "Hits": [30, 60],
            "fear": [0.10, 0.12],
            "anger": [0.05, 0.08],
            "anticipation": [0.15, 0.14],
            "trust": [0.25, 0.22],
            "surprise": [0.08, 0.06],
            "sadness": [0.07, 0.09],
            "disgust": [0.05, 0.04],
            "joy": [0.25, 0.25],
        }
    )
    res = prepare_panel("nrc_emotion_profile", df, {}, source=Source("emotion_nrc.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "ranked_bars"
    assert len(panel.marks) == 8


def test_nrc_emotion_over_time_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Document ID": ["1", "2", "3"],
            "Document": ["doc_1933.txt", "doc_1934.txt", "doc_1935.txt"],
            "Tokens": [100, 200, 150],
            "Hits": [30, 60, 45],
            "fear": [0.10, 0.12, 0.11],
            "anger": [0.05, 0.08, 0.06],
            "anticipation": [0.15, 0.14, 0.16],
            "trust": [0.25, 0.22, 0.24],
            "surprise": [0.08, 0.06, 0.07],
            "sadness": [0.07, 0.09, 0.08],
            "disgust": [0.05, 0.04, 0.05],
            "joy": [0.25, 0.25, 0.23],
        }
    )
    # Undated and unordered: refused, as every change-over-time figure is.
    refused = prepare_panel("nrc_emotion_over_time", df, {}, source=Source("emotion_nrc.csv"))
    assert not refused.ok
    assert "PANEL_NO_AXIS" in [d.code for d in refused.diagnostics]

    df["Date"] = ["1933-03-04", "1934-01-03", "1935-01-04"]
    res = prepare_panel("nrc_emotion_over_time", df, {}, source=Source("emotion_nrc.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "line_series"
    assert len(panel.marks) == 15  # 5 key emotions x 3 docs


def test_nrc_emotion_over_time_places_a_dated_corpus_by_year() -> None:
    """A dated corpus once crashed here ('AxisInfo' has no attribute 'lower'):
    the undated fixture above only ever reached the fallback branch."""
    emotions = ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")
    df = pd.DataFrame(
        {
            "Document ID": ["1", "2"],
            "Document": ["1990-01-01_a.txt", "1995-06-01_b.txt"],
            "Date": ["1990-01-01", "1995-06-01"],
            **{emotion: [0.1, 0.2] for emotion in emotions},
        }
    )
    res = prepare_panel("nrc_emotion_over_time", df, {}, source=Source("emotion_nrc.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.x_label == "Year"
    assert panel.x_axis == "time"
    assert sorted({mark.x for mark in panel.marks}) == pytest.approx([1990.0, 1995.4], abs=0.05)


def test_doc_similarity_figures_read_the_full_pair_table_not_duplicates() -> None:
    """A real doc_similarity run writes doc_pairs.csv (every pair) and
    duplicates.csv (same columns minus Band, usually empty). The neighbours
    network chose the narrower empty one and failed with PANEL_EMPTY."""
    pair = ["Document ID A", "Document A", "Document ID B", "Document B", "Similarity"]
    tables = [("doc_pairs.csv", [*pair, "Band"]), ("duplicates.csv", pair)]
    for panel in panels_for_tool("doc_similarity"):
        assert best_table(panel, tables) == "doc_pairs.csv", panel.name


def test_nrc_emotion_distribution_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Document ID": ["1", "2"],
            "Document": ["doc1.txt", "doc2.txt"],
            "Tokens": [100, 200],
            "Hits": [30, 60],
            "fear": [0.10, 0.12],
            "anger": [0.05, 0.08],
            "anticipation": [0.15, 0.14],
            "trust": [0.25, 0.22],
            "surprise": [0.08, 0.06],
            "sadness": [0.07, 0.09],
            "disgust": [0.05, 0.04],
            "joy": [0.25, 0.25],
        }
    )
    res = prepare_panel("nrc_emotion_distribution", df, {}, source=Source("emotion_nrc.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "distribution"
    assert len(panel.y_categories) == 8
    assert len(panel.marks) == 16  # 8 emotions x 2 docs


def test_keyness_top_distinctive_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Word": ["freedom", "duty", "the", "common"],
            "G2 (log-likelihood)": [45.2, 22.1, 1.2, 18.5],
            "Log Ratio": [2.5, 1.8, 0.02, -1.9],
            "Freq Group A (pattern docs)": [50, 30, 1000, 5],
            "Freq Group B (other docs)": [5, 4, 980, 40],
            "Overrepresented in": ["Group A", "Group A", "Group A", "Group B"],
        }
    )
    res = prepare_panel(
        "keyness_top_distinctive", df, {"top-n": 5, "significance": "0.05"}, source=Source("keyness.csv")
    )
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "ranked_bars"
    # 'the' (G2=1.2) filtered out because < 3.84 threshold!
    assert len(panel.marks) == 3
    assert panel.marks[0].key == "freedom"


def test_readability_grade_comparison_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Document ID": ["doc1", "doc2"],
            "Document": ["doc1.txt", "doc2.txt"],
            "Flesch-Kincaid Grade": [8.5, 10.2],
            "Gunning Fog Index": [9.1, 11.0],
            "Coleman-Liau Index": [8.0, 9.8],
            "Automated Readability Index": [7.8, 10.5],
            "SMOG Index": [9.0, 10.8],
        }
    )
    res = prepare_panel("readability_grade_comparison", df, {}, source=Source("readability.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "distribution"
    assert len(panel.y_categories) == 5
    assert len(panel.marks) == 10


def test_gender_annotator_top_names_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Document": ["doc1.txt", "doc2.txt", "doc2.txt"],
            "Document ID": ["1", "2", "2"],
            "Name": ["Lincoln", "Washington", "Lincoln"],
            "Gender": ["male", "male", "male"],
            "Dictionary": ["census", "census", "census"],
            "Mentions": [12, 10, 8],
        }
    )
    res = prepare_panel("gender_annotator_top_names", df, {"top-n": 5}, source=Source("names.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "ranked_bars"
    assert len(panel.marks) == 2
    assert panel.marks[0].key == "Lincoln"
    assert panel.marks[0].x == 20.0
    assert panel.marks[0].group == "Male"


def test_gender_annotator_corpus_breakdown_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Gender": ["male", "female", "both", "unknown"],
            "Mentions": [150, 50, 10, 40],
        }
    )
    res = prepare_panel("gender_annotator_corpus_breakdown", df, {}, source=Source("names.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "ranked_bars"
    assert len(panel.marks) == 4
    # Total mentions = 250, male share = 60.0%
    assert panel.marks[0].key == "male"
    assert pytest.approx(panel.marks[0].x, 0.1) == 60.0


def test_dispersion_top_adjusted_terms_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Term": ["freedom", "government", "the", "ephemeral"],
            "Frequency": [100, 120, 5000, 100],
            "Range": [45, 50, 50, 2],
            "Juilland D": [0.85, 0.90, 0.99, 0.05],
            "Adjusted Frequency": [85.0, 108.0, 4950.0, 5.0],
        }
    )
    res = prepare_panel(
        "dispersion_top_adjusted_terms", df, {"top-n": 5, "hide-function-words": True}, source=Source("dispersion.csv")
    )
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "ranked_bars"
    # 'the' is function word -> filtered out; 'government' (108) > 'freedom' (85) > 'ephemeral' (5)
    assert len(panel.marks) == 3
    assert panel.marks[0].key == "government"
    assert panel.marks[0].x == 108.0


def test_coreference_top_clusters_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Cluster ID": ["1", "1", "1", "2", "2"],
            "Lemma": ["president", "president", "president", "nation", "nation"],
            "Mention": ["President", "he", "Mr. President", "nation", "country"],
            "Document ID": ["d1", "d1", "d2", "d1", "d2"],
        }
    )
    res = prepare_panel("coreference_top_clusters", df, {"top-n": 5}, source=Source("coref.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "ranked_bars"
    assert len(panel.marks) == 2
    assert panel.marks[0].x == 3.0
    assert "president" in panel.marks[0].label


def test_ngrams_heatmap_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "N-gram": ["united states", "white house", "united states", "of the"],
            "Frequency in Document": [5, 3, 4, 10],
            "Frequency in Corpus": [9, 3, 9, 10],
            "Document ID": ["d1", "d1", "d2", "d1"],
            "Document": ["doc1.txt", "doc1.txt", "doc2.txt", "doc1.txt"],
        }
    )
    res = prepare_panel("ngrams_heatmap", df, {"top-n": 5, "hide-function-words": True}, source=Source("ngrams.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "heatmap"
    assert "united states" in panel.x_categories
    assert "of the" not in panel.x_categories  # filtered function word


def test_doc_duplicates_cluster_sizes_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Group": ["1", "2"],
            "Size": [4, 2],
            "Members": ["docA, docB, docC, docD", "docE, docF"],
        }
    )
    res = prepare_panel("doc_duplicates_cluster_sizes", df, {}, source=Source("duplicates.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "ranked_bars"
    assert len(panel.marks) == 2
    assert panel.marks[0].key == "1"
    assert panel.marks[0].x == 4.0


def test_knowledge_graph_predicates_prepares_cleanly() -> None:
    df = pd.DataFrame(
        {
            "Subject": ["Paris", "London", "Obama"],
            "Predicate": ["isA", "isA", "role"],
            "Object": ["City", "City", "President"],
            "Source": ["stub", "stub", "stub"],
        }
    )
    res = prepare_panel("knowledge_graph_predicates", df, {}, source=Source("kg.csv"))
    assert res.ok, [str(d) for d in res.diagnostics]
    panel = res.unwrap()
    assert panel.shape == "ranked_bars"
    assert len(panel.marks) == 2
    assert panel.marks[0].key == "isA"
    assert panel.marks[0].x == 2.0


def test_spellcheck_and_ngrams_showcases_render_cleanly() -> None:
    from core.viz.showcase import render_showcase, showcase_for

    sc_spell = showcase_for("spellcheck_showcase")
    assert sc_spell is not None
    spell_df = pd.DataFrame(
        {
            "Document": ["doc1.txt", "doc1.txt", "doc2.txt"],
            "Word": ["teh", "recieve", "teh"],
            "Count": [5, 2, 3],
            "Suggestion": ["the", "receive", "the"],
            "Score": [0.85, 0.90, 0.85],
        }
    )
    res_spell = render_showcase(sc_spell, {"spell_check.csv": spell_df}, fmt="png")
    assert res_spell.ok, [str(d) for d in res_spell.diagnostics]
    assert len(res_spell.unwrap()) > 0

    sc_ngrams = showcase_for("ngrams_showcase")
    assert sc_ngrams is not None
    ngrams_df = pd.DataFrame(
        {
            "N-gram": ["united states", "fellow citizens", "american people"],
            "Frequency in Document": [5, 3, 2],
            "Frequency in Corpus": [25, 12, 10],
            "Document ID": ["1", "1", "1"],
            "Document": ["doc1.txt", "doc1.txt", "doc1.txt"],
        }
    )
    res_ngrams = render_showcase(sc_ngrams, {"ngrams.csv": ngrams_df}, fmt="png")
    assert res_ngrams.ok, [str(d) for d in res_ngrams.diagnostics]
    assert len(res_ngrams.unwrap()) > 0
