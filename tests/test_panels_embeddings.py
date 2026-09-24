"""The nearest-neighbour panels use the published Word2Vec result rows."""

from __future__ import annotations

import pandas as pd
import pytest

from core.viz.panels_embeddings import (
    WORD2VEC_BERT_NEIGHBOURS,
    WORD2VEC_BERT_TSNE,
    WORD2VEC_BERT_VECTOR_QUERY,
    WORD2VEC_GENSIM_NEIGHBOURS,
    WORD2VEC_GENSIM_TSNE,
    WORD2VEC_GENSIM_VECTOR_QUERY,
)
from core.viz.panelspec import Provenance


@pytest.mark.parametrize("definition", [WORD2VEC_GENSIM_NEIGHBOURS, WORD2VEC_BERT_NEIGHBOURS])
def test_marks_keep_ranked_cosine_and_resolve_to_exact_source_rows(definition) -> None:
    # This is the public distances() schema from each implementation:
    # Word / Neighbor / Cosine.
    frame = pd.DataFrame(
        [
            {"Word": "bank", "Neighbor": "shore", "Cosine": 0.91},
            {"Word": "bank", "Neighbor": "river", "Cosine": 0.87},
            {"Word": "bank", "Neighbor": "money", "Cosine": 0.72},
        ]
    )
    result = definition.build(frame, definition.defaults(), Provenance(tool=definition.tool, panel=definition.name))
    prepared = result.unwrap()

    assert [mark.label for mark in prepared.marks] == ["shore", "river", "money"]
    assert [mark.x for mark in prepared.marks] == [0.91, 0.87, 0.72]
    assert "cosine similarity" in prepared.x_label.lower()
    assert "t-SNE" in " ".join(prepared.notes)
    for mark in prepared.marks:
        evidence = mark.evidence
        assert evidence.scope == "rows"
        matching = frame
        for column, value in evidence.filters:
            matching = matching[matching[column].astype(str) == value]
        assert len(matching) == evidence.count == 1
        assert matching.iloc[0]["Neighbor"] == mark.label


def test_top_n_is_applied_after_score_ordering_and_bad_scores_are_reported() -> None:
    frame = pd.DataFrame(
        [
            {"Word": "bank", "Neighbor": "low", "Cosine": 0.2},
            {"Word": "bank", "Neighbor": "bad", "Cosine": "NaN"},
            {"Word": "bank", "Neighbor": "high", "Cosine": 0.9},
        ]
    )
    result = WORD2VEC_GENSIM_NEIGHBOURS.build(
        frame,
        WORD2VEC_GENSIM_NEIGHBOURS.defaults() | {"top-n": 1},
        Provenance(tool="word2vec_gensim", panel=WORD2VEC_GENSIM_NEIGHBOURS.name),
    )
    prepared = result.unwrap()

    assert [mark.label for mark in prepared.marks] == ["high"]
    assert [diagnostic.code for diagnostic in result.diagnostics] == ["PANEL_BAD_NUMERIC"]


@pytest.mark.parametrize("definition", [WORD2VEC_GENSIM_TSNE, WORD2VEC_BERT_TSNE])
def test_tsne_projection_marks_resolve_uniquely_and_labels_are_limited(definition) -> None:
    # Both engine project_tsne implementations publish Word / X / Y.
    frame = pd.DataFrame(
        [
            {"Word": word, "X": x, "Y": y}
            for word, x, y in [
                ("alpha", -4, 0),
                ("bravo", -2, 3),
                ("charlie", 0, -3),
                ("delta", 2, 2),
                ("echo", 4, -1),
                ("foxtrot", 1, 0),
            ]
        ]
    )
    result = definition.build(
        frame, definition.defaults() | {"label-count": 3}, Provenance(tool=definition.tool, panel=definition.name)
    )
    prepared = result.unwrap()

    assert prepared.shape == "scatter_labelled"
    assert len(prepared.marks) == len(frame)
    assert sum(mark.labelled for mark in prepared.marks) == 3
    assert {mark.label for mark in prepared.marks if mark.labelled} == {"alpha", "echo", "charlie"}
    assert prepared.x_label == "t-SNE dimension 1"
    assert "not cosine similarities" in " ".join(prepared.notes)
    assert "original embedding space" in " ".join(prepared.notes)
    for mark in prepared.marks:
        assert mark.evidence.scope == "rows"
        matching = frame
        for column, value in mark.evidence.filters:
            matching = matching[matching[column].astype(str) == value]
        assert len(matching) == mark.evidence.count == 1
        assert matching.iloc[0]["Word"] == mark.label


def test_tsne_projection_drops_bad_coordinates_and_rejects_ambiguous_source_rows() -> None:
    frame = pd.DataFrame(
        [
            {"Word": "valid", "X": 1.0, "Y": 2.0},
            {"Word": "bad", "X": "NaN", "Y": 3.0},
        ]
    )
    result = WORD2VEC_GENSIM_TSNE.build(
        frame,
        WORD2VEC_GENSIM_TSNE.defaults(),
        Provenance(tool="word2vec_gensim", panel=WORD2VEC_GENSIM_TSNE.name),
    )
    assert [mark.label for mark in result.unwrap().marks] == ["valid"]
    assert "PANEL_BAD_NUMERIC" in [diagnostic.code for diagnostic in result.diagnostics]

    duplicate = pd.DataFrame([{"Word": "same", "X": 1.0, "Y": 2.0}] * 2)
    ambiguous = WORD2VEC_BERT_TSNE.build(
        duplicate,
        WORD2VEC_BERT_TSNE.defaults(),
        Provenance(tool="word2vec_bert", panel=WORD2VEC_BERT_TSNE.name),
    )
    assert [diagnostic.code for diagnostic in ambiguous.diagnostics] == ["PANEL_AMBIGUOUS_EVIDENCE"]


@pytest.mark.parametrize("definition", [WORD2VEC_GENSIM_VECTOR_QUERY, WORD2VEC_BERT_VECTOR_QUERY])
def test_saved_vectors_can_be_queried_without_retraining(definition) -> None:
    frame = pd.DataFrame(
        [
            {"Word": "government", "Count": 12, "Vector": "1,0,0"},
            {"Word": "state", "Count": 10, "Vector": "0.9,0.1,0"},
            {"Word": "nation", "Count": 7, "Vector": "0,1,0"},
            {"Word": "opposition", "Count": 4, "Vector": "-1,0,0"},
        ]
    )
    result = definition.build(
        frame,
        definition.defaults() | {"query": "GOVERNMENT", "top-n": 2},
        Provenance(tool=definition.tool, panel=definition.name),
    )
    prepared = result.unwrap()

    assert [mark.label for mark in prepared.marks] == ["state", "nation"]
    assert prepared.marks[0].x == pytest.approx(0.9939, abs=0.0001)
    assert prepared.data.columns.tolist() == ["Word", "Neighbor", "Cosine"]
    for mark in prepared.marks:
        evidence = mark.evidence
        assert evidence.scope == "terms"
        assert evidence.phrase == mark.label
        rows = prepared.data
        for column, value in evidence.filters:
            rows = rows[rows[column].astype(str) == value]
        assert len(rows) == evidence.count == 1
        assert rows.iloc[0]["Neighbor"] == mark.label


def test_saved_vector_query_reports_missing_words_and_bad_rows() -> None:
    frame = pd.DataFrame(
        [
            {"Word": "known", "Count": 4, "Vector": "1,0"},
            {"Word": "other", "Count": 3, "Vector": "0.5,0"},
            {"Word": "broken", "Count": 1, "Vector": "NaN,0"},
        ]
    )
    missing = WORD2VEC_GENSIM_VECTOR_QUERY.build(
        frame,
        WORD2VEC_GENSIM_VECTOR_QUERY.defaults() | {"query": "absent"},
        Provenance(tool="word2vec_gensim", panel=WORD2VEC_GENSIM_VECTOR_QUERY.name),
    )
    assert missing.value is None
    assert missing.diagnostics[0].code == "PANEL_QUERY_NOT_FOUND"

    found = WORD2VEC_GENSIM_VECTOR_QUERY.build(
        frame,
        WORD2VEC_GENSIM_VECTOR_QUERY.defaults() | {"query": "known"},
        Provenance(tool="word2vec_gensim", panel=WORD2VEC_GENSIM_VECTOR_QUERY.name),
    )
    assert found.unwrap().marks[0].label == "other"
    assert [diagnostic.code for diagnostic in found.diagnostics] == ["PANEL_BAD_VECTORS"]


def test_blank_saved_vector_query_starts_with_most_frequent_usable_word() -> None:
    frame = pd.DataFrame(
        [
            {"Word": "government", "Count": 12, "Vector": "1,0"},
            {"Word": "state", "Count": 10, "Vector": "0.9,0.1"},
        ]
    )
    result = WORD2VEC_GENSIM_VECTOR_QUERY.build(
        frame,
        WORD2VEC_GENSIM_VECTOR_QUERY.defaults(),
        Provenance(tool="word2vec_gensim", panel=WORD2VEC_GENSIM_VECTOR_QUERY.name),
    )
    prepared = result.unwrap()
    assert prepared.title == "Words nearest to government"
    assert prepared.provenance.params["query"] == "government"
    assert "most frequent word" in prepared.subtitle


def test_tsne_maps_the_most_frequent_content_words_when_counts_are_present() -> None:
    """Regression: 7,478 words drew one cloud, labelled at its outliers."""
    rows = [{"Word": "the", "X": 0.0, "Y": 0.0, "Count": 10_000}]
    rows += [{"Word": f"w{i}", "X": float(i), "Y": float(-i), "Count": 1000 - i} for i in range(50)]
    result = WORD2VEC_GENSIM_TSNE.build(
        pd.DataFrame(rows),
        WORD2VEC_GENSIM_TSNE.defaults() | {"words": 20, "label-count": 5},
        Provenance(tool="word2vec_gensim", panel=WORD2VEC_GENSIM_TSNE.name),
    )
    prepared = result.unwrap()
    assert "the" not in [mark.label for mark in prepared.marks], "function words are hidden by default"
    assert len(prepared.marks) == 20
    assert [mark.label for mark in prepared.marks if mark.labelled] == ["w0", "w1", "w2", "w3", "w4"]
    assert "PANEL_WORDS_CAPPED" in [d.code for d in result.diagnostics]
