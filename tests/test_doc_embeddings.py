"""doc_embeddings: vectors, pairs in doc_similarity's shape, neighbours, map, search.

Runs through the real ONNX runtime on the tiny fixture models (the vectors
mean nothing; the shapes, contracts and plumbing are what is checked).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from core.analysis.doc_embeddings import embed_corpus, semantic_search

pytest.importorskip("onnxruntime")

TEXTS = {
    "1934_roosevelt.txt": ["the bank raised rates", "the nation faces hard times", "we will work"],
    "1962_kennedy.txt": ["we sat on the river bank", "the children played"],
    "1990_bush.txt": ["the economy grew", "taxes and trade and jobs", "the budget"],
    "2010_obama.txt": ["health care for every family", "schools must teach"],
    "2020_trump.txt": ["the economy is strong", "jobs and more jobs"],
}


def frame() -> pd.DataFrame:
    rows = []
    for doc_id, (name, sentences) in enumerate(TEXTS.items(), 1):
        for sent_id, sentence in enumerate(sentences, 1):
            for index, form in enumerate(sentence.split(), 1):
                rows.append(
                    {
                        "ID": index,
                        "Form": form,
                        "Lemma": form,
                        "POS": "NN",
                        "NER": "O",
                        "Head": 0,
                        "DepRel": "dep",
                        "Sentence ID": sent_id,
                        "Document ID": doc_id,
                        "Document": name,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def granite(tiny_models: Any) -> Any:
    return tiny_models("granite-embedding-english-r2")


class TestEmbedCorpus:
    def test_documents_get_vectors_pairs_neighbours_and_a_map(self, granite: Any) -> None:
        result = embed_corpus(frame(), top_n=2)
        assert result.ok, result.diagnostics
        tables = result.unwrap()
        assert len(tables.vectors) == 5
        width = len(tables.vectors["Vector"].iloc[0].split(","))
        assert width == 32
        vector = np.array([float(v) for v in tables.vectors["Vector"].iloc[0].split(",")])
        assert abs(np.linalg.norm(vector) - 1) < 1e-3
        # Every unordered pair once, in doc_similarity's columns.
        assert len(tables.pairs) == 10
        assert list(tables.pairs.columns)[:6] == [
            "Document ID A",
            "Document A",
            "Document ID B",
            "Document B",
            "Similarity",
            "Band",
        ]
        # Relative to the corpus: centred, so pairs fall on both sides of 0.
        assert tables.pairs["Similarity"].min() < 0 < tables.pairs["Similarity"].max()
        assert tables.pairs["Cosine"].between(-1, 1).all()
        assert tables.pairs["Similarity"].is_monotonic_decreasing
        assert (tables.neighbours.groupby("Document")["Rank"].max() == 2).all()
        assert set(tables.map.columns) >= {"Document", "X", "Y", "Cluster"}
        assert tables.map["Cluster"].str.startswith("Cluster ").all()
        assert any(d.code == "DOC_EMBED_CLUSTERS" for d in result.diagnostics)

    def test_the_pairs_draw_on_document_similaritys_heatmap(self, granite: Any) -> None:
        from core.viz.panels import prepare_panel

        tables = embed_corpus(frame()).unwrap()
        built = prepare_panel("doc_embeddings_heatmap", tables.pairs)
        assert built.value is not None, built.diagnostics
        prepared = built.unwrap()
        assert prepared.title == "Documents by meaning"
        assert prepared.value_label == "Similarity, relative to the corpus (%)"
        assert prepared.color_scale == "diverging"

    def test_sentences_get_vectors_and_a_map_but_no_pair_explosion(self, granite: Any) -> None:
        tables = embed_corpus(frame(), unit="sentence").unwrap()
        assert len(tables.vectors) == sum(len(s) for s in TEXTS.values())
        assert tables.pairs.empty and tables.neighbours.empty
        assert len(tables.map) == len(tables.vectors)

    def test_bert_works_as_a_sentence_reader_too(self, tiny_models: Any) -> None:
        tiny_models("bert-base-uncased")
        result = embed_corpus(frame(), model="bert-base-uncased")
        assert result.ok, result.diagnostics

    def test_a_missing_model_names_the_models_page(self) -> None:
        result = embed_corpus(frame(), model="qwen3-embedding-0.6b")
        assert not result.ok
        assert "Open Models" in result.diagnostics[0].message

    @pytest.mark.parametrize(
        ("unit", "top_n", "code"), [("chapter", 5, "DOC_EMBED_BAD_UNIT"), ("document", 0, "DOC_EMBED_BAD_K")]
    )
    def test_bad_parameters(self, unit: str, top_n: int, code: str) -> None:
        assert embed_corpus(frame(), unit=unit, top_n=top_n).diagnostics[0].code == code


class TestSearch:
    def test_ranks_every_sentence_by_meaning(self, tiny_models: Any) -> None:
        tiny_models("qwen3-embedding-0.6b")
        found = semantic_search(frame(), "interest rates", model="qwen3-embedding-0.6b", top_n=4)
        assert found.ok, found.diagnostics
        table = found.unwrap()
        assert list(table["Rank"]) == [1, 2, 3, 4]
        assert table["Cosine"].is_monotonic_decreasing
        assert set(table.columns) >= {"Document", "Sentence", "Cosine"}

    def test_an_empty_query_is_refused(self, granite: Any) -> None:
        assert semantic_search(frame(), "  ").diagnostics[0].code == "DOC_EMBED_BAD_QUERY"


class TestAdapter:
    def test_the_desktop_run_writes_every_table(self, granite: Any) -> None:
        from core.profiler.executor import ADAPTERS, BatchContext

        ctx = BatchContext(corpus=None, table=frame())
        result = ADAPTERS["doc_embeddings"](
            ctx, {"model": "granite-embedding-english-r2", "unit": "document", "top-n": 3, "query": "jobs", "seed": 1}
        )
        assert result.ok, result.diagnostics
        assert set(result.unwrap()) == {
            "doc_vectors.csv",
            "doc_pairs.csv",
            "doc_neighbours.csv",
            "doc_map.csv",
            "search_results.csv",
        }
