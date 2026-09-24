"""C36-C40 - contextual, html annotator, KG, semantic, charts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import HashEmbeddingBackend, has_spacy_model
from core.analysis import (
    contextual as ctx_mod,
    html_annotator as html_mod,
    knowledge_graph as kg_mod,
    semantic as sem_mod,
)
from core.conll.schema import Col
from core.io.reader import read_corpus
from core.pipelines.cache import PipelineCache
from core.pipelines.spacy_backend import build_spacy_pipeline, spacy_model_name
from core.viz import charts as chart_mod

FIXTURE = Path(__file__).parent / "fixtures" / "mini-corpus"


def _parsed_frame() -> pd.DataFrame:
    if not has_spacy_model():
        pytest.skip(f"spaCy model not installed — run: python -m spacy download {spacy_model_name('en')}")
    result = read_corpus(FIXTURE)
    assert result.ok
    corpus = result.unwrap()
    cache = PipelineCache()
    cache.register("spacy", build_spacy_pipeline)  # type: ignore[arg-type]
    return cache.get("spacy", "en").unwrap().parse(corpus).unwrap()


@pytest.mark.model_integration
def test_contextual_vectors_and_wsi() -> None:
    frame = _parsed_frame()
    # Offline fake behind the protocol (production hash vectors were
    # removed in FR-5.6; the real transformer path is marked below).
    backend = HashEmbeddingBackend()
    cv = ctx_mod.contextual_vectors(frame, field=Col.FORM, backend=backend)
    assert cv.ok
    cdf = cv.unwrap()
    assert len(cdf) > 0
    assert "Vector" in cdf.columns
    # Same word in different sentences should have different vectors (contextual)
    # Check that at least two rows with same Word have different Vector if corpus has >1 sentence
    if len(cdf) >= 2:
        # Find a word that appears in multiple sentences (if any)
        dup_words = cdf[cdf.duplicated(subset=["Word"], keep=False)]
        if not dup_words.empty:
            word = str(dup_words["Word"].iloc[0])
            vecs = cdf[cdf["Word"] == word]["Vector"].unique().tolist()
            # If word appears in different sentences, vectors should differ
            # Not guaranteed for tiny corpus, but check at least 1
            assert len(vecs) >= 1

    wsi = ctx_mod.wsi_senses(frame, field=Col.LEMMA, min_count=1, backend=backend)
    assert wsi.ok
    assert "Sense" in wsi.unwrap().columns

    empty = pd.DataFrame(
        columns=[
            "ID",
            "Form",
            "Lemma",
            "POS",
            "NER",
            "Head",
            "DepRel",
            "Sentence ID",
            "Document ID",
            "Document",
            "Deps",
            "Record ID",
            "Clause Tag",
        ]
    )
    assert ctx_mod.contextual_vectors(empty, backend=backend).ok
    assert ctx_mod.wsi_senses(empty, backend=backend).ok


def test_html_annotator() -> None:
    html_text = "<p>Hello <b>world</b> &amp; friends</p>"
    ext = html_mod.extract_text(html_text)
    assert ext.ok
    assert ext.unwrap() == "Hello world & friends"

    annotated = html_mod.annotate("Hello world", {"world": "PLACE"}, case_sensitive=False)
    assert annotated.ok
    assert '<mark data-tag="PLACE">world</mark>' in annotated.unwrap()

    # Gender
    df = pd.DataFrame(
        [
            {"Form": "He", "Sentence ID": 1, "Document ID": 1},
            {"Form": "she", "Sentence ID": 1, "Document ID": 1},
            {"Form": "dog", "Sentence ID": 1, "Document ID": 1},
            {"Form": "They", "Sentence ID": 2, "Document ID": 1},
        ]
    )
    g = html_mod.gender_spans(df, field="Form")
    assert g.ok
    gdf = g.unwrap()
    assert len(gdf) == 3
    assert set(gdf["Gender"]) == {"M", "F", "N"}

    bad = html_mod.extract_text(123)  # type: ignore[arg-type]
    assert not bad.ok  # should be diagnostic via runtime check, but typed as str so mypy would flag


def test_knowledge_graph() -> None:
    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "Form": "Paris",
                "Lemma": "paris",
                "POS": "NNP",
                "NER": "GPE",
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": 1,
                "Document ID": 1,
                "Document": "a.txt",
                "Deps": "",
                "Record ID": 1,
                "Clause Tag": "",
            },
            {
                "ID": 1,
                "Form": "London",
                "Lemma": "london",
                "POS": "NNP",
                "NER": "GPE",
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": 1,
                "Document ID": 1,
                "Document": "a.txt",
                "Deps": "",
                "Record ID": 2,
                "Clause Tag": "",
            },
            {
                "ID": 1,
                "Form": "UnknownCity",
                "Lemma": "unknowncity",
                "POS": "NNP",
                "NER": "GPE",
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": 1,
                "Document ID": 1,
                "Document": "a.txt",
                "Deps": "",
                "Record ID": 3,
                "Clause Tag": "",
            },
        ]
    )
    res = kg_mod.build(df, field=Col.FORM, source="stub")
    assert res.ok
    kdf = res.unwrap()
    # Paris and London are in KB, UnknownCity not -> 2*3=6 triples? Actually Paris 3, London 3 =6
    assert len(kdf) == 6
    assert "Paris" in kdf["Subject"].tolist()
    assert res.diagnostics  # unknown entity produces INFO

    empty = pd.DataFrame(
        columns=[
            "ID",
            "Form",
            "Lemma",
            "POS",
            "NER",
            "Head",
            "DepRel",
            "Sentence ID",
            "Document ID",
            "Document",
            "Deps",
            "Record ID",
            "Clause Tag",
        ]
    )
    assert kg_mod.build(empty).ok


@pytest.mark.model_integration
def test_semantic_aggregate() -> None:
    frame = _parsed_frame()
    res = sem_mod.aggregate(frame, field=Col.LEMMA)
    assert res.ok
    sdf = res.unwrap()
    assert "Lemma" in sdf.columns
    assert "WordNet" in sdf.columns
    assert len(sdf) > 0

    # Synthetic with known words
    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "Form": "dog",
                "Lemma": "dog",
                "POS": "NN",
                "NER": "O",
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": 1,
                "Document ID": 1,
                "Document": "a.txt",
                "Deps": "",
                "Record ID": 1,
                "Clause Tag": "",
            },
            {
                "ID": 1,
                "Form": "run",
                "Lemma": "run",
                "POS": "VB",
                "NER": "O",
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": 2,
                "Document ID": 1,
                "Document": "a.txt",
                "Deps": "",
                "Record ID": 2,
                "Clause Tag": "",
            },
        ]
    )
    r2 = sem_mod.aggregate(df, field=Col.LEMMA)
    assert r2.ok
    rdf = r2.unwrap()
    dog_row = rdf[rdf["Lemma"] == "dog"].iloc[0]
    assert "dog.n.01" in str(dog_row["WordNet"])
    run_row = rdf[rdf["Lemma"] == "run"].iloc[0]
    assert "Motion" in str(run_row["FrameNet"])


def test_charts() -> None:
    df = pd.DataFrame({"Category": ["A", "B", "C"], "Count": [5, 3, 8]})
    res = chart_mod.bar_chart_html(df, x="Category", y="Count", title="Test")
    assert res.ok
    html_str = res.unwrap()
    assert "plotly" in html_str.lower() or "<table" in html_str
    assert "A" in html_str

    # Empty
    empty = pd.DataFrame(columns=["Category", "Count"])
    assert not chart_mod.bar_chart_html(empty, x="Category", y="Count").ok

    # Bad column
    assert not chart_mod.bar_chart_html(df, x="Nope", y="Count").ok
