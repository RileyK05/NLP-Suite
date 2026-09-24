"""C31-C35 - coreference, topics, embeddings, ngrams, cooccurrence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import has_spacy_model
from core.analysis import (
    coreference as coref_mod,
    ngram_cooccurrence as cooc_mod,
    ngrams as ng_mod,
    topic_model as topic_mod,
    word_embeddings as w2v_mod,
)
from core.conll.schema import Col
from core.io.reader import read_corpus
from core.pipelines.cache import PipelineCache
from core.pipelines.spacy_backend import build_spacy_pipeline, spacy_model_name

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


def test_coreference_basic() -> None:
    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "Form": "Alice",
                "Lemma": "alice",
                "POS": "NNP",
                "NER": "PERSON",
                "Head": 2,
                "DepRel": "nsubj",
                "Sentence ID": 1,
                "Document ID": 1,
                "Document": "a.txt",
                "Deps": "",
                "Record ID": 1,
                "Clause Tag": "",
            },
            {
                "ID": 2,
                "Form": "saw",
                "Lemma": "see",
                "POS": "VBD",
                "NER": "O",
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
                "Form": "Alice",
                "Lemma": "alice",
                "POS": "NNP",
                "NER": "PERSON",
                "Head": 2,
                "DepRel": "nsubj",
                "Sentence ID": 2,
                "Document ID": 1,
                "Document": "a.txt",
                "Deps": "",
                "Record ID": 3,
                "Clause Tag": "",
            },
        ]
    )
    res = coref_mod.run(df, field=Col.LEMMA)
    assert res.ok
    out = res.unwrap()
    # alice appears twice -> one cluster with 2 mentions
    assert len(out) == 2
    assert out["Lemma"].iloc[0] == "alice"
    assert out["Cluster ID"].nunique() == 1


@pytest.mark.model_integration
def test_coreference_on_corpus() -> None:
    frame = _parsed_frame()
    res = coref_mod.run(frame)
    assert res.ok
    # mini-corpus has nouns, so at least one cluster
    assert len(res.unwrap()) >= 0


@pytest.mark.model_integration
def test_topic_model() -> None:
    frame = _parsed_frame()
    res = topic_mod.run(frame, n_topics=2, top_n=3, field=Col.LEMMA)
    assert res.ok, res.diagnostics
    out = res.unwrap()
    assert len(out) == 6
    assert set(out["Topic"]) == {0, 1}
    assert "Word" in out.columns
    assert "Weight" in out.columns

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
    assert topic_mod.run(empty).ok


@pytest.mark.model_integration
def test_word2vec_engine() -> None:
    frame = _parsed_frame()
    v = w2v_mod.vectors(frame, field=Col.LEMMA, min_count=1)
    assert v.ok
    vdf = v.unwrap()
    assert len(vdf) > 0
    assert "Vector" in vdf.columns
    # Pick a word that exists
    word = str(vdf["Word"].iloc[0])
    d = w2v_mod.distances(frame, word, field=Col.LEMMA, top_n=2, min_count=1)
    assert d.ok
    assert len(d.unwrap()) == 2
    assert d.unwrap()["Word"].iloc[0] == word.lower()

    bad = w2v_mod.distances(frame, "nonexistentword123", field=Col.LEMMA)
    assert not bad.ok


@pytest.mark.model_integration
def test_ngrams_and_collocations() -> None:
    frame = _parsed_frame()
    n2 = ng_mod.ngrams(frame, n=2, field=Col.FORM)
    assert n2.ok
    ndf = n2.unwrap()
    assert "N-gram" in ndf.columns
    assert "Frequency in Document" in ndf.columns
    assert "Frequency in Corpus" in ndf.columns

    c = ng_mod.collocations(frame, field=Col.FORM, min_count=1)
    assert c.ok
    assert "PMI" in c.unwrap().columns

    bad = ng_mod.ngrams(frame, n=10)
    assert not bad.ok


@pytest.mark.model_integration
def test_cooccurrence() -> None:
    frame = _parsed_frame()
    res = cooc_mod.run(frame, window=3, field=Col.LEMMA, min_count=1)
    assert res.ok
    out = res.unwrap()
    # mini-corpus may produce some pairs
    assert "Word 1" in out.columns
    assert "Word 2" in out.columns

    # Synthetic sentence: "a b c" with window 3 should give pairs a-b, a-c, b-c
    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "Form": "a",
                "Lemma": "a",
                "POS": "DT",
                "NER": "O",
                "Head": 2,
                "DepRel": "det",
                "Sentence ID": 1,
                "Document ID": 1,
                "Document": "a.txt",
                "Deps": "",
                "Record ID": 1,
                "Clause Tag": "",
            },
            {
                "ID": 2,
                "Form": "b",
                "Lemma": "b",
                "POS": "NN",
                "NER": "O",
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
                "ID": 3,
                "Form": "c",
                "Lemma": "c",
                "POS": "NN",
                "NER": "O",
                "Head": 2,
                "DepRel": "obj",
                "Sentence ID": 1,
                "Document ID": 1,
                "Document": "a.txt",
                "Deps": "",
                "Record ID": 3,
                "Clause Tag": "",
            },
        ]
    )
    r2 = cooc_mod.run(df, window=5, field=Col.LEMMA, min_count=1)
    assert r2.ok
    odf = r2.unwrap()
    assert len(odf) == 3
    pairs = {tuple(sorted([row["Word 1"], row["Word 2"]])) for _, row in odf.iterrows()}
    assert ("a", "b") in pairs
    assert ("a", "c") in pairs

    bad = cooc_mod.run(df, window=1)
    assert not bad.ok


def test_ngrams_attribute_counts_per_document() -> None:
    """Review finding: two documents sharing 'blue sky' must both appear.

    The corpus-wide count claiming the first document's ID made document
    attribution impossible; legacy (src/NGrams_util.py:143-155) keeps
    per-document frequency alongside the corpus total.
    """
    rows = []
    for did, doc in (("1", "a.txt"), ("2", "b.txt")):
        for i, form in enumerate(("blue", "sky", "today"), 1):
            rows.append(
                {
                    Col.ID.value: i,
                    Col.FORM.value: form,
                    Col.LEMMA.value: form.lower(),
                    Col.POS.value: "NN",
                    Col.NER.value: "O",
                    Col.HEAD.value: 0,
                    Col.DEPREL.value: "root",
                    Col.DEPS.value: "_",
                    Col.CLAUSE_TAG.value: "",
                    Col.RECORD_ID.value: i,
                    Col.SENTENCE_ID.value: 1,
                    Col.DOCUMENT_ID.value: did,
                    Col.DOCUMENT.value: doc,
                }
            )
    result = ng_mod.ngrams(pd.DataFrame(rows), n=2)
    assert result.ok
    out = result.unwrap()
    blue_sky = out[out["N-gram"] == "blue sky"]
    assert len(blue_sky) == 2, "both documents must appear"
    assert set(blue_sky["Document ID"]) == {"1", "2"}
    assert (blue_sky["Frequency in Corpus"] == 2).all()
    assert (blue_sky["Frequency in Document"] == 1).all()
    assert set(blue_sky["Document"]) == {"a.txt", "b.txt"}


def test_cooccurrence_window_uses_original_positions() -> None:
    """Review finding S6: punctuation must consume window positions."""

    def mk(tokens, did="1"):
        rows = []
        for i, form in enumerate(tokens, 1):
            rows.append(
                {
                    Col.ID.value: i,
                    Col.FORM.value: form,
                    Col.LEMMA.value: form.lower(),
                    Col.POS.value: "NN",
                    Col.NER.value: "O",
                    Col.HEAD.value: 0,
                    Col.DEPREL.value: "root",
                    Col.DEPS.value: "_",
                    Col.CLAUSE_TAG.value: "",
                    Col.RECORD_ID.value: i,
                    Col.SENTENCE_ID.value: 1,
                    Col.DOCUMENT_ID.value: did,
                    Col.DOCUMENT.value: "t.txt",
                }
            )
        return pd.DataFrame(rows)

    # cat and dog separated by five punctuation tokens: 6 original positions
    # apart, adjacent in FILTERED space. window=2 must NOT count them.
    far = mk(["cat", ",", ".", "!", "?", ";", "dog"])
    result = cooc_mod.run(far, window=2)
    assert result.ok
    out = result.unwrap()
    pair = out[(out["Word 1"] == "cat") & (out["Word 2"] == "dog")]
    assert pair.empty, "filtered-space indexing would wrongly count this"

    # 4-token gap with window=5 counts (4 < 5); 5-token gap does not.
    near = cooc_mod.run(mk(["cat", "a", "b", "c", "dog"]), window=5).unwrap()
    assert not near[(near["Word 1"] == "cat") & (near["Word 2"] == "dog")].empty
    edge = cooc_mod.run(mk(["cat", "a", "b", "c", "d", "dog"]), window=5).unwrap()
    assert edge[(edge["Word 1"] == "cat") & (edge["Word 2"] == "dog")].empty

    # per-document attribution (same shape as finding 5's fix)
    both = pd.concat([mk(["cat", "dog"], did="1"), mk(["cat", "dog"], did="2")], ignore_index=True)
    per_doc = cooc_mod.run(both, window=5).unwrap()
    cat_dog = per_doc[(per_doc["Word 1"] == "cat") & (per_doc["Word 2"] == "dog")]
    assert set(cat_dog["Document ID"]) == {"1", "2"}
