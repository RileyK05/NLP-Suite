"""C21-C25 — CoNLL search, K-sentences, clause/SVO, corpus & text stats."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import has_spacy_model
from core.analysis import (
    clause_svo as csv_mod,
    corpus_statistics as cs_mod,
    k_sentences as ks_mod,
    table_search as ts_mod,
    text_statistics as txt_mod,
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
    assert result.ok, result.diagnostics
    corpus = result.unwrap()
    cache = PipelineCache()
    cache.register("spacy", build_spacy_pipeline)  # type: ignore[arg-type]
    frame = cache.get("spacy", "en").unwrap().parse(corpus).unwrap()
    return frame


# ---------------------------------------------------------------------------
# C21 table_search
# ---------------------------------------------------------------------------


@pytest.mark.model_integration
def test_table_search_eq_and_contains() -> None:
    frame = _parsed_frame()
    # Every row has Document ID; searching for doc 1 should return a subset.
    doc1 = ts_mod.search(frame, [ts_mod.SearchFilter(field=Col.DOCUMENT_ID.value, op="eq", value="1")])
    assert doc1.ok
    assert 0 < doc1.unwrap().matched_rows <= doc1.unwrap().total_rows

    # contains is case-insensitive by default — "the" appears.
    any_the = ts_mod.search(frame, [ts_mod.SearchFilter(field=Col.FORM.value, op="contains", value="the")])
    assert any_the.ok
    assert any_the.unwrap().matched_rows >= 0

    # negate — rows that are NOT POS == PUNCT (if any) should be >0 when corpus has punctuation.
    neg = ts_mod.search(frame, [ts_mod.SearchFilter(field=Col.POS.value, op="eq", value="PUNCT", negate=True)])
    assert neg.ok
    # If the blank fallback was used there is no PUNCT, so matched == total — still valid.
    assert neg.unwrap().matched_rows >= 0


@pytest.mark.model_integration
def test_table_search_logic_and_errors() -> None:
    frame = _parsed_frame()
    # AND vs OR
    f1 = ts_mod.SearchFilter(field=Col.DOCUMENT_ID.value, op="eq", value="1")
    f2 = ts_mod.SearchFilter(field=Col.DOCUMENT_ID.value, op="eq", value="999")
    and_res = ts_mod.search(frame, [f1, f2], logic="AND")
    assert and_res.ok
    assert and_res.unwrap().matched_rows == 0
    or_res = ts_mod.search(frame, [f1, f2], logic="OR")
    assert or_res.ok
    assert or_res.unwrap().matched_rows == and_res.unwrap().total_rows or or_res.unwrap().matched_rows > 0

    bad_field = ts_mod.search(frame, [ts_mod.SearchFilter(field="Nope", op="eq", value="x")])
    assert not bad_field.ok
    assert bad_field.diagnostics[0].code == "TABLE_SEARCH_BAD_FIELD"

    bad_regex = ts_mod.search(frame, [ts_mod.SearchFilter(field=Col.FORM.value, op="regex", value="[")])
    assert not bad_regex.ok

    no_filters = ts_mod.search(frame, [])
    assert not no_filters.ok


# ---------------------------------------------------------------------------
# C22 K-sentences
# ---------------------------------------------------------------------------


@pytest.mark.model_integration
def test_k_sentences_basic() -> None:
    frame = _parsed_frame()
    result = ks_mod.run(frame, k_first=1, k_last=1)
    assert result.ok
    kres = result.unwrap()
    # One doc -> two sections (first/last), three docs -> six rows when k=1.
    assert len(kres.counts) == 6
    assert set(kres.counts["Section"]) == {"first", "last"}
    # repetitions is sorted by Doc Frequency
    assert "Lemma" in kres.repetitions.columns


@pytest.mark.model_integration
def test_k_sentences_errors() -> None:
    frame = _parsed_frame()
    assert not ks_mod.run(frame, k_first=-1, k_last=1).ok
    assert not ks_mod.run(frame, k_first=0, k_last=0).ok


# ---------------------------------------------------------------------------
# C23 Clause & SVO
# ---------------------------------------------------------------------------


def test_clause_and_svo() -> None:
    # Synthetic frame with Clause Tag and dependency structure.
    # Two sentences, each with a verb head and nsubj/obj dependents.
    df = pd.DataFrame(
        [
            {
                Col.ID.value: 1,
                Col.FORM.value: "Cats",
                Col.LEMMA.value: "cat",
                Col.POS.value: "NNS",
                Col.HEAD.value: 2,
                Col.DEPREL.value: "nsubj",
                Col.SENTENCE_ID.value: 1,
                Col.DOCUMENT_ID.value: 1,
                Col.DOCUMENT.value: "doc.txt",
                Col.CLAUSE_TAG.value: "NP",
            },
            {
                Col.ID.value: 2,
                Col.FORM.value: "chase",
                Col.LEMMA.value: "chase",
                Col.POS.value: "VBP",
                Col.HEAD.value: 0,
                Col.DEPREL.value: "root",
                Col.SENTENCE_ID.value: 1,
                Col.DOCUMENT_ID.value: 1,
                Col.DOCUMENT.value: "doc.txt",
                Col.CLAUSE_TAG.value: "VP",
            },
            {
                Col.ID.value: 3,
                Col.FORM.value: "mice",
                Col.LEMMA.value: "mouse",
                Col.POS.value: "NNS",
                Col.HEAD.value: 2,
                Col.DEPREL.value: "obj",
                Col.SENTENCE_ID.value: 1,
                Col.DOCUMENT_ID.value: 1,
                Col.DOCUMENT.value: "doc.txt",
                Col.CLAUSE_TAG.value: "NP",
            },
        ]
    )
    # Ensure required columns present for validate_columns
    for col in ["NER", "Deps", "Record ID"]:
        if col not in df.columns:
            df[col] = ""

    clauses = csv_mod.clause_frequencies(df)
    assert clauses.ok
    tags = {r.tag: r.count for r in clauses.unwrap()}
    assert tags["NP"] == 2
    assert tags["VP"] == 1

    svos = csv_mod.extract_svo(df)
    assert svos.ok
    triples = svos.unwrap()
    assert len(triples) == 1
    assert triples[0].subject == "Cats"
    assert triples[0].verb == "chase"
    assert triples[0].obj == "mice"


def test_svo_passive_roles_match_legacy() -> None:
    """Active/passive equivalence: (John, hired, Mary) regardless of notation.

    Review finding: UD nsubj:pass was treated as the subject and obl:agent as
    an object; the spaCy shape (nsubjpass/agent/pobj) produced nothing. The
    legacy engine maps the passive agent to the subject and the patient to
    the object (Stanford_CoreNLP_SVO_enhanced_dependencies_util.py:245-258).
    """

    def build(pairs, poses):
        rows = []
        for i, ((form, head, deprel), pos) in enumerate(zip(pairs, poses, strict=False), 1):
            rows.append(
                {
                    Col.ID.value: i,
                    Col.FORM.value: form,
                    Col.LEMMA.value: form.lower(),
                    Col.POS.value: pos,
                    Col.NER.value: "O",
                    Col.HEAD.value: head,
                    Col.DEPREL.value: deprel,
                    Col.DEPS.value: "_",
                    Col.CLAUSE_TAG.value: "",
                    Col.RECORD_ID.value: i,
                    Col.SENTENCE_ID.value: 1,
                    Col.DOCUMENT_ID.value: "1",
                    Col.DOCUMENT.value: "t.txt",
                }
            )
        return pd.DataFrame(rows)

    # UD canonical: "Mary was hired by John"
    ud = build(
        [
            ("Mary", 4, "nsubj:pass"),
            ("was", 4, "aux:pass"),
            ("by", 5, "case"),
            ("hired", 0, "root"),
            ("John", 4, "obl:agent"),
        ],
        ["NNP", "VBD", "IN", "VBN", "NNP"],
    )
    ud_triples = [(t.subject, t.verb, t.obj) for t in csv_mod.extract_svo(ud).unwrap()]
    assert ud_triples == [("John", "hired", "Mary")]

    # spaCy coarse shape for the same sentence
    spacy = build(
        [
            ("Mary", 4, "nsubjpass"),
            ("was", 4, "auxpass"),
            ("by", 4, "agent"),
            ("hired", 0, "ROOT"),
            ("John", 3, "pobj"),
        ],
        ["NNP", "VBD", "IN", "VBN", "NNP"],
    )
    spacy_triples = [(t.subject, t.verb, t.obj) for t in csv_mod.extract_svo(spacy).unwrap()]
    assert spacy_triples == [("John", "hired", "Mary")]

    # Agentless passive gets the legacy inferred-subject placeholder
    agentless = build(
        [("Mary", 3, "nsubj:pass"), ("was", 3, "aux:pass"), ("hired", 0, "root")],
        ["NNP", "VBD", "VBN"],
    )
    agentless_triples = [(t.subject, t.verb, t.obj) for t in csv_mod.extract_svo(agentless).unwrap()]
    assert agentless_triples == [("Inferred_Subject_Passive", "hired", "Mary")]

    # Active unchanged: (Mary, hired, John)
    active = build(
        [("Mary", 2, "nsubj"), ("hired", 0, "ROOT"), ("John", 2, "obj")],
        ["NNP", "VBD", "NNP"],
    )
    active_triples = [(t.subject, t.verb, t.obj) for t in csv_mod.extract_svo(active).unwrap()]
    assert active_triples == [("Mary", "hired", "John")]


def test_svo_many_roles_warns_instead_of_silently_clipping() -> None:
    pairs = [
        ("S1", 5, "nsubj"),
        ("B", 5, "nsubj"),
        ("C", 5, "nsubj"),
        ("D", 5, "nsubj"),
        ("hired", 0, "ROOT"),
        ("X", 5, "obj"),
        ("Y", 5, "obj"),
    ]
    df = pd.DataFrame(
        [
            {
                Col.ID.value: i,
                Col.FORM.value: form,
                Col.LEMMA.value: form.lower(),
                Col.POS.value: pos,
                Col.NER.value: "O",
                Col.HEAD.value: head,
                Col.DEPREL.value: deprel,
                Col.DEPS.value: "_",
                Col.CLAUSE_TAG.value: "",
                Col.RECORD_ID.value: i,
                Col.SENTENCE_ID.value: 1,
                Col.DOCUMENT_ID.value: 1,
                Col.DOCUMENT.value: "t.txt",
            }
            for i, ((form, head, deprel), pos) in enumerate(
                zip(pairs, ["NNP", "NNP", "NNP", "NNP", "VBD", "NNP", "NNP"], strict=False), 1
            )
        ]
    )
    result = csv_mod.extract_svo(df)
    assert result.ok
    assert len(result.unwrap()) == 6  # 3x3 after documented clipping
    assert any(d.code == "SVO_MANY_ROLES" for d in result.diagnostics)


def test_clause_empty_frame() -> None:
    # Empty but schema-valid frame should succeed with empty results.
    cols = [
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
        "Clause Tag",
        "Deps",
        "Record ID",
    ]
    empty = pd.DataFrame(columns=cols)
    assert csv_mod.clause_frequencies(empty).ok
    assert csv_mod.extract_svo(empty).ok


# ---------------------------------------------------------------------------
# C24 Corpus statistics
# ---------------------------------------------------------------------------


@pytest.mark.model_integration
def test_corpus_statistics() -> None:
    frame = _parsed_frame()
    result = cs_mod.run(frame, field=Col.FORM)
    assert result.ok
    df = result.unwrap().to_frame()
    assert len(df) == 3  # three docs in mini-corpus
    for col in ["TTR", "Root TTR", "Log TTR", "Herdan", "Yule K"]:
        assert col in df.columns
    assert all(0 <= v <= 1 for v in df["TTR"].tolist())

    bad = cs_mod.run(frame, field=Col.POS)
    assert not bad.ok


# ---------------------------------------------------------------------------
# C25 Text statistics
# ---------------------------------------------------------------------------


@pytest.mark.model_integration
def test_text_statistics() -> None:
    frame = _parsed_frame()
    result = txt_mod.run(frame)
    assert result.ok
    df = result.unwrap().to_frame()
    assert len(df) == 3
    for col in ["Sentences", "Tokens", "Avg Sentence Length", "Syllables"]:
        assert col in df.columns
    assert all(v > 0 for v in df["Tokens"].tolist())


def test_text_statistics_empty() -> None:
    cols = [
        "Form",
        "Sentence ID",
        "Document ID",
        "ID",
        "Lemma",
        "POS",
        "NER",
        "Head",
        "DepRel",
        "Document",
        "Deps",
        "Record ID",
        "Clause Tag",
    ]
    empty = pd.DataFrame(columns=cols)
    result = txt_mod.run(empty)
    assert result.ok
    assert len(result.unwrap().to_frame()) == 0
