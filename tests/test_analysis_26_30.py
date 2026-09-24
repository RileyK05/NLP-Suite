"""C26-C30 - CSV stats, sentiment, NER, SVO compare."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import has_spacy_model
from core.analysis import (
    csv_stats as csv_mod,
    ner as ner_mod,
    sentiment_swn_hedono as swn_mod,
    sentiment_vader_anew as va_mod,
    svo_compare as svc_mod,
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


def test_csv_stats_describe_and_correlation() -> None:
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [2, 4, 6, 8], "Group": ["x", "x", "y", "y"]})
    res = csv_mod.describe(df)
    assert res.ok
    out = res.unwrap().to_frame()
    assert len(out) == 2
    assert set(out["Column"]) == {"A", "B"}

    # Measurement values, not the key/case regression (Count/Mean all-NaN).
    first = out.iloc[0]
    assert first["Count"] == 4
    assert first["Mean"] == 2.5
    assert abs(first["Std"] - 1.290994) < 1e-5
    assert first["Min"] == 1.0 and first["Max"] == 4.0
    assert first["Median"] == 2.5

    grp = csv_mod.describe(df, group_by="Group")
    assert grp.ok
    grouped = grp.unwrap().to_frame()
    assert len(grouped) == 4
    x_a = grouped[(grouped["Group"] == "x") & (grouped["Column"] == "A")].iloc[0]
    assert x_a["Count"] == 2 and x_a["Mean"] == 1.5 and x_a["Max"] == 2.0

    corr = csv_mod.correlation(df, "A", "B")
    assert corr.ok
    assert abs(corr.unwrap() - 1.0) < 1e-6

    bad_grp = csv_mod.describe(df, group_by="Nope")
    assert not bad_grp.ok

    none_num = csv_mod.describe(pd.DataFrame({"X": ["a", "b"]}))
    assert not none_num.ok


def test_csv_stats_correlation_matrix_is_square_and_named() -> None:
    """The correlation heatmap reads ``correlation_matrix.csv``: a square with
    ``Column`` naming the row axis, so a panel can index both axes."""
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [2, 4, 5, 4, 5],
            "C": [5, 3, 2, -1, -4],
            "note": ["x"] * 5,
        }
    )
    res = csv_mod.correlation_matrix(df)
    assert res.ok, res.diagnostics
    matrix = res.unwrap()
    assert list(matrix["Column"]) == ["A", "B", "C"]
    assert list(matrix.columns) == ["Column", "A", "B", "C"]
    # Perfectly known values, checked by hand: corr(A, B) and corr(A, C).
    ab = float(matrix[matrix["Column"] == "A"]["B"].iloc[0])
    ac = float(matrix[matrix["Column"] == "A"]["C"].iloc[0])
    assert abs(ab - 0.774597) < 1e-5
    assert abs(ac - (-0.98387)) < 1e-5
    # The diagonal is 1 by construction and still written, so the square can
    # be checked by eye.
    assert all(float(matrix[matrix["Column"] == c][c].iloc[0]) == 1.0 for c in ("A", "B", "C"))

    # Rank method answers with the same shape.
    ranked = csv_mod.correlation_matrix(df, method="spearman")
    assert ranked.ok
    assert list(ranked.unwrap().columns) == ["Column", "A", "B", "C"]

    # Refusals name their cause.
    constant = csv_mod.correlation_matrix(pd.DataFrame({"A": [1, 1, 1], "B": [1, 2, 3]}))
    assert not constant.ok
    assert any(d.code == "CSV_CORR_CONSTANT" for d in constant.diagnostics)
    too_few = csv_mod.correlation_matrix(pd.DataFrame({"A": [1, 2], "note": ["x", "y"]}))
    assert not too_few.ok
    assert any(d.code == "CSV_CORR_TOO_FEW_COLUMNS" for d in too_few.diagnostics)
    bad_method = csv_mod.correlation_matrix(df, method="cosine")
    assert not bad_method.ok
    assert any(d.code == "CSV_CORR_BAD_METHOD" for d in bad_method.diagnostics)
    missing = csv_mod.correlation_matrix(df, columns=["A", "Nope"])
    assert not missing.ok
    assert any(d.code == "CSV_CORR_BAD_COLUMN" for d in missing.diagnostics)


@pytest.mark.model_integration
def test_csv_stats_empty() -> None:
    empty = pd.DataFrame(columns=["A", "B"])
    res = csv_mod.describe(empty)
    assert res.ok
    assert len(res.unwrap().to_frame()) == 0


@pytest.mark.model_integration
def test_vader_anew_on_mini_corpus() -> None:
    pytest.importorskip("vaderSentiment")
    frame = _parsed_frame()
    v = va_mod.vader(frame, field=Col.FORM)
    assert v.ok, v.diagnostics
    vdf = v.unwrap()
    assert len(vdf) == 3
    assert set(vdf.columns) == {"Document ID", "Document", "Sentences", "Neg", "Neu", "Pos", "Compound"}

    lex = va_mod.load_anew_lexicon(Path(__file__).parent / "fixtures" / "sentiment" / "anew_mini.csv")
    a = va_mod.anew(frame, field=Col.LEMMA, lexicon=lex)
    assert a.ok, a.diagnostics
    adf = a.unwrap()
    assert len(adf) == 3
    assert "Valence" in adf.columns

    empty = pd.DataFrame(
        columns=[
            "Form",
            "Lemma",
            "POS",
            "NER",
            "Head",
            "DepRel",
            "Sentence ID",
            "Document ID",
            "Document",
            "ID",
            "Deps",
            "Record ID",
            "Clause Tag",
        ]
    )
    assert va_mod.vader(empty).ok
    assert va_mod.anew(empty).ok


def test_vader_negation() -> None:
    good = pd.DataFrame(
        [
            {
                "ID": 1,
                "Form": "good",
                "Lemma": "good",
                "POS": "JJ",
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
        ]
    )
    not_good = pd.DataFrame(
        [
            {
                "ID": 1,
                "Form": "not",
                "Lemma": "not",
                "POS": "RB",
                "NER": "O",
                "Head": 2,
                "DepRel": "advmod",
                "Sentence ID": 1,
                "Document ID": 1,
                "Document": "a.txt",
                "Deps": "",
                "Record ID": 1,
                "Clause Tag": "",
            },
            {
                "ID": 2,
                "Form": "good",
                "Lemma": "good",
                "POS": "JJ",
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
        ]
    )
    pytest.importorskip("vaderSentiment")  # package scorer required
    gv = va_mod.vader(good).unwrap()
    ngv = va_mod.vader(not_good).unwrap()
    assert gv["Compound"].iloc[0] > 0
    assert ngv["Compound"].iloc[0] < 0


@pytest.mark.model_integration
def test_swn_hedono() -> None:
    frame = _parsed_frame()

    def fake_resolver(lemma: str, pos: str) -> tuple[str, float, float] | None:
        table = {("happy", "a"): ("happy.a.01", 0.875, 0.0), ("sad", "a"): ("sad.a.01", 0.0, 0.625)}
        return table.get((lemma, pos))

    s = swn_mod.sentiwordnet(frame, resolver=fake_resolver)
    assert s.ok, s.diagnostics
    assert len(s.unwrap()) == 3
    hedo_lex = swn_mod.load_hedonometer_lexicon(
        Path(__file__).parent / "fixtures" / "sentiment" / "hedonometer_mini.json"
    )
    h = swn_mod.hedonometer(frame, lexicon=hedo_lex)
    assert h.ok, h.diagnostics
    assert len(h.unwrap()) == 3

    def one(word: str, pos: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ID": 1,
                    "Form": word,
                    "Lemma": word,
                    "POS": pos,
                    "NER": "O",
                    "Head": 0,
                    "DepRel": "root",
                    "Sentence ID": 1,
                    "Document ID": 1,
                    "Document": "a.txt",
                    "Deps": "",
                    "Record ID": 1,
                    "Clause Tag": "",
                }
            ]
        )

    pos = one("happy", "ADJ")
    neg = one("sad", "ADJ")
    ps = swn_mod.sentiwordnet(pos, resolver=fake_resolver).unwrap()
    ns = swn_mod.sentiwordnet(neg, resolver=fake_resolver).unwrap()
    assert ps["Pos"].iloc[0] > ns["Pos"].iloc[0]
    assert ps["Net"].iloc[0] > ns["Net"].iloc[0]

    ph = swn_mod.hedonometer(pos, lexicon=hedo_lex).unwrap()
    nh = swn_mod.hedonometer(neg, lexicon=hedo_lex).unwrap()
    assert ph["Happiness"].iloc[0] > nh["Happiness"].iloc[0]


def test_ner_timeline_and_location() -> None:
    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "Form": "Barack",
                "Lemma": "Barack",
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
                "Form": "Obama",
                "Lemma": "Obama",
                "POS": "NNP",
                "NER": "PERSON",
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
                "Form": "Paris",
                "Lemma": "Paris",
                "POS": "NNP",
                "NER": "GPE",
                "Head": 2,
                "DepRel": "obl",
                "Sentence ID": 1,
                "Document ID": 1,
                "Document": "a.txt",
                "Deps": "",
                "Record ID": 3,
                "Clause Tag": "",
            },
            {
                "ID": 1,
                "Form": "London",
                "Lemma": "London",
                "POS": "NNP",
                "NER": "LOC",
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": 2,
                "Document ID": 1,
                "Document": "a.txt",
                "Deps": "",
                "Record ID": 4,
                "Clause Tag": "",
            },
            {
                "ID": 1,
                "Form": "dog",
                "Lemma": "dog",
                "POS": "NN",
                "NER": "O",
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": 1,
                "Document ID": 2,
                "Document": "b.txt",
                "Deps": "",
                "Record ID": 5,
                "Clause Tag": "",
            },
        ]
    )
    tl = ner_mod.entity_timeline(df)
    assert tl.ok
    tdf = tl.unwrap()
    # "Barack Obama" is ONE mention span (consecutive same-type tokens in one
    # sentence) — the token-level grouping split it into two rows.
    assert len(tdf) == 3
    assert set(tdf["NER Tag"]) == {"PERSON", "GPE", "LOC"}
    barack = tdf[tdf["NER Tag"] == "PERSON"].iloc[0]
    assert barack["Entity"] == "Barack Obama"
    assert barack["Count"] == 1

    loc = ner_mod.location_tracking(df)
    assert loc.ok
    ldf = loc.unwrap()
    assert len(ldf) == 2
    assert set(ldf["NER Tag"]) == {"GPE", "LOC"}


def test_ner_multiword_mentions_are_whole() -> None:
    """Review finding: 'New York' split into New and York entities."""
    rows = []
    record = 0
    for sid in (1, 2):  # "New York" mentioned once per sentence
        for form, ner in (("New", "B-GPE"), ("York", "I-GPE")):
            record += 1
            rows.append(
                {
                    "ID": record,
                    "Form": form,
                    "Lemma": form.lower(),
                    "POS": "NNP",
                    "NER": ner,
                    "Head": 0,
                    "DepRel": "root",
                    "Sentence ID": sid,
                    "Document ID": 1,
                    "Document": "a.txt",
                    "Deps": "",
                    "Record ID": record,
                    "Clause Tag": "",
                }
            )
    tl = ner_mod.entity_timeline(pd.DataFrame(rows))
    assert tl.ok
    tdf = tl.unwrap()
    assert len(tdf) == 1
    assert tdf.iloc[0]["Entity"] == "New York"
    assert tdf.iloc[0]["Count"] == 2  # two mentions, one per sentence

    loc = ner_mod.location_tracking(pd.DataFrame(rows))
    assert loc.ok
    ldf = loc.unwrap()
    assert ldf.iloc[0]["Location"] == "New York"


def test_ner_empty() -> None:
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
        "Deps",
        "Record ID",
        "Clause Tag",
    ]
    empty = pd.DataFrame(columns=cols)
    assert ner_mod.entity_timeline(empty).ok
    assert ner_mod.location_tracking(empty).ok


def test_svo_compare() -> None:
    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "Form": "Cats",
                "Lemma": "cat",
                "POS": "NNS",
                "NER": "O",
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
                "Form": "chase",
                "Lemma": "chase",
                "POS": "VBP",
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
                "Form": "mice",
                "Lemma": "mouse",
                "POS": "NNS",
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
            {
                "ID": 1,
                "Form": "Cats",
                "Lemma": "cat",
                "POS": "NNS",
                "NER": "O",
                "Head": 2,
                "DepRel": "nsubj",
                "Sentence ID": 1,
                "Document ID": 2,
                "Document": "b.txt",
                "Deps": "",
                "Record ID": 4,
                "Clause Tag": "",
            },
            {
                "ID": 2,
                "Form": "chase",
                "Lemma": "chase",
                "POS": "VBP",
                "NER": "O",
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": 1,
                "Document ID": 2,
                "Document": "b.txt",
                "Deps": "",
                "Record ID": 5,
                "Clause Tag": "",
            },
            {
                "ID": 3,
                "Form": "dogs",
                "Lemma": "dog",
                "POS": "NNS",
                "NER": "O",
                "Head": 2,
                "DepRel": "obj",
                "Sentence ID": 1,
                "Document ID": 2,
                "Document": "b.txt",
                "Deps": "",
                "Record ID": 6,
                "Clause Tag": "",
            },
        ]
    )
    res = svc_mod.compare(df)
    assert res.ok
    out = res.unwrap()
    assert len(out) == 1
    assert out["Subject Jaccard"].iloc[0] == 1.0
    assert out["Object Jaccard"].iloc[0] == 0.0
    assert out["Triple Jaccard"].iloc[0] == 0.0


def test_svo_compare_no_svos() -> None:
    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "Form": "hello",
                "Lemma": "hello",
                "POS": "UH",
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
                "Form": "world",
                "Lemma": "world",
                "POS": "NN",
                "NER": "O",
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": 1,
                "Document ID": 2,
                "Document": "b.txt",
                "Deps": "",
                "Record ID": 2,
                "Clause Tag": "",
            },
        ]
    )
    res = svc_mod.compare(df)
    assert res.ok
    assert len(res.unwrap()) == 1
