"""FR-5.5 C2 — Word2Vec contract tests (oracle: tests/fixtures/w2v/).

Keeps the stub's frame contracts ((Word, Count, Vector),
(Word, Neighbor, Cosine)) and failure codes, now backed by real Gensim
training. Gensim-backed tests are marked model_integration; guards,
save/load plumbing around a hand-built model, and t-SNE shape run offline.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import pandas as pd
import pytest

from core.analysis import word_embeddings as w2v_mod
from core.conll.schema import Col

FIX = Path(__file__).resolve().parent / "fixtures" / "w2v"
W2V_CONFIG = {"vector_size": 20, "window": 2, "min_count": 1, "seed": 7, "epochs": 20, "sg": 1}


def _sentences() -> list[list[str]]:
    with (FIX / "sentences.csv").open(encoding="utf-8") as fh:
        return [row["Text"].split() for row in csv.DictReader(fh)]


def _frame(sentences: list[list[str]] | None = None) -> pd.DataFrame:
    columns = [
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
    rows = []
    rec = 1
    for sent_id, toks in enumerate(sentences if sentences is not None else _sentences(), start=1):
        for tok in toks:
            rows.append(
                {
                    "ID": rec,
                    "Form": tok,
                    "Lemma": tok,
                    "POS": "NN",
                    "NER": "O",
                    "Head": 0,
                    "DepRel": "root",
                    "Sentence ID": sent_id,
                    "Document ID": 1,
                    "Document": "s.txt",
                    "Deps": "",
                    "Record ID": rec,
                    "Clause Tag": "",
                }
            )
            rec += 1
    return pd.DataFrame(rows, columns=columns)


def _expected_sims() -> list[tuple[str, str, float]]:
    with (FIX / "similarities_expected.csv").open(encoding="utf-8") as fh:
        return [(row["WordA"], row["WordB"], float(row["Cosine"])) for row in csv.DictReader(fh)]


# ---------------------------------------------------------------------------
# Offline: contracts, guards, plumbing
# ---------------------------------------------------------------------------
class TestContracts:
    def test_vectors_shape_and_counts(self) -> None:
        pytest.importorskip("gensim")
        res = w2v_mod.vectors(_frame(), field=Col.LEMMA, **W2V_CONFIG)  # type: ignore[arg-type]
        assert res.ok, res.diagnostics
        frame = res.unwrap()
        assert list(frame.columns) == ["Word", "Count", "Vector"]
        counts = dict(zip(frame["Word"], frame["Count"], strict=True))
        assert counts["cat"] == 4 and counts["engine"] == 4 and counts["trip"] == 1
        dims = {len(str(v).split(",")) for v in frame["Vector"]}
        assert dims == {20}

    def test_vectors_empty_frame(self) -> None:
        res = w2v_mod.vectors(_frame([]), field=Col.LEMMA)
        assert res.ok and len(res.unwrap()) == 0

    def test_vectors_min_count_filters_to_empty(self) -> None:
        res = w2v_mod.vectors(_frame(), field=Col.LEMMA, min_count=99)
        assert res.ok and len(res.unwrap()) == 0

    def test_vectors_bad_field(self) -> None:
        res = w2v_mod.vectors(_frame(), field=Col.POS)  # type: ignore[arg-type]
        assert not res.ok and any(d.code == "W2V_BAD_FIELD" for d in res.diagnostics)

    def test_distances_contract(self) -> None:
        pytest.importorskip("gensim")
        res = w2v_mod.distances(_frame(), "cat", field=Col.LEMMA, top_n=3, **W2V_CONFIG)  # type: ignore[arg-type]
        assert res.ok, res.diagnostics
        frame = res.unwrap()
        assert list(frame.columns) == ["Word", "Neighbor", "Cosine"]
        assert len(frame) == 3
        assert set(frame["Word"]) == {"cat"}
        assert "cat" not in set(frame["Neighbor"])
        cosines = frame["Cosine"].tolist()
        assert cosines == sorted(cosines, reverse=True)

    def test_distances_unknown_word(self) -> None:
        pytest.importorskip("gensim")
        res = w2v_mod.distances(_frame(), "zxqfrbl", field=Col.LEMMA, **W2V_CONFIG)  # type: ignore[arg-type]
        assert not res.ok and any(d.code == "W2V_UNKNOWN_WORD" for d in res.diagnostics)

    def test_distances_bad_query(self) -> None:
        res = w2v_mod.distances(_frame(), "   ", field=Col.LEMMA)
        assert not res.ok and any(d.code == "W2V_BAD_QUERY" for d in res.diagnostics)

    def test_missing_gensim_is_actionable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "gensim", None)
        monkeypatch.setitem(sys.modules, "gensim.models", None)
        res = w2v_mod.train(_frame())
        assert res.value is None
        assert any(d.code == "W2V_BACKEND_MISSING" for d in res.diagnostics)

    def test_save_load_roundtrip_without_gensim(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("gensim")
        trained = w2v_mod.train(_frame(), **W2V_CONFIG).unwrap()  # type: ignore[arg-type]
        path = tmp_path / "model.npz"
        assert w2v_mod.save_model(trained, path).ok
        monkeypatch.setitem(sys.modules, "gensim", None)
        monkeypatch.setitem(sys.modules, "gensim.models", None)
        loaded = w2v_mod.load_model(path)
        assert loaded.ok, loaded.diagnostics
        back = loaded.unwrap()
        assert back.words == trained.words and back.counts == trained.counts
        for a, b in (("cat", "kitten"), ("engine", "wheel")):
            assert abs(back.similarity(a, b) - trained.similarity(a, b)) < 1e-9
        with pytest.raises(KeyError):
            back.similarity("cat", "zxqfrbl")

    def test_tsne_shape(self) -> None:
        pytest.importorskip("gensim")
        trained = w2v_mod.train(_frame(), **W2V_CONFIG).unwrap()  # type: ignore[arg-type]
        res = w2v_mod.project_tsne(trained, seed=7)
        assert res.ok, res.diagnostics
        frame = res.unwrap()
        assert list(frame.columns) == ["Word", "X", "Y"]
        assert len(frame) == len(trained.words)
        again = w2v_mod.project_tsne(trained, seed=7).unwrap()
        pd.testing.assert_frame_equal(frame, again)

    def test_tsne_too_few_words(self) -> None:
        pytest.importorskip("gensim")
        trained = w2v_mod.train(_frame([["cat", "kitten"]]), min_count=1).unwrap()
        res = w2v_mod.project_tsne(trained, seed=7)
        assert res.ok and len(res.unwrap()) == 0
        assert any(d.code == "W2V_TSNE_SKIPPED" for d in res.diagnostics)


# ---------------------------------------------------------------------------
# Gensim-backed: recorded reference (oracle: ORACLE_NOTES.md)
# ---------------------------------------------------------------------------
@pytest.mark.model_integration
class TestRecordedReference:
    def test_similarities_match_oracle(self) -> None:
        trained = w2v_mod.train(_frame(), **W2V_CONFIG).unwrap()  # type: ignore[arg-type]
        for word_a, word_b, expected in _expected_sims():
            assert abs(trained.similarity(word_a, word_b) - expected) < 1e-3, (word_a, word_b)

    def test_within_beats_cross_theme(self) -> None:
        trained = w2v_mod.train(_frame(), **W2V_CONFIG).unwrap()  # type: ignore[arg-type]
        assert trained.similarity("cat", "kitten") > trained.similarity("cat", "car")

    def test_determinism(self) -> None:
        frame = _frame()
        first = w2v_mod.train(frame, **W2V_CONFIG).unwrap()  # type: ignore[arg-type]
        second = w2v_mod.train(frame, **W2V_CONFIG).unwrap()  # type: ignore[arg-type]
        assert first.vectors == second.vectors

    def test_self_similarity_is_one(self) -> None:
        trained = w2v_mod.train(_frame(), **W2V_CONFIG).unwrap()  # type: ignore[arg-type]
        assert abs(trained.similarity("cat", "cat") - 1.0) < 1e-6
