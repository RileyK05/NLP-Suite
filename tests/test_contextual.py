"""FR-5.6 C2 — contextual embedding contract tests.

Offline: the ``HashEmbeddingBackend`` test double (context-sensitive,
deterministic, protocol-conformant) drives vector shape, WSI splits,
and failure contracts. The real transformer path is marked: a bogus
model name must fail with the documented code, never a traceback
(``HF_HUB_OFFLINE=1`` forces the fast offline failure).
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import HashEmbeddingBackend
from core.analysis import contextual as ctx_mod
from core.conll.schema import Col


def _frame() -> pd.DataFrame:
    # (form, pos, sent_id, doc_id, doc): "bank" in 3 sentences
    tokens = [
        ("the", "DET", 1, 1, "d1.txt"),
        ("bank", "NOUN", 1, 1, "d1.txt"),
        ("closed", "VERB", 1, 1, "d1.txt"),
        ("by", "ADP", 2, 1, "d1.txt"),
        ("the", "DET", 2, 1, "d1.txt"),
        ("bank", "NOUN", 2, 1, "d1.txt"),
        ("sat", "VERB", 2, 1, "d1.txt"),
        ("the", "DET", 3, 1, "d1.txt"),
        ("bank", "NOUN", 3, 1, "d1.txt"),
        ("opened", "VERB", 3, 1, "d1.txt"),
    ]
    return pd.DataFrame(
        [
            {
                "ID": rec,
                "Form": form,
                "Lemma": form.lower(),
                "POS": pos,
                "NER": "O",
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": sent,
                "Document ID": doc_id,
                "Document": doc,
            }
            for rec, (form, pos, sent, doc_id, doc) in enumerate(tokens, start=1)
        ]
    )


class TestVectors:
    def test_shape_and_columns(self) -> None:
        res = ctx_mod.contextual_vectors(_frame(), field=Col.FORM, backend=HashEmbeddingBackend())
        assert res.ok, res.diagnostics
        frame = res.unwrap()
        assert list(frame.columns) == ["Word", "Lemma", "Sentence ID", "Document ID", "Vector"]
        assert len(frame) == 10
        assert all(len(str(v).split(",")) == 8 for v in frame["Vector"])

    def test_same_word_different_sentences_differ(self) -> None:
        frame = ctx_mod.contextual_vectors(_frame(), field=Col.LEMMA, backend=HashEmbeddingBackend()).unwrap()
        vecs = frame[frame["Lemma"] == "bank"]["Vector"].unique().tolist()
        assert len(vecs) == 3

    def test_deterministic(self) -> None:
        backend = HashEmbeddingBackend()
        first = ctx_mod.contextual_vectors(_frame(), backend=backend).unwrap()["Vector"].tolist()
        second = ctx_mod.contextual_vectors(_frame(), backend=backend).unwrap()["Vector"].tolist()
        assert first == second

    def test_empty_frame(self) -> None:
        empty = pd.DataFrame(
            columns=["ID", "Form", "Lemma", "POS", "NER", "Head", "DepRel", "Sentence ID", "Document ID", "Document"]
        )
        res = ctx_mod.contextual_vectors(empty, backend=HashEmbeddingBackend())
        assert res.ok and res.unwrap().empty

    def test_bad_field(self) -> None:
        res = ctx_mod.contextual_vectors(_frame(), field=Col.NER, backend=HashEmbeddingBackend())
        assert not res.ok
        assert res.diagnostics[0].code == "CTX_BAD_FIELD"

    def test_missing_column(self) -> None:
        res = ctx_mod.contextual_vectors(_frame().drop(columns=["Form"]), backend=HashEmbeddingBackend())
        assert not res.ok
        assert res.diagnostics[0].code == "CONLL_MISSING_COLUMN"


class TestWSI:
    def test_splits_two_sentence_lemma(self) -> None:
        res = ctx_mod.wsi_senses(_frame(), field=Col.LEMMA, min_count=2, backend=HashEmbeddingBackend())
        assert res.ok, res.diagnostics
        frame = res.unwrap()
        assert set(frame.columns) == {"Lemma", "Sense", "Sentence ID", "Document ID"}
        assert set(frame["Sense"].unique()) <= {1, 2}

    def test_min_count_filters_rare_lemmas(self) -> None:
        res = ctx_mod.wsi_senses(_frame(), field=Col.LEMMA, min_count=99, backend=HashEmbeddingBackend())
        assert res.ok, res.diagnostics
        assert res.unwrap().empty

    def test_empty_frame(self) -> None:
        empty = pd.DataFrame(
            columns=["ID", "Form", "Lemma", "POS", "NER", "Head", "DepRel", "Sentence ID", "Document ID", "Document"]
        )
        res = ctx_mod.wsi_senses(empty, backend=HashEmbeddingBackend())
        assert res.ok and res.unwrap().empty


class TestBackendFailures:
    def test_blocked_package_fails_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "transformers", None)
        res = ctx_mod.default_backend()
        assert not res.ok
        assert res.diagnostics[0].code == "CTX_BACKEND_MISSING"
        assert "embeddings" in res.diagnostics[0].context.get("fix", "")

    def test_bogus_model_fails_with_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("transformers")
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        backend = ctx_mod.TransformerBackend("no-such-model-xyzzy-123")
        with pytest.raises(RuntimeError, match="cannot load embedding model"):
            backend.embed(["bank"], ["the bank closed"])

    def test_resolver_names_default_model(self) -> None:
        pytest.importorskip("transformers")
        resolved = ctx_mod.default_backend()
        assert resolved.ok, resolved.diagnostics
        assert resolved.unwrap().model_name == "bert-base-uncased"
