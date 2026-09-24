"""FR-5.9 — BERT extractive summarization contract tests.

Offline: the hash test double drives ranking structure (top-k,
original order, determinism, single-sentence docs, bad counts).
"""

from __future__ import annotations

import pandas as pd

from conftest import HashEmbeddingBackend
from core.analysis import bert_extract as bert_mod


def _frame() -> pd.DataFrame:
    # doc 1: 4 sentences; doc 2: 1 sentence
    tokens = [
        ("alpha", "NOUN", 1, 1, "d1.txt"),
        ("beta", "NOUN", 2, 1, "d1.txt"),
        ("gamma", "NOUN", 3, 1, "d1.txt"),
        ("delta", "NOUN", 4, 1, "d1.txt"),
        ("only", "ADJ", 1, 2, "d2.txt"),
    ]
    return pd.DataFrame(
        [
            {
                "ID": rec,
                "Form": form,
                "Lemma": form,
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


class TestSummarize:
    def test_top_k_in_original_order(self) -> None:
        res = bert_mod.summarize(_frame(), sentences=2, backend=HashEmbeddingBackend())
        assert res.ok, res.diagnostics
        first = res.unwrap().iloc[0]
        assert first["Summary Sentences"] == 2
        ids = [int(v) for v in str(first["Sentence IDs"]).split(",")]
        assert ids == sorted(ids) and len(ids) == 2
        words = str(first["Summary"]).split()
        assert len(words) == 2 and set(words) <= {"alpha", "beta", "gamma", "delta"}

    def test_single_sentence_doc_kept_whole(self) -> None:
        res = bert_mod.summarize(_frame(), sentences=3, backend=HashEmbeddingBackend())
        assert res.ok, res.diagnostics
        second = res.unwrap().iloc[1]
        assert second["Summary"] == "only"
        assert second["Summary Sentences"] == 1

    def test_more_requested_than_available(self) -> None:
        res = bert_mod.summarize(_frame(), sentences=99, backend=HashEmbeddingBackend())
        assert res.ok, res.diagnostics
        assert res.unwrap().iloc[0]["Summary Sentences"] == 4

    def test_deterministic(self) -> None:
        backend = HashEmbeddingBackend()
        first = bert_mod.summarize(_frame(), backend=backend).unwrap()["Summary"].tolist()
        second = bert_mod.summarize(_frame(), backend=backend).unwrap()["Summary"].tolist()
        assert first == second

    def test_bad_count_rejected(self) -> None:
        for bad in (0, -2, True):
            res = bert_mod.summarize(_frame(), sentences=bad, backend=HashEmbeddingBackend())  # type: ignore[arg-type]
            assert not res.ok
            assert res.diagnostics[0].code == "BERT_BAD_COUNT"

    def test_empty_frame(self) -> None:
        empty = pd.DataFrame(
            columns=["ID", "Form", "Lemma", "POS", "NER", "Head", "DepRel", "Sentence ID", "Document ID", "Document"]
        )
        res = bert_mod.summarize(empty, backend=HashEmbeddingBackend())
        assert res.ok and res.unwrap().empty

    def test_summary_covers_distinct_content(self) -> None:
        # every summary sentence comes from its own document's sentences
        res = bert_mod.summarize(_frame(), sentences=2, backend=HashEmbeddingBackend())
        assert res.ok, res.diagnostics
        frame = res.unwrap()
        assert "only" not in str(frame.iloc[0]["Summary"])
        assert "alpha" not in str(frame.iloc[1]["Summary"])
