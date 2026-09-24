"""FR-6.2 — character mention + arc contract tests.

Offline: hand-built NER frames (multi-token spans via adjacency,
case-insensitive grouping, non-PERSON exclusion) and a fake VADER
analyzer with hand-computed compounds.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from core.narrative import characters as char_mod


def _frame() -> pd.DataFrame:
    # sent 1: "Harry met Hermione ." / sent 2: "harry waved ."
    tokens = [
        ("Harry", "PERSON", 1, 1, "d1.txt"),
        ("met", "O", 1, 1, "d1.txt"),
        ("Hermione", "PERSON", 1, 1, "d1.txt"),
        ("Granger", "PERSON", 1, 1, "d1.txt"),  # adjacent: one span
        (".", "O", 1, 1, "d1.txt"),
        ("harry", "PERSON", 2, 1, "d1.txt"),
        ("waved", "O", 2, 1, "d1.txt"),
        (".", "O", 2, 1, "d1.txt"),
        ("London", "GPE", 2, 1, "d1.txt"),  # not a person: excluded
    ]
    return pd.DataFrame(
        [
            {
                "ID": rec,
                "Form": form,
                "Lemma": form.lower(),
                "POS": "NOUN",
                "NER": ner,
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": sent,
                "Document ID": doc_id,
                "Document": doc,
            }
            for rec, (form, ner, sent, doc_id, doc) in enumerate(tokens, start=1)
        ]
    )


class FakeAnalyzer:
    lexicon: ClassVar[dict[str, float]] = {"love": 1.0}

    def polarity_scores(self, text: str) -> dict[str, float]:
        compound = 0.5 if "waved" in text else -0.25
        return {"neg": 0.0, "neu": 0.5, "pos": 0.5, "compound": compound}


class TestMentions:
    def test_spans_group_and_count(self) -> None:
        res = char_mod.character_mentions(_frame())
        assert res.ok, res.diagnostics
        frame = res.unwrap()
        assert frame["Character"].tolist() == ["Harry", "Hermione Granger"]
        assert frame["Mentions"].tolist() == [2, 1]  # harry + Harry
        assert frame["First Sentence"].tolist() == [1, 1]
        assert frame["Last Sentence"].tolist() == [2, 1]

    def test_non_person_excluded(self) -> None:
        res = char_mod.character_mentions(_frame())
        assert res.ok, res.diagnostics
        assert "London" not in res.unwrap()["Character"].tolist()

    def test_empty_frame(self) -> None:
        empty = pd.DataFrame(
            columns=["ID", "Form", "Lemma", "POS", "NER", "Head", "DepRel", "Sentence ID", "Document ID", "Document"]
        )
        res = char_mod.character_mentions(empty)
        assert res.ok and res.unwrap().empty

    def test_missing_column(self) -> None:
        res = char_mod.character_mentions(_frame().drop(columns=["NER"]))
        assert not res.ok
        assert res.diagnostics[0].code == "CONLL_MISSING_COLUMN"


class TestArcs:
    def test_arc_scores_mention_sentences(self) -> None:
        res = char_mod.character_arcs(_frame(), analyzer=FakeAnalyzer())
        assert res.ok, res.diagnostics
        frame = res.unwrap()
        harry = frame[frame["Character"] == "Harry"].sort_values("Sentence ID")
        assert harry["Compound"].tolist() == [-0.25, 0.5]
        hermione = frame[frame["Character"] == "Hermione Granger"]
        assert hermione["Compound"].tolist() == [-0.25]

    def test_empty_frame(self) -> None:
        empty = pd.DataFrame(
            columns=["ID", "Form", "Lemma", "POS", "NER", "Head", "DepRel", "Sentence ID", "Document ID", "Document"]
        )
        res = char_mod.character_arcs(empty, analyzer=FakeAnalyzer())
        assert res.ok and res.unwrap().empty
