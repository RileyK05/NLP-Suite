"""FR-2.8 C2 — style contract tests.

Offline: hand-built mini lexicons with hand-computed sentence
statistics, loader validation, stopword/alpha filtering, iconic-word
thresholds, empty frames, missing assets. Expected values below are
hand-computed (sample stdev, rounded to 2 decimals), not copied from
the implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd

from core.analysis import style as style_mod
from core.conll.schema import Col

CONC_LEX = {"apple": 4.50, "run": 2.00, "sad": 1.00, "bad": 2.00, "gloom": 5.00}
ICON_LEX = {"buzz": (6.50, 0.50), "murmur": (5.50, 1.00), "table": (2.00, 0.30)}


def _frame(doc_tokens: list[tuple[str, str, int, int, str]]) -> pd.DataFrame:
    """(form, pos, sent_id, doc_id, doc) -> canonical parse frame (lemma=form)."""
    rows = []
    for rec, (form, pos, sent, doc_id, doc) in enumerate(doc_tokens, start=1):
        rows.append(
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
        )
    return pd.DataFrame(rows)


SENTS = [
    ("apple", "NOUN", 1, 1, "d1.txt"),
    ("run", "VERB", 1, 1, "d1.txt"),
    ("xyzzy", "NOUN", 1, 1, "d1.txt"),  # alpha, unscored: counted, not found
    ("the", "DET", 1, 1, "d1.txt"),  # stopword: excluded entirely
    ("sad", "ADJ", 2, 1, "d1.txt"),
    ("bad", "ADJ", 2, 1, "d1.txt"),
    ("gloom", "NOUN", 2, 1, "d1.txt"),
]


class TestConcreteness:
    def test_sentence_statistics_match_hand_computation(self) -> None:
        res = style_mod.concreteness(_frame(SENTS), lexicon=CONC_LEX)
        assert res.ok, res.diagnostics
        frame = res.unwrap()
        assert len(frame) == 2
        first = frame.iloc[0]
        assert (first["Concreteness Mean"], first["Concreteness Median"], first["Concreteness SD"]) == (
            3.25,
            3.25,
            1.77,
        )
        assert first["Words Found"] == "2 out of 3"
        assert first["Coverage %"] == "67.0%"
        assert first["Found Words"] == "(apple, 4.5), (run, 2.0)"
        assert first["All Words"] == "apple, run, xyzzy"
        second = frame.iloc[1]
        assert (second["Concreteness Mean"], second["Concreteness Median"], second["Concreteness SD"]) == (
            2.67,
            2.0,
            2.08,
        )
        assert second["Words Found"] == "3 out of 3"

    def test_unscored_sentence_skipped(self) -> None:
        rows = [("xyzzy", "NOUN", 1, 1, "d.txt"), ("plugh", "NOUN", 2, 1, "d.txt")]
        res = style_mod.concreteness(_frame(rows), lexicon=CONC_LEX)
        assert res.ok, res.diagnostics
        assert res.unwrap().empty

    def test_punctuation_and_stopwords_excluded(self) -> None:
        rows = [("apple", "NOUN", 1, 1, "d.txt"), ("!", ".", 1, 1, "d.txt"), ("the", "DET", 1, 1, "d.txt")]
        res = style_mod.concreteness(_frame(rows), lexicon=CONC_LEX)
        assert res.ok, res.diagnostics
        assert res.unwrap()["All Words"].tolist() == ["apple"]

    def test_loader_reads_brysbaert_columns(self, tmp_path: Path) -> None:
        target = tmp_path / "conc.csv"
        target.write_text("Word,Conc.M,Conc.SD\napple,4.5,0.8\nrun,2.0,1.1\n", encoding="utf-8")
        loaded = style_mod.load_concreteness_lexicon(target)
        assert loaded.ok, loaded.diagnostics
        assert loaded.unwrap() == {"apple": 4.5, "run": 2.0}

    def test_loader_rejects_wrong_columns(self, tmp_path: Path) -> None:
        target = tmp_path / "conc.csv"
        target.write_text("term,score\napple,4.5\n", encoding="utf-8")
        loaded = style_mod.load_concreteness_lexicon(target)
        assert not loaded.ok
        assert loaded.diagnostics[0].code == "STYLE_CONCRETENESS_BAD_COLUMNS"

    def test_missing_asset_fails_loudly(self) -> None:
        res = style_mod.concreteness(_frame(SENTS), lexicon=Path("no-such-dir/no-such-file.csv"))
        assert not res.ok
        assert res.diagnostics[0].code == "STYLE_LEXICON_NOT_FOUND"


class TestIconicity:
    ICON_SENTS: ClassVar[list[tuple[str, str, int, int, str]]] = [
        ("buzz", "NOUN", 1, 1, "d1.txt"),
        ("murmur", "VERB", 1, 1, "d1.txt"),
        ("table", "NOUN", 1, 1, "d1.txt"),
        ("zzz", "NOUN", 1, 1, "d1.txt"),
    ]

    def test_sentence_statistics_match_hand_computation(self) -> None:
        res = style_mod.iconicity(_frame(self.ICON_SENTS), lexicon=ICON_LEX)
        assert res.ok, res.diagnostics
        row = res.unwrap().iloc[0]
        assert (row["Iconicity Mean"], row["Iconicity Median"], row["Iconicity SD"]) == (4.67, 5.5, 2.36)
        assert row["Words Found"] == "3 out of 4"

    def test_iconic_words_defaults(self) -> None:
        res = style_mod.iconic_words(_frame(self.ICON_SENTS), lexicon=ICON_LEX)
        assert res.ok, res.diagnostics
        frame = res.unwrap()
        assert frame["Word"].tolist() == ["buzz", "murmur"]
        assert frame["Rating"].tolist() == [6.5, 5.5]

    def test_iconic_words_custom_thresholds(self) -> None:
        res = style_mod.iconic_words(_frame(self.ICON_SENTS), lexicon=ICON_LEX, min_rating=6.0, max_rating_sd=0.4)
        assert res.ok, res.diagnostics
        assert res.unwrap().empty  # buzz's sd 0.5 exceeds 0.4

    def test_loader_reads_winter_columns(self, tmp_path: Path) -> None:
        target = tmp_path / "icon.csv"
        target.write_text("word,rating,rating_sd\nbuzz,6.5,0.5\n", encoding="utf-8")
        loaded = style_mod.load_iconicity_lexicon(target)
        assert loaded.ok, loaded.diagnostics
        assert loaded.unwrap() == {"buzz": (6.5, 0.5)}

    def test_loader_rejects_wrong_columns(self, tmp_path: Path) -> None:
        target = tmp_path / "icon.csv"
        target.write_text("word,score\nbuzz,6.5\n", encoding="utf-8")
        loaded = style_mod.load_iconicity_lexicon(target)
        assert not loaded.ok
        assert loaded.diagnostics[0].code == "STYLE_ICONICITY_BAD_COLUMNS"


class TestShared:
    def test_empty_frame_gives_empty_frames(self) -> None:
        empty = pd.DataFrame(
            columns=["ID", "Form", "Lemma", "POS", "NER", "Head", "DepRel", "Sentence ID", "Document ID", "Document"]
        )
        assert style_mod.concreteness(empty, lexicon=CONC_LEX).unwrap().empty
        assert style_mod.iconicity(empty, lexicon=ICON_LEX).unwrap().empty
        assert style_mod.iconic_words(empty, lexicon=ICON_LEX).unwrap().empty

    def test_bad_field(self) -> None:
        res = style_mod.concreteness(_frame(SENTS), field=Col.NER, lexicon=CONC_LEX)
        assert not res.ok
        assert res.diagnostics[0].code == "STYLE_BAD_FIELD"

    def test_form_field_supported(self) -> None:
        rows = [("Apples", "NOUN", 1, 1, "d.txt")]
        lex = {"apples": 4.0}
        res = style_mod.concreteness(_frame(rows), field=Col.FORM, lexicon=lex)
        assert res.ok, res.diagnostics
        assert res.unwrap()["Concreteness Mean"].tolist() == [4.0]
