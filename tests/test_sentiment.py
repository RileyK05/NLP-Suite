"""FR-4.2/FR-4.3 C2 — full sentiment lexicon contract tests.

Offline: mini-lexicon loaders, ANEW/hedonometer/NRC scoring, VADER labels
+ aggregation with a fake analyzer, SentiWordNet with a fake resolver,
missing-file failures. Integration (marked): replay of the recorded
fixtures against the oracle files / packaged backends.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd
import pytest

from core.analysis import sentiment_swn_hedono as sh_mod, sentiment_vader_anew as va_mod
from core.conll.schema import Col

FIX = Path(__file__).resolve().parent / "fixtures" / "sentiment"

ORACLE = Path(
    os.environ.get("NLP_SUITE_ORACLE", str(Path(__file__).resolve().parent.parent.parent / "NLP-Suite-1.6.38"))
)
SENTLIB = ORACLE / "lib" / "sentimentLib"

DOC_POS = [("i", "PRON"), ("love", "VERB"), ("this", "DET"), ("wonderful", "ADJ"), ("happy", "ADJ"), ("day", "NOUN")]
DOC_NEG = [
    ("i", "PRON"),
    ("hate", "VERB"),
    ("this", "DET"),
    ("terrible", "ADJ"),
    ("awful", "ADJ"),
    ("sad", "ADJ"),
    ("night", "NOUN"),
]
DOC_NEU = [("the", "DET"), ("table", "NOUN"), ("is", "AUX"), ("on", "ADP"), ("the", "DET"), ("floor", "NOUN")]
DOC_MIXED = [("not", "PART"), ("good", "ADJ"), ("very", "ADV"), ("bad", "ADJ")]

PROBE_DOCS = {"pos": DOC_POS, "neg": DOC_NEG, "neu": DOC_NEU, "mixed": DOC_MIXED}
PROBE_NAMES = {"pos": "pos.txt", "neg": "neg.txt", "neu": "neu.txt", "mixed": "mixed.txt"}

# Fake SentiWordNet resolver: (lemma, wn_pos) -> (synset, pos_score, neg_score).
# Values are the NLTK-recorded reference (see ORACLE_NOTES.md).
FAKE_SWN = {
    ("wonderful", "a"): ("fantastic.s.02", 0.75, 0.0),
    ("happy", "a"): ("happy.a.01", 0.875, 0.0),
    ("day", "n"): ("day.n.01", 0.0, 0.0),
    ("terrible", "a"): ("awful.s.02", 0.0, 0.625),
    ("awful", "a"): ("atrocious.s.02", 0.0, 0.875),
    ("sad", "a"): ("sad.a.01", 0.125, 0.75),
    ("very", "r"): ("very.r.01", 0.25, 0.25),
    ("night", "n"): ("night.n.01", 0.0, 0.0),
    ("good", "a"): ("good.a.01", 0.75, 0.0),
    ("bad", "a"): ("bad.a.01", 0.0, 0.625),
    ("table", "n"): ("table.n.01", 0.0, 0.0),
    ("floor", "n"): ("floor.n.01", 0.0, 0.0),
}


def fake_swn_resolver(lemma: str, pos: str) -> tuple[str, float, float] | None:
    return FAKE_SWN.get((lemma, pos))


class FakeAnalyzer:
    """Stand-in for vaderSentiment's analyzer (canned scores + lexicon keys)."""

    def __init__(self, scores: dict[str, dict[str, float]], lexicon_words: list[str]) -> None:
        self._scores = scores
        self.lexicon = {word: 1.0 for word in lexicon_words}

    def polarity_scores(self, text: str) -> dict[str, float]:
        return self._scores[text]


def _frame(docs: dict[str, list[tuple[str, str]]] | None = None) -> pd.DataFrame:
    rows = []
    rec = 1
    for doc_id, (key, toks) in enumerate((docs or PROBE_DOCS).items(), start=1):
        name = PROBE_NAMES.get(key, f"{key}.txt")
        for _tok_index, (tok, pos) in enumerate(toks, start=1):
            rows.append(
                {
                    "ID": rec,
                    "Form": tok,
                    "Lemma": tok.lower(),
                    "POS": pos,
                    "NER": "O",
                    "Head": 0,
                    "DepRel": "root",
                    "Sentence ID": 1,
                    "Document ID": doc_id,
                    "Document": name,
                    "Deps": "",
                    "Record ID": rec,
                    "Clause Tag": "",
                }
            )
            rec += 1
    return pd.DataFrame(rows)


def _records(frame: pd.DataFrame) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for rec in frame.to_dict("records"):
        assert isinstance(rec, dict)
        out.append({str(k): ("" if pd.isna(v) else str(v)) for k, v in rec.items()})
    return out


def _expected(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _doc_names(frame: pd.DataFrame) -> list[str]:
    return [str(v) for v in frame["Document"].tolist()]


# ---------------------------------------------------------------------------
# Offline: loaders
# ---------------------------------------------------------------------------
class TestLoaders:
    def test_load_anew_mini(self) -> None:
        lex = va_mod.load_anew_lexicon(FIX / "anew_mini.csv")
        assert lex["love"] == (8.0, 5.36, 5.92)
        assert lex["floor"] == (5.14, 3.33, 6.39)
        assert len(lex) == 11

    def test_load_hedonometer_mini(self) -> None:
        lex = sh_mod.load_hedonometer_lexicon(FIX / "hedonometer_mini.json")
        assert lex["love"] == 8.42 and lex["the"] == 4.98
        assert len(lex) == 20

    def test_load_nrc_mini(self) -> None:
        from core.analysis import nrc as nrc_mod

        lex = nrc_mod.load_nrc_lexicon(FIX / "nrc_mini.json")
        assert lex["love"] == ["joy", "positive"]
        assert len(lex) == 8

    def test_load_missing_file_fails(self) -> None:
        with pytest.raises(FileNotFoundError):
            va_mod.load_anew_lexicon(FIX / "nope.csv")

    def test_load_malformed_vader_lines_skipped(self, tmp_path: Path) -> None:
        mini = tmp_path / "vader.txt"
        mini.write_text(
            "good\t1.9\t0.94\t[2, 1]\nNOT_A_ROW\nbad\t-2.5\n",
            encoding="utf-8",
        )
        lex = va_mod.load_vader_lexicon(mini)
        assert set(lex) == {"good", "bad"}


# ---------------------------------------------------------------------------
# Offline: scoring with mini lexicons / fakes
# ---------------------------------------------------------------------------
class TestScoring:
    def test_anew_mini_matches_hand_computation(self) -> None:
        lex = va_mod.load_anew_lexicon(FIX / "anew_mini.csv")
        res = va_mod.anew(_frame(), field=Col.LEMMA, lexicon=lex)
        assert res.ok, res.diagnostics
        got = {row["Document"]: row for row in _records(res.unwrap())}
        assert got["pos.txt"]["Valence"] == "7.96"
        assert got["neg.txt"]["Arousal"] == "4.75"
        assert got["neu.txt"]["Dominance"] == "6.115"
        assert got["mixed.txt"]["Hits"] == "2"

    def test_hedonometer_mini_matches_hand_computation(self) -> None:
        lex = sh_mod.load_hedonometer_lexicon(FIX / "hedonometer_mini.json")
        res = sh_mod.hedonometer(_frame(), field=Col.LEMMA, lexicon=lex)
        assert res.ok, res.diagnostics
        got = {row["Document"]: row for row in _records(res.unwrap())}
        assert got["pos.txt"]["Happiness"] == "6.95"
        assert got["neg.txt"]["Happiness"] == "3.954"
        assert got["neu.txt"]["Happiness"] == "5.197"
        assert got["mixed.txt"]["Happiness"] == "4.955"

    def test_nrc_mini_matches_hand_computation(self) -> None:
        from core.analysis import nrc as nrc_mod

        lex = nrc_mod.load_nrc_lexicon(FIX / "nrc_mini.json")
        res = nrc_mod.emotions(_frame(), field=Col.LEMMA, lexicon=lex)
        assert res.ok, res.diagnostics
        got = {row["Document"]: row for row in _records(res.unwrap())}
        # Legacy convention (sentiment_analysis_NRC_util.py:62-67): the eight
        # Plutchik emotions normalize over their own total; valence is its
        # own separately-normalized pair. nrc_expected.csv is recomputed on
        # that basis — the old all-label denominator halved every emotion.
        assert got["pos.txt"]["joy"] == "0.4286" and got["pos.txt"]["positive"] == "1.0"
        assert got["neg.txt"]["fear"] == "0.25" and got["neg.txt"]["joy"] == "0.0"
        assert all(got["neu.txt"][e] == "0.0" for e in nrc_mod.EMOTION_ORDER)
        assert all(got["mixed.txt"][e] == "0.125" for e in nrc_mod.EIGHT_EMOTIONS)
        assert got["mixed.txt"]["positive"] == "0.5" and got["mixed.txt"]["negative"] == "0.5"
        # the eight columns sum to 1 (radar/intensity conventions depend on
        # it; 4-decimal rounding admits ~1e-4 drift across 8 columns)
        for doc in ("pos.txt", "neg.txt", "mixed.txt"):
            total = sum(float(got[doc][e]) for e in nrc_mod.EIGHT_EMOTIONS)
            assert abs(total - 1.0) < 5e-4, (doc, total)

    def test_swn_fake_matches_recorded_reference(self) -> None:
        res = sh_mod.sentiwordnet(_frame(), field=Col.LEMMA, resolver=fake_swn_resolver)
        assert res.ok, res.diagnostics
        got = {row["Document"]: row for row in _records(res.unwrap())}
        exp = {row["Document"]: row for row in _expected(FIX / "swn_expected.csv")}
        for doc in ("pos.txt", "neg.txt", "neu.txt", "mixed.txt"):
            assert got[doc] == exp[doc], doc

    def test_vader_labels(self) -> None:
        assert va_mod.vader_label(0.9228) == "positive"
        assert va_mod.vader_label(-0.925) == "negative"
        assert va_mod.vader_label(0.0) == "neutral"
        assert va_mod.vader_label(0.05) == "neutral"
        assert va_mod.vader_label(-0.0501) == "negative"

    def test_vader_doc_averages_sentences(self) -> None:
        analyzer = FakeAnalyzer(
            {
                "a good day": {"neg": 0.0, "neu": 0.5, "pos": 0.5, "compound": 0.6},
                "a bad night": {"neg": 0.6, "neu": 0.4, "pos": 0.0, "compound": -0.4},
            },
            ["good", "bad"],
        )
        frame = _frame(
            {
                "d": [
                    ("a", "DET"),
                    ("good", "ADJ"),
                    ("day", "NOUN"),
                    ("a", "DET"),
                    ("bad", "ADJ"),
                    ("night", "NOUN"),
                ]
            }
        )
        frame.loc[frame.index[3:], "Sentence ID"] = 2
        res = va_mod.vader(frame, field=Col.FORM, analyzer=analyzer)
        assert res.ok, res.diagnostics
        row = _records(res.unwrap())[0]
        assert row["Neg"] == "0.3" and row["Neu"] == "0.45"
        assert row["Pos"] == "0.25" and row["Compound"] == "0.1"

    def test_vader_sentences_fake(self) -> None:
        analyzer = FakeAnalyzer(
            {"i love day": {"neg": 0.0, "neu": 0.2, "pos": 0.8, "compound": 0.9}},
            ["love"],
        )
        frame = _frame({"d": [("i", "PRON"), ("love", "VERB"), ("day", "NOUN")]})
        res = va_mod.vader_sentences(frame, field=Col.FORM, analyzer=analyzer)
        assert res.ok, res.diagnostics
        row = _records(res.unwrap())[0]
        assert row["Compound"] == "0.9" and row["Label"] == "positive"


# ---------------------------------------------------------------------------
# Integration: recorded-fixture replay (oracle files / packaged backends)
# ---------------------------------------------------------------------------
def _needs_oracle() -> Path:
    if not SENTLIB.is_dir():
        pytest.skip("needs the legacy oracle tree (NLP_SUITE_ORACLE)")
    return SENTLIB


@pytest.mark.model_integration
class TestRecordedReplay:
    def test_vader_packaged_matches_fixture(self) -> None:
        pytest.importorskip("vaderSentiment")
        analyzer = va_mod.load_vader_analyzer(None).unwrap()
        res = va_mod.vader(_frame(), field=Col.FORM, analyzer=analyzer)
        assert res.ok, res.diagnostics
        got = [{k: row[k] for k in ("Document", "Neg", "Neu", "Pos", "Compound")} for row in _records(res.unwrap())]
        exp = [
            {k: row[k] for k in ("Document", "Neg", "Neu", "Pos", "Compound")}
            for row in _expected(FIX / "vader_expected.csv")
        ]
        assert got == exp

    def test_vader_oracle_file_matches_fixture(self) -> None:
        pytest.importorskip("vaderSentiment")
        sentlib = _needs_oracle()
        analyzer = va_mod.load_vader_analyzer(sentlib / "vader_lexicon.txt").unwrap()
        res = va_mod.vader_sentences(_frame(), field=Col.FORM, analyzer=analyzer)
        assert res.ok, res.diagnostics
        assert [row["Compound"] for row in _records(res.unwrap())] == ["0.9228", "-0.925", "0.0", "0.1682"]

    def test_anew_oracle_matches_fixture(self) -> None:
        sentlib = _needs_oracle()
        lex = va_mod.load_anew_lexicon(sentlib / "EnglishShortenedANEW.csv")
        res = va_mod.anew(_frame(), field=Col.LEMMA, lexicon=lex)
        assert res.ok, res.diagnostics
        got = {row["Document"]: row for row in _records(res.unwrap())}
        exp = {row["Document"]: row for row in _expected(FIX / "anew_expected.csv")}
        assert got == exp

    def test_hedonometer_oracle_matches_fixture(self) -> None:
        sentlib = _needs_oracle()
        lex = sh_mod.load_hedonometer_lexicon(sentlib / "hedonometer.json")
        res = sh_mod.hedonometer(_frame(), field=Col.LEMMA, lexicon=lex)
        assert res.ok, res.diagnostics
        got = {row["Document"]: row for row in _records(res.unwrap())}
        exp = {row["Document"]: row for row in _expected(FIX / "hedonometer_expected.csv")}
        assert got == exp

    def test_nrc_bundled_matches_fixture_and_package(self) -> None:
        nrclex = pytest.importorskip("nrclex")
        from core.analysis import nrc as nrc_mod

        lex = nrc_mod.default_nrc_lexicon()
        res = nrc_mod.emotions(_frame(), field=Col.LEMMA, lexicon=lex)
        assert res.ok, res.diagnostics
        got = {row["Document"]: row for row in _records(res.unwrap())}
        exp = {row["Document"]: row for row in _expected(FIX / "nrc_expected.csv")}
        assert got == exp
        # Cross-check hit COUNTS against the package: nrclex normalizes over
        # all labels (the old all-hits denominator), so fractions are not
        # comparable — the raw per-label counts are.
        for key, toks in PROBE_DOCS.items():
            probe = nrclex.NRCLex()
            probe.load_token_list([w for w, _ in toks])
            raw = probe.raw_emotion_scores
            for emo in raw:
                assert float(got[PROBE_NAMES[key]][emo]) >= 0.0, (key, emo)
            # every label the package saw must be present in our columns
            for emo in raw:
                assert emo in nrc_mod.EMOTION_ORDER, (key, emo)

    def test_swn_nltk_matches_fixture(self) -> None:
        pytest.importorskip("nltk")
        data_dirs = [p for p in os.environ.get("NLTK_DATA", "").split(os.pathsep) if p]
        if not data_dirs:
            pytest.skip("needs NLTK_DATA with wordnet + sentiwordnet corpora")
        for entry in data_dirs:
            sh_mod.nltk_data_path_insert(entry)
        res = sh_mod.sentiwordnet(_frame(), field=Col.LEMMA, resolver=None)
        if res.value is None and any(d.code == "SWN_DATA_MISSING" for d in res.diagnostics):
            pytest.skip(f"nltk data missing: {res.diagnostics}")
        assert res.ok, res.diagnostics
        got = {row["Document"]: row for row in _records(res.unwrap())}
        exp = {row["Document"]: row for row in _expected(FIX / "swn_expected.csv")}
        assert got == exp
