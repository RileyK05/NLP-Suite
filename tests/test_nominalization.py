"""FR-2.7 C2 — nominalization contract tests.

Offline: fake-backend detection, suffix prefilter, curated bypass,
direction constraint on the NLTK backend via a stub wordnet module,
empty frames, bad columns. Integration (marked): replay against the
real NLTK WordNet corpus when the ``wordnet`` extra + data exist.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from core.analysis import nominalization as nom_mod
from core.conll.schema import Col

# Hand-built derivational map: noun lemma -> base verb (already validated).
FAKE_MAP = {
    "destruction": "destroy",
    "decision": "decide",
    "killing": "kill",
    "celebration": "celebrate",
    "attack": "attack",  # irregular: not longer than its verb
}


class FakeBackend:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._map = mapping
        self.calls: list[str] = []

    def base_verb(self, noun_lemma: str) -> str | None:
        self.calls.append(noun_lemma)
        return self._map.get(noun_lemma)


def _frame(rows: list[tuple[str, str, str, int, int, str]]) -> pd.DataFrame:
    """(form, lemma, pos, sent_id, doc_id, doc) -> canonical parse frame."""
    rec = 1
    out = []
    for form, lemma, pos, sent, doc_id, doc in rows:
        out.append(
            {
                "ID": rec,
                "Form": form,
                "Lemma": lemma,
                "POS": pos,
                "NER": "O",
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": sent,
                "Document ID": doc_id,
                "Document": doc,
            }
        )
        rec += 1
    return pd.DataFrame(out)


DOC = [
    ("The", "the", "DET", 1, 1, "d1.txt"),
    ("destruction", "destruction", "NN", 1, 1, "d1.txt"),
    ("of", "of", "ADP", 1, 1, "d1.txt"),
    ("the", "the", "DET", 1, 1, "d1.txt"),
    ("city", "city", "NN", 1, 1, "d1.txt"),
    ("was", "be", "AUX", 1, 1, "d1.txt"),
    ("total", "total", "ADJ", 1, 1, "d1.txt"),
    (".", ".", ".", 1, 1, "d1.txt"),
    ("They", "they", "PRON", 2, 1, "d1.txt"),
    ("celebrate", "celebrate", "VERB", 2, 1, "d1.txt"),  # verb, not a noun: ignored
    ("the", "the", "DET", 2, 1, "d1.txt"),
    ("celebration", "celebration", "NN", 2, 1, "d1.txt"),
    ("table", "table", "NN", 2, 1, "d1.txt"),  # base noun: backend has no verb
    (".", ".", ".", 2, 1, "d1.txt"),
]


class TestDetection:
    def test_finds_deverbal_nouns(self) -> None:
        res = nom_mod.detect_nominalizations(_frame(DOC), backend=FakeBackend(FAKE_MAP))
        assert res.ok, res.diagnostics
        frame = res.unwrap()
        assert frame["Word"].tolist() == ["destruction", "celebration"]
        assert frame["Base Verb"].tolist() == ["destroy", "celebrate"]
        assert frame["Sentence ID"].tolist() == [1, 2]

    def test_verbs_and_plain_nouns_ignored(self) -> None:
        res = nom_mod.detect_nominalizations(_frame(DOC), backend=FakeBackend(FAKE_MAP))
        assert res.ok, res.diagnostics
        words = res.unwrap()["Word"].tolist()
        assert "celebrate" not in words and "city" not in words and "table" not in words

    def test_memoizes_repeated_words(self) -> None:
        backend = FakeBackend(FAKE_MAP)
        rows = [("Destruction", "destruction", "NN", 1, 1, "d.txt")] * 3
        res = nom_mod.detect_nominalizations(_frame(rows), backend=backend)
        assert res.ok, res.diagnostics
        assert backend.calls.count("destruction") == 1
        assert len(res.unwrap()) == 3

    def test_suffix_prefilter_blocks_irregulars(self) -> None:
        rows = [("attack", "attack", "NN", 1, 1, "d.txt")]
        res = nom_mod.detect_nominalizations(_frame(rows), backend=FakeBackend(FAKE_MAP))
        assert res.ok, res.diagnostics
        assert res.unwrap().empty  # 'attack' has no nominal suffix

    def test_curated_list_bypasses_prefilter(self) -> None:
        rows = [("attack", "attack", "NN", 1, 1, "d.txt")]
        res = nom_mod.detect_nominalizations(_frame(rows), backend=FakeBackend(FAKE_MAP), curated_words=["attack"])
        assert res.ok, res.diagnostics
        assert res.unwrap()["Base Verb"].tolist() == ["attack"]

    def test_no_check_ending_runs_pure_backend(self) -> None:
        rows = [("attack", "attack", "NN", 1, 1, "d.txt")]
        res = nom_mod.detect_nominalizations(_frame(rows), backend=FakeBackend(FAKE_MAP), check_ending=False)
        assert res.ok, res.diagnostics
        assert len(res.unwrap()) == 1

    def test_universal_pos_nouns_count(self) -> None:
        rows = [("destruction", "destruction", "NOUN", 1, 1, "d.txt")]
        res = nom_mod.detect_nominalizations(_frame(rows), backend=FakeBackend(FAKE_MAP))
        assert res.ok, res.diagnostics
        assert len(res.unwrap()) == 1


class TestSentenceFrequency:
    def test_per_sentence_counts(self) -> None:
        res = nom_mod.sentence_frequency(_frame(DOC), backend=FakeBackend(FAKE_MAP))
        assert res.ok, res.diagnostics
        frame = res.unwrap()
        # sentence 1: 7 words (punct excluded), 1 hit; sentence 2: 5 words, 1 hit
        assert frame["Nominalizations in Sentence"].tolist() == [1, 1]
        assert frame["Words in Sentence"].tolist() == [7, 5]
        assert frame["Nominalization %"].tolist() == [round(100 / 7, 2), 20.0]
        assert frame["Nominalized Words"].tolist() == ["destruction", "celebration"]

    def test_empty_frame_gives_empty_frames(self) -> None:
        empty = pd.DataFrame(
            columns=["ID", "Form", "Lemma", "POS", "NER", "Head", "DepRel", "Sentence ID", "Document ID", "Document"]
        )
        for fn in (nom_mod.detect_nominalizations, nom_mod.sentence_frequency):
            res = fn(empty, backend=FakeBackend(FAKE_MAP))
            assert res.ok, res.diagnostics
            assert res.unwrap().empty


class TestFailures:
    def test_missing_backend_without_nltk_data(self, without_wordnet: None) -> None:
        # Was guarded by `nltk.data.find("corpora/wordnet")`, which looks for an
        # unzipped directory. The downloader leaves wordnet.zip, so the guard
        # read a working install as a missing one and asserted a failure that
        # could not happen. The fixture makes the corpus actually absent.
        res = nom_mod.detect_nominalizations(_frame(DOC))
        assert not res.ok
        assert res.diagnostics[0].code == "NOMINALIZATION_DATA_MISSING"

    def test_bad_field(self) -> None:
        res = nom_mod.detect_nominalizations(_frame(DOC), field=Col.NER, backend=FakeBackend(FAKE_MAP))
        assert not res.ok
        assert res.diagnostics[0].code == "NOMINALIZATION_BAD_FIELD"

    def test_missing_column(self) -> None:
        frame = _frame(DOC).drop(columns=["Document ID"])
        res = nom_mod.detect_nominalizations(frame, backend=FakeBackend(FAKE_MAP))
        assert not res.ok
        assert res.diagnostics[0].code == "CONLL_MISSING_COLUMN"

    def test_curated_list_missing_file(self, tmp_path: Path) -> None:
        res = nom_mod.load_curated_words(tmp_path / "nope.csv")
        assert not res.ok
        assert res.diagnostics[0].code == "NOMINALIZATION_LIST_NOT_FOUND"


class FakeLemma:
    def __init__(self, verb: str | None) -> None:
        self._verb = verb

    def derivationally_related_forms(self) -> list[FakeForm]:
        return [FakeForm(self._verb)] if self._verb is not None else []


class FakeForm:
    def __init__(self, verb: str) -> None:
        self._verb = verb

    def name(self) -> str:
        return self._verb

    def synset(self) -> FakeSynset:
        return FakeSynset()


class FakeSynset:
    def pos(self) -> str:
        return "v"


class StubWordNet:
    """Stub NLTK wordnet module: noun -> verb candidates with real lengths."""

    NOUN = "n"
    MAP: ClassVar[dict[str, str]] = {
        "destruction": "destroy",
        "table": "tabulate",
        "quality": "qualify",
        "belief": "believe",
    }

    def lemmas(self, word: str, pos: str) -> list[FakeLemma]:
        assert pos == "n"
        return [FakeLemma(self.MAP[word])] if word in self.MAP else []


class TestNltkBackendDirection:
    def test_accepts_longer_noun(self) -> None:
        assert nom_mod.NltkNominalization(StubWordNet()).base_verb("destruction") == "destroy"

    def test_rejects_denominal_verbs(self) -> None:
        backend = nom_mod.NltkNominalization(StubWordNet())
        # table->tabulate and quality->qualify: the verb is LONGER, wrong direction
        assert backend.base_verb("table") is None
        assert backend.base_verb("quality") is None

    def test_rejects_same_length_irregular(self) -> None:
        # belief<-believe: real WordNet misses it too (not longer than its verb)
        assert nom_mod.NltkNominalization(StubWordNet()).base_verb("belief") is None

    def test_unknown_word_is_none(self) -> None:
        assert nom_mod.NltkNominalization(StubWordNet()).base_verb("xyzzy") is None


ORACLE = Path(
    os.environ.get("NLP_SUITE_ORACLE", str(Path(__file__).resolve().parent.parent.parent / "NLP-Suite-1.6.38"))
)


@pytest.mark.model_integration
def test_nltk_wordnet_replay() -> None:
    """destruction->destroy / decision->decide / death->die against real WordNet."""
    resolved = nom_mod.default_wordnet_module()
    if resolved.value is None:
        pytest.skip(f"WordNet data unavailable: {resolved.diagnostics[0].message}")
    backend = nom_mod.NltkNominalization(resolved.unwrap())
    assert backend.base_verb("destruction") == "destroy"
    assert backend.base_verb("decision") == "decide"
    assert backend.base_verb("killing") == "kill"
