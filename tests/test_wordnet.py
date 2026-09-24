"""FR-4.4 C2 — WordNet contract tests (fake backend; oracle: tests/fixtures/wordnet/).

The fake below is hand-built from ORACLE_NOTES.md, not generated from the
implementation. CSV expectations are the NLTK-recorded oracle values.
A test that needs the real NLTK backend is marked model_integration.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys

import pandas as pd
import pytest

from core.analysis import wordnet as wn_mod

FIX = Path(__file__).resolve().parent / "fixtures" / "wordnet"


# ---------------------------------------------------------------------------
# Fake backend (test-only; never production). Mirrors the oracle structure for
# the fixture vocabulary: first-sense order, own-lexname hits, the person
# noun.Tops trap, the canine tooth first sense.
# ---------------------------------------------------------------------------
class _FakeLemma:
    def __init__(self, name: str, count: int = 0) -> None:
        self._name = name
        self._count = count

    def name(self) -> str:
        return self._name

    def count(self) -> int:
        return self._count


class _FakeSynset:
    def __init__(
        self,
        name: str,
        lexname: str,
        lemmas: list[tuple[str, int]],
        definition: str = "",
        examples: tuple[str, ...] = (),
    ) -> None:
        self._name = name
        self._lexname = lexname
        self._lemmas = [_FakeLemma(n, c) for n, c in lemmas]
        self._hyper: list[_FakeSynset] = []
        self._hypo: list[_FakeSynset] = []
        self._definition = definition
        self._examples = list(examples)

    def name(self) -> str:
        return self._name

    def lexname(self) -> str:
        return self._lexname

    def lemmas(self) -> list[_FakeLemma]:
        return list(self._lemmas)

    def hypernyms(self) -> list[_FakeSynset]:
        return list(self._hyper)

    def hyponyms(self) -> list[_FakeSynset]:
        return list(self._hypo)

    def definition(self) -> str:
        return self._definition

    def examples(self) -> list[str]:
        return list(self._examples)


_CITY_DEF = "a large and densely populated urban area; may include several independent administrative districts"
_CITY_EX = ("Ancient Troy was a great city",)

_FAKE_SPECS: dict[str, tuple[str, list[tuple[str, int]], list[str], list[str], str, tuple[str, ...]]] = {
    # name: (lexname, lemmas, hypernyms, hyponyms, definition, examples)
    "dog.n.01": ("noun.animal", [("dog", 42), ("domestic_dog", 0)], ["canine.n.02"], [], "domestic canine", ()),
    "canine.n.01": ("noun.body", [("canine", 0), ("eyetooth", 0)], [], [], "pointed tooth", ()),
    "canine.n.02": ("noun.animal", [("canine", 0)], ["carnivore.n.01"], [], "fissiped mammal", ()),
    "carnivore.n.01": ("noun.animal", [("carnivore", 0)], [], [], "flesh-eater", ()),
    "feline.n.01": ("noun.animal", [("feline", 0)], ["carnivore.n.01"], [], "feline mammal", ()),
    "cat.n.01": ("noun.animal", [("cat", 0)], ["feline.n.01"], [], "domestic feline", ()),
    "city.n.01": (
        "noun.location",
        # per-lemma semcor counts (103 + 7 + 2 = 112, the recorded synset sum)
        [("city", 103), ("metropolis", 7), ("urban_center", 2)],
        [],
        ["provincial_capital.n.01", "state_capital.n.01", "national_capital.n.01"],
        _CITY_DEF,
        _CITY_EX,
    ),
    "provincial_capital.n.01": (
        "noun.location",
        [("provincial_capital", 0)],
        [],
        [],
        "the capital city of a province",
        (),
    ),
    "state_capital.n.01": (
        "noun.location",
        [("state_capital", 0)],
        [],
        [],
        "the capital city of a political subdivision of a country",
        (),
    ),
    "national_capital.n.01": ("noun.location", [("national_capital", 0)], [], [], "the capital city of a nation", ()),
    "person.n.01": ("noun.Tops", [("person", 0)], [], [], "a human being", ()),
    "love.n.01": ("noun.feeling", [("love", 0)], [], [], "a strong affection", ()),
    "table.n.01": ("noun.group", [("table", 0)], [], [], "a set of data", ()),
    "run.v.01": ("verb.motion", [("run", 0)], [], [], "move quickly", ()),
    "love.v.01": ("verb.emotion", [("love", 0)], [], [], "feel affection", ()),
    "chase.v.01": ("verb.motion", [("chase", 0)], [], [], "pursue", ()),
    "give.v.01": ("verb.possession", [("give", 0)], [], [], "transfer possession", ()),
    "see.v.01": ("verb.perception", [("see", 0)], [], [], "perceive visually", ()),
}


class FakeWordNet:
    """Hand-built stand-in for the NLTK backend (fixture vocabulary only)."""

    def __init__(self) -> None:
        self._by_name: dict[str, _FakeSynset] = {}
        for name, (lex, lemmas, _hyper, _hypo, definition, examples) in _FAKE_SPECS.items():
            self._by_name[name] = _FakeSynset(name, lex, lemmas, definition, examples)
        for name, (_lex, _lemmas, hyper, hypo, _d, _e) in _FAKE_SPECS.items():
            node = self._by_name[name]
            node._hyper = [self._by_name[h] for h in hyper]
            node._hypo = [self._by_name[h] for h in hypo]

    def synsets(self, word: str, pos: str) -> list[_FakeSynset]:
        want = ".n." if pos == "NOUN" else ".v."
        return [s for s in self._by_name.values() if want in s.name() and word in [lm.name() for lm in s.lemmas()]]

    def synset(self, name: str) -> _FakeSynset:
        return self._by_name[name]


@pytest.fixture()
def fake() -> FakeWordNet:
    return FakeWordNet()


def _words(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as fh:
        return [row["Word"] for row in csv.DictReader(fh)]


def _records(frame: pd.DataFrame) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for rec in frame.to_dict("records"):
        assert isinstance(rec, dict)
        out.append({str(k): ("" if pd.isna(v) else str(v)) for k, v in rec.items()})
    return out


def _expected(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def test_up_noun_matches_recorded_oracle(fake: FakeWordNet) -> None:
    res = wn_mod.aggregate_up(_words(FIX / "up_noun_words.csv"), pos="NOUN", backend=fake)
    assert res.ok, res.diagnostics
    assert list(res.unwrap().columns) == ["Word", "WordNet Category", "Intermediate synset 1"]
    assert _records(res.unwrap()) == _expected(FIX / "up_noun_expected.csv")


def test_up_verb_matches_recorded_oracle(fake: FakeWordNet) -> None:
    res = wn_mod.aggregate_up(_words(FIX / "up_verb_words.csv"), pos="VERB", backend=fake)
    assert res.ok, res.diagnostics
    assert _records(res.unwrap()) == _expected(FIX / "up_verb_expected.csv")


def test_up_anchors_match_recorded_oracle(fake: FakeWordNet) -> None:
    cases = _expected(FIX / "up_anchors_expected.csv")
    for row in cases:
        res = wn_mod.aggregate_up([row["Word"]], pos="NOUN", anchors=[row["Anchor"]], backend=fake)
        assert res.ok, res.diagnostics
        got = _records(res.unwrap())[0]
        assert got["WordNet Category"] == row["WordNet Category"], row
        for col in ("Intermediate synset 1", "Intermediate synset 2", "Intermediate synset 3"):
            assert got.get(col, "") == row[col], (row, col)
    # the unresolved anchor warns but still classifies
    res = wn_mod.aggregate_up(["dog"], pos="NOUN", anchors=["zxqfrbl"], backend=fake)
    assert res.ok
    assert any(d.code == "WORDNET_ANCHOR_UNRESOLVED" for d in res.diagnostics)
    assert _records(res.unwrap())[0]["WordNet Category"] == "animal"


def test_down_city_matches_recorded_oracle(fake: FakeWordNet) -> None:
    res = wn_mod.expand_down("city", pos="NOUN", backend=fake)
    assert res.ok, res.diagnostics
    assert list(res.unwrap().columns) == ["Term", "WordNet Category", "Definition", "Frequency", "Examples"]
    assert _records(res.unwrap()) == _expected(FIX / "down_city_expected.csv")


def test_down_unknown_keyword_fails(fake: FakeWordNet) -> None:
    res = wn_mod.expand_down("zxqfrbl", pos="NOUN", backend=fake)
    assert not res.ok
    assert any(d.code == "WORDNET_KEYWORD_NOT_FOUND" for d in res.diagnostics)


def test_bad_pos_fails(fake: FakeWordNet) -> None:
    res = wn_mod.aggregate_up(["dog"], pos="ADJ", backend=fake)  # type: ignore[arg-type]
    assert not res.ok
    assert any(d.code == "WORDNET_BAD_POS" for d in res.diagnostics)


def test_empty_input_warns(fake: FakeWordNet) -> None:
    res = wn_mod.aggregate_up([], pos="NOUN", backend=fake)
    assert res.ok
    assert len(res.unwrap()) == 0
    assert any(d.code == "WORDNET_EMPTY_INPUT" for d in res.diagnostics)


def test_all_not_found_warns_but_returns_rows(fake: FakeWordNet) -> None:
    res = wn_mod.aggregate_up(["zxqfrbl", "happy"], pos="NOUN", backend=fake)
    assert res.ok
    assert [r["WordNet Category"] for r in _records(res.unwrap())] == ["Not found", "Not found"]
    assert any(d.code == "WORDNET_ALL_NOT_FOUND" for d in res.diagnostics)


def test_blank_word_is_not_found(fake: FakeWordNet) -> None:
    res = wn_mod.aggregate_up([""], pos="NOUN", backend=fake)
    assert res.ok
    assert _records(res.unwrap())[0]["WordNet Category"] == "Not found"


def test_category_counts_excludes_not_found(fake: FakeWordNet) -> None:
    res = wn_mod.aggregate_up(["dog", "cat", "zxqfrbl", "city"], pos="NOUN", backend=fake)
    assert res.ok
    counted = wn_mod.category_counts(res.unwrap())
    assert counted.ok
    assert _records(counted.unwrap()) == [
        {"WordNet Category": "animal", "Frequency": "2"},
        {"WordNet Category": "location", "Frequency": "1"},
    ]


def test_missing_nltk_backend_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "nltk", None)
    res = wn_mod.default_backend()
    assert res.value is None
    assert any(d.code == "WORDNET_BACKEND_MISSING" for d in res.diagnostics)
    res2 = wn_mod.aggregate_up(["dog"], pos="NOUN", backend=None)
    assert res2.value is None


def test_missing_wordnet_data_is_actionable(without_wordnet: None) -> None:
    # `without_wordnet` also drops the loaded corpus, not just the search path:
    # emptying the path alone left an already-materialised reader answering,
    # so this passed alone and failed once anything else had used WordNet.
    res = wn_mod.default_backend()
    assert res.value is None
    assert any(d.code == "WORDNET_DATA_MISSING" for d in res.diagnostics)


@pytest.mark.model_integration
def test_up_noun_against_real_nltk() -> None:
    """Tier-1 oracle replay: real NLTK backend must reproduce the CSV fixtures."""
    data_dir = os.environ.get("NLTK_DATA", "")
    if not data_dir:
        pytest.skip("needs NLTK_DATA with the wordnet corpus")
    backend = wn_mod.default_backend(data_dir=data_dir)
    if backend.value is None:
        pytest.skip(f"nltk backend unusable: {backend.diagnostics}")
    res = wn_mod.aggregate_up(_words(FIX / "up_noun_words.csv"), pos="NOUN", backend=backend.unwrap())
    assert res.ok, res.diagnostics
    assert _records(res.unwrap()) == _expected(FIX / "up_noun_expected.csv")
    down = wn_mod.expand_down("city", pos="NOUN", backend=backend.unwrap())
    assert down.ok, down.diagnostics
    assert _records(down.unwrap()) == _expected(FIX / "down_city_expected.csv")


def test_cli_up_missing_backend_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools import wordnet as cli

    monkeypatch.setitem(sys.modules, "nltk", None)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["up", str(FIX / "up_noun_words.csv"), str(tmp_path / "out")])
    assert exc_info.value.code == 1


@pytest.mark.model_integration
def test_cli_up_and_down_write_artifacts(tmp_path: Path) -> None:
    """C4: the same core API through the CLI, with envelope provenance."""
    import json

    from tools import wordnet as cli

    data_dir = os.environ.get("NLTK_DATA", "")
    if not data_dir:
        pytest.skip("needs NLTK_DATA with the wordnet corpus")
    out = tmp_path / "out"
    assert cli.main(["up", str(FIX / "up_noun_words.csv"), str(out), "--pos", "NOUN"]) == 0
    runs = [p for p in out.iterdir() if p.is_dir()]
    assert len(runs) == 1
    assert (runs[0] / "wordnet_up.csv").is_file()
    assert (runs[0] / "wordnet_up_frequency.csv").is_file()
    envelope = json.loads((runs[0] / "result.json").read_text(encoding="utf-8"))
    assert envelope["tool"] == "wordnet-up"

    out2 = tmp_path / "out2"
    assert cli.main(["down", str(out2), "--keyword", "city"]) == 0
    runs2 = [p for p in out2.iterdir() if p.is_dir()]
    assert (runs2[0] / "wordnet_down.csv").is_file()
