"""FR-4.5 C2 — VerbNet/FrameNet contract tests.

Offline: fake backends (first-wins, Not-found rows, counts), bad-POS
rejection, empty input, NLTK adapter behavior via stub corpora.
Integration (marked): replay against the real NLTK corpora when the
``wordnet`` extra + data exist.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd
import pytest

from core.analysis import framenet as fn_mod, verbnet as vn_mod


class FakeVerbNet:
    MAP: ClassVar[dict[str, list[str]]] = {
        "run": ["run-51.3.2", "run-51.3.1"],
        "give": ["give-13.1"],
        "eat": ["eat-39.1-1"],
    }

    def class_ids(self, lemma: str) -> list[str]:
        return list(self.MAP.get(lemma, []))


class FakeFrameNet:
    MAP: ClassVar[dict[tuple[str, str], list[str]]] = {
        ("run", "v"): ["Self_motion", "Fluidic_motion"],
        ("dog", "n"): ["Animals"],
        ("run", "n"): ["Leadership"],
    }

    def frames_for(self, lemma: str, pos: str) -> list[str]:
        return sorted(self.MAP.get((lemma, pos), []))


class TestVerbNet:
    def test_first_class_wins(self) -> None:
        res = vn_mod.aggregate_verbs(["run", "give", "xyzzy"], backend=FakeVerbNet())
        assert res.ok, res.diagnostics
        frame = res.unwrap()
        assert frame["Word"].tolist() == ["run", "give", "xyzzy"]
        assert frame["VerbNet Class"].tolist() == ["run-51.3.2", "give-13.1", "Not found"]

    def test_counts_exclude_not_found(self) -> None:
        agg = vn_mod.aggregate_verbs(["run", "run", "give", "xyzzy"], backend=FakeVerbNet()).unwrap()
        counted = vn_mod.category_counts(agg)
        assert counted.ok, counted.diagnostics
        frame = counted.unwrap()
        assert frame["VerbNet Class"].tolist() == ["run-51.3.2", "give-13.1"]
        assert frame["Frequency"].tolist() == [2, 1]

    def test_empty_input_warns(self) -> None:
        res = vn_mod.aggregate_verbs([], backend=FakeVerbNet())
        assert res.ok
        assert res.unwrap().empty
        assert res.diagnostics[0].code == "VERBNET_EMPTY_INPUT"

    def test_all_not_found_warns(self) -> None:
        res = vn_mod.aggregate_verbs(["xyzzy"], backend=FakeVerbNet())
        assert res.ok
        assert res.diagnostics[0].code == "VERBNET_ALL_NOT_FOUND"

    def test_bad_frame_rejected(self) -> None:
        res = vn_mod.category_counts(pd.DataFrame({"Nope": []}))
        assert not res.ok
        assert res.diagnostics[0].code == "VERBNET_BAD_FRAME"

    def test_missing_data_fails_loudly(self) -> None:
        pytest.importorskip("nltk")
        import nltk

        try:
            nltk.data.find("corpora/verbnet")
            pytest.skip("VerbNet data present — backend resolves")
        except LookupError:
            pass
        res = vn_mod.aggregate_verbs(["run"])
        assert not res.ok
        assert res.diagnostics[0].code == "VERBNET_DATA_MISSING"


class StubVnCorpus:
    def classids(self, lemma: str = "") -> list[str]:
        assert lemma == "run"
        return ["run-51.3.2", "run-51.3.1"]


class TestNltkVerbNetAdapter:
    def test_delegates_to_classids(self) -> None:
        assert vn_mod.NltkVerbNet(StubVnCorpus()).class_ids("run") == ["run-51.3.2", "run-51.3.1"]


class TestFrameNet:
    def test_first_frame_wins_sorted(self) -> None:
        res = fn_mod.aggregate(["run", "dog", "xyzzy"], pos="VERB", backend=FakeFrameNet())
        assert res.ok, res.diagnostics
        frame = res.unwrap()
        # ('run','v') frames sorted: Fluidic_motion first
        assert frame["FrameNet Frame"].tolist() == ["Fluidic_motion", "Not found", "Not found"]

    def test_noun_pos_selects_noun_frames(self) -> None:
        res = fn_mod.aggregate(["run", "dog"], pos="NOUN", backend=FakeFrameNet())
        assert res.ok, res.diagnostics
        assert res.unwrap()["FrameNet Frame"].tolist() == ["Leadership", "Animals"]

    def test_bad_pos_rejected(self) -> None:
        res = fn_mod.aggregate(["run"], pos="ADJ", backend=FakeFrameNet())
        assert not res.ok
        assert res.diagnostics[0].code == "FRAMENET_BAD_POS"

    def test_counts_exclude_not_found(self) -> None:
        agg = fn_mod.aggregate(["run", "run", "dog", "xyzzy"], pos="NOUN", backend=FakeFrameNet()).unwrap()
        counted = fn_mod.category_counts(agg)
        assert counted.ok, counted.diagnostics
        assert counted.unwrap()["Frequency"].tolist() == [2, 1]

    def test_missing_data_fails_loudly(self) -> None:
        pytest.importorskip("nltk")
        import nltk

        try:
            nltk.data.find("corpora/framenet_v17")
            pytest.skip("FrameNet data present — backend resolves")
        except LookupError:
            pass
        res = fn_mod.aggregate(["run"])
        assert not res.ok
        assert res.diagnostics[0].code == "FRAMENET_DATA_MISSING"


class StubFnFrame:
    def __init__(self, name: str, units: list[str]) -> None:
        self.name = name
        self.lexUnit = {unit: 1 for unit in units}


class StubFnCorpus:
    def __init__(self) -> None:
        self.calls = 0
        self._frames = [
            StubFnFrame("Self_motion", ["run.v", "walk.v"]),
            StubFnFrame("Fluidic_motion", ["run.v"]),
            StubFnFrame("Animals", ["dog.n"]),
        ]

    def frames(self) -> list[StubFnFrame]:
        self.calls += 1
        return self._frames


class TestNltkFrameNetAdapter:
    def test_index_built_once_and_sorted(self) -> None:
        backend = fn_mod.NltkFrameNet(StubFnCorpus())
        assert backend.frames_for("run", "v") == ["Fluidic_motion", "Self_motion"]
        assert backend.frames_for("DOG", "n") == ["Animals"]
        assert backend.frames_for("xyzzy", "n") == []
        assert backend._index is not None


@pytest.mark.model_integration
def test_nltk_corpora_replay() -> None:
    """run -> a run-* class; run.v -> Self_motion against the real corpora."""
    vn = vn_mod.default_backend()
    if vn.value is None:
        pytest.skip(f"VerbNet data unavailable: {vn.diagnostics[0].message}")
    assert vn.unwrap().class_ids("run")[0].startswith("run-")
    fn = fn_mod.default_backend()
    if fn.value is None:
        pytest.skip(f"FrameNet data unavailable: {fn.diagnostics[0].message}")
    assert "Self_motion" in fn.unwrap().frames_for("run", "v")
