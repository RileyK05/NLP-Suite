"""FR-4.6 C2 — symbolic + actor typology contract tests.

Offline: hand-built hypernym graphs behind the backend protocol
(exact hits, last-token rule, stopword/generic guards, instance
skipping, first-sense-only climb, depth cap, loaders). Integration
(marked): replay against the real NLTK WordNet corpus when the
``wordnet`` extra + data exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from core.analysis import symbolic as sym_mod


class FakeGraph:
    """Hand-built WordNet fragment.

    kitchen.n.01 -> room.n.01 (domestic anchor); london.n.01 is an
    instance (skipped); sheriff.n.01 -> lawman.n.01 (military_police
    anchor); wanderer.n.01 dead-ends while its second sense would hit
    (proves first-sense-only); deep.n.01 sits atop a 13-link chain
    (proves the depth cap).
    """

    SYNSETS: ClassVar[dict[str, list[str]]] = {
        "kitchen": ["kitchen.n.01"],
        "london": ["london.n.01"],
        "sheriff": ["sheriff.n.01"],
        "wanderer": ["wanderer.n.01", "roamer.n.02"],
        "deep": ["deep.n.01"],
        "way": ["way.n.06"],
    }
    PARENTS: ClassVar[dict[str, list[str]]] = {
        "kitchen.n.01": ["room.n.01"],
        "room.n.01": ["area.n.01"],
        "sheriff.n.01": ["lawman.n.01"],
        "lawman.n.01": [],
        "wanderer.n.01": ["deadend.n.01"],
        "deadend.n.01": [],
        "roamer.n.02": ["traveler.n.01"],
        "traveler.n.01": ["worker.n.01"],  # laborer anchor — must NOT be reached
        "deep.n.01": ["link01"],
        **{f"link{i:02d}": [f"link{i + 1:02d}"] for i in range(1, 13)},
        "link13": ["worker.n.01"],  # anchor at depth 13 — beyond the cap
        "worker.n.01": [],
    }
    INSTANCES: ClassVar[frozenset[str]] = frozenset({"london.n.01"})
    ANCHORS: ClassVar[frozenset[str]] = frozenset(
        {name for names in (sym_mod.SPACE_ANCHORS | sym_mod.ACTOR_ANCHORS).values() for name in names}
        | {"area.n.01", "deadend.n.01", "traveler.n.01", "link01", "link13"}
    )

    def noun_synsets(self, word: str) -> list[str]:
        return list(self.SYNSETS.get(word, []))

    def hypernyms(self, synset: str) -> list[str]:
        return list(self.PARENTS.get(synset, []))

    def is_instance(self, synset: str) -> bool:
        return synset in self.INSTANCES

    def resolve(self, name: str) -> str | None:
        return name if name in self.ANCHORS else None


SPACE_LEX = {"castle": "royal_court", "room": "domestic_interior"}
ACTOR_LEX = {"harry": "child_youth", "sheriff": "authority_official"}


class TestSpace:
    def test_exact_lexicon_hit(self) -> None:
        res = sym_mod.classify_space("castle", lexicon=SPACE_LEX, backend=FakeGraph())
        assert res.ok, res.diagnostics
        assert res.unwrap() == "royal_court"

    def test_last_token_hit(self) -> None:
        res = sym_mod.classify_space("throne room", lexicon=SPACE_LEX, backend=FakeGraph())
        assert res.ok, res.diagnostics
        assert res.unwrap() == "domestic_interior"

    def test_wordnet_climb(self) -> None:
        res = sym_mod.classify_space("kitchen", lexicon=SPACE_LEX, backend=FakeGraph())
        assert res.ok, res.diagnostics
        assert res.unwrap() == "domestic_interior"

    def test_stopword_never_classifies(self) -> None:
        # way.n.06 IS an anchor, but 'way' is a generic abstract noun
        res = sym_mod.classify_space("way", lexicon=SPACE_LEX, backend=FakeGraph())
        assert res.ok, res.diagnostics
        assert res.unwrap() == "unclassified"

    def test_instance_skipped(self) -> None:
        res = sym_mod.classify_space("london", lexicon=SPACE_LEX, backend=FakeGraph())
        assert res.ok, res.diagnostics
        assert res.unwrap() == "unclassified"

    def test_unknown_is_unclassified(self) -> None:
        res = sym_mod.classify_space("xyzzy", lexicon=SPACE_LEX, backend=FakeGraph())
        assert res.ok, res.diagnostics
        assert res.unwrap() == "unclassified"

    def test_no_wordnet_means_lexicon_only(self) -> None:
        res = sym_mod.classify_space("kitchen", lexicon=SPACE_LEX, backend=FakeGraph(), use_wordnet=False)
        assert res.ok, res.diagnostics
        assert res.unwrap() == "unclassified"

    def test_lexicon_beats_stopword(self) -> None:
        res = sym_mod.classify_space("way", lexicon={"way": "water_passage"}, backend=FakeGraph())
        assert res.ok, res.diagnostics
        assert res.unwrap() == "water_passage"


class TestActor:
    def test_lexicon_hit(self) -> None:
        res = sym_mod.classify_actor("sheriff", lexicon=ACTOR_LEX, backend=FakeGraph())
        assert res.ok, res.diagnostics
        assert res.unwrap() == "authority_official"

    def test_proper_noun_never_classified(self) -> None:
        res = sym_mod.classify_actor("Harry", pos="PROPN", lexicon=ACTOR_LEX, backend=FakeGraph())
        assert res.ok, res.diagnostics
        assert res.unwrap() == "unclassified"

    def test_named_character_via_csv_without_pos(self) -> None:
        res = sym_mod.classify_actor("Harry", lexicon=ACTOR_LEX, backend=FakeGraph())
        assert res.ok, res.diagnostics
        assert res.unwrap() == "child_youth"

    def test_generic_bucket(self) -> None:
        res = sym_mod.classify_actor("man", lexicon=ACTOR_LEX, backend=FakeGraph())
        assert res.ok, res.diagnostics
        assert res.unwrap() == "generic_person"

    def test_first_sense_only(self) -> None:
        # wanderer.n.01 dead-ends; the second sense would reach worker.n.01
        res = sym_mod.classify_actor("wanderer", lexicon={}, backend=FakeGraph())
        assert res.ok, res.diagnostics
        assert res.unwrap() == "unclassified"

    def test_depth_cap(self) -> None:
        # worker.n.01 sits 13 links down — beyond the cap
        res = sym_mod.classify_actor("deep", lexicon={}, backend=FakeGraph())
        assert res.ok, res.diagnostics
        assert res.unwrap() == "unclassified"

    def test_climb_hit(self) -> None:
        res = sym_mod.classify_actor("sheriff", lexicon={}, backend=FakeGraph())
        assert res.ok, res.diagnostics
        assert res.unwrap() == "military_police"


class TestAggregation:
    def test_spaces(self) -> None:
        res = sym_mod.aggregate_spaces(["castle", "kitchen", "xyzzy"], lexicon=SPACE_LEX, backend=FakeGraph())
        assert res.ok, res.diagnostics
        frame = res.unwrap()
        assert frame["Space Type"].tolist() == ["royal_court", "domestic_interior", "unclassified"]
        counted = sym_mod.category_counts(frame, "Space Type")
        assert counted.ok, counted.diagnostics
        assert counted.unwrap()["Space Type"].tolist() == ["royal_court", "domestic_interior"]

    def test_actors(self) -> None:
        res = sym_mod.aggregate_actors(["sheriff", "man", "xyzzy"], lexicon=ACTOR_LEX, backend=FakeGraph())
        assert res.ok, res.diagnostics
        assert res.unwrap()["Actor Type"].tolist() == ["authority_official", "generic_person", "unclassified"]

    def test_empty_input_warns(self) -> None:
        res = sym_mod.aggregate_spaces([], lexicon=SPACE_LEX, backend=FakeGraph())
        assert res.ok
        assert res.diagnostics[0].code == "SYMBOLIC_EMPTY_INPUT"

    def test_bad_frame_rejected(self) -> None:
        import pandas as pd

        res = sym_mod.category_counts(pd.DataFrame({"Nope": []}), "Space Type")
        assert not res.ok
        assert res.diagnostics[0].code == "SYMBOLIC_BAD_FRAME"


class TestLoaders:
    def test_space_loader(self, tmp_path: Path) -> None:
        target = tmp_path / "space.csv"
        target.write_text("term,category,source\nkitchen,domestic_interior,curated\n", encoding="utf-8")
        loaded = sym_mod.load_space_lexicon(target)
        assert loaded.ok, loaded.diagnostics
        assert loaded.unwrap() == {"kitchen": "domestic_interior"}

    def test_unknown_category_skipped_with_warning(self, tmp_path: Path) -> None:
        target = tmp_path / "actor.csv"
        target.write_text("term,category\nsheriff,space_marshal\nmother,kin_family\n", encoding="utf-8")
        loaded = sym_mod.load_actor_lexicon(target)
        assert loaded.ok, loaded.diagnostics
        assert loaded.unwrap() == {"mother": "kin_family"}
        assert loaded.diagnostics[0].code == "SYMBOLIC_LEXICON_SKIPPED_ROWS"

    def test_bad_columns(self, tmp_path: Path) -> None:
        target = tmp_path / "space.csv"
        target.write_text("word,cat\nkitchen,domestic_interior\n", encoding="utf-8")
        loaded = sym_mod.load_space_lexicon(target)
        assert not loaded.ok
        assert loaded.diagnostics[0].code == "SYMBOLIC_LEXICON_BAD_COLUMNS"

    def test_missing_file(self, tmp_path: Path) -> None:
        res = sym_mod.classify_space("kitchen", lexicon=tmp_path / "nope.csv", backend=FakeGraph())
        assert not res.ok
        assert res.diagnostics[0].code == "SYMBOLIC_LEXICON_NOT_FOUND"

    def test_missing_backend_without_data(self, without_wordnet: None) -> None:
        # See test_nominalization: the old guard probed for an unzipped
        # corpora/wordnet directory and so never matched a real install.
        res = sym_mod.classify_space("kitchen", lexicon={})
        assert not res.ok
        assert res.diagnostics[0].code == "SYMBOLIC_DATA_MISSING"


@pytest.mark.model_integration
def test_nltk_wordnet_replay() -> None:
    """kitchen climbs to a domestic anchor; London the instance is skipped."""
    resolved = sym_mod.default_backend()
    if resolved.value is None:
        pytest.skip(f"WordNet data unavailable: {resolved.diagnostics[0].message}")
    backend = resolved.unwrap()
    assert sym_mod.classify_space("kitchen", lexicon={}, backend=backend).unwrap() == "domestic_interior"
    assert sym_mod.classify_space("london", lexicon={}, backend=backend).unwrap() == "unclassified"
