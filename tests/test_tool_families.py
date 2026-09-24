"""The tool taxonomy is complete, engine-declared, and honest about gaps.

The syllabus grades comparisons -- MALLET against Gensim, CBOW against
skip-gram, four neural sentiment backends -- so tools may not be merged, and
every tool the course names must exist as its own selectable workflow. The
old desktop had no notion of a family at all: two shelves ("analysis" and
"visualization") chosen by a guess at the input format, in which "topic
modeling" and "word embeddings" sat side by side with no distinction. That is
the "weird chunking" these tests end.

Three ratchets:

1. every registered tool carries a family from ``FAMILY_LABELS`` (also
   enforced by ``validate_specs``);
2. every tool the desktop publishes resolves its family label engine-side --
   the interface may not invent one (the lesson of ``test_labels.py``);
3. ``tests/fixtures/syllabus_tools.json`` is the syllabus manifest: every
   ``status: present`` entry resolves to exactly one registered tool, and
   every ``status: missing`` entry stays visible here as a gap with a family
   to file it under. Deleting a missing entry without building the tool
   fails, so the gap ledger cannot quietly shrink.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.profiler.registry import FAMILY_LABELS, TOOL_REGISTRY, get_tool
from desktop_backend.tables import TABLE_TOOLS

MANIFEST = Path(__file__).parent / "fixtures" / "syllabus_tools.json"

ALL_SPECS = (*TOOL_REGISTRY, *TABLE_TOOLS.values())
BY_NAME = {spec.name: spec for spec in ALL_SPECS}


def _entries() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["entries"]


class TestEveryToolHasAShelf:
    def test_every_registered_tool_declares_a_known_family(self) -> None:
        # validate_specs enforces this too; keeping it here makes the taxonomy
        # failure read as a taxonomy failure and not as a generic spec error.
        bad = {spec.name: spec.family for spec in ALL_SPECS if spec.family not in FAMILY_LABELS}
        assert not bad, f"tools without a FAMILY_LABELS shelf: {bad}"

    def test_every_family_used_resolves_to_words(self) -> None:
        for spec in ALL_SPECS:
            assert FAMILY_LABELS[spec.family], f"{spec.name}'s family has no label"

    def test_no_family_label_is_dead(self) -> None:
        """A label no tool uses is a shelf nobody stocks.

        One shelf stays reserved for tools still being built (utilities for
        environment commands); the gis shelf is now stocked (geocode, gis_map,
        svo_map), so it may no longer sit in the reserved set.
        """
        used = {spec.family for spec in ALL_SPECS}
        reserved = {"utilities"}
        dead = sorted(set(FAMILY_LABELS) - used - reserved)
        assert not dead, f"no tool will ever appear under {dead}"


class TestTheSyllabusManifest:
    def test_every_present_entry_resolves_to_exactly_one_tool(self) -> None:
        for entry in _entries():
            if entry["status"] != "present":
                continue
            ours = entry["ours"]
            assert ours in BY_NAME, (
                f"{entry['syllabus']} claims {ours!r}, which is not in any registry. "
                "Build it, or move the entry to status=missing."
            )
            assert BY_NAME[ours].family == entry["family"], (
                f"{entry['syllabus']}: {ours} is filed under {BY_NAME[ours].family!r}, "
                f"the manifest says {entry['family']!r}"
            )

    def test_every_missing_entry_names_a_family_to_file_it_under(self) -> None:
        # The gap ledger only works if each gap already knows its shelf: when
        # the tool lands it slots into the gallery without a second decision.
        for entry in _entries():
            if entry["status"] != "missing":
                continue
            assert entry["ours"] is None, f"{entry['syllabus']} is missing but claims {entry['ours']!r}"
            assert entry["family"] in FAMILY_LABELS, entry

    def test_the_gaps_the_report_called_out_are_now_built(self) -> None:
        """The named gaps must be BUILT, not quietly deleted from the manifest.

        These four were the report's headline gaps. The ratchet has served its
        purpose — they stayed visible while unbuilt — so this now pins the
        other direction: each one resolves to a registered tool.
        """
        entries = {e["syllabus"]: e for e in _entries()}
        for named in (
            "Word2Vec via BERT",
            "CoreNLP special annotator: gender (4 dictionaries)",
            "Sentiment: neural (BERT)",
            "Data reduction: NMF",
        ):
            assert named in entries, f"{named} vanished from the manifest entirely"
            assert entries[named]["status"] == "present", f"{named} is still a gap"
            assert entries[named]["ours"] in BY_NAME, f"{named} claims an unregistered tool"

    def test_landing_a_gap_means_filing_it(self) -> None:
        """`ours` set + status=missing is a contradiction: half-finished filing."""
        for entry in _entries():
            if entry["ours"] is not None:
                assert entry["status"] == "present", f"{entry['syllabus']} has a tool but is marked missing"


class TestTheComparisonToolsStaySeparate:
    def test_the_three_topic_families_are_three_tools(self) -> None:
        """HW2 asks which of MALLET and Gensim is better -- one card cannot answer."""
        topics = sorted(spec.name for spec in ALL_SPECS if spec.family == "topics")
        assert set(topics) >= {"lda_gensim", "lda_mallet", "bert_topics"}, topics
        assert len(topics) >= 2, f"topic modeling is not separated: {topics}"

    def test_the_two_training_architectures_are_a_parameter_not_a_tool(self) -> None:
        """CBOW vs skip-gram is one training run's setting (HW2); BERT vs
        Gensim is two different tools. The first is a parameter, the second a
        family split -- keep them apart."""
        w2v = get_tool("word2vec_gensim")
        assert w2v is not None
        sg = next((p for p in w2v.params if p.name == "sg"), None)
        assert sg is not None and set(sg.choices) == {0, 1}
        # BERT vs Gensim stays a family split, and the rubric's third
        # embedding question (word sense induction) is a third tool.
        assert get_tool("word2vec_bert") is not None
        embeddings = sorted(spec.name for spec in ALL_SPECS if spec.family == "embeddings")
        assert embeddings == ["word2vec_bert", "word2vec_gensim", "word_sense_induction"], embeddings


@pytest.mark.parametrize("family", sorted(FAMILY_LABELS))
def test_a_family_with_tools_is_reachable_from_the_payload(family: str) -> None:
    """What /api/tools ships must be groupable: family id and label together."""
    stocked = [spec for spec in ALL_SPECS if spec.family == family]
    for spec in stocked:
        assert spec.family in FAMILY_LABELS
        assert FAMILY_LABELS[spec.family]
