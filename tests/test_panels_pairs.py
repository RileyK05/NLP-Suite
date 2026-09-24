"""Pairs and networks: co-occurrence, document similarity, SVO overlap, graphs.

Fixtures use the real tables' columns (see the audit of the 87-speech
corpus). What these pin:

* co-occurrence is read per PAIR, summed over documents -- Word 1 alone means
  nothing, because it is only the alphabetically first word of the pair;
* PPMI is computed from the table's own marginals, and checked by hand here;
* function words and thin evidence are left out by default, and said so;
* a network carries its edges, every edge joins two drawn nodes, and an edge's
  evidence resolves against the published data (the bug that shipped first:
  edges were built and never attached, so every network drew bare nodes);
* a document heatmap is symmetric, dated, and leaves its diagonal blank;
* svo_compare's documents are named when the run carries names.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from core.result import Result
from core.viz.panels_pairs import PAIRS_PANELS
from core.viz.panelspec import PanelDefinition, PreparedPanel, Provenance

BY_NAME = {panel.name: panel for panel in PAIRS_PANELS}


def build(name: str, frame: pd.DataFrame, params: dict[str, Any] | None = None) -> Result[PreparedPanel]:
    definition: PanelDefinition = BY_NAME[name]
    return definition.build(
        frame, definition.defaults() | (params or {}), Provenance(tool=definition.tool, panel=name, source="t.csv")
    )


def ok(name: str, frame: pd.DataFrame, params: dict[str, Any] | None = None) -> PreparedPanel:
    result = build(name, frame, params)
    assert result.ok, [str(d) for d in result.diagnostics]
    return result.unwrap()


def rows_behind(panel: PreparedPanel, filters: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    selected = panel.data
    for column, value in filters:
        selected = selected[selected[column].astype(str) == value]
    return selected


def cooccurrence_frame() -> pd.DataFrame:
    """cooccurrences.csv rows: Word 1 < Word 2, Count = sentences per document."""
    rows = [
        ("social", "security", 12, "1"),
        ("social", "security", 9, "2"),
        ("states", "united", 30, "1"),
        ("states", "united", 25, "2"),
        ("of", "united", 40, "1"),
        ("peace", "war", 11, "1"),
        ("peace", "year", 2, "2"),
        ("security", "year", 10, "2"),
    ]
    return pd.DataFrame(
        [
            {
                "Word 1": min(a, b),
                "Word 2": max(a, b),
                "Count": count,
                "Document ID": doc,
                "Document": f"19{doc}0-01-01_x y_sotu.txt",
            }
            for a, b, count, doc in rows
        ]
    )


class TestCooccurrenceAssociation:
    def test_pairs_are_summed_over_documents(self) -> None:
        panel = ok("ngram_cooccurrence_association", cooccurrence_frame(), {"min-total": 1})
        states = next(m for m in panel.marks if m.key == "states::united")
        assert states.x == pytest.approx(math.log10(55))
        assert "2 of 2" in states.evidence.describe

    def test_ppmi_uses_the_tables_own_marginals(self) -> None:
        """By hand, over the unfiltered table (function-word pairs included,
        since they are part of every marginal)."""
        panel = ok("ngram_cooccurrence_association", cooccurrence_frame(), {"min-total": 1})
        totals = {
            "social security": 21,
            "states united": 55,
            "of united": 40,
            "peace war": 11,
            "peace year": 2,
            "security year": 10,
        }
        grand = sum(totals.values())
        mass = {"social": 21, "security": 31, "states": 55, "united": 95, "of": 40, "peace": 13, "war": 11, "year": 12}
        expected = max(0.0, math.log2((21 / grand) / ((mass["social"] / grand) * (mass["security"] / grand))))
        social = next(m for m in panel.marks if m.key == "security::social")
        assert social.y == pytest.approx(expected)

    def test_function_word_pairs_are_hidden_by_default(self) -> None:
        panel = ok("ngram_cooccurrence_association", cooccurrence_frame(), {"min-total": 1})
        assert "of::united" not in {m.key for m in panel.marks}
        shown = ok(
            "ngram_cooccurrence_association", cooccurrence_frame(), {"min-total": 1, "hide-function-words": False}
        )
        assert "of::united" in {m.key for m in shown.marks}

    def test_thin_evidence_is_left_out_and_said_so(self) -> None:
        result = build("ngram_cooccurrence_association", cooccurrence_frame(), {"min-total": 5})
        assert "peace::year" not in {m.key for m in result.unwrap().marks}
        assert any(d.code == "PANEL_FREQUENCY_FILTERED" for d in result.diagnostics)

    def test_evidence_selects_the_pair(self) -> None:
        panel = ok("ngram_cooccurrence_association", cooccurrence_frame(), {"min-total": 1})
        for mark in panel.marks:
            assert len(rows_behind(panel, mark.evidence.filters)) >= 1

    def test_the_label_says_the_pair_is_unordered(self) -> None:
        panel = ok("ngram_cooccurrence_association", cooccurrence_frame(), {"min-total": 1})
        assert next(m for m in panel.marks if m.key == "states::united").label == "states · united"


class TestCooccurrencePartners:
    def test_the_default_word_is_a_content_word(self) -> None:
        panel = ok("ngram_cooccurrence_partners", cooccurrence_frame(), {"min-total": 1})
        assert panel.title == 'Partners of "united"'
        assert {m.label for m in panel.marks} == {"states"}, "'of' is a function word"

    def test_a_partner_below_the_floor_is_left_out(self) -> None:
        """PPMI's rare-word bias: on the real corpus 'year' was topped by names seen once."""
        panel = ok("ngram_cooccurrence_partners", cooccurrence_frame(), {"word": "year", "min-total": 5})
        assert {m.label for m in panel.marks} == {"security"}

    def test_an_absent_word_is_refused_with_suggestions(self) -> None:
        result = build("ngram_cooccurrence_partners", cooccurrence_frame(), {"word": "zebra"})
        assert not result.ok
        assert "Some words that do" in result.diagnostics[0].message


class TestNetworks:
    def test_the_cooccurrence_network_has_edges_between_its_nodes(self) -> None:
        panel = ok("ngram_cooccurrence_network", cooccurrence_frame(), {"min-total": 1})
        keys = {m.key for m in panel.marks}
        assert panel.edges, "edges were once built and never attached"
        assert all(edge.source in keys and edge.target in keys for edge in panel.edges)
        assert all(0.0 <= m.x <= 1.0 and 0.0 <= m.y <= 1.0 for m in panel.marks)

    def test_the_network_is_built_on_hub_words_not_rare_dyads(self) -> None:
        """Regression: taking the top pairs by PPMI drew forty disconnected
        dyads on the real corpus, because the rarest pairs score highest and
        share no words. Hub words first, then their links, is a network."""
        rows = [("social", "security", 40), ("security", "tax", 30), ("social", "tax", 25), ("budget", "tax", 35)]
        rows += [(f"rare{i}", f"once{i}", 10) for i in range(6)]
        frame = pd.DataFrame(
            [
                {"Word 1": min(a, b), "Word 2": max(a, b), "Count": c, "Document ID": "1", "Document": "a.txt"}
                for a, b, c in rows
            ]
        )
        panel = ok("ngram_cooccurrence_network", frame, {"min-total": 1, "top-n": 4})
        assert {m.key for m in panel.marks} == {"social", "security", "tax", "budget"}
        degree = {m.key: 0 for m in panel.marks}
        for edge in panel.edges:
            degree[edge.source] += 1
            degree[edge.target] += 1
        assert max(degree.values()) >= 2, "a word with two links: a network, not a set of dyads"
        assert all(m.labelled for m in panel.marks), "a node without its word is a dot"

    def test_edge_evidence_resolves(self) -> None:
        panel = ok("ngram_cooccurrence_network", cooccurrence_frame(), {"min-total": 1})
        for edge in panel.edges:
            assert len(rows_behind(panel, edge.evidence.filters)) >= 1


def similarity_frame() -> pd.DataFrame:
    names = [
        ("1", "1934-01-03_franklin d roosevelt_sotu.txt"),
        ("2", "1962-01-11_john f kennedy_sotu.txt"),
        ("3", "2024-03-07_joseph r biden_sotu.txt"),
    ]
    values = {("1", "2"): 71.0, ("1", "3"): 55.0, ("2", "3"): 64.0}
    return pd.DataFrame(
        [
            {
                "Document ID A": a,
                "Document A": dict(names)[a],
                "Document ID B": b,
                "Document B": dict(names)[b],
                "Similarity": value,
                "Band": "similar",
            }
            for (a, b), value in values.items()
        ]
    )


class TestDocumentSimilarity:
    def test_heatmap_is_symmetric_dated_and_blank_on_the_diagonal(self) -> None:
        panel = ok("doc_similarity_heatmap", similarity_frame())
        assert panel.y_categories == ("1934 Roosevelt", "1962 Kennedy", "2024 Biden")
        cells = {(int(m.x), int(m.y)): m.value for m in panel.marks}
        assert cells[(0, 1)] == cells[(1, 0)] == 71.0
        assert not any(x == y for x, y in cells), "a document compared with itself is not a finding"

    def test_neighbours_link_each_document_to_its_closest(self) -> None:
        panel = ok("doc_similarity_neighbours", similarity_frame(), {"k": 1})
        assert {frozenset((e.source, e.target)) for e in panel.edges} >= {
            frozenset(("1934-01-03_franklin d roosevelt_sotu.txt", "1962-01-11_john f kennedy_sotu.txt"))
        }


class TestSvoCompare:
    def frame(self, named: bool) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "Doc A": ["1", "1", "2"],
                "Doc B": ["2", "3", "3"],
                "Common Triples": [1, 0, 2],
                "Subject Jaccard": [0.12, 0.14, 0.2],
                "Verb Jaccard": [0.09, 0.06, 0.1],
                "Object Jaccard": [0.07, 0.1, 0.09],
                "Triple Jaccard": [0.004, 0.0, 0.01],
            }
        )
        if named:
            names = {
                "1": "1934-01-03_franklin d roosevelt_sotu.txt",
                "2": "1935-01-04_franklin d roosevelt_sotu.txt",
                "3": "1962-01-11_john f kennedy_sotu.txt",
            }
            frame["Document A"] = frame["Doc A"].map(names)
            frame["Document B"] = frame["Doc B"].map(names)
        return frame

    def test_documents_are_named_when_the_run_carries_names(self) -> None:
        panel = ok("svo_compare_heatmap", self.frame(named=True))
        assert panel.y_categories == ("1934 Roosevelt", "1935 Roosevelt", "1962 Kennedy")

    def test_an_older_run_without_names_falls_back_to_ids(self) -> None:
        panel = ok("svo_compare_heatmap", self.frame(named=False))
        assert panel.y_categories[0] == "Doc 1"

    def test_the_neighbours_network_has_edges(self) -> None:
        panel = ok("svo_compare_neighbours", self.frame(named=True), {"k": 1})
        assert panel.edges
