"""Model figures: dendrogram, eras, components, topic words, stability matches.

Fixtures mirror the real tables (shape_hc_merges.csv is scipy linkage in
table form; the shape features are "<Measure> <slice>"). What these pin:

* the dendrogram reads the linkage ids correctly (below n a document, n and
  above a merge), puts every leaf once, down the page in tree order, each
  merge at its height, and joins parent to child with elbow links;
* grouping-by-date figures place every document at its year, one row per
  cluster or topic;
* loadings are drawn along the speech, one small multiple per measure;
* stability is a topic x seed heatmap whose cells resolve to one match row.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from core.result import Result
from core.viz.panels_models import MODELS_PANELS
from core.viz.panelspec import PanelDefinition, PreparedPanel, Provenance

BY_NAME = {panel.name: panel for panel in MODELS_PANELS}
NAMES = [
    "1934-01-03_franklin d roosevelt_sotu.txt",
    "1962-01-11_john f kennedy_sotu.txt",
    "1985-02-06_ronald reagan_sotu.txt",
    "2024-03-07_joseph r biden_sotu.txt",
]


def build(name: str, frame: pd.DataFrame, params: dict[str, Any] | None = None) -> Result[PreparedPanel]:
    definition: PanelDefinition = BY_NAME[name]
    return definition.build(
        frame, definition.defaults() | (params or {}), Provenance(tool=definition.tool, panel=name, source="t.csv")
    )


def ok(name: str, frame: pd.DataFrame, params: dict[str, Any] | None = None) -> PreparedPanel:
    result = build(name, frame, params)
    assert result.ok, [str(d) for d in result.diagnostics]
    return result.unwrap()


def merges() -> pd.DataFrame:
    """Four documents: (0,1) join at 10, (2,3) at 20, then both clusters at 50."""
    return pd.DataFrame(
        [
            {"Merge": 1, "Left": NAMES[0], "Right": NAMES[1], "Height": 10.0, "Left id": 0, "Right id": 1},
            {"Merge": 2, "Left": NAMES[2], "Right": NAMES[3], "Height": 20.0, "Left id": 2, "Right id": 3},
            {"Merge": 3, "Left": "cluster 1", "Right": "cluster 2", "Height": 50.0, "Left id": 4, "Right id": 5},
        ]
    )


class TestDendrogram:
    def test_every_document_is_a_named_leaf_once(self) -> None:
        panel = ok("shape_hc_dendrogram", merges())
        leaves = [m for m in panel.marks if m.group == "Document"]
        assert sorted(m.label for m in leaves) == ["1934 Roosevelt", "1962 Kennedy", "1985 Reagan", "2024 Biden"]
        assert all(m.x == 0.0 for m in leaves)

    def test_merges_sit_at_their_height(self) -> None:
        panel = ok("shape_hc_dendrogram", merges())
        heights = {m.key: m.x for m in panel.marks if m.group == "Merge"}
        assert heights == {"merge:1": pytest.approx(0.2), "merge:2": pytest.approx(0.4), "merge:3": 1.0}

    def test_leaves_run_down_the_page_in_tree_order(self) -> None:
        panel = ok("shape_hc_dendrogram", merges())
        leaves = sorted((m for m in panel.marks if m.group == "Document"), key=lambda m: m.y)
        assert [m.label for m in leaves] == ["1934 Roosevelt", "1962 Kennedy", "1985 Reagan", "2024 Biden"]
        merge_one = next(m for m in panel.marks if m.key == "merge:1")
        assert merge_one.y == pytest.approx((leaves[0].y + leaves[1].y) / 2), "a merge sits between its children"

    def test_links_join_each_merge_to_its_two_children_as_elbows(self) -> None:
        panel = ok("shape_hc_dendrogram", merges())
        assert panel.edge_style == "elbow"
        assert len(panel.edges) == 6
        assert {e.target for e in panel.edges if e.source == "merge:3"} == {"merge:1", "merge:2"}

    def test_evidence_resolves_to_one_node_row(self) -> None:
        panel = ok("shape_hc_dendrogram", merges())
        for mark in panel.marks:
            ((column, value),) = mark.evidence.filters
            assert (panel.data[column] == value).sum() == 1

    def test_too_few_documents_is_refused(self) -> None:
        assert not build("shape_hc_dendrogram", merges().head(1)).ok


class TestEras:
    def assignments(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Document ID": ["1", "2", "3", "4"],
                "Document": NAMES,
                "Date": ["1934-01-03", "1962-01-11", "1985-02-06", "2024-03-07"],
                "Year": [1934, 1962, 1985, 2024],
                "Cluster": [2, 2, 1, 1],
                "Distance": [10.0, 12.0, 9.0, 11.0],
            }
        )

    def test_every_document_sits_at_its_year_in_its_clusters_row(self) -> None:
        panel = ok("shape_hc_eras", self.assignments())
        assert panel.y_categories == ("Cluster 1", "Cluster 2")
        biden = next(m for m in panel.marks if m.label == "2024 Biden")
        assert biden.y == 0.0 and biden.x == pytest.approx(2024.18, abs=0.01)


class TestComponents:
    def loadings(self) -> pd.DataFrame:
        rows = []
        for measure, scale in (("Tokens", 1.0), ("Noun Ratio", 0.01)):
            for position in range(1, 5):
                rows.append(
                    {
                        "Feature": f"{measure} {position:02d}",
                        "Component 1": scale * position,
                        "Component 2": -scale * position,
                    }
                )
        return pd.DataFrame(rows)

    def test_loadings_are_drawn_along_the_speech_per_measure(self) -> None:
        panel = ok("shape_svd_loadings", self.loadings())
        assert panel.facets == ("Tokens", "Noun Ratio")
        assert panel.groups == ("Component 1", "Component 2")
        tokens = sorted((m for m in panel.marks if m.facet == "Tokens" and m.group == "Component 1"), key=lambda m: m.x)
        assert [m.x for m in tokens] == [0.25, 0.5, 0.75, 1.0]

    def test_the_notes_warn_that_length_drives_unscaled_components(self) -> None:
        assert any("not standardised" in note for note in BY_NAME["shape_svd_loadings"].notes)

    def test_explained_labels_components_as_integers(self) -> None:
        frame = pd.DataFrame({"Component": [1.0, 2.0], "Explained variance": [0.54, 0.46], "Cumulative": [0.54, 1.0]})
        panel = ok("shape_svd_explained", frame)
        assert [m.label for m in panel.marks] == ["Component 1", "Component 2"]


class TestStability:
    def test_topics_by_seeds_resolve_to_one_match(self) -> None:
        frame = pd.DataFrame(
            {
                "Topic": [0, 1, 0, 1],
                "Seed": [101, 101, 102, 102],
                "Matched topic": [0, 1, 1, 0],
                "Jaccard": [0.82, 1.0, 0.25, 0.9],
                "Shared words": ["year, world", "america, people", "year", "people, work"],
                "Reference words": ["a", "b", "a", "b"],
            }
        )
        panel = ok("lda_stability_matches", frame)
        assert panel.x_categories == ("seed 101", "seed 102")
        assert panel.y_categories == ("Topic 0", "Topic 1")
        for mark in panel.marks:
            selected = panel.data
            for column, value in mark.evidence.filters:
                selected = selected[selected[column].astype(str) == value]
            assert len(selected) == 1
