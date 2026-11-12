"""Domain figures preserve their denominator and the rows behind each mark."""

from __future__ import annotations

import pandas as pd
import pytest

from core.gis.mapping import heatmap_html, kml
from core.viz.panel_plotters import build_panel_figure
from core.viz.panels import panels_for_tool, prepare_panel
from core.viz.panelspec import PanelMark, PreparedPanel, Source
from desktop_backend.panels import public_panel

GEO = pd.DataFrame(
    {
        "Place": ["Paris", "Lyon", "London", "Dateline E", "Dateline W", "Real origin", "Unknown", "Bad latitude"],
        "Lat": [48.85, 45.76, 51.51, 90, 90, 0, 0, 91],
        "Lon": [2.35, 4.84, -0.13, 180, -180, 0, 0, 10],
        "Status": ["OK", "OK", "OK", "OK", "OK", "OK", "UNKNOWN", "OK"],
        "Mentions": [8, 2, 4, 1, 1, 3, 100, 3],
    }
)
SENTENCES = pd.DataFrame(
    {
        "Document ID": [1, 1, 1, 1, 2],
        "Document": ["a.txt", "a.txt", "a.txt", "a.txt", "b.txt"],
        "Sentence ID": [1, 2, 6, 10, 1],
        "Sentence": ["First", "Second", "Middle", "Last", "Only"],
        "Label": ["Very negative", "Negative", "neutral", "Positive", "Very positive"],
        "Compound": [-0.8, -0.6, 0, 0.7, 0.9],
    }
)


def _draw(name: str, frame: pd.DataFrame, **params: object) -> PreparedPanel:
    result = prepare_panel(name, frame, params, source=Source(path="fixture.csv"))
    assert result.ok, result.diagnostics
    return result.unwrap()


def _rows(panel: PreparedPanel, mark: PanelMark) -> pd.DataFrame:
    rows = panel.data
    for column, value in mark.evidence.filters:
        rows = rows[rows[column].astype(str).eq(value)]
    assert len(rows) == mark.evidence.count
    return rows


def test_pin_map_groups_coincident_names_and_keeps_real_zero_zero() -> None:
    panel = _draw("geocode_pins", GEO)
    assert "Pin area shows mentions" in panel.subtitle
    assert sum(mark.size or 0 for mark in panel.marks) == 19
    assert _rows(panel, next(mark for mark in panel.marks if mark.x == mark.y == 0))["Place"].tolist() == [
        "Real origin"
    ]
    edge = next(mark for mark in panel.marks if mark.x == -180)
    assert edge.y == 90
    assert set(_rows(panel, edge)["Place"]) == {"Dateline E", "Dateline W"}
    older = _draw("geocode_pins", GEO.drop(columns="Mentions"))
    assert sum(mark.size or 0 for mark in older.marks) == 6
    assert "resolved names" in older.subtitle


def test_concentration_sums_mentions_and_supplies_the_same_cells_to_renderers() -> None:
    panel = _draw("geocode_concentration", GEO, cell_size="10")
    assert panel.map_cell_degrees == 10
    paris = next(mark for mark in panel.marks if mark.x == 5 and mark.y == 45)
    assert paris.value == 10
    assert set(_rows(panel, paris)["Place"]) == {"Paris", "Lyon"}
    assert sum(mark.value or 0 for mark in panel.marks) == 19
    browser = public_panel(panel, [])
    assert browser["mapCellDegrees"] == 10
    assert len(browser["mapLand"]) > 100
    figure = build_panel_figure(panel).unwrap()
    assert len(figure.data) == 1 + len(panel.marks)
    assert figure.data[0].fill == "toself"
    assert figure.layout.yaxis.scaleanchor == "x"
    assert all(trace.hoveron == "fills" for trace in figure.data[1:])


def test_unresolved_names_stay_visible_in_coverage_but_never_reach_maps() -> None:
    panel = _draw("geocode_coverage", GEO)
    assert {mark.label: len(_rows(panel, mark)) for mark in panel.marks} == {
        "Resolved": 6,
        "No match": 1,
        "Invalid coordinates": 1,
    }
    no_maps = prepare_panel("geocode_pins", GEO[GEO["Status"].eq("UNKNOWN")], None)
    assert no_maps.value is None and no_maps.diagnostics[-1].code == "PANEL_NO_MAP_COORDS"
    standalone = heatmap_html(GEO, weight_col="Mentions", cell_degrees=10)
    assert standalone.ok
    html = standalone.unwrap()
    assert "<svg" in html and "Total 10 from 2 rows" in html
    assert "Unknown" not in html and "Bad latitude" not in html
    assert "Unknown" not in kml(GEO).unwrap()


@pytest.mark.parametrize(
    "tool",
    [
        "polarity_vader",
        "polarity_distilbert_sst2",
        "polarity_spacy",
        "polarity_stanza",
        "polarity_corenlp",
    ],
)
def test_sentence_figures_resolve_the_same_evidence_for_every_backend(tool: str) -> None:
    mix = _draw(f"{tool}_sentence_mix", SENTENCES)
    assert mix.value_domain == (0, 100)
    for y in (0, 1):
        assert sum(mark.value or 0 for mark in mix.marks if mark.y == y) == 100
    assert {mark.label: len(_rows(mix, mark)) for mark in mix.marks if mark.y == 0} == {
        "a · Negative": 2,
        "a · Neutral": 1,
        "a · Positive": 1,
    }

    progress = _draw(f"{tool}_sentence_progression", SENTENCES, document="a.txt", sections="5")
    assert progress.value_domain == (-1, 1)
    assert [mark.x for mark in progress.marks] == [0, 2, 4]
    assert [mark.value for mark in progress.marks] == pytest.approx([-0.7, 0, 0.7])
    assert [_rows(progress, mark)["Sentence ID"].tolist() for mark in progress.marks] == [[1, 2], [6], [10]]
    plot = build_panel_figure(progress).unwrap()
    assert (plot.data[0].zmin, plot.data[0].zmax) == (-1, 1)
    assert plot.data[0].z[0][1] is None

    spread = _draw(f"{tool}_sentence_distribution", SENTENCES)
    assert len(spread.marks) == len(SENTENCES)
    assert all(mark.evidence.scope == "sentences" for mark in spread.marks)
    assert [_rows(spread, mark).iloc[0]["Sentence"] for mark in spread.marks] == SENTENCES["Sentence"].tolist()


def test_lexicon_scatter_reports_coverage_and_leaves_unscored_documents_out() -> None:
    frame = pd.DataFrame(
        {
            "Document ID": [1, 2],
            "Document": ["a.txt", "b.txt"],
            "Tokens": [100, 50],
            "Hits": [12, 0],
            "Valence": [6, 5],
        }
    )
    panel = _draw("valence_anew_lexicon_coverage", frame)
    assert panel.marks[0].x == 12
    assert panel.marks[0].y == 6
    assert "1 of 2 documents" in panel.subtitle
    assert _rows(panel, panel.marks[0]).iloc[0]["Hits"] == 12


def test_other_tools_draw_their_own_evidence() -> None:
    locations = pd.DataFrame(
        {
            "Document ID": [1, 1, 2],
            "Document": ["a.txt", "a.txt", "b.txt"],
            "Location": ["Paris", "Paris", "Rome"],
            "Count": [4, 2, 1],
        }
    )
    places = _draw("ner_places_by_document", locations)
    assert [mark.value for mark in places.marks] == [6, 1]
    assert len(_rows(places, places.marks[0])) == 2
    svo = pd.DataFrame({"Place": ["Rome", "Rome"], "Verb": ["Saw", "saw"], "Subject": ["a", "b"], "Object": ["b", "a"]})
    verbs = _draw("svo_map_verbs", svo)
    assert verbs.marks[0].value == 2
    assert len(_rows(verbs, verbs.marks[0])) == 2

    topics = pd.DataFrame(
        {
            "Document ID": [1, 2],
            "Document": ["a.txt", "b.txt"],
            "Topic": [0, 1],
            "Distance": [1.25, 0.5],
            "Dominant topic": [0, 1],
            "Contribution": [0.55, 0.9],
            "Topic keywords": ["a", "b"],
        }
    )
    assert [mark.x for mark in _draw("lda_gensim_topic_fit", topics).marks] == [0.55, 0.9]
    assert [mark.x for mark in _draw("topic_clusters_bert_topic_fit", topics).marks] == [1.25, 0.5]

    kwic = pd.DataFrame(
        {
            "Hit": ["river", "river"],
            "Left Context": ["along the", "THE"],
            "Right Context": ["flows north", ""],
            "Document ID": [1, 2],
            "Sentence ID": [1, 1],
        }
    )
    neighbors = _draw("kwic_neighbor_balance", kwic)
    assert [(mark.label, mark.x) for mark in neighbors.marks] == [("the", 2), ("flows", 1)]
    assert sum(len(_rows(neighbors, mark)) for mark in neighbors.marks) == 3


def test_default_geocode_run_publishes_map_and_coverage_images() -> None:
    pytest.importorskip("seaborn")
    from core.viz.run_figures import run_figures

    figures, diagnostics = run_figures("geocode", {"geocoded.csv": GEO}, dpi=80)
    assert not [notice for notice in diagnostics if notice.code in {"RUN_FIGURE_FAILED", "RUN_FIGURE_SKIPPED"}]
    artifacts = {figure.filename: figure.data for figure in figures}
    for panel in panels_for_tool("geocode"):
        assert artifacts[f"figures/{panel.name}.png"].startswith(b"\x89PNG")
        assert b"<svg" in artifacts[f"figures/{panel.name}.svg"]
