"""Culturomics panel preparation preserves the dated row evidence."""

from __future__ import annotations

import pandas as pd

from core.viz.panels_ngram_viewer import NGRAM_FREQUENCY_OVER_TIME, ngram_frequency_over_time
from core.viz.panelspec import Provenance


def test_ngram_panel_is_line_series_and_maps_selected_metric() -> None:
    source = pd.DataFrame(
        {
            "N-gram": ["war", "war", "peace"],
            "Year": [1900, 1901, 1900],
            "Count": [2, 0, 1],
            "Per Million": [10.0, 0.0, 5.0],
            "Share of Documents": [1.0, 0.0, 0.5],
        }
    )
    provenance = Provenance(source="ngram-viewer.csv", tool="ngram_viewer", panel="ngram_frequency_over_time")
    result = ngram_frequency_over_time(source, {"metric": "Count"}, provenance)
    assert result.ok, result.diagnostics
    panel = result.unwrap()
    assert panel.shape == "line_series"
    assert panel.groups == ("peace", "war")
    war_zero = next(mark for mark in panel.marks if mark.group == "war" and mark.x == 1901)
    assert war_zero.y == 0
    assert war_zero.evidence.as_query() == {"N-gram": "war", "Year": "1901"}
    assert war_zero.evidence.count == 0
    assert NGRAM_FREQUENCY_OVER_TIME.shape == "line_series"


def test_smoothed_point_keeps_raw_hits_and_exposure_in_its_evidence() -> None:
    source = pd.DataFrame(
        [
            {
                "N-gram": "war",
                "Year": 1901,
                "Count": 0.5,
                "Per Million": 250000.0,
                "Share of Documents": 0.0,
                "Raw Count": 0,
                "Raw Per Million": 0.0,
                "Corpus Tokens": 4,
                "Corpus Documents": 2,
                "Documents With Hit": 0,
            }
        ]
    )
    provenance = Provenance(
        source="ngram_series.csv",
        tool="ngram_viewer",
        panel="ngram_frequency_over_time",
        settings={"smooth": 3},
    )
    panel = ngram_frequency_over_time(source, {"metric": "Count"}, provenance).unwrap()
    mark = panel.marks[0]
    assert mark.y == 0.5
    assert mark.evidence.count == 0
    assert "0 raw matches" in mark.evidence.describe
    assert "2 documents / 4 tokens" in mark.evidence.describe
