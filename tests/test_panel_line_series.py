"""Annual panel exports preserve absent-year gaps and observed zeroes."""

import pandas as pd
import pytest

from core.viz.panel_plotters import build_panel_figure
from core.viz.panelspec import Evidence, PanelMark, PreparedPanel, Provenance


def _mark(year: int, value: float) -> PanelMark:
    return PanelMark(
        key=f"war-{year}",
        label=str(year),
        x=float(year),
        y=value,
        group="war",
        evidence=Evidence(scope="rows", filters=(("Year", str(year)),), count=1, describe=f"war in {year}"),
    )


def test_export_breaks_line_at_no_exposure_year_but_retains_observed_zero() -> None:
    pytest.importorskip("plotly")
    panel = PreparedPanel(
        panel="test_year_series",
        shape="line_series",
        title="war over time",
        marks=(_mark(2001, 4), _mark(2002, 0), _mark(2004, 6)),
        x_label="Year",
        y_label="Occurrences per million",
        groups=("war",),
        provenance=Provenance(tool="ngram_viewer", panel="test_year_series", source="ngram_series.csv"),
        data=pd.DataFrame({"Year": [2001, 2002, 2004]}),
    )
    figure = build_panel_figure(panel).unwrap()
    assert len(figure.data) == 1
    line = figure.data[0]
    assert list(line.x) == [2001.0, 2002.0, None, 2004.0]
    assert list(line.y) == [4, 0, None, 6]
    assert line.mode == "lines+markers"
    assert list(line.hovertext) == ["war in 2001", "war in 2002", None, "war in 2004"]
