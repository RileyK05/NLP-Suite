"""The verb-profile figures preserve each speech's denominators and evidence."""

from __future__ import annotations

import pandas as pd

from core.viz.panel_plotters import build_panel_figure
from core.viz.panels import prepare_panel
from core.viz.panelspec import Source
from desktop_backend.live import LiveResult
from desktop_backend.live_panels import live_panels_offered


def verb_frame() -> pd.DataFrame:
    rows = [
        # 1930 Alpha: 2 active, 1 passive, 1 unknown; one of each modality.
        ("1", "1930_alpha_sotu.txt", "active", "ability"),
        ("1", "1930_alpha_sotu.txt", "active", "possibility"),
        ("1", "1930_alpha_sotu.txt", "passive", "obligation"),
        ("1", "1930_alpha_sotu.txt", "unknown", "none"),
        # 1940 Beta: 1 active, 2 passive; one obligation verb.
        ("2", "1940_beta_sotu.txt", "active", "none"),
        ("2", "1940_beta_sotu.txt", "passive", "obligation"),
        ("2", "1940_beta_sotu.txt", "passive", "none"),
        # Too short for the scatter's default threshold.
        ("3", "1950_gamma_sotu.txt", "passive", "obligation"),
    ]
    return pd.DataFrame(rows, columns=["Document ID", "Document", "Voice", "Modality"])


def panel(name: str, params: dict[str, object] | None = None):
    result = prepare_panel(name, verb_frame(), params or {}, source=Source(path="verbs.csv"))
    assert result.ok, [diagnostic.message for diagnostic in result.diagnostics]
    return result.unwrap()


def test_voice_mix_is_a_per_speech_composition_and_click_filters_its_table_row() -> None:
    prepared = panel("verb_analysis_voice_mix")
    first_speech = prepared.data[prepared.data["Document ID"] == "1"]
    assert set(first_speech["Category"]) == {"Active", "Passive", "Unknown voice"}
    assert first_speech["Share"].sum() == 1.0

    passive = next(mark for mark in prepared.marks if mark.key == "1:voice:passive")
    assert passive.evidence.count == 1
    assert passive.evidence.filters == (("Document ID", "1"), ("Category", "Passive"))
    matching = prepared.data[(prepared.data["Document ID"] == "1") & (prepared.data["Category"] == "Passive")]
    assert len(matching) == 1
    assert matching.iloc[0]["Count"] == passive.evidence.count


def test_modality_mix_keeps_no_modality_as_an_explicit_category() -> None:
    prepared = panel("verb_analysis_modality_mix")
    first_speech = prepared.data[prepared.data["Document ID"] == "1"]
    assert set(first_speech["Category"]) == {
        "Ability",
        "Possibility",
        "Obligation",
        "No modality",
    }
    assert first_speech["Share"].sum() == 1.0
    assert any(mark.group == "No modality" for mark in prepared.marks)


def test_agency_commitment_scatter_uses_speech_rates_and_shows_sample_size() -> None:
    prepared = panel("verb_analysis_agency_commitment", {"minimum-verbs": 3, "group-by": "decade"})
    by_key = {mark.key: mark for mark in prepared.marks}
    assert set(by_key) == {"1", "2"}
    assert by_key["1"].x == 25.0
    assert by_key["1"].y == 25.0
    assert by_key["1"].size == 4
    assert by_key["1"].group == "1930s"
    assert len(prepared.annotations) == 2
    assert "not statistical cutoffs" in prepared.notes[1]


def test_agency_commitment_reports_hidden_short_speeches_and_can_colour_by_speaker() -> None:
    result = prepare_panel(
        "verb_analysis_agency_commitment",
        verb_frame(),
        {"minimum-verbs": 2, "group-by": "speaker"},
        source=Source(path="verbs.csv"),
    )
    assert result.ok
    assert any(item.code == "PANEL_SHORT_SPEECHES_HIDDEN" for item in result.diagnostics)
    assert {mark.group for mark in result.unwrap().marks} == {"Alpha", "Beta"}


def test_new_panels_render_through_the_existing_interactive_shapes() -> None:
    for name in (
        "verb_analysis_voice_mix",
        "verb_analysis_modality_mix",
        "verb_analysis_agency_commitment",
    ):
        values = {"minimum-verbs": 1} if name == "verb_analysis_agency_commitment" else None
        figure = build_panel_figure(panel(name, values))
        assert figure.ok
        assert figure.unwrap().data


def test_live_verb_run_offers_all_three_new_views_as_tabs() -> None:
    result = LiveResult(tool="verb_analysis", ok=True, frames={"verbs.csv": verb_frame()})
    names = {panel["name"] for panel in live_panels_offered(result)}
    assert {
        "verb_analysis_voice_mix",
        "verb_analysis_modality_mix",
        "verb_analysis_agency_commitment",
    } <= names
