"""Maps and diagnostics for geocoded names and narrative triples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_maps import cell_center
from core.viz.panelspec import Evidence, PanelDefinition, PanelMark, PanelParam, PreparedPanel, Provenance

_COORDS = ("Place", "Lat", "Lon")
_GEOCODE = (*_COORDS, "Status")
_BASEMAP = "Land outlines: Natural Earth 1:110m."
_SVO_NOTE = (
    "Each triple is associated with every geocoded place mentioned in its document. "
    "One triple can count at multiple places; the text may describe the event somewhere else."
)
_GRID = PanelParam(
    name="cell_size",
    type="choice",
    default="10",
    choices=("5", "10", "15", "30"),
    label="Grid cell width (degrees)",
    help="Smaller cells separate nearby places. Cells are latitude/longitude bins.",
)
_TOP_PLACES = PanelParam(
    name="places", type="int", default=12, minimum=3, maximum=30, label="Places", help="Most mapped triples first."
)
_TOP_VERBS = PanelParam(
    name="verbs", type="int", default=10, minimum=3, maximum=25, label="Verbs", help="Most frequent verbs first."
)


def _valid_coords(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["Lat"], errors="coerce").between(-90, 90) & pd.to_numeric(
        frame["Lon"], errors="coerce"
    ).between(-180, 180)


def _map_data(frame: pd.DataFrame, weight: str) -> pd.DataFrame:
    valid = _valid_coords(frame)
    if "Status" in frame:
        valid &= frame["Status"].eq("OK")
    data = frame.loc[valid].copy()
    data["Lat"] = data["Lat"].astype(float)
    data["Lon"] = data["Lon"].astype(float).replace(180.0, -180.0)
    counts = pd.to_numeric(data[weight], errors="coerce") if weight in data else pd.Series(1, index=data.index)
    data["Map weight"] = counts.where(counts.between(0, float("inf"), inclusive="neither"), 0)
    return data[data["Map weight"] > 0].copy()


def _map_marks(data: pd.DataFrame, degrees: int, unit: str) -> tuple[PanelMark, ...]:
    locations = [
        cell_center(x, y, degrees) if degrees else (x, y) for x, y in zip(data["Lon"], data["Lat"], strict=True)
    ]
    data["Map location"] = [f"{x:.12g}, {y:.12g}" for x, y in locations]
    marks = []
    for key, rows in data.groupby("Map location", sort=False):
        lon, lat = (float(part) for part in key.split(", "))
        total = float(rows["Map weight"].sum())
        names = list(dict.fromkeys(rows["Place"].astype(str)))
        label = ", ".join(names[:3])
        if len(names) > 3:
            label += f" and {len(names) - 3} more"
        location = f"{degrees}° cell at {lat:g}°, {lon:g}°" if degrees else f"{lat:.2f}°, {lon:.2f}°"
        marks.append(
            PanelMark(
                key=key,
                label=label,
                x=lon,
                y=lat,
                size=None if degrees else total,
                value=total if degrees else None,
                evidence=Evidence(
                    "rows", (("Map location", key),), len(rows), f"{label} · {total:,g} {unit} · {location}"
                ),
            )
        )
    return tuple(sorted(marks, key=lambda mark: -(mark.value or mark.size or 0)))


def _map_panel(tool: str, *, grid: bool, name: str, title: str, question: str) -> PanelDefinition:
    geocode = tool == "geocode"
    weight = {"geocode": "Mentions", "gis_map": "Weight", "svo_map": "SVO rows"}[tool]
    requires: tuple[str, ...] = (*_COORDS, weight) if tool == "svo_map" else _GEOCODE
    if tool == "gis_map":
        requires += (weight,)
    caveat = (
        "Only successful lookups with valid coordinates are mapped. Older runs without mention counts count each name once."
        if geocode
        else _SVO_NOTE
        if tool == "svo_map"
        else "The grid totals or pin sizes use the input map weights."
    )
    notes: tuple[str, ...] = (caveat, _BASEMAP)
    if grid:
        notes += (
            "Cells show totals. A degree covers less ground near the poles, so colors do not measure density per square kilometer.",
        )

    def build(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
        data = _map_data(frame, weight)
        if data.empty:
            return Result.failure(Diagnostic.error("PANEL_NO_MAP_COORDS", "No mappable places with positive counts."))
        degrees = int(params["cell_size"]) if grid else 0
        unit = (
            ("mentions" if weight in frame else "resolved names")
            if geocode
            else "triple-place pairs"
            if tool == "svo_map"
            else "weight units"
        )
        marks = _map_marks(data, degrees, unit)
        encoding = f"Color shows {unit} per {degrees}° cell" if grid else f"Pin area shows {unit}"
        omitted = len(frame) - len(data)
        omission = f" · {omitted:,} rows omitted" if omitted else ""
        return Result.success(
            PreparedPanel(
                panel=name,
                title=title,
                shape="geographic",
                marks=marks,
                data=data,
                provenance=provenance,
                x_label="",
                y_label="",
                map_cell_degrees=degrees,
                value_label=unit,
                notes=notes,
                subtitle=f"{len(data):,} place rows · {encoding}{omission}",
                width=950,
                height=588,
            )
        )

    return PanelDefinition(
        name=name,
        title=title,
        tool=tool,
        shape="geographic",
        question=question,
        summary="Counts summed in latitude/longitude cells." if grid else "Pins sized by the count at each location.",
        requires=requires,
        params=(_GRID,) if grid else (),
        build=build,
        notes=notes,
    )


_OUTCOMES = {
    "OK": "Resolved",
    "UNKNOWN": "No match",
    "UNAVAILABLE": "Provider unavailable",
}
_COVERAGE_NOTES = (
    "Coverage includes the place names sent to the geocoder. Extraction thresholds and the lookup limit can leave other names out.",
)


def _coverage(frame: pd.DataFrame, _params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    data = frame.copy()
    data["Outcome"] = data["Status"].map(_OUTCOMES).fillna("Other status")
    data.loc[data["Status"].eq("OK") & ~_valid_coords(data), "Outcome"] = "Invalid coordinates"
    totals = data["Outcome"].value_counts()
    outcomes = ("Resolved", "No match", "Provider unavailable", "Invalid coordinates", "Other status")
    marks: list[PanelMark] = []
    for outcome in outcomes:
        count = int(totals.get(outcome, 0))
        if count:
            marks.append(
                PanelMark(
                    key=outcome,
                    label=outcome,
                    x=float(count),
                    y=float(len(marks)),
                    evidence=Evidence(
                        "rows",
                        (("Outcome", outcome),),
                        count,
                        f"{outcome} · {count:,} of {len(data):,} names ({count / len(data):.1%})",
                    ),
                )
            )
    resolved = int(totals.get("Resolved", 0))
    return Result.success(
        PreparedPanel(
            panel="geocode_coverage",
            shape="ranked_bars",
            title="Geocoding coverage",
            marks=tuple(marks),
            x_label="Place names submitted",
            y_label="",
            data=data,
            provenance=provenance,
            notes=_COVERAGE_NOTES,
            subtitle=f"{resolved:,} of {len(data):,} place names resolved ({resolved / len(data):.1%})",
            height=310,
        )
    )


def _verbs_by_place(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    data = frame.dropna(subset=["Place", "Verb"]).copy()
    data["Verb"] = data["Verb"].astype(str).str.strip().str.casefold()
    data = data[data["Verb"].ne("")]
    if data.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_MAPPED_VERBS", "No mapped rows carry a verb."))
    places = tuple(str(place) for place in data["Place"].value_counts().head(params["places"]).index)
    data = data[data["Place"].isin(places)]
    verbs = tuple(data["Verb"].value_counts().head(params["verbs"]).index)
    data = data[data["Verb"].isin(verbs)]
    counts = data.groupby(["Place", "Verb"]).size()
    marks = tuple(
        PanelMark(
            key=f"{x}:{y}",
            label=f"{place} · {verb}",
            x=float(x),
            y=float(y),
            value=float(count),
            evidence=Evidence(
                "rows",
                (("Place", place), ("Verb", verb)),
                int(count),
                f"{place} · {verb} · {count:,} mapped triples",
            ),
        )
        for y, place in enumerate(places)
        for x, verb in enumerate(verbs)
        if (count := counts.get((place, verb), 0))
    )
    return Result.success(
        PreparedPanel(
            panel="svo_map_verbs",
            shape="heatmap",
            title="Mapped verbs by place",
            marks=marks,
            x_label="Verb",
            y_label="Place",
            x_categories=verbs,
            y_categories=places,
            color_scale="sequential",
            value_label="Mapped triples",
            data=data,
            provenance=provenance,
            notes=(_SVO_NOTE,),
            subtitle=f"{len(places)} places · {len(verbs)} verbs · {len(data):,} triple-place pairs shown",
        )
    )


GEOGRAPHY_PANELS = (
    _map_panel(
        "geocode",
        grid=False,
        name="geocode_pins",
        title="Places mentioned",
        question="Where are the places named in this corpus?",
    ),
    _map_panel(
        "geocode",
        grid=True,
        name="geocode_concentration",
        title="Place mention concentration",
        question="Which parts of the map account for the most place mentions?",
    ),
    PanelDefinition(
        name="geocode_coverage",
        tool="geocode",
        shape="ranked_bars",
        title="Geocoding coverage",
        question="How many place names resolved, and what happened to the rest?",
        summary="Counts of attempted names by lookup outcome.",
        requires=_GEOCODE,
        params=(),
        build=_coverage,
        notes=_COVERAGE_NOTES,
    ),
    _map_panel(
        "svo_map",
        grid=False,
        name="svo_map_pins",
        title="Places associated with narrative triples",
        question="Which places are associated with the most subject-verb-object triples?",
    ),
    _map_panel(
        "svo_map",
        grid=True,
        name="svo_map_concentration",
        title="Narrative triple concentration",
        question="Where do document-level triple-place associations concentrate?",
    ),
    PanelDefinition(
        name="svo_map_verbs",
        tool="svo_map",
        shape="heatmap",
        title="Mapped verbs by place",
        question="Which actions appear in documents associated with each place?",
        summary="A matrix of mapped triples by place and verb.",
        requires=("Place", "Verb", "Subject", "Object"),
        params=(_TOP_PLACES, _TOP_VERBS),
        build=_verbs_by_place,
        notes=(_SVO_NOTE,),
    ),
    _map_panel(
        "gis_map",
        grid=False,
        name="gis_map_pins",
        title="Input coordinate map",
        question="Where are the supplied coordinates, and how much weight does each place carry?",
    ),
    _map_panel(
        "gis_map",
        grid=True,
        name="gis_map_concentration",
        title="Coordinate concentration",
        question="Which latitude and longitude cells have the greatest total input weight?",
    ),
)
