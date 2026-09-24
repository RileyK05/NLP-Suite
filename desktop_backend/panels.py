"""Panels for the desktop: which figures a finished run can draw, and its data.

``core/viz/panels.py`` knows what panels exist and how to prepare one. This
module is the thin layer that connects them to a run the reader is looking
at: it finds the run's result table, asks the registry to prepare the panel,
and renders the result into JSON the app can draw.

Two decisions worth stating.

**The app draws the panel itself, from marks.** The prepared panel crosses
the boundary as geometry plus evidence, not as an HTML blob, for the same
reason ``desktop/src/ChartCanvas.tsx`` draws charts with plain SVG: every
mark has to stay clickable and identifiable, so that pointing at a bubble
can say what it stands for and filter the table beneath it. A rendered
figure is a picture; marks are a thing you can interrogate. The HTML and
image renderers still exist for export, where the opposite is true.

**Provenance comes from the envelope, not from the request.** The source
path, its hash and the settings that produced it are read from the run the
panel is drawn over, so a caption cannot be talked into saying something the
run did not do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from core.artifacts.envelope import Envelope
from core.io.reader import hash_file
from core.result import Diagnostic, Result
from core.viz.panels import PANELS, best_table, get_panel, panels_for_tool, prepare_panel
from core.viz.panelspec import PanelDefinition, PreparedPanel, Source

__all__ = [
    "MAX_PANEL_ROWS",
    "MEDIA_TYPES",
    "BundleBody",
    "PanelBody",
    "StaticPanelBody",
    "bundles_for_envelope",
    "panel_declarations",
    "panels_for_envelope",
    "prepare_run_panel",
    "public_panel",
    "render_prepared",
    "render_run_bundle",
    "result_tables",
    "table_for_panel",
]

#: Rows of the drawn table sent alongside the figure. The figure itself is
#: complete -- every mark crosses -- but the table beside it is a preview,
#: and a 50,000-row result should not travel to draw one.
MAX_PANEL_ROWS = 500

INPUT_MANIFEST = "input_files.csv"


class PanelBody(BaseModel):
    """A request to draw one panel over one finished run."""

    model_config = ConfigDict(extra="forbid")

    panel: str = Field(description="registered panel name, e.g. keyness_volcano")
    params: dict[str, Any] = Field(default_factory=dict, description="panel parameters; defaults fill the rest")


def panel_declarations() -> list[dict[str, Any]]:
    """Every panel and what it takes, for the app to build controls from.

    Shipped rather than restated in TypeScript, for the same reason
    ``/api/tools`` ships its labels: a second copy on the far side of a
    language boundary drifts without anything failing.
    """
    return [definition.to_dict() for definition in PANELS]


def panels_for_envelope(directory: Path, envelope: Envelope) -> list[dict[str, Any]]:
    """The panels this run can actually draw.

    Two tests, both required. The panel must belong to the tool that produced
    the run, and the run must contain a table carrying every column the panel
    reads. The second test is not redundant: a run can be partial, or written
    by an older version of the tool, and offering a figure that fails with a
    missing column the moment it is clicked is worse than not offering it.
    """
    tables = result_tables(directory, envelope)
    return [
        definition.to_dict()
        for definition in panels_for_tool(envelope.tool)
        if _table_carrying(tables, definition) is not None
    ]


def result_tables(directory: Path, envelope: Envelope) -> list[tuple[str, list[str]]]:
    """Every result CSV in the run, with its header, in envelope order.

    A run can publish several tables -- ``lda_gensim`` writes the topic words,
    the dominant topic per document, the relevance ranking and the intertopic
    placement, each a different shape -- so "the run's table" is not one
    thing. Reading only the header keeps this cheap on a large result.

    The input manifest records what was read rather than what was found, so
    it is skipped: a panel drawn over the list of input files would be
    technically valid and completely useless. An unreadable table costs its
    own entry, not the whole list.
    """
    found: list[tuple[str, list[str]]] = []
    for artifact in envelope.artifacts:
        if not artifact.path.endswith(".csv") or Path(artifact.path).name == INPUT_MANIFEST:
            continue
        try:
            header = pd.read_csv(directory / artifact.path, nrows=0, encoding="utf-8-sig")
        except (OSError, ValueError):
            continue
        found.append((artifact.path, [str(column) for column in header.columns]))
    return found


def table_for_panel(directory: Path, envelope: Envelope, definition: PanelDefinition) -> str | None:
    """The run's first table carrying every column this panel reads, or None.

    Chosen by what the panel needs rather than by position. Picking the first
    CSV is what made every topic-model panel fail: the first table an LDA run
    writes is its topic words, and none of the three LDA panels draws those.
    """
    return _table_carrying(result_tables(directory, envelope), definition)


def _table_carrying(tables: list[tuple[str, list[str]]], definition: PanelDefinition) -> str | None:
    return best_table(definition, tables)


def public_panel(prepared: PreparedPanel, diagnostics: list[Any]) -> dict[str, Any]:
    """A prepared panel as the app receives it: geometry, evidence, words.

    Every mark crosses, including the evidence behind it, because a click has
    to be answerable without a second round trip. The rows are capped; the
    marks are not.
    """
    frame = prepared.data.head(MAX_PANEL_ROWS)
    return {
        "ok": True,
        "panel": prepared.panel,
        "shape": prepared.shape,
        "title": prepared.title,
        "subtitle": prepared.subtitle,
        "xLabel": prepared.x_label,
        "yLabel": prepared.y_label,
        "groups": list(prepared.groups),
        "caption": prepared.caption,
        "notes": list(prepared.notes),
        "marks": [
            {
                "key": mark.key,
                "label": mark.label,
                "x": mark.x,
                "y": mark.y,
                "group": mark.group,
                "size": mark.size,
                "labelled": mark.labelled,
                "evidence": mark.evidence.to_dict(),
                "value": mark.value,
                "facet": mark.facet,
            }
            for mark in prepared.marks
        ],
        "edges": [
            {
                "key": edge.key,
                "source": edge.source,
                "target": edge.target,
                "weight": edge.weight,
                "label": edge.label,
                "evidence": edge.evidence.to_dict(),
            }
            for edge in prepared.edges
        ],
        "xCategories": list(prepared.x_categories),
        "yCategories": list(prepared.y_categories),
        "colorScale": prepared.color_scale,
        "edgeStyle": prepared.edge_style,
        "facets": list(prepared.facets),
        "lineGap": prepared.line_gap,
        "pointsOnly": list(prepared.points_only),
        "xLog10": prepared.x_log10,
        "valueLabel": prepared.value_label,
        "width": prepared.width,
        "height": prepared.height,
        "annotations": [
            {"kind": a.kind, "value": a.value, "label": a.label, "note": a.note} for a in prepared.annotations
        ],
        "table": {
            "columns": [str(column) for column in frame.columns],
            "rows": [{str(k): _cell(v) for k, v in row.items()} for row in frame.to_dict(orient="records")],
            "total": len(prepared.data),
            "truncated": bool(len(prepared.data) > MAX_PANEL_ROWS),
        },
        "diagnostics": [{"severity": d.severity.value, "code": d.code, "message": d.message} for d in diagnostics],
    }


def panel_failure(diagnostics: list[Any]) -> dict[str, Any]:
    """A refusal in the same shape as a success (R-C7): ``ok`` plus reasons."""
    return {
        "ok": False,
        "diagnostics": [{"severity": d.severity.value, "code": d.code, "message": d.message} for d in diagnostics],
    }


def _cell(value: Any) -> str:
    """One table cell as text. Missing stays blank rather than becoming 'nan'."""
    if value is None:
        return ""
    text = str(value)
    return "" if text in ("nan", "NaT", "None") else text


def prepare_run_panel(directory: Path, envelope: Envelope, body: PanelBody) -> Result[PreparedPanel]:
    """One panel prepared over one run: the step both renderers share."""
    definition = get_panel(body.panel)
    if definition is None:
        return Result.failure(Diagnostic.error("PANEL_UNKNOWN", f"no panel named {body.panel!r}", panel=body.panel))
    if definition.tool != envelope.tool:
        return Result.failure(
            Diagnostic.error(
                "PANEL_WRONG_TOOL",
                f"{body.panel} draws {definition.tool} results; this run is {envelope.tool}",
                panel=body.panel,
            )
        )
    relative = table_for_panel(directory, envelope, definition)
    if relative is None:
        needed = ", ".join(definition.requires)
        return Result.failure(
            Diagnostic.error(
                "PANEL_NO_TABLE",
                f"this run has no table carrying the columns {body.panel} reads ({needed}); "
                "it may be partial or from an older version of the tool. Re-running it should fix this.",
                panel=body.panel,
            )
        )

    path = directory / relative
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, ValueError) as exc:
        return Result.failure(Diagnostic.error("PANEL_TABLE_UNREADABLE", f"could not read {relative}: {exc}"))

    # The caption's facts come from the run, not from the caller.
    source = Source(
        path=Path(relative).name,
        sha256=hash_file(path),
        settings=dict(envelope.params),
        library=_renderer(),
    )
    return prepare_panel(body.panel, frame, body.params, source=source)


def draw_panel(directory: Path, envelope: Envelope, body: PanelBody) -> dict[str, Any]:
    """Prepare one panel over one run, as a response the app can render."""
    prepared = prepare_run_panel(directory, envelope, body)
    if prepared.value is None:
        return panel_failure(list(prepared.diagnostics))
    return public_panel(prepared.unwrap(), list(prepared.diagnostics))


class StaticPanelBody(PanelBody):
    """A request for a panel's publication figure (matplotlib + seaborn)."""

    format: Literal["png", "svg", "pdf"] = Field(default="png", description="image format")
    dpi: int = Field(default=200, ge=72, le=400, description="resolution for PNG")
    notes: bool = Field(default=False, description="print the figure's limits under the caption")


#: What each format is served as.
MEDIA_TYPES = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}


class BundleBody(BaseModel):
    """A request for one of a run's publication-only figures."""

    model_config = ConfigDict(extra="forbid")

    bundle: str = Field(description="registered bundle name, e.g. readability_measure_correlations")
    format: Literal["png", "svg", "pdf"] = "png"
    dpi: int = Field(default=200, ge=72, le=400)


def bundles_for_envelope(directory: Path, envelope: Envelope) -> list[dict[str, Any]]:
    """The publication-only figures this run's tables can feed."""
    from core.viz.static.bundles import bundles_for  # noqa: PLC0415 - matplotlib loads on first figure

    tables = result_tables(directory, envelope)
    return [bundle.to_dict() for bundle in bundles_for(envelope.tool) if best_table(bundle, tables) is not None]


def render_run_bundle(directory: Path, envelope: Envelope, body: BundleBody) -> Result[bytes]:
    """One publication-only figure over this run, captioned from the run."""
    from core.viz.static.bundles import get_bundle, render_bundle  # noqa: PLC0415 - matplotlib loads on first figure

    bundle = get_bundle(body.bundle)
    if bundle is None or bundle.tool != envelope.tool:
        return Result.failure(
            Diagnostic.error("BUNDLE_UNKNOWN", f"{body.bundle!r} is not one of {envelope.tool}'s figures")
        )
    relative = best_table(bundle, result_tables(directory, envelope))
    if relative is None:
        return Result.failure(Diagnostic.error("PANEL_NO_TABLE", f"this run has no table {bundle.title} can draw"))
    path = directory / relative
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, ValueError) as exc:
        return Result.failure(Diagnostic.error("PANEL_TABLE_UNREADABLE", f"could not read {relative}: {exc}"))
    return render_bundle(
        bundle,
        frame,
        body.format,
        source=Path(relative).name,
        settings=dict(envelope.params),
        sha256=hash_file(path),
        dpi=body.dpi,
    )


def render_prepared(prepared: Result[PreparedPanel], fmt: str, dpi: int, notes: bool) -> Result[bytes]:
    """A prepared panel as publication-figure bytes, diagnostics carried."""
    if prepared.value is None:
        return Result[bytes](None, prepared.diagnostics)
    from core.viz.static import render_static  # noqa: PLC0415 - matplotlib loads on first figure, not at startup

    image = render_static(prepared.unwrap(), fmt, dpi=dpi, notes=notes)
    return Result[bytes](image.value, (*prepared.diagnostics, *image.diagnostics))


def _renderer() -> str:
    """What the app draws with. The desktop draws marks itself (SVG), so the
    caption says so rather than crediting a library that was not used."""
    return "NLP Suite desktop (SVG)"
