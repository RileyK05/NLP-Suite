"""Purpose-built figures from an unsaved live answer, when they fit."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panels import best_table, get_panel, panels_for_tool, prepare_panel
from core.viz.panelspec import PanelDefinition, PreparedPanel, Source
from desktop_backend.live import LiveResult
from desktop_backend.panels import panel_failure, public_panel


def _eligible(result: LiveResult) -> list[tuple[PanelDefinition, str, pd.DataFrame]]:
    """Each of the tool's panels with the first frame that can feed it.

    A tool can write several tables (LDA writes five), and each panel reads
    one of them, so eligibility is per panel: the relevance bars come from
    one frame and the topic flow from another.
    """
    if not result.ok:
        return []
    found: list[tuple[PanelDefinition, str, pd.DataFrame]] = []
    usable = [(name, list(frame.columns)) for name, frame in result.frames.items() if not frame.empty]
    for definition in panels_for_tool(result.tool):
        name = best_table(definition, usable)
        if name is not None:
            found.append((definition, name, result.frames[name]))
    return found


def live_panels_offered(result: LiveResult) -> list[dict[str, Any]]:
    """The figures this answer can draw, as the app builds its tabs from."""
    return [definition.to_dict() for definition, _, _ in _eligible(result)]


def _draw(
    definition: PanelDefinition, name: str, frame: pd.DataFrame, settings: dict[str, Any], params: dict[str, Any] | None
) -> dict[str, Any]:
    """Live analysis has no run directory, so the caption says it is unsaved
    and identifies these exact CSV bytes by hash."""
    digest = hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
    prepared = prepare_panel(
        definition.name,
        frame,
        params,
        source=Source(path=f"unsaved live {name}", sha256=digest, settings=settings, library="NLP Suite desktop (SVG)"),
    )
    if prepared.value is None:
        return panel_failure(list(prepared.diagnostics))
    return public_panel(prepared.unwrap(), list(prepared.diagnostics))


def live_panel(result: LiveResult, settings: dict[str, Any]) -> dict[str, Any] | None:
    """The first eligible figure, drawn with its defaults; None when none fits.

    A partial or older result whose columns do not satisfy a panel simply
    returns no panel; the ordinary table remains available to the reader.
    """
    eligible = _eligible(result)
    if not eligible:
        return None
    definition, name, frame = eligible[0]
    return _draw(definition, name, frame, settings, None)


def prepare_live_panel(
    result: LiveResult, settings: dict[str, Any], panel: str, params: dict[str, Any]
) -> Result[PreparedPanel]:
    """One named figure of a remembered answer, prepared for any renderer."""
    definition = get_panel(panel)
    if definition is None or definition.tool != result.tool:
        return Result.failure(
            Diagnostic.error("PANEL_WRONG_TOOL", f"{panel!r} is not one of {result.tool}'s figures", panel=panel)
        )
    for candidate, name, frame in _eligible(result):
        if candidate.name == panel:
            digest = hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
            return prepare_panel(
                candidate.name,
                frame,
                params,
                source=Source(path=f"unsaved live {name}", sha256=digest, settings=settings),
            )
    return Result.failure(
        Diagnostic.error("PANEL_NO_TABLE", f"this answer has no table that {panel} can draw", panel=panel)
    )


def draw_live_panel(result: LiveResult, settings: dict[str, Any], panel: str, params: dict[str, Any]) -> dict[str, Any]:
    """One named figure over a remembered answer, with the reader's settings."""
    definition = get_panel(panel)
    if definition is None or definition.tool != result.tool:
        return panel_failure(
            [Diagnostic.error("PANEL_WRONG_TOOL", f"{panel!r} is not one of {result.tool}'s figures", panel=panel)]
        )
    for candidate, name, frame in _eligible(result):
        if candidate.name == panel:
            return _draw(candidate, name, frame, settings, params)
    return panel_failure(
        [
            Diagnostic.error(
                "PANEL_NO_TABLE",
                f"this answer has no table with the columns {panel} reads: {', '.join(definition.requires)}",
                panel=panel,
            )
        ]
    )
