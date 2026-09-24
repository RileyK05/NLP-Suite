"""The figures a finished run publishes beside its tables.

A run's tables are the result; its figures are how most readers will first
meet it, and the files they will put in a paper. So a run the desktop
finishes renders each of its tool's panels with their defaults, through the
publication renderer (``core/viz/static``), into ``figures/`` inside the run
-- written by the run's own OutputWriter, so they are sealed, hashed and
listed in the envelope like every other artifact (R3).

A figure that cannot be drawn is a warning on the run, never a failed run:
the tables are the result, and they stand without their pictures.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from typing import Any

import pandas as pd

from core.result import Diagnostic
from core.viz.panels import best_table, fit_to_rows, panels_for_tool, prepare_panel
from core.viz.panelspec import Source

__all__ = ["FIGURE_FORMATS", "RunFigure", "run_figures"]

#: PNG to look at, SVG to edit or print at any size.
FIGURE_FORMATS: tuple[str, ...] = ("png", "svg")
#: A tool with more panels than this publishes the first ones; the rest are
#: a click away in the app.
MAX_FIGURES = 6
_SKIP = {"input_files.csv"}


@dataclass(frozen=True, slots=True)
class RunFigure:
    """One rendered figure file, ready for ``OutputWriter.write_bytes``."""

    filename: str
    data: bytes
    description: str


def run_figures(
    tool: str,
    frames: Mapping[str, pd.DataFrame],
    *,
    settings: Mapping[str, Any] | None = None,
    hashes: Mapping[str, str] | None = None,
    dpi: int = 200,
) -> tuple[list[RunFigure], list[Diagnostic]]:
    """Every default figure the run's tables can feed, plus an index page.

    ``hashes`` are the written tables' SHA-256 digests, so each caption can
    name the exact file it was drawn from.
    """
    from core.viz.static import LIBRARY, render_static

    tables = [
        (name, list(frame.columns))
        for name, frame in frames.items()
        if name.endswith(".csv") and not frame.empty and name not in _SKIP
    ]
    figures: list[RunFigure] = []
    notices: list[Diagnostic] = []
    index: list[str] = [f"# Figures for this {tool} run", ""]
    for definition in panels_for_tool(tool)[:MAX_FIGURES]:
        table = best_table(definition, tables)
        if table is None:
            continue
        digest = (hashes or {}).get(table) or hashlib.sha256(frames[table].to_csv(index=False).encode()).hexdigest()
        prepared = prepare_panel(
            definition.name,
            frames[table],
            None,
            source=Source(path=table, sha256=digest, settings=dict(settings or {}), library=LIBRARY),
        )
        if prepared.value is None:
            reason = "; ".join(d.message for d in prepared.diagnostics if d.severity.value == "ERROR")[:300]
            notices.append(
                Diagnostic.info(
                    "RUN_FIGURE_SKIPPED", f"{definition.title}: not drawn ({reason})", panel=definition.name
                )
            )
            continue
        panel = fit_to_rows(prepared.unwrap())
        drawn = []
        for fmt in FIGURE_FORMATS:
            image = render_static(panel, fmt, dpi=dpi)
            if image.value is None:
                notices.append(
                    Diagnostic.warning(
                        "RUN_FIGURE_FAILED",
                        f"{definition.title} ({fmt}): {'; '.join(d.message for d in image.diagnostics)[:300]}",
                        panel=definition.name,
                    )
                )
                continue
            filename = f"figures/{definition.name}.{fmt}"
            figures.append(RunFigure(filename, image.unwrap(), f"{definition.title} ({fmt.upper()})"))
            drawn.append(filename)
        if drawn:
            index += [f"## {definition.title}", "", definition.question or definition.summary, ""]
            index += [f"- [{name.split('/')[-1]}]({name.split('/')[-1]})" for name in drawn]
            index += ["", f"Drawn from `{table}`. {panel.caption}", ""]
    _bundle_figures(
        tool,
        frames=frames,
        tables=tables,
        settings=settings,
        hashes=hashes,
        dpi=dpi,
        figures=figures,
        notices=notices,
        index=index,
    )
    if figures:
        figures.append(RunFigure("figures/README.md", "\n".join(index).encode("utf-8"), "Index of this run's figures"))
    return figures, notices


def _bundle_figures(  # noqa: PLR0913 - the run's accumulators, threaded through
    tool: str,
    *,
    frames: Mapping[str, pd.DataFrame],
    tables: list[tuple[str, list[str]]],
    settings: Mapping[str, Any] | None,
    hashes: Mapping[str, str] | None,
    dpi: int,
    figures: list[RunFigure],
    notices: list[Diagnostic],
    index: list[str],
) -> None:
    """The tool's publication-only figures (``core/viz/static/bundles``)."""
    from core.viz.static.bundles import bundles_for, render_bundle

    for bundle in bundles_for(tool):
        table = best_table(bundle, tables)
        if table is None:
            continue
        drawn = []
        for fmt in FIGURE_FORMATS:
            image = render_bundle(
                bundle,
                frames[table],
                fmt,
                source=table,
                settings=settings,
                sha256=(hashes or {}).get(table, ""),
                dpi=dpi,
            )
            if image.value is None:
                notices.extend(image.diagnostics)
                break
            filename = f"figures/{bundle.name}.{fmt}"
            figures.append(RunFigure(filename, image.unwrap(), f"{bundle.title} ({fmt.upper()})"))
            drawn.append(filename)
        if drawn:
            index += [f"## {bundle.title} (publication only)", "", bundle.question, ""]
            index += [f"- [{name.split('/')[-1]}]({name.split('/')[-1]})" for name in drawn]
            index += ["", f"Drawn from `{table}`.", ""]
