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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

import pandas as pd

from core.result import Diagnostic
from core.viz.panels import best_table, fit_to_rows, panels_for_tool, prepare_panel
from core.viz.panelspec import Source
from core.viz.showcase import render_showcase_multi, showcases_for

__all__ = ["FIGURE_FORMATS", "RunFigure", "run_figures"]

#: PNG to look at, SVG to edit or print at any size.
FIGURE_FORMATS: tuple[str, ...] = ("png", "svg")
#: A tool with more panels than this publishes the first ones; the rest are
#: a click away in the app.
MAX_FIGURES = 6
# Must equal desktop_backend.runner.INPUT_MANIFEST (source); core must not
# import desktop_backend, so the literal is spelled here and checked by tests.
_SKIP = {"desktop_inputs.csv"}


@dataclass(frozen=True, slots=True)
class RunFigure:
    """One rendered figure file, ready for ``OutputWriter.write_bytes``."""

    filename: str
    data: bytes
    description: str


def run_figures(  # noqa: PLR0913 - drawing options (settings, hashes, dpi, events) are keyword-only
    tool: str,
    frames: Mapping[str, pd.DataFrame],
    *,
    settings: Mapping[str, Any] | None = None,
    hashes: Mapping[str, str] | None = None,
    dpi: int = 200,
    events: Sequence[tuple[float, str]] = (),
) -> tuple[list[RunFigure], list[Diagnostic]]:
    """Every default figure the run's tables can feed, plus an index page.

    ``hashes`` are the written tables' SHA-256 digests, so each caption can
    name the exact file it was drawn from. ``events`` are the project's dated
    events, drawn as stoplines on every dated showcase and panel.
    """
    from core.viz.static import LIBRARY, render_static_multi

    tables = [
        (name, list(frame.columns))
        for name, frame in frames.items()
        if name.endswith(".csv") and not frame.empty and name not in _SKIP
    ]
    digest_cache: dict[str, str] = dict(hashes or {})

    def table_digest(table_name: str, frame: pd.DataFrame) -> str:
        cached = digest_cache.get(table_name)
        if cached is not None:
            return cached
        computed = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
        digest_cache[table_name] = computed
        return computed

    figures: list[RunFigure] = []
    notices: list[Diagnostic] = []
    index: list[str] = [f"# Figures for this {tool} run", ""]
    _showcase_figures(
        tool,
        frames=frames,
        settings=settings,
        hashes=digest_cache,
        dpi=dpi,
        events=events,
        figures=figures,
        notices=notices,
        index=index,
    )
    for definition in panels_for_tool(tool)[:MAX_FIGURES]:
        table = best_table(definition, tables)
        if table is None:
            continue
        digest = table_digest(table, frames[table])
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
        rendered = render_static_multi(panel, FIGURE_FORMATS, dpi=dpi)
        if rendered.value is None:
            notices.append(
                Diagnostic.warning(
                    "RUN_FIGURE_FAILED",
                    f"{definition.title}: {'; '.join(d.message for d in rendered.diagnostics)[:300]}",
                    panel=definition.name,
                )
            )
        else:
            notices.extend(rendered.diagnostics)
            by_fmt = rendered.unwrap()
            for fmt in FIGURE_FORMATS:
                filename = f"figures/{definition.name}.{fmt}"
                figures.append(RunFigure(filename, by_fmt[fmt], f"{definition.title} ({fmt.upper()})"))
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
        hashes=digest_cache,
        dpi=dpi,
        figures=figures,
        notices=notices,
        index=index,
    )
    if figures:
        figures.append(RunFigure("figures/README.md", "\n".join(index).encode("utf-8"), "Index of this run's figures"))
    return figures, notices


def _showcase_figures(  # noqa: PLR0913 - the run's accumulators, threaded through
    tool: str,
    *,
    frames: Mapping[str, pd.DataFrame],
    settings: Mapping[str, Any] | None,
    hashes: dict[str, str],
    dpi: int,
    events: Sequence[tuple[float, str]],
    figures: list[RunFigure],
    notices: list[Diagnostic],
    index: list[str],
) -> None:
    """The tool's showcase figure, when it can support one (docs/viz-panels.md)."""
    for showcase in showcases_for(tool):
        present = {name: frames[name] for name in showcase.tables if name in frames and not frames[name].empty}
        if not present:
            continue
        source = next(name for name in showcase.tables if name in present)
        digest = hashes.get(source)
        if digest is None:
            digest = hashlib.sha256(present[source].to_csv(index=False).encode()).hexdigest()
            hashes[source] = digest
        drawn = []
        rendered = render_showcase_multi(
            showcase,
            present,
            FIGURE_FORMATS,
            source=source,
            settings=settings,
            sha256=digest,
            dpi=dpi,
            events=events,
        )
        if rendered.value is None:
            notices.extend(rendered.diagnostics)
            continue
        notices.extend(rendered.diagnostics)
        by_fmt = rendered.unwrap()
        for fmt in FIGURE_FORMATS:
            filename = f"figures/{showcase.name}.{fmt}"
            figures.append(RunFigure(filename, by_fmt[fmt], f"{showcase.title} ({fmt.upper()})"))
            drawn.append(filename)
        if drawn:
            index += [f"## {showcase.title}", "", showcase.question, ""]
            index += [f"- [{name.split('/')[-1]}]({name.split('/')[-1]})" for name in drawn]
            index += ["", f"Drawn from `{source}`.", ""]


def _bundle_figures(  # noqa: PLR0913 - the run's accumulators, threaded through
    tool: str,
    *,
    frames: Mapping[str, pd.DataFrame],
    tables: list[tuple[str, list[str]]],
    settings: Mapping[str, Any] | None,
    hashes: dict[str, str],
    dpi: int,
    figures: list[RunFigure],
    notices: list[Diagnostic],
    index: list[str],
) -> None:
    """The tool's publication-only figures (``core/viz/static/bundles``)."""
    from core.viz.static.bundles import bundles_for, render_bundle_multi

    for bundle in bundles_for(tool):
        table = best_table(bundle, tables)
        if table is None:
            continue
        digest = hashes.get(table)
        if digest is None:
            digest = hashlib.sha256(frames[table].to_csv(index=False).encode()).hexdigest()
            hashes[table] = digest
        drawn = []
        rendered = render_bundle_multi(
            bundle,
            frames[table],
            FIGURE_FORMATS,
            source=table,
            settings=settings,
            sha256=digest,
            dpi=dpi,
        )
        if rendered.value is None:
            notices.extend(rendered.diagnostics)
            continue
        notices.extend(rendered.diagnostics)
        by_fmt = rendered.unwrap()
        for fmt in FIGURE_FORMATS:
            filename = f"figures/{bundle.name}.{fmt}"
            figures.append(RunFigure(filename, by_fmt[fmt], f"{bundle.title} ({fmt.upper()})"))
            drawn.append(filename)
        if drawn:
            index += [f"## {bundle.title} (publication only)", "", bundle.question, ""]
            index += [f"- [{name.split('/')[-1]}]({name.split('/')[-1]})" for name in drawn]
            index += ["", f"Drawn from `{table}`.", ""]
