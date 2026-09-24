"""Culturomics time-series panel for the n-gram viewer output."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panelspec import Evidence, PanelDefinition, PanelMark, PanelParam, PreparedPanel, Provenance

__all__ = ["NGRAM_FREQUENCY_OVER_TIME", "ngram_frequency_over_time"]

NGRAM = "N-gram"
YEAR = "Year"
COUNT = "Count"
PER_MILLION = "Per Million"
SHARE = "Share of Documents"
_METRICS = (COUNT, PER_MILLION, SHARE)
_MARK_CAP = 5000


def ngram_frequency_over_time(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Prepare one independent dated series per n-gram."""
    metric = str(params["metric"])
    if metric not in _METRICS:
        return Result.failure(Diagnostic.error("PANEL_BAD_PARAM", f"unsupported n-gram metric {metric!r}"))

    working = frame.copy()
    working[YEAR] = pd.to_numeric(working[YEAR], errors="coerce")
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    valid = working[YEAR].notna() & working[metric].notna() & working[NGRAM].notna()
    dropped = int((~valid).sum())
    working = working.loc[valid].sort_values([NGRAM, YEAR], kind="stable")
    diagnostics: list[Diagnostic] = []
    if dropped:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_BAD_NUMERIC",
                f"{dropped} n-gram year row(s) had a missing or non-numeric year, metric, or query and were omitted.",
                dropped=dropped,
            )
        )
    if working.empty:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", "no dated n-gram rows are available to plot"), *diagnostics
        )
    if len(working) > _MARK_CAP:
        return Result.failure(
            Diagnostic.error(
                "PANEL_TOO_MANY_MARKS",
                f"{len(working)} n-gram year rows exceed the readable panel limit of {_MARK_CAP}; reduce queries or years.",
                available=len(working),
                maximum=_MARK_CAP,
            ),
            *diagnostics,
        )

    groups = tuple(sorted(working[NGRAM].astype(str).unique()))
    marks: list[PanelMark] = []
    for _, row in working.iterrows():
        query = str(row[NGRAM])
        year = int(row[YEAR])
        value = float(row[metric])
        key = f"{query}\u241f{year}"
        raw_value = row.get("Raw Count")
        if raw_value is None and int(provenance.settings.get("smooth", 1)) == 1:
            raw_value = row.get(COUNT)
        try:
            raw_count = int(raw_value) if raw_value is not None and math.isfinite(float(raw_value)) else None
        except (TypeError, ValueError):
            raw_count = None
        tokens = row.get("Corpus Tokens")
        documents = row.get("Corpus Documents")
        exposure = (
            f"; corpus exposure {int(documents)} documents / {int(tokens)} tokens"
            if pd.notna(documents) and pd.notna(tokens)
            else ""
        )
        matches = f"; {raw_count} raw matches" if raw_count is not None else "; raw matches unavailable in this result"
        marks.append(
            PanelMark(
                key=key,
                label=query,
                x=float(year),
                y=value,
                group=query,
                evidence=Evidence(
                    scope="rows",
                    filters=((NGRAM, query), (YEAR, str(year))),
                    count=max(0, raw_count) if raw_count is not None else 1,
                    describe=f"{query}, {year}: {metric} {value:g}{matches}{exposure}",
                ),
            )
        )

    prepared = PreparedPanel(
        panel=NGRAM_FREQUENCY_OVER_TIME.name,
        shape="line_series",
        title="N-gram frequency over time",
        subtitle=f"{len(groups)} n-gram series; {len(marks)} corpus-exposure year points",
        marks=tuple(marks),
        x_label="Year",
        y_label=metric,
        provenance=provenance,
        data=working,
        groups=groups,
        notes=(
            "Each line is a separate query. A zero marks a year with dated corpus documents and no hits; "
            "a missing year has no dated corpus exposure and remains a gap.",
            "Per Million uses the year's total dated-corpus token count. Share of Documents is the fraction "
            "of that year's dated documents containing at least one match.",
            "When the source run uses smoothing, Count and Per Million are smoothed while Share of Documents "
            "remains the raw per-year share.",
        ),
    )
    return Result.success(prepared, *diagnostics)


NGRAM_FREQUENCY_OVER_TIME = PanelDefinition(
    name="ngram_frequency_over_time",
    title="N-gram frequency over time",
    question="How often is each phrase used, year by year, allowing for how much text each year has?",
    tool="ngram_viewer",
    shape="line_series",
    summary="Independent n-gram frequency lines over dated corpus exposure years.",
    requires=(NGRAM, YEAR, COUNT, PER_MILLION, SHARE),
    params=(
        PanelParam(
            name="metric",
            type="choice",
            default=PER_MILLION,
            choices=_METRICS,
            label="Y-axis measure",
            help="Choose raw matches, matches per million dated-corpus tokens, or the share of dated documents with a match.",
        ),
    ),
    build=ngram_frequency_over_time,
    notes=(
        "No-corpus years have no row and are drawn as gaps; zero-hit years with corpus exposure are zero-valued points.",
    ),
)
