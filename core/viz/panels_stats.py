"""Figures for the statistics tools over an analyst's own table.

The same engines serve two doors -- the desktop's ``table_*`` workflows
(``desktop_backend/tables.py``) and the CLI's ``stats_*`` tools -- and write
the same tables, so every figure here is registered once per tool name by a
factory. Their category and value columns are whatever the analyst chose
("NER Tag" by "Decade"), so the builders find them by the fixed columns
beside them, never by name.

What each figure is for:

* **chi-square** -- which cells drive the association: Pearson residuals on
  a diverging scale, observed and expected in the hover. On the real NER
  table by decade, PERSON is over-represented from the 1990s on and ORG
  before, which a single chi-square and p-value cannot say.
* **crosstab** -- the counts as a heatmap, and each row as a composition.
* **table keyness** -- effect (log ratio) against evidence (G2).
* **Kruskal-Wallis / Mann-Whitney** -- the group medians with their spread,
  and for Kruskal-Wallis which pairs of groups differ (adjusted p).
* **Mann-Kendall** -- the observations and the fitted Sen's slope line.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import decimal_year
from core.viz.panelspec import (
    Annotation,
    Evidence,
    PanelBuilder,
    PanelDefinition,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
)

__all__ = ["STATS_PANELS"]

OBSERVED = "Observed"
EXPECTED = "Expected"
RESIDUAL = "Pearson Residual"
TOTAL = "Total"
#: Cochran's rule: chi-square is unreliable when more than a fifth of the
#: expected counts are under five.
_SMALL_EXPECTED = 5.0
_SMALL_SHARE = 0.2
_CAP_CATEGORIES = 60


def _ordered(values: pd.Series) -> list[str]:
    """Category order as the analyst's data gives it, decades and years in order."""
    unique = list(dict.fromkeys(values.astype(str)))
    return sorted(
        unique, key=lambda value: (0, float(value.rstrip("s"))) if value.rstrip("s").isdigit() else (1, value)
    )


def _chi2_residuals(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, name: str
) -> Result[PreparedPanel]:
    del params
    categories = [column for column in frame.columns if column not in (OBSERVED, EXPECTED, RESIDUAL)]
    if len(categories) != 2:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", "the observed/expected table should name exactly two category columns")
        )
    row_column, column_column = categories
    working = frame.copy()
    for column in (OBSERVED, EXPECTED, RESIDUAL):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    rows = _ordered(working[row_column])
    columns = _ordered(working[column_column])
    if len(rows) > _CAP_CATEGORIES or len(columns) > _CAP_CATEGORIES:
        return Result.failure(
            Diagnostic.error(
                "PANEL_TOO_MANY_CATEGORIES",
                f"{len(rows)} x {len(columns)} categories is more than a readable heatmap holds; group the rarer "
                "categories before testing.",
            )
        )
    diagnostics: list[Diagnostic] = []
    small = float((working[EXPECTED] < _SMALL_EXPECTED).mean())
    if small > _SMALL_SHARE:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_SMALL_EXPECTED",
                f"{small:.0%} of cells expect fewer than {_SMALL_EXPECTED:g} observations; the chi-square test is "
                "unreliable here (Cochran's rule). Merge sparse categories, or read the residuals as descriptive.",
                share=round(small, 3),
            )
        )
    marks = tuple(
        PanelMark(
            key=f"{row[row_column]}|{row[column_column]}",
            label=f"{row[row_column]} / {row[column_column]}",
            x=float(columns.index(str(row[column_column]))),
            y=float(rows.index(str(row[row_column]))),
            value=float(row[RESIDUAL]),
            evidence=Evidence(
                scope="rows",
                filters=((row_column, str(row[row_column])), (column_column, str(row[column_column]))),
                count=int(row[OBSERVED]),
                describe=(
                    f"{row[row_column]} in {row[column_column]}: observed {int(row[OBSERVED])}, expected "
                    f"{row[EXPECTED]:.1f}, residual {row[RESIDUAL]:+.2f}"
                ),
            ),
        )
        for _, row in working.iterrows()
        if not math.isnan(float(row[RESIDUAL]))
    )
    prepared = PreparedPanel(
        panel=name,
        shape="heatmap",
        title=f"Where {row_column} and {column_column} depart from independence",
        subtitle="Pearson residuals: red = fewer than expected, blue = more than expected",
        marks=marks,
        x_label=column_column,
        y_label=row_column,
        provenance=provenance,
        data=working,
        x_categories=tuple(columns),
        y_categories=tuple(rows),
        color_scale="diverging",
        value_label="Pearson residual",
        height=max(500, 26 * len(rows) + 160),
        notes=CHI2_NOTES,
    )
    return Result.success(prepared, *diagnostics)


CHI2_NOTES = (
    "A residual is (observed - expected) / sqrt(expected). Beyond about +/-2 a cell contributes noticeably to the "
    "association; the colour says which way.",
    "The chi-square statistic and p-value say that the two variables are associated, not how strongly: read "
    "Cramer's V in the summary table for the effect size, and these cells for where it lives.",
    "Counts in text are rarely independent (one speech contributes many entities), so a p-value here is "
    "optimistic; the pattern of residuals is the more robust reading.",
)


def _crosstab_heatmap(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, name: str
) -> Result[PreparedPanel]:
    del params
    row_column = frame.columns[0]
    columns = [column for column in frame.columns[1:] if column != TOTAL]
    body = frame[frame[row_column].astype(str) != TOTAL]
    rows = [str(value) for value in body[row_column]]
    marks: list[PanelMark] = []
    for row_index, (_, row) in enumerate(body.iterrows()):
        for column_index, column in enumerate(columns):
            count = float(pd.to_numeric(row[column], errors="coerce"))
            if math.isnan(count):
                continue
            share = count / float(row[TOTAL]) if TOTAL in row and float(row[TOTAL]) else float("nan")
            marks.append(
                PanelMark(
                    key=f"{row[row_column]}|{column}",
                    label=f"{row[row_column]} / {column}",
                    x=float(column_index),
                    y=float(row_index),
                    value=count,
                    evidence=Evidence(
                        scope="rows",
                        filters=((row_column, str(row[row_column])),),
                        count=int(count),
                        describe=f"{row[row_column]} in {column}: {int(count)}"
                        + (f" ({share:.0%} of the row)" if not math.isnan(share) else ""),
                    ),
                )
            )
    prepared = PreparedPanel(
        panel=name,
        shape="heatmap",
        title=f"{row_column} by {frame.columns.name or 'category'}",
        subtitle=f"{len(rows)} x {len(columns)} counts",
        marks=tuple(marks),
        x_label="",
        y_label=str(row_column),
        provenance=provenance,
        data=body.reset_index(drop=True),
        x_categories=tuple(str(column) for column in columns),
        y_categories=tuple(rows),
        color_scale="sequential",
        value_label="Count",
        height=max(500, 26 * len(rows) + 160),
        notes=(
            "Raw counts: a row or column that is simply larger (a decade with more speeches) is darker everywhere. "
            "Read the composition figure, or the chi-square residuals, to compare shapes rather than sizes.",
        ),
    )
    return Result.success(prepared)


def _crosstab_composition(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, name: str
) -> Result[PreparedPanel]:
    del params
    row_column = frame.columns[0]
    columns = [str(column) for column in frame.columns[1:] if column != TOTAL]
    body = frame[frame[row_column].astype(str) != TOTAL].reset_index(drop=True)
    marks: list[PanelMark] = []
    for row_index, row in body.iterrows():
        counts = [max(0.0, float(pd.to_numeric(row[column], errors="coerce") or 0.0)) for column in columns]
        total = sum(counts)
        if total <= 0:
            continue
        start = 0.0
        for column, count in zip(columns, counts, strict=True):
            share = count / total
            if share > 0:
                marks.append(
                    PanelMark(
                        key=f"{row[row_column]}|{column}",
                        label=str(row[row_column]),
                        x=start,
                        y=float(row_index),
                        size=share,
                        group=column,
                        evidence=Evidence(
                            scope="rows",
                            filters=((row_column, str(row[row_column])),),
                            count=int(count),
                            describe=f"{row[row_column]}: {share:.0%} in {column} ({int(count)})",
                        ),
                    )
                )
            start += share
    prepared = PreparedPanel(
        panel=name,
        shape="ribbon",
        title=f"How each {row_column} divides",
        subtitle=f"{len(body)} row(s), each drawn to the same length",
        marks=tuple(marks),
        x_label="Share of the row",
        y_label=str(row_column),
        provenance=provenance,
        data=body,
        groups=tuple(columns),
        height=max(500, 26 * len(body) + 120),
        notes=(
            "Every row is stretched to the same length, so rows of very different sizes compare by shape; the "
            "counts heatmap shows the sizes.",
        ),
    )
    return Result.success(prepared)


def _table_keyness(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, name: str
) -> Result[PreparedPanel]:
    label_top = int(params.get("label-top", 15))
    working = frame.copy()
    for column in ("G2 (log-likelihood)", "Log Ratio", "Freq A", "Freq B"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=["G2 (log-likelihood)", "Log Ratio"])
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no rows with both G2 and a log ratio"))
    labelled = set(working.nlargest(label_top, "G2 (log-likelihood)")["Word"].astype(str))
    groups = tuple(sorted(working["Overrepresented in"].astype(str).unique()))
    marks = tuple(
        PanelMark(
            key=str(row["Word"]),
            label=str(row["Word"]),
            x=float(row["Log Ratio"]),
            y=float(row["G2 (log-likelihood)"]),
            group=str(row["Overrepresented in"]),
            labelled=str(row["Word"]) in labelled,
            evidence=Evidence(
                scope="rows",
                filters=(("Word", str(row["Word"])),),
                count=int((row["Freq A"] or 0) + (row["Freq B"] or 0)),
                describe=(
                    f"{row['Word']}: log ratio {row['Log Ratio']:+.2f}, G2 {row['G2 (log-likelihood)']:.1f} "
                    f"({int(row['Freq A'])} in A, {int(row['Freq B'])} in B)"
                ),
            ),
        )
        for _, row in working.iterrows()
    )
    prepared = PreparedPanel(
        panel=name,
        shape="scatter_labelled",
        title="Keyness: effect against evidence",
        subtitle=f"{len(marks)} word(s); the {len(labelled)} with the most evidence are named",
        marks=marks,
        x_label="Log ratio (effect: + = more in A, - = more in B)",
        y_label="G2 (evidence)",
        provenance=provenance,
        data=working.reset_index(drop=True),
        groups=groups,
        annotations=(Annotation(kind="hline", value=3.84, label="G2 = 3.84", note="p < 0.05 at 1 degree of freedom"),),
        notes=(
            "Far from zero left or right is a big difference between the columns; high up is a lot of evidence. "
            "A word high but near zero is common in both and barely different; far out but low is too rare to trust.",
            "G2 grows with the size of the counts, so on a large table nearly every word clears p < 0.05; judge "
            "by the log ratio.",
        ),
    )
    return Result.success(prepared)


def _medians(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, name: str
) -> Result[PreparedPanel]:
    """Group medians, ranked; the quartiles in the hover where the result has them.

    The results carry medians (and for Mann-Whitney the quartiles), not the
    raw values, so a box and points cannot be drawn honestly from them; the
    notes send the reader to the chart builder for that.
    """
    del params
    working = frame.copy()
    working["Median"] = pd.to_numeric(working["Median"], errors="coerce")
    order = _ordered(working["Group"])
    marks = tuple(
        PanelMark(
            key=str(row["Group"]),
            label=str(row["Group"]),
            x=float(row["Median"]),
            y=float(order.index(str(row["Group"]))),
            evidence=Evidence(
                scope="rows",
                filters=(("Group", str(row["Group"])),),
                count=1,
                describe=f"{row['Group']}: median {row['Median']:g}"
                + (f" (middle half {row['Q25']:g} to {row['Q75']:g})" if "Q25" in row and "Q75" in row else ""),
            ),
        )
        for _, row in working.iterrows()
    )
    prepared = PreparedPanel(
        panel=name,
        shape="ranked_bars",
        title="Group medians",
        subtitle=f"{len(marks)} group(s), in their own order",
        marks=marks,
        x_label="Median",
        y_label="Group",
        provenance=provenance,
        data=working,
        notes=(
            "These tests compare whole distributions by rank, not means; the median is the summary they come "
            "with. The test statistic, p-value and effect size are in the summary table beside this one.",
            "The result table holds medians, not every value, so no box or points can be drawn from it; chart the "
            "original column by group in the chart builder to see the spread.",
        ),
    )
    return Result.success(prepared)


def _posthoc(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, name: str
) -> Result[PreparedPanel]:
    """Which pairs differ: -log10 of the adjusted p, so darker is stronger."""
    del params
    adjusted = next((column for column in frame.columns if column.startswith("p-value (")), "p-value")
    working = frame.copy()
    working[adjusted] = pd.to_numeric(working[adjusted], errors="coerce")
    groups = _ordered(pd.concat([working["Group A"], working["Group B"]]))
    marks: list[PanelMark] = []
    for _, row in working.iterrows():
        p = float(row[adjusted])
        strength = min(10.0, -math.log10(max(p, 1e-10)))
        for a, b in ((row["Group A"], row["Group B"]), (row["Group B"], row["Group A"])):
            marks.append(
                PanelMark(
                    key=f"{a}|{b}",
                    label=f"{a} vs {b}",
                    x=float(groups.index(str(b))),
                    y=float(groups.index(str(a))),
                    value=strength,
                    evidence=Evidence(
                        scope="rows",
                        filters=(("Group A", str(row["Group A"])), ("Group B", str(row["Group B"]))),
                        count=1,
                        describe=f"{a} vs {b}: adjusted p {p:.3g}" + (" (differ)" if p < 0.05 else ""),
                    ),
                )
            )
    prepared = PreparedPanel(
        panel=name,
        shape="heatmap",
        title="Which groups differ from which",
        subtitle=f"{len(groups)} groups; darker = stronger evidence of a difference ({adjusted})",
        marks=tuple(marks),
        x_label="Group",
        y_label="Group",
        provenance=provenance,
        data=working,
        x_categories=tuple(groups),
        y_categories=tuple(groups),
        color_scale="sequential",
        value_label="Evidence of a difference (-log10 p)",
        height=max(500, 26 * len(groups) + 160),
        notes=(
            "Colour is -log10 of the adjusted p-value: 1.3 is p = 0.05, 2 is p = 0.01, capped at 10. It is "
            "evidence of a difference, not its size.",
            "Adjusted for the number of pairs compared; with many groups, only large differences survive.",
        ),
    )
    return Result.success(prepared)


def _trend(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, name: str
) -> Result[PreparedPanel]:
    del params
    date_column, value_column = frame.columns[0], frame.columns[1]
    working = frame.copy()
    working["_x"] = working[date_column].map(decimal_year)
    if working["_x"].isna().all():
        working["_x"] = pd.to_numeric(working[date_column], errors="coerce")
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working["Trend Line"] = pd.to_numeric(working["Trend Line"], errors="coerce")
    working = working.dropna(subset=["_x", value_column])
    if len(working) < 3:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "a trend needs at least three dated observations"))
    marks: list[PanelMark] = []
    for index, row in working.iterrows():
        marks.append(
            PanelMark(
                key=f"obs:{index}",
                label=str(row[date_column]),
                x=float(row["_x"]),
                y=float(row[value_column]),
                group="Observations",
                evidence=Evidence(
                    scope="rows",
                    filters=((date_column, str(row[date_column])),),
                    count=1,
                    describe=f"{row[date_column]}: {value_column} {row[value_column]:g}",
                ),
            )
        )
        marks.append(
            PanelMark(
                key=f"trend:{index}",
                label="Sen's slope",
                x=float(row["_x"]),
                y=float(row["Trend Line"]),
                group="Sen's slope",
                evidence=Evidence(
                    scope="rows",
                    filters=((date_column, str(row[date_column])),),
                    count=1,
                    describe=f"Trend at {row[date_column]}: {row['Trend Line']:g}",
                ),
            )
        )
    span = float(working["_x"].max() - working["_x"].min()) or 1.0
    prepared = PreparedPanel(
        panel=name,
        shape="line_series",
        title=f"{value_column} over {date_column}, with its monotonic trend",
        subtitle=f"{len(working)} observation(s); the line is the Sen's slope fit",
        marks=tuple(marks),
        x_label=str(date_column),
        y_label=str(value_column),
        provenance=provenance,
        data=working.drop(columns=["_x"]).reset_index(drop=True),
        groups=("Observations", "Sen's slope"),
        points_only=("Observations",),
        line_gap=span,
        notes=(
            "Mann-Kendall asks whether values tend to rise (or fall) over time, not whether they rise in a straight "
            "line; Sen's slope is the median of every pairwise slope, so one outlier cannot bend it.",
            "Kendall's tau and the p-value are in the summary table. A significant trend over speeches is a trend "
            "in these documents, not necessarily in the language of the time.",
        ),
    )
    return Result.success(prepared)


def _definition(  # noqa: PLR0913 - one declaration per tool door; keyword-only, each a real field
    tool: str,
    suffix: str,
    *,
    title: str,
    question: str,
    shape: str,
    summary: str,
    requires: tuple[str, ...],
    build: Callable[..., Result[PreparedPanel]],
    notes: tuple[str, ...],
    params: tuple[PanelParam, ...] = (),
) -> PanelDefinition:
    name = f"{tool}_{suffix}"

    def builder(frame: pd.DataFrame, values: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
        return build(frame, values, provenance, name=name)

    typed: PanelBuilder = builder
    return PanelDefinition(
        name=name,
        title=title,
        question=question,
        tool=tool,
        shape=shape,  # type: ignore[arg-type]
        summary=summary,
        requires=requires,
        params=params,
        build=typed,
        notes=notes,
    )


def _categorical(tool: str, *, chi2: bool, crosstab: bool, keyness: bool) -> tuple[PanelDefinition, ...]:
    out: list[PanelDefinition] = []
    if chi2:
        out.append(
            _definition(
                tool,
                "residuals",
                title="Chi-square residuals",
                question="Which combinations occur more, or less, than chance would give?",
                shape="heatmap",
                summary="Pearson residual per cell, diverging colour, observed and expected in the hover.",
                requires=(OBSERVED, EXPECTED, RESIDUAL),
                build=_chi2_residuals,
                notes=CHI2_NOTES,
            )
        )
    if crosstab:
        out.append(
            _definition(
                tool,
                "counts",
                title="Crosstab counts",
                question="How are the counts spread across the two categories?",
                shape="heatmap",
                summary="The cross-tabulated counts as a heatmap.",
                requires=(TOTAL,),
                build=_crosstab_heatmap,
                notes=("Raw counts: larger rows and columns are darker everywhere.",),
            )
        )
        out.append(
            _definition(
                tool,
                "composition",
                title="Crosstab composition",
                question="How does each row divide across the columns?",
                shape="ribbon",
                summary="Each row's shares across the columns, drawn to one length.",
                requires=(TOTAL,),
                build=_crosstab_composition,
                notes=("Rows are stretched to one length, so they compare by shape, not size.",),
            )
        )
    if keyness:
        out.append(
            _definition(
                tool,
                "keyness",
                title="Keyness: effect against evidence",
                question="Which words differ most between the two columns, and on how much evidence?",
                shape="scatter_labelled",
                summary="Log ratio against G2, one point per word.",
                requires=("Word", "Freq A", "Freq B", "G2 (log-likelihood)", "Log Ratio", "Overrepresented in"),
                build=_table_keyness,
                params=(
                    PanelParam(
                        name="label-top",
                        type="int",
                        default=15,
                        minimum=0,
                        maximum=100,
                        label="Words to name",
                        help="How many words, by G2, to write on the plot.",
                    ),
                ),
                notes=("G2 grows with the counts; on a large table judge by the log ratio.",),
            )
        )
    return tuple(out)


def _groups(tool: str, *, posthoc: bool) -> tuple[PanelDefinition, ...]:
    out = [
        _definition(
            tool,
            "medians",
            title="Group medians",
            question="Where does each group's middle value sit?",
            shape="ranked_bars",
            summary="Each group's median, with its quartiles where the test reports them.",
            requires=("Group", "Median"),
            build=_medians,
            notes=("The result holds medians, not every value; chart the original column for the spread.",),
        )
    ]
    if posthoc:
        out.append(
            _definition(
                tool,
                "posthoc",
                title="Which groups differ",
                question="Which pairs of groups differ, once the number of comparisons is allowed for?",
                shape="heatmap",
                summary="Dunn's post-hoc adjusted p-values between every pair of groups.",
                requires=("Group A", "Group B", "Z statistic"),
                build=_posthoc,
                notes=("Colour is evidence of a difference (-log10 adjusted p), not its size.",),
            )
        )
    return tuple(out)


def _trends(tool: str) -> tuple[PanelDefinition, ...]:
    return (
        _definition(
            tool,
            "trend",
            title="Trend over time",
            question="Do the values tend to rise or fall over time?",
            shape="line_series",
            summary="Every observation, with the Sen's slope trend line.",
            requires=("Trend Line",),
            build=_trend,
            notes=("Mann-Kendall tests a monotonic tendency; Sen's slope is robust to outliers.",),
        ),
    )


def _correlation_matrix(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, name: str
) -> Result[PreparedPanel]:
    """Every pairwise correlation as a diverging heatmap.

    ``correlation_matrix.csv`` is a square: one row per column, one column
    per column, ``Column`` naming the row axis. Values sit in [-1, 1] by
    construction, so the diverging scale is centred without a rescale.
    """
    del params
    categories = [str(column) for column in frame.columns if column != "Column"]
    if len(categories) < 2:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", "the correlation matrix needs at least two numeric columns")
        )
    rows = [str(value) for value in frame["Column"]]
    if sorted(rows) != sorted(categories):
        return Result.failure(
            Diagnostic.error(
                "PANEL_NO_DATA",
                "the correlation matrix is not square: its rows and columns name different variables",
            )
        )
    working = frame.copy()
    for column in categories:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    cells: list[PanelMark] = []
    for _, row in working.iterrows():
        row_name = str(row["Column"])
        for column in categories:
            value = row[column]
            if pd.isna(value):
                continue
            cells.append(
                PanelMark(
                    key=f"{row_name}|{column}",
                    label=f"{row_name} / {column}",
                    x=float(categories.index(column)),
                    y=float(rows.index(row_name)),
                    value=float(value),
                    evidence=Evidence(
                        scope="rows",
                        filters=(("Column", row_name),),
                        count=1,
                        describe=f"{row_name} against {column}: r = {float(value):+.3f}",
                    ),
                )
            )
    prepared = PreparedPanel(
        panel=name,
        shape="heatmap",
        title="How the numeric columns move together",
        subtitle="Every pairwise correlation; red moves one way, blue the other",
        marks=tuple(cells),
        x_label="Column",
        y_label="Column",
        provenance=provenance,
        data=working,
        x_categories=tuple(categories),
        y_categories=tuple(rows),
        color_scale="diverging",
        value_label="Correlation",
        height=max(500, 40 * len(rows) + 160),
        width=max(900, 90 * len(categories) + 200),
        notes=(
            "Colour is the correlation coefficient: red near +1 (they rise together), blue near -1 (one rises "
            "as the other falls), pale near 0 (no straight-line relationship). A strong correlation is not a "
            "cause: both columns may simply move with a third factor, such as document length.",
            "A correlation needs variation in both columns. A column that never changes is refused by the "
            "engine, not drawn as a blank row.",
        ),
    )
    return Result.success(prepared)


STATS_PANELS: tuple[PanelDefinition, ...] = (
    *_categorical("table_chi2", chi2=True, crosstab=False, keyness=False),
    *_categorical("table_crosstab", chi2=False, crosstab=True, keyness=False),
    *_categorical("table_keyness", chi2=False, crosstab=False, keyness=True),
    *_categorical("stats_categorical", chi2=True, crosstab=True, keyness=True),
    *_groups("table_mw", posthoc=False),
    *_groups("table_kw", posthoc=True),
    *_groups("stats_groups", posthoc=True),
    *_trends("table_trend"),
    *_trends("stats_trends"),
    _definition(
        "csv_stats",
        "correlations",
        title="How the numeric columns move together",
        question="Which pairs of numeric columns rise and fall together, and which move opposite?",
        shape="heatmap",
        summary="Every pairwise correlation, diverging colour, one cell per pair.",
        requires=("Column",),
        build=_correlation_matrix,
        notes=(
            "Colour is the correlation coefficient: red near +1, blue near -1, pale near 0. A strong "
            "correlation is not a cause.",
        ),
    ),
)
