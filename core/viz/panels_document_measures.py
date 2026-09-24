"""Per-document measures: one factory, eight tools.

``readability``, ``lexical_diversity``, ``corpus_statistics``,
``text_statistics``, ``sentence_complexity``, ``verb_analysis`` and
``nominalization`` all produce (or reduce to) one row per document: a score,
a rate, or a handful of them. ``style`` does too, when its lexicon asset is
installed. Eight tools asking the same four questions of a per-document
number -- has it moved over time, does it differ by decade or speaker, is it
really just standing in for document length, and how do several of a tool's
measures move together -- is exactly the situation one factory should serve,
so a reader sees one consistent design across all of them instead of eight
subtly different ones.

:class:`MeasureTool` is the one thing each tool supplies: how to turn its
raw result table into a canonical per-document frame (``Document ID``,
``Document``, ``Date``, one column per measure, a "how big is this document"
column for the hover and the length check), which measures exist, which of
them are known to be length-sensitive, and which deserve a small-multiples
view. Four builders --:func:`_over_time`, :func:`_by_group`,
:func:`_length_check`, :func:`_all_measures` -- read only that contract, so
adding a ninth per-document tool is "write a ``MeasureTool``", not "write
another builder".

Two tools need a real aggregation step before they fit the contract:
``sentence_complexity`` (dependency distance et al. are per sentence) and
``nominalization`` (the by-sentence table lists only sentences that contain
a nominalization). Both aggregate to one row per document in ``prepare``
-- median for sentence scores, sum-of-parts for nominalization's rate -- and
carry the sentence count into the hover so a three-sentence document does not
read as equally solid evidence as an eighty-sentence one. ``style``'s two
facets (concreteness, iconicity) do the same, but its lexicon asset is not
installed on this machine (``core/analysis/style.py``'s
``_resolve_asset_path`` fails), so those panels are built and tested against
hand-shaped fixtures only and are UNVERIFIED against real output; see the
class docstring on ``_STYLE_NOTE`` for what that means for a reader. Their
per-sentence table also carries no ``Date`` column at all, so they get only
a by-group panel -- a trend or a length check needs an axis this table does
not have.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import GROUPINGS, decimal_year, document_labels, group_of, rolling_median
from core.viz.panelspec import Evidence, PanelDefinition, PanelMark, PanelParam, PreparedPanel, Provenance

__all__ = [
    "CORPUS_STATISTICS_ALL_MEASURES",
    "CORPUS_STATISTICS_BY_GROUP",
    "CORPUS_STATISTICS_LENGTH_CHECK",
    "CORPUS_STATISTICS_OVER_TIME",
    "DOCUMENT_MEASURES_PANELS",
    "LEXICAL_DIVERSITY_ALL_MEASURES",
    "LEXICAL_DIVERSITY_BY_GROUP",
    "LEXICAL_DIVERSITY_LENGTH_CHECK",
    "LEXICAL_DIVERSITY_OVER_TIME",
    "NOMINALIZATION_BY_GROUP",
    "NOMINALIZATION_OVER_TIME",
    "READABILITY_ALL_MEASURES",
    "READABILITY_BY_GROUP",
    "READABILITY_LENGTH_CHECK",
    "READABILITY_OVER_TIME",
    "SENTENCE_COMPLEXITY_ALL_MEASURES",
    "SENTENCE_COMPLEXITY_BY_GROUP",
    "SENTENCE_COMPLEXITY_LENGTH_CHECK",
    "SENTENCE_COMPLEXITY_OVER_TIME",
    "STYLE_CONCRETENESS_BY_GROUP",
    "STYLE_ICONICITY_BY_GROUP",
    "TEXT_STATISTICS_ALL_MEASURES",
    "TEXT_STATISTICS_BY_GROUP",
    "TEXT_STATISTICS_OVER_TIME",
    "VERB_ANALYSIS_ALL_MEASURES",
    "VERB_ANALYSIS_BY_GROUP",
    "VERB_ANALYSIS_OVER_TIME",
    "MeasureTool",
    "measure_panels",
    "prepare_passthrough",
]

DOC_ID = "Document ID"
DOC = "Document"
DATE = "Date"

#: A trend needs at least this many dated documents, or "the line" is one or
#: two points wearing a trend's clothing.
_MIN_DATED = 3
#: A dated corpus of SOTU-style annual documents sits about one year apart;
#: the rolling median should read as one line across that normal cadence and
#: break only across a real hole -- several Congresses' worth of silence.
_LINE_GAP_YEARS = 5.0
_WINDOW_DEFAULT = 9
_LABEL_TOP_N = 5


# ------------------------------------------------------------------ contract --


@dataclass(frozen=True, slots=True)
class MeasureTool:
    """What one tool supplies to the shared four-panel factory.

    ``prepare`` turns the tool's raw result table into a canonical
    per-document frame: ``Document ID``, ``Document``, ``Date`` (may be
    absent -- ``style`` has none), one column per name in ``measures``, and
    ``size_column`` (how big this document is, in ``size_unit``, for the
    hover and for a length check).
    """

    tool: str
    requires: tuple[str, ...]
    prepare: Callable[[pd.DataFrame], Result[pd.DataFrame]]
    measures: tuple[str, ...]
    default_measure: str
    size_column: str
    size_unit: str
    length_measures: tuple[str, ...] = ()
    length_column: str = ""
    length_label: str = ""
    default_length_measure: str = ""
    facet_measures: tuple[str, ...] = ()
    notes_by_measure: Mapping[str, str] = field(default_factory=dict)
    general_notes: tuple[str, ...] = ()
    has_dates: bool = True


def _missing(frame: pd.DataFrame, required: tuple[str, ...]) -> list[Diagnostic]:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return [Diagnostic.error("PANEL_MISSING_COLUMN", f"missing column(s): {missing}", missing=missing)]
    if frame.empty:
        return [Diagnostic.error("PANEL_NO_DATA", "the result table is empty")]
    return []


def _fmt_size(value: object, unit: str) -> str:
    """ "; 2,224 words", or "" when the size is unknown."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if math.isnan(number):
        return ""
    text = f"{round(number):,}" if number == int(number) else f"{number:,.2f}"
    return f"; {text} {unit}"


# ------------------------------------------------------------ prepare: plain --
# readability, lexical_diversity, corpus_statistics and text_statistics are
# already one row per document; nothing to aggregate.


def _prepare_passthrough(required: tuple[str, ...]) -> Callable[[pd.DataFrame], Result[pd.DataFrame]]:
    def prepare(frame: pd.DataFrame) -> Result[pd.DataFrame]:
        diags = _missing(frame, required)
        if diags:
            return Result.failure(*diags)
        return Result.success(frame.copy())

    return prepare


# ------------------------------------------------------- prepare: aggregated --


def _prepare_sentence_complexity(frame: pd.DataFrame) -> Result[pd.DataFrame]:
    """Median dependency distance et al. per document, from the sentence rows."""
    required = (
        DOC_ID,
        DOC,
        DATE,
        "Sentence ID",
        "Tokens",
        "Mean Dependency Distance",
        "Max Dependency Distance",
        "Depth",
        "Subordinate Clauses",
    )
    diags = _missing(frame, required)
    if diags:
        return Result.failure(*diags)
    working = frame.copy()
    for column in ("Tokens", "Mean Dependency Distance", "Max Dependency Distance", "Depth", "Subordinate Clauses"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    grouped = working.groupby(DOC_ID, as_index=False).agg(
        **{
            DOC: (DOC, "first"),
            DATE: (DATE, "first"),
            "Mean Dependency Distance": ("Mean Dependency Distance", "median"),
            "Max Dependency Distance": ("Max Dependency Distance", "median"),
            "Depth": ("Depth", "median"),
            "Subordinate Clauses": ("Subordinate Clauses", "median"),
            # The length check's x axis: the document's typical sentence
            # length, not its total word count -- dependency distance is a
            # per-sentence measure, so it is sentence length it can trade on.
            "Sentence Tokens (median)": ("Tokens", "median"),
            "_n": ("Sentence ID", "count"),
        }
    )
    return Result.success(grouped)


def _prepare_nominalization(frame: pd.DataFrame) -> Result[pd.DataFrame]:
    """A document's nominalization rate: its nominalizations over its words,
    both summed from the by-sentence table.

    That table lists only sentences containing at least one nominalization
    (``core/analysis/nominalization.py::sentence_frequency``), so the word
    count summed here is smaller than the document's true word total --
    sentences with zero nominalizations never appear. The rate is honest
    about what it counts (nominalizations per word *among nominalization-
    bearing sentences*), and the notes say so; it is the only per-document
    nominalization rate this table can produce without the per-word table,
    which belongs to another panel.
    """
    required = (DOC_ID, DOC, DATE, "Sentence ID", "Words in Sentence", "Nominalizations in Sentence")
    diags = _missing(frame, required)
    if diags:
        return Result.failure(*diags)
    working = frame.copy()
    working["Words in Sentence"] = pd.to_numeric(working["Words in Sentence"], errors="coerce")
    working["Nominalizations in Sentence"] = pd.to_numeric(working["Nominalizations in Sentence"], errors="coerce")
    grouped = working.groupby(DOC_ID, as_index=False).agg(
        **{
            DOC: (DOC, "first"),
            DATE: (DATE, "first"),
            "_words": ("Words in Sentence", "sum"),
            "_noms": ("Nominalizations in Sentence", "sum"),
            "_n": ("Sentence ID", "count"),
        }
    )
    grouped["Nominalization %"] = np.where(grouped["_words"] > 0, 100.0 * grouped["_noms"] / grouped["_words"], np.nan)
    return Result.success(grouped)


_VERB_RATE_COLUMNS: dict[str, str] = {
    "% Passive": "Passive",
    "% Active": "Active",
    "% Past": "Past",
    "% Present": "Present",
    "% Future": "Future",
    "% Gerund": "Gerund",
    "% Ability": "Ability",
    "% Possibility": "Possibility",
    "% Obligation": "Obligation",
}


def _prepare_verb_analysis(frame: pd.DataFrame) -> Result[pd.DataFrame]:
    """Voice/tense/modality rates per document, from ``verb_summary.csv``'s
    raw counts -- a house rule (normalise pooled counts) applied to counts
    that would otherwise just measure how many verbs a speech had."""
    required = (DOC_ID, DOC, DATE, "Verbs", *_VERB_RATE_COLUMNS.values())
    diags = _missing(frame, required)
    if diags:
        return Result.failure(*diags)
    working = frame.copy()
    working["Verbs"] = pd.to_numeric(working["Verbs"], errors="coerce")
    for label, column in _VERB_RATE_COLUMNS.items():
        working[column] = pd.to_numeric(working[column], errors="coerce")
        working[label] = np.where(working["Verbs"] > 0, 100.0 * working[column] / working["Verbs"], np.nan)
    return Result.success(working)


def _prepare_style(mean_column: str, extra_required: tuple[str, ...]) -> Callable[[pd.DataFrame], Result[pd.DataFrame]]:
    """Median per-sentence rating -> one value per document.

    ``style``'s per-sentence tables carry no ``Date``; the caller adds a
    placeholder so the shared grouping code still has a column to read.
    """

    def prepare(frame: pd.DataFrame) -> Result[pd.DataFrame]:
        required = (DOC_ID, DOC, "Sentence ID", mean_column, *extra_required)
        diags = _missing(frame, required)
        if diags:
            return Result.failure(*diags)
        working = frame.copy()
        working[mean_column] = pd.to_numeric(working[mean_column], errors="coerce")
        grouped = working.groupby(DOC_ID, as_index=False).agg(
            **{
                DOC: (DOC, "first"),
                f"{mean_column} (median)": (mean_column, "median"),
                "_n": ("Sentence ID", "count"),
            }
        )
        grouped[DATE] = None
        return Result.success(grouped)

    return prepare


# ------------------------------------------------------------------ builders --


def _canonical(frame: pd.DataFrame, tool: MeasureTool) -> Result[pd.DataFrame]:
    prepared = tool.prepare(frame)
    if prepared.value is None:
        return prepared
    canon = prepared.unwrap()
    if DATE not in canon.columns:
        canon = canon.copy()
        canon[DATE] = None
    return Result.success(canon, *prepared.diagnostics)


def _over_time(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
    *,
    tool: MeasureTool,
    definition: PanelDefinition,
) -> Result[PreparedPanel]:
    canon = _canonical(frame, tool)
    if canon.value is None:
        return Result.failure(*canon.diagnostics)
    working = canon.unwrap()
    diags = list(canon.diagnostics)

    measure = str(params["measure"])
    window = int(params["window"])
    if window < 1 or window % 2 == 0:
        return Result.failure(
            Diagnostic.error("PANEL_BAD_WINDOW", f"window must be a positive odd number of documents, got {window}"),
            *diags,
        )

    working = working.copy()
    working["_year"] = working[DATE].map(decimal_year)
    working[measure] = pd.to_numeric(working[measure], errors="coerce")
    undated = int(working["_year"].isna().sum())
    usable = working.dropna(subset=["_year", measure]).copy()
    if len(usable) < _MIN_DATED:
        return Result.failure(
            Diagnostic.error(
                "PANEL_TOO_FEW_DATED",
                f"only {len(usable)} dated document(s) have a usable {measure} value; a trend needs at least "
                f"{_MIN_DATED}. {undated} document(s) have no usable date and were left off.",
                dated=len(usable),
                undated=undated,
            ),
            *diags,
        )
    if undated:
        diags.append(
            Diagnostic.info(
                "PANEL_UNDATED_DROPPED",
                f"{undated} document(s) have no usable date and were left off this trend.",
                dropped=undated,
            )
        )
    if window > len(usable):
        diags.append(
            Diagnostic.warning(
                "PANEL_WINDOW_TOO_WIDE",
                f"window ({window}) is wider than the {len(usable)} dated document(s); no rolling median can be drawn.",
                window=window,
                dated=len(usable),
            )
        )

    labels = document_labels(usable[DOC])
    marks: list[PanelMark] = []
    for _, row in usable.iterrows():
        doc_id = str(row[DOC_ID])
        label = labels[str(row[DOC])]
        value = float(row[measure])
        year = float(row["_year"])
        size_text = _fmt_size(row.get(tool.size_column), tool.size_unit)
        marks.append(
            PanelMark(
                key=f"doc:{doc_id}",
                label=label,
                x=year,
                y=value,
                group="Documents",
                evidence=Evidence(
                    scope="rows",
                    filters=((DOC_ID, doc_id),),
                    count=1,
                    describe=f"{label}: {measure} {value:g}{size_text}",
                ),
            )
        )

    median_group = f"Rolling median ({window} documents)"
    groups = ("Documents", median_group)
    if window <= len(usable):
        sortable = sorted(
            zip(
                usable["_year"],
                usable[measure],
                usable[DOC_ID].astype(str),
                usable[DOC].astype(str),
                usable[DATE],
                strict=True,
            ),
            key=lambda item: (item[0], item[1]),
        )
        xs = [item[0] for item in sortable]
        ys = [item[1] for item in sortable]
        pairs = rolling_median(xs, ys, window)
        for offset, (mx, my) in enumerate(pairs):
            covered = sortable[_window_start(offset, window, len(sortable)) :][:window]
            _center_year, _center_val, center_id, center_doc, _center_date = sortable[offset]
            center_label = labels.get(center_doc, center_doc)
            span_dates = sorted(str(item[4]) for item in covered if item[4])
            span = (
                f"{span_dates[0]} to {span_dates[-1]}" if span_dates else f"{covered[0][0]:.1f} to {covered[-1][0]:.1f}"
            )
            marks.append(
                PanelMark(
                    key=f"median:{center_id}:{offset}",
                    label="Rolling median",
                    x=float(mx),
                    y=float(my),
                    group=median_group,
                    evidence=Evidence(
                        scope="rows",
                        filters=((DOC_ID, center_id),),
                        count=window,
                        describe=(
                            f"Rolling median of {window} documents from {span}: {measure} {my:g} (at {center_label})"
                        ),
                    ),
                )
            )

    prepared = PreparedPanel(
        panel=definition.name,
        shape="line_series",
        title=definition.title,
        subtitle=f"{len(usable)} dated document(s) · rolling median over {window} documents",
        marks=tuple(marks),
        x_label="Year",
        y_label=measure,
        provenance=provenance,
        data=usable.drop(columns=["_year"]).reset_index(drop=True),
        groups=groups,
        points_only=("Documents",),
        line_gap=_LINE_GAP_YEARS,
        notes=_notes_for(tool, measure),
    )
    return Result.success(prepared, *diags)


def _group_order(working: pd.DataFrame, group_by: str) -> list[str]:
    groups = sorted(set(working["_group"]))
    if group_by in ("year", "decade"):

        def key(name: str) -> tuple[int, int]:
            if not name[:1].isdigit():
                return (1, 0)
            return (0, int(name.rstrip("s")))

        return sorted(groups, key=key)
    # speaker: chronological by the group's earliest document.
    first_year: dict[str, float] = {}
    for name in groups:
        years = working.loc[working["_group"] == name, "_year"].dropna()
        first_year[name] = float(years.min()) if not years.empty else math.inf
    return sorted(groups, key=lambda name: (first_year[name] == math.inf, first_year[name]))


def _window_start(index: int, window: int, count: int) -> int:
    """Where ``rolling_median``'s window for point ``index`` starts: centred
    where it fits, the first or last ``window`` points near an end."""
    return min(max(0, index - window // 2), count - window)


#: More groups than this share one colour (the row already names the group).
_MAX_COLOURED_GROUPS = 8


def _by_group(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
    *,
    tool: MeasureTool,
    definition: PanelDefinition,
) -> Result[PreparedPanel]:
    canon = _canonical(frame, tool)
    if canon.value is None:
        return Result.failure(*canon.diagnostics)
    working = canon.unwrap().copy()
    diags = list(canon.diagnostics)

    measure = str(params["measure"])
    group_by = str(params["group-by"])
    working[measure] = pd.to_numeric(working[measure], errors="coerce")
    bad = int(working[measure].isna().sum())
    working = working.dropna(subset=[measure]).copy()
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"no document has a usable {measure} value"), *diags)
    if bad:
        diags.append(
            Diagnostic.info("PANEL_BAD_NUMERIC", f"{bad} document(s) had no usable {measure} value and were left off.")
        )
    if not tool.has_dates and group_by in ("year", "decade"):
        diags.append(
            Diagnostic.warning(
                "PANEL_NO_DATES",
                f"{tool.tool}'s table carries no per-document date, so every document falls into one "
                "'(undated)' group; group by speaker instead to separate documents.",
            )
        )

    working["_year"] = working[DATE].map(decimal_year)
    working["_group"] = [
        group_of(str(doc), when, group_by) for doc, when in zip(working[DOC], working[DATE], strict=True)
    ]
    order = _group_order(working, group_by)
    index_of = {name: index for index, name in enumerate(order)}
    labels = document_labels(working[DOC])
    sizes = working["_group"].value_counts()
    if len(order) > 1 and float(sizes.median()) <= 1:
        # One speech a year: grouping by year makes every "distribution" a
        # single point, and the figure a slower way to draw a timeline.
        diags.append(
            Diagnostic.warning(
                "PANEL_SINGLETON_GROUPS",
                f"Most {group_by} groups hold a single document, so each box is one point. Group by "
                "decade to compare spreads, or use the over-time figure to follow single documents.",
            )
        )
    # Colour repeats the row a point sits in; past eight groups it only adds
    # a legend as long as the axis, so the points share one colour.
    coloured = len(order) <= _MAX_COLOURED_GROUPS

    marks: list[PanelMark] = []
    for _, row in working.iterrows():
        doc_id = str(row[DOC_ID])
        label = labels[str(row[DOC])]
        group = row["_group"]
        value = float(row[measure])
        size_text = _fmt_size(row.get(tool.size_column), tool.size_unit)
        marks.append(
            PanelMark(
                key=f"doc:{doc_id}",
                label=label,
                x=value,
                y=float(index_of[group]),
                group=group if coloured else "",
                evidence=Evidence(
                    scope="rows",
                    filters=((DOC_ID, doc_id),),
                    count=1,
                    describe=f"{label} ({group}): {measure} {value:g}{size_text}",
                ),
            )
        )

    prepared = PreparedPanel(
        panel=definition.name,
        shape="distribution",
        title=definition.title,
        subtitle=f"{len(working)} document(s) across {len(order)} {group_by} group(s)",
        marks=tuple(marks),
        x_label=measure,
        y_label=group_by.capitalize(),
        provenance=provenance,
        data=working.drop(columns=["_year", "_group"]).reset_index(drop=True),
        groups=tuple(order) if coloured else (),
        y_categories=tuple(order),
        notes=_notes_for(tool, measure),
    )
    return Result.success(prepared, *diags)


def _length_check(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
    *,
    tool: MeasureTool,
    definition: PanelDefinition,
) -> Result[PreparedPanel]:
    canon = _canonical(frame, tool)
    if canon.value is None:
        return Result.failure(*canon.diagnostics)
    working = canon.unwrap().copy()
    diags = list(canon.diagnostics)

    measure = str(params["measure"])
    working[measure] = pd.to_numeric(working[measure], errors="coerce")
    working[tool.length_column] = pd.to_numeric(working[tool.length_column], errors="coerce")
    usable = working.dropna(subset=[measure, tool.length_column]).copy()
    if len(usable) < _MIN_DATED:
        return Result.failure(
            Diagnostic.error(
                "PANEL_NO_DATA",
                f"only {len(usable)} document(s) have both a usable {measure} and {tool.length_column} value; "
                f"a length check needs at least {_MIN_DATED}.",
            ),
            *diags,
        )

    labels = document_labels(usable[DOC])
    median_value = float(usable[measure].median())
    usable["_extremity"] = (usable[measure] - median_value).abs()
    labelled_ids = set(usable.sort_values("_extremity", ascending=False).head(_LABEL_TOP_N)[DOC_ID].astype(str))

    marks: list[PanelMark] = []
    for _, row in usable.iterrows():
        doc_id = str(row[DOC_ID])
        label = labels[str(row[DOC])]
        value = float(row[measure])
        length = float(row[tool.length_column])
        marks.append(
            PanelMark(
                key=f"doc:{doc_id}",
                label=label,
                x=length,
                y=value,
                labelled=doc_id in labelled_ids,
                evidence=Evidence(
                    scope="rows",
                    filters=((DOC_ID, doc_id),),
                    count=1,
                    describe=f"{label}: {measure} {value:g}; {tool.length_label} {length:g}",
                ),
            )
        )

    prepared = PreparedPanel(
        panel=definition.name,
        shape="scatter_labelled",
        title=definition.title,
        subtitle=f"{len(usable)} document(s) · {measure} against {tool.length_label.lower()}",
        marks=tuple(marks),
        x_label=tool.length_label,
        y_label=measure,
        provenance=provenance,
        data=usable.drop(columns=["_extremity"]).reset_index(drop=True),
        notes=(
            *_notes_for(tool, measure),
            "A strong trend across this scatter means the measure is standing in for document length rather "
            "than describing something distinct about the document's content.",
        ),
    )
    return Result.success(prepared, *diags)


def _all_measures(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
    *,
    tool: MeasureTool,
    definition: PanelDefinition,
) -> Result[PreparedPanel]:
    canon = _canonical(frame, tool)
    if canon.value is None:
        return Result.failure(*canon.diagnostics)
    working = canon.unwrap().copy()
    diags = list(canon.diagnostics)

    working["_year"] = working[DATE].map(decimal_year)
    undated = int(working["_year"].isna().sum())
    usable = working.dropna(subset=["_year"]).copy()
    if len(usable) < _MIN_DATED:
        return Result.failure(
            Diagnostic.error(
                "PANEL_TOO_FEW_DATED",
                f"only {len(usable)} dated document(s) are available; small multiples need at least {_MIN_DATED}. "
                f"{undated} document(s) have no usable date and were left off.",
            ),
            *diags,
        )
    if undated:
        diags.append(
            Diagnostic.info(
                "PANEL_UNDATED_DROPPED",
                f"{undated} document(s) have no usable date and were left off.",
                dropped=undated,
            )
        )

    labels = document_labels(usable[DOC])
    window = min(_WINDOW_DEFAULT, len(usable) if len(usable) % 2 else len(usable) - 1)
    window = max(window, 1)
    marks: list[PanelMark] = []
    for facet in tool.facet_measures:
        facet_values = pd.to_numeric(usable[facet], errors="coerce")
        facet_frame = usable.assign(_value=facet_values).dropna(subset=["_value"])
        if facet_frame.empty:
            diags.append(Diagnostic.warning("PANEL_NO_DATA", f"no document has a usable {facet} value; facet skipped."))
            continue
        for _, row in facet_frame.iterrows():
            doc_id = str(row[DOC_ID])
            label = labels[str(row[DOC])]
            value = float(row["_value"])
            year = float(row["_year"])
            size_text = _fmt_size(row.get(tool.size_column), tool.size_unit)
            marks.append(
                PanelMark(
                    key=f"doc:{facet}:{doc_id}",
                    label=label,
                    x=year,
                    y=value,
                    group="Documents",
                    facet=facet,
                    evidence=Evidence(
                        scope="rows",
                        filters=((DOC_ID, doc_id),),
                        count=1,
                        describe=f"{label}: {facet} {value:g}{size_text}",
                    ),
                )
            )
        if window <= len(facet_frame) and window >= 3:
            sortable = sorted(
                zip(facet_frame["_year"], facet_frame["_value"], facet_frame[DOC_ID].astype(str), strict=True)
            )
            xs = [item[0] for item in sortable]
            ys = [item[1] for item in sortable]
            pairs = rolling_median(xs, ys, window)
            for offset, (mx, my) in enumerate(pairs):
                center_id = sortable[offset][2]
                marks.append(
                    PanelMark(
                        key=f"median:{facet}:{center_id}:{offset}",
                        label="Rolling median",
                        x=float(mx),
                        y=float(my),
                        group="Rolling median",
                        facet=facet,
                        evidence=Evidence(
                            scope="rows",
                            filters=((DOC_ID, center_id),),
                            count=window,
                            describe=f"Rolling median of {window} documents: {facet} {my:g} (at document id {center_id})",
                        ),
                    )
                )

    if not marks:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no facet had a usable value"), *diags)

    prepared = PreparedPanel(
        panel=definition.name,
        shape="small_multiples",
        title=definition.title,
        subtitle=f"{len(usable)} dated document(s) · {len(tool.facet_measures)} measure(s), each its own scale",
        marks=tuple(marks),
        x_label="Year",
        y_label="",
        provenance=provenance,
        data=usable.drop(columns=["_year"]).reset_index(drop=True),
        groups=("Documents", "Rolling median"),
        points_only=("Documents",),
        facets=tool.facet_measures,
        line_gap=_LINE_GAP_YEARS,
        notes=tuple(tool.general_notes)
        + tuple(tool.notes_by_measure.get(m, "") for m in tool.facet_measures if tool.notes_by_measure.get(m)),
    )
    return Result.success(prepared, *diags)


def _notes_for(tool: MeasureTool, measure: str) -> tuple[str, ...]:
    specific = tool.notes_by_measure.get(measure, "")
    return tuple(x for x in (specific, *tool.general_notes) if x)


# ------------------------------------------------------------------ registry --


def _measure_param(tool: MeasureTool) -> PanelParam:
    return PanelParam(
        name="measure",
        type="choice",
        default=tool.default_measure,
        choices=tool.measures,
        label="Measure",
        help="Which of this tool's per-document measures to plot.",
    )


def _window_param() -> PanelParam:
    return PanelParam(
        name="window",
        type="int",
        default=_WINDOW_DEFAULT,
        minimum=3,
        maximum=51,
        label="Smoothing window",
        help="Rolling median width, in documents (must be odd).",
    )


def _group_by_param() -> PanelParam:
    return PanelParam(
        name="group-by",
        type="choice",
        default="decade",
        choices=tuple(g for g in GROUPINGS if g != "none"),
        label="Group by",
        help="How to bucket documents: by year, by decade, or by speaker (parsed from the file name).",
    )


def _length_measure_param(tool: MeasureTool) -> PanelParam:
    return PanelParam(
        name="measure",
        type="choice",
        default=tool.default_length_measure,
        choices=tool.length_measures,
        label="Measure",
        help="Which length-sensitive measure to check against document length.",
    )


def _definition_notes(tool: MeasureTool) -> tuple[str, ...]:
    """The limits a generated panel declares: the tool's general notes plus
    each measure's own, de-duplicated in order.

    The registry requires every panel to state what it cannot show. A tool
    whose caveats all live per measure (``lexical_diversity``: TTR's length
    trap; ``text_statistics``: what each column means) would otherwise emit a
    definition with empty notes and be rejected, even though the figure it
    draws carries them.
    """
    notes = [*tool.general_notes]
    notes.extend(note for note in tool.notes_by_measure.values() if note)
    return tuple(dict.fromkeys(notes))


def _in_sentence(title: str) -> str:
    """A title word as it reads mid-sentence: "Readability" -> "readability",
    but "VADER tone" and "BERT tone" keep their acronyms."""
    words = title.split(" ", 1)
    first = words[0]
    if len(first) > 1 and first[1:].islower():
        first = first.lower()
    return " ".join([first, *words[1:]])


#: Every MeasureTool the factory has built, by tool name, in build order --
#: for figures drawn over a tool's whole table (``core/viz/static/bundles``).
MEASURE_TOOLS: dict[str, list[MeasureTool]] = {}


def _factory(tool: MeasureTool, *, title_word: str, prefix: str) -> tuple[PanelDefinition, ...]:
    """Build every panel this tool qualifies for, from its ``MeasureTool`` contract."""
    MEASURE_TOOLS.setdefault(tool.tool, []).append(tool)
    definitions: list[PanelDefinition] = []
    definition_notes = _definition_notes(tool)

    if tool.has_dates:
        holder: dict[str, PanelDefinition] = {}

        def build_over_time(
            frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, _tool: MeasureTool = tool
        ) -> Result[PreparedPanel]:
            return _over_time(frame, params, provenance, tool=_tool, definition=holder["definition"])

        over_time = PanelDefinition(
            name=f"{prefix}_over_time",
            title=f"{title_word} over time",
            tool=tool.tool,
            shape="line_series",
            summary="Each dated document's score on the chosen measure, plotted over time, with a rolling median.",
            question=f"Has {_in_sentence(title_word)} changed over time in this corpus?",
            requires=tool.requires,
            params=(_measure_param(tool), _window_param()),
            build=build_over_time,
            notes=definition_notes,
        )
        holder["definition"] = over_time
        definitions.append(over_time)

    holder2: dict[str, PanelDefinition] = {}

    def build_by_group(
        frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, _tool: MeasureTool = tool
    ) -> Result[PreparedPanel]:
        return _by_group(frame, params, provenance, tool=_tool, definition=holder2["definition"])

    by_group = PanelDefinition(
        name=f"{prefix}_by_group",
        title=f"{title_word} by group",
        tool=tool.tool,
        shape="distribution",
        summary="Every document's score on the chosen measure, grouped by decade, speaker or year: box plus points.",
        question=f"How does {_in_sentence(title_word)} differ across decades or speakers?",
        requires=tool.requires,
        params=(_measure_param(tool), _group_by_param()),
        build=build_by_group,
        notes=definition_notes,
    )
    holder2["definition"] = by_group
    definitions.append(by_group)

    if tool.length_measures:
        holder3: dict[str, PanelDefinition] = {}

        def build_length_check(
            frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, _tool: MeasureTool = tool
        ) -> Result[PreparedPanel]:
            return _length_check(frame, params, provenance, tool=_tool, definition=holder3["definition"])

        length_check = PanelDefinition(
            name=f"{prefix}_length_check",
            title=f"{title_word}: is it just document length?",
            tool=tool.tool,
            shape="scatter_labelled",
            summary=f"The chosen measure against {tool.length_label.lower()}, to check whether it only tracks length.",
            question=f"Is this {_in_sentence(title_word)} measure actually just a proxy for how long the document is?",
            requires=tool.requires,
            params=(_length_measure_param(tool),),
            build=build_length_check,
            notes=definition_notes,
        )
        holder3["definition"] = length_check
        definitions.append(length_check)

    if len(tool.facet_measures) >= 3:
        holder4: dict[str, PanelDefinition] = {}

        def build_all_measures(
            frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, _tool: MeasureTool = tool
        ) -> Result[PreparedPanel]:
            return _all_measures(frame, params, provenance, tool=_tool, definition=holder4["definition"])

        all_measures = PanelDefinition(
            name=f"{prefix}_all_measures",
            title=f"{title_word}: all measures",
            tool=tool.tool,
            shape="small_multiples",
            summary=f"Every {_in_sentence(title_word)} measure over time, each on its own scale, side by side.",
            question=f"How do {_in_sentence(title_word)}'s measures move together across this corpus?",
            requires=tool.requires,
            params=(),
            build=build_all_measures,
            notes=definition_notes,
        )
        holder4["definition"] = all_measures
        definitions.append(all_measures)

    return tuple(definitions)


# --------------------------------------------------------------------- tools --

_READABILITY_GRADE_NOTE = (
    "Flesch Reading Ease runs the opposite way to the grade measures: higher Flesch Reading Ease means "
    "easier reading (0-121 clamp), while the Flesch-Kincaid, Gunning Fog, Coleman-Liau, Automated Readability "
    "and SMOG indices all run the other way -- a higher number means a higher US school grade is needed to "
    "read it comfortably."
)
_READABILITY = MeasureTool(
    tool="readability",
    requires=(
        DOC_ID,
        DOC,
        DATE,
        "Words",
        "Flesch Reading Ease",
        "Flesch-Kincaid Grade",
        "Gunning Fog Index",
        "Coleman-Liau Index",
        "Automated Readability Index",
        "SMOG Index",
    ),
    prepare=_prepare_passthrough(
        (
            DOC_ID,
            DOC,
            DATE,
            "Words",
            "Flesch Reading Ease",
            "Flesch-Kincaid Grade",
            "Gunning Fog Index",
            "Coleman-Liau Index",
            "Automated Readability Index",
            "SMOG Index",
        )
    ),
    measures=(
        "Flesch Reading Ease",
        "Flesch-Kincaid Grade",
        "Gunning Fog Index",
        "Coleman-Liau Index",
        "Automated Readability Index",
        "SMOG Index",
    ),
    default_measure="Flesch Reading Ease",
    size_column="Words",
    size_unit="words",
    length_measures=(
        "Flesch-Kincaid Grade",
        "Gunning Fog Index",
        "Coleman-Liau Index",
        "Automated Readability Index",
        "SMOG Index",
    ),
    length_column="Words",
    length_label="Words",
    default_length_measure="Flesch-Kincaid Grade",
    facet_measures=(
        "Flesch Reading Ease",
        "Flesch-Kincaid Grade",
        "Gunning Fog Index",
        "Coleman-Liau Index",
        "Automated Readability Index",
        "SMOG Index",
    ),
    notes_by_measure=dict.fromkeys(
        (
            "Flesch Reading Ease",
            "Flesch-Kincaid Grade",
            "Gunning Fog Index",
            "Coleman-Liau Index",
            "Automated Readability Index",
            "SMOG Index",
        ),
        _READABILITY_GRADE_NOTE,
    ),
    general_notes=(
        "Each formula is a ratio of sentence length and word/syllable length, not a pooled count, so it does "
        "not mechanically grow with document length the way a raw count would -- but a short document (few "
        "sentences to average over) can still give a noisy, unstable estimate.",
        "SMOG below 3 sentences is degenerate and reported as 0.0 by the analysis module.",
    ),
)

_LEXDIV_NOTE = (
    "TTR falls mechanically as a document gets longer, because new tokens are increasingly likely to repeat an "
    "earlier type -- it is not comparable across documents of different lengths. Root TTR (Guiraud) and Log TTR "
    "(Herdan) correct for this only partially. MTLD and vocd-D are built specifically to be length-corrected and "
    "are the safer measures for comparing documents of different lengths."
)
_LEXICAL_DIVERSITY = MeasureTool(
    tool="lexical_diversity",
    requires=(DOC_ID, DOC, DATE, "Total Tokens", "TTR", "Root TTR (Guiraud)", "Log TTR (Herdan)", "MTLD", "vocd-D"),
    prepare=_prepare_passthrough(
        (DOC_ID, DOC, DATE, "Total Tokens", "TTR", "Root TTR (Guiraud)", "Log TTR (Herdan)", "MTLD", "vocd-D")
    ),
    measures=("TTR", "Root TTR (Guiraud)", "Log TTR (Herdan)", "MTLD", "vocd-D"),
    default_measure="MTLD",
    size_column="Total Tokens",
    size_unit="tokens",
    length_measures=("TTR", "Root TTR (Guiraud)", "Log TTR (Herdan)"),
    length_column="Total Tokens",
    length_label="Total Tokens",
    default_length_measure="TTR",
    facet_measures=("TTR", "Root TTR (Guiraud)", "Log TTR (Herdan)", "MTLD", "vocd-D"),
    notes_by_measure=dict.fromkeys(("TTR", "Root TTR (Guiraud)", "Log TTR (Herdan)", "MTLD", "vocd-D"), _LEXDIV_NOTE),
    general_notes=(),
)

_CORPUS_STATS_NOTE = (
    "TTR, Root TTR and Log TTR fall mechanically as a document gets longer -- they are not comparable across "
    "documents of different lengths. Yule's K is comparatively robust to document length, but it runs the "
    "opposite way to TTR: a HIGHER Yule K means MORE repetition (less lexical diversity), not more."
)
_CORPUS_STATISTICS = MeasureTool(
    tool="corpus_statistics",
    requires=(DOC_ID, DOC, DATE, "Tokens", "TTR", "Root TTR", "Log TTR", "Yule K"),
    prepare=_prepare_passthrough((DOC_ID, DOC, DATE, "Tokens", "TTR", "Root TTR", "Log TTR", "Yule K")),
    measures=("TTR", "Root TTR", "Log TTR", "Yule K"),
    default_measure="Yule K",
    size_column="Tokens",
    size_unit="tokens",
    length_measures=("TTR", "Root TTR", "Log TTR"),
    length_column="Tokens",
    length_label="Tokens",
    default_length_measure="TTR",
    facet_measures=("TTR", "Root TTR", "Log TTR", "Yule K"),
    notes_by_measure=dict.fromkeys(("TTR", "Root TTR", "Log TTR", "Yule K"), _CORPUS_STATS_NOTE),
    general_notes=("Herdan is this table's alias for Log TTR (identical values); it is not offered separately here.",),
)

_TEXT_STATISTICS = MeasureTool(
    tool="text_statistics",
    requires=(
        DOC_ID,
        DOC,
        DATE,
        "Tokens",
        "Sentences",
        "Avg Sentence Length",
        "Avg Word Length",
        "Avg Syllables per Word",
    ),
    prepare=_prepare_passthrough(
        (DOC_ID, DOC, DATE, "Tokens", "Sentences", "Avg Sentence Length", "Avg Word Length", "Avg Syllables per Word")
    ),
    measures=("Tokens", "Sentences", "Avg Sentence Length", "Avg Word Length", "Avg Syllables per Word"),
    default_measure="Avg Sentence Length",
    size_column="Tokens",
    size_unit="tokens",
    facet_measures=("Tokens", "Sentences", "Avg Sentence Length"),
    notes_by_measure={
        "Tokens": "A raw document length in tokens -- longer documents are not 'better' or 'worse', just longer.",
        "Sentences": "A raw sentence count; it moves with document length, unlike the per-sentence averages.",
        "Avg Sentence Length": "Tokens per sentence, already a per-document rate; short documents (few sentences) give noisier averages.",
        "Avg Word Length": "Characters per word, already a per-document rate.",
        "Avg Syllables per Word": "Syllables per word from this tool's own syllable heuristic, which differs slightly from the readability tool's.",
    },
    general_notes=(),
)

_DEP_DISTANCE_NOTE = (
    "Mean Dependency Distance is known to correlate with sentence length: a longer sentence has more chance "
    "of a long dependency link. This document-level figure is the MEDIAN across the document's scored "
    "sentences, so one unusually long or short sentence does not dominate it."
)
_SENTENCE_COMPLEXITY = MeasureTool(
    tool="sentence_complexity",
    requires=(
        DOC_ID,
        DOC,
        DATE,
        "Sentence ID",
        "Tokens",
        "Mean Dependency Distance",
        "Max Dependency Distance",
        "Depth",
        "Subordinate Clauses",
    ),
    prepare=_prepare_sentence_complexity,
    measures=("Mean Dependency Distance", "Max Dependency Distance", "Depth", "Subordinate Clauses"),
    default_measure="Mean Dependency Distance",
    size_column="_n",
    size_unit="scored sentences",
    length_measures=("Mean Dependency Distance",),
    length_column="Sentence Tokens (median)",
    length_label="Median sentence length (tokens)",
    default_length_measure="Mean Dependency Distance",
    facet_measures=("Mean Dependency Distance", "Max Dependency Distance", "Depth", "Subordinate Clauses"),
    notes_by_measure={"Mean Dependency Distance": _DEP_DISTANCE_NOTE},
    general_notes=(
        "Every value here is a median over one document's sentences, not a single measurement; the hover "
        "names how many sentences it was computed from.",
        "Single-token sentences score a dependency distance of 0 (no head to measure against) and pull the "
        "document's median toward 0 when they are common.",
    ),
)

_VERB_ANALYSIS_NOTE_BASE = (
    "This is a share of the document's own verb tokens (0-100%), not a share of sentences -- one sentence can "
    "contain both active and passive verbs. Documents with few verb tokens give noisy rates; check the verb "
    "count in the hover before trusting an extreme value."
)
_VERB_ANALYSIS = MeasureTool(
    tool="verb_analysis",
    requires=(DOC_ID, DOC, DATE, "Verbs", *_VERB_RATE_COLUMNS.values()),
    prepare=_prepare_verb_analysis,
    measures=tuple(_VERB_RATE_COLUMNS.keys()),
    default_measure="% Passive",
    size_column="Verbs",
    size_unit="verb tokens",
    facet_measures=("% Passive", "% Past", "% Present", "% Future", "% Gerund", "% Obligation"),
    notes_by_measure=dict.fromkeys(_VERB_RATE_COLUMNS.keys(), _VERB_ANALYSIS_NOTE_BASE),
    general_notes=(
        "Tense here is 'past'/'present'/'future'/'gerund' as detected by morphology, POS tag or word ending, in "
        "that preference order (documented in core/analysis/verb_analysis.py); 'infinitive', 'participle' and "
        "'unknown' verbs count in the document's total Verbs but have no dedicated rate here.",
        "'% Passive' and '% Active' are computed from the same Verbs denominator; a verb with unknown voice "
        "counts in neither, so the two need not sum to 100%.",
    ),
)

_NOMINALIZATION = MeasureTool(
    tool="nominalization",
    requires=(DOC_ID, DOC, DATE, "Sentence ID", "Words in Sentence", "Nominalizations in Sentence"),
    prepare=_prepare_nominalization,
    measures=("Nominalization %",),
    default_measure="Nominalization %",
    size_column="_words",
    size_unit="words (nominalization-bearing sentences only)",
    general_notes=(
        "This rate is nominalizations divided by words, both summed only over the sentences that contain at "
        "least one nominalization -- sentences with none never appear in this table, so the word count in the "
        "denominator is smaller than the document's true word total. Read it as a rate among nominalization-"
        "bearing sentences, not a whole-document rate.",
        "The hover names how many nominalization-bearing sentences the document had; a document with very few "
        "such sentences gives a noisy rate.",
    ),
)

_STYLE_NOTE = (
    "UNVERIFIED: the lexicon asset this measure needs (Brysbaert concreteness / Winter iconicity norms) is not "
    "installed on this machine, so core/analysis/style.py cannot produce real output here and this panel has "
    "never been checked against a real run. It is built and tested only against hand-shaped fixtures using the "
    "analysis module's own column names."
)
_STYLE_CONCRETENESS = MeasureTool(
    tool="style",
    requires=(DOC_ID, DOC, "Sentence ID", "Concreteness Mean", "Words Found", "Coverage %"),
    prepare=_prepare_style("Concreteness Mean", ("Words Found", "Coverage %")),
    measures=("Concreteness Mean (median)",),
    default_measure="Concreteness Mean (median)",
    size_column="_n",
    size_unit="scored sentences",
    has_dates=False,
    general_notes=(
        _STYLE_NOTE,
        "Concreteness Mean is the mean Brysbaert concreteness rating (roughly 1 = abstract, 5 = concrete) of "
        "this sentence's matched alpha, non-stopword tokens; this figure is the MEDIAN of that mean across the "
        "document's scored sentences. Sentences with no lexicon-listed content words are not counted.",
        "This table carries no per-document date, so it cannot be grouped by year or decade -- only by speaker.",
    ),
)
_STYLE_ICONICITY = MeasureTool(
    tool="style",
    requires=(DOC_ID, DOC, "Sentence ID", "Iconicity Mean", "Words Found", "Coverage %"),
    prepare=_prepare_style("Iconicity Mean", ("Words Found", "Coverage %")),
    measures=("Iconicity Mean (median)",),
    default_measure="Iconicity Mean (median)",
    size_column="_n",
    size_unit="scored sentences",
    has_dates=False,
    general_notes=(
        _STYLE_NOTE,
        "Iconicity Mean is the mean Winter iconicity rating (higher = the word's sound is felt to resemble its "
        "meaning more) of this sentence's matched alpha tokens (stopwords kept, unlike concreteness); this "
        "figure is the MEDIAN of that mean across the document's scored sentences.",
        "This table carries no per-document date, so it cannot be grouped by year or decade -- only by speaker.",
    ),
)


# -------------------------------------------------------------------- panels --

(READABILITY_OVER_TIME, READABILITY_BY_GROUP, READABILITY_LENGTH_CHECK, READABILITY_ALL_MEASURES) = _factory(
    _READABILITY, title_word="Readability", prefix="readability"
)
(
    LEXICAL_DIVERSITY_OVER_TIME,
    LEXICAL_DIVERSITY_BY_GROUP,
    LEXICAL_DIVERSITY_LENGTH_CHECK,
    LEXICAL_DIVERSITY_ALL_MEASURES,
) = _factory(_LEXICAL_DIVERSITY, title_word="Lexical diversity", prefix="lexical_diversity")
(
    CORPUS_STATISTICS_OVER_TIME,
    CORPUS_STATISTICS_BY_GROUP,
    CORPUS_STATISTICS_LENGTH_CHECK,
    CORPUS_STATISTICS_ALL_MEASURES,
) = _factory(_CORPUS_STATISTICS, title_word="Corpus statistics", prefix="corpus_statistics")
(TEXT_STATISTICS_OVER_TIME, TEXT_STATISTICS_BY_GROUP, TEXT_STATISTICS_ALL_MEASURES) = _factory(
    _TEXT_STATISTICS, title_word="Text statistics", prefix="text_statistics"
)
(
    SENTENCE_COMPLEXITY_OVER_TIME,
    SENTENCE_COMPLEXITY_BY_GROUP,
    SENTENCE_COMPLEXITY_LENGTH_CHECK,
    SENTENCE_COMPLEXITY_ALL_MEASURES,
) = _factory(_SENTENCE_COMPLEXITY, title_word="Sentence complexity", prefix="sentence_complexity")
(VERB_ANALYSIS_OVER_TIME, VERB_ANALYSIS_BY_GROUP, VERB_ANALYSIS_ALL_MEASURES) = _factory(
    _VERB_ANALYSIS, title_word="Verb analysis", prefix="verb_analysis"
)
(NOMINALIZATION_OVER_TIME, NOMINALIZATION_BY_GROUP) = _factory(
    _NOMINALIZATION, title_word="Nominalization", prefix="nominalization"
)
(STYLE_CONCRETENESS_BY_GROUP,) = _factory(_STYLE_CONCRETENESS, title_word="Concreteness", prefix="style_concreteness")
(STYLE_ICONICITY_BY_GROUP,) = _factory(_STYLE_ICONICITY, title_word="Iconicity", prefix="style_iconicity")

#: The shared factory and its simplest prepare step, for other families whose
#: tables are also one row per document (sentiment scores, style guesses):
#: their trends and groupings then behave exactly as every other measure's.
measure_panels = _factory
prepare_passthrough = _prepare_passthrough

DOCUMENT_MEASURES_PANELS: tuple[PanelDefinition, ...] = (
    READABILITY_OVER_TIME,
    READABILITY_BY_GROUP,
    READABILITY_LENGTH_CHECK,
    READABILITY_ALL_MEASURES,
    LEXICAL_DIVERSITY_OVER_TIME,
    LEXICAL_DIVERSITY_BY_GROUP,
    LEXICAL_DIVERSITY_LENGTH_CHECK,
    LEXICAL_DIVERSITY_ALL_MEASURES,
    CORPUS_STATISTICS_OVER_TIME,
    CORPUS_STATISTICS_BY_GROUP,
    CORPUS_STATISTICS_LENGTH_CHECK,
    CORPUS_STATISTICS_ALL_MEASURES,
    TEXT_STATISTICS_OVER_TIME,
    TEXT_STATISTICS_BY_GROUP,
    TEXT_STATISTICS_ALL_MEASURES,
    SENTENCE_COMPLEXITY_OVER_TIME,
    SENTENCE_COMPLEXITY_BY_GROUP,
    SENTENCE_COMPLEXITY_LENGTH_CHECK,
    SENTENCE_COMPLEXITY_ALL_MEASURES,
    VERB_ANALYSIS_OVER_TIME,
    VERB_ANALYSIS_BY_GROUP,
    VERB_ANALYSIS_ALL_MEASURES,
    NOMINALIZATION_OVER_TIME,
    NOMINALIZATION_BY_GROUP,
    STYLE_CONCRETENESS_BY_GROUP,
    STYLE_ICONICITY_BY_GROUP,
)
