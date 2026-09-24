"""Panels for the "where in time, where in the document" family of tools.

Nine tools share one question that no generic column-typed chart can ask:
*where* does something fall -- in the calendar, or along the length of a
document -- rather than merely how often it happens. A lexicon term's rate
over the corpus's years, a scatter of a date mentioned against the date it
was spoken, sentiment across the relative position of a speech, an entity's
mentions decade by decade: each needs the builder to know what a "position"
means for that specific table, because none of these tables is a plain (x, y)
pair by accident of its columns.

Two binning rules recur across the family and are kept in one place so two
panels cannot disagree about what "zero" means:

* **A count of matches over a real time axis gets an explicit zero** for a
  facet the corpus actually has documents in, and a gap for a facet it does
  not -- the rule ``panels_ngram_viewer.py`` established. ``lexicon_series``
  and ``ner_entity_timeline`` both follow it.
* **A mean of a per-sentence score (sentiment, noun ratio) over a relative
  position bin has no such zero.** An empty bin is a gap: there is no honest
  "0" sentiment for a stretch of a speech nobody scored, unlike an honest "0
  matches". ``narrative_emotion_arc`` and ``shapes_story_arc`` share
  :func:`_bin_by_relative_position`, which never fabricates a bin.

Every builder here is a pure function of (frame, params, provenance): no
plotly import, no filesystem, no corpus access, per the contract in
``core/viz/panelspec.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import decimal_year, document_labels, is_function_word
from core.viz.panelspec import (
    Annotation,
    Evidence,
    PanelDefinition,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
    counted_on_lemmas,
)

__all__ = [
    "COREFERENCE_CHAIN_POSITIONS",
    "DATE_ANNOTATOR_TIMELINE",
    "DATE_ANNOTATOR_TYPE_COUNTS",
    "DISPERSION_FREQUENCY_VS_CONCENTRATION",
    "GENDER_ANNOTATOR_MENTIONS_OVER_TIME",
    "KWIC_HIT_POSITIONS",
    "LEXICON_SERIES_HEATMAP",
    "LEXICON_SERIES_RATE",
    "NARRATIVE_CHARACTER_POSITIONS",
    "NARRATIVE_EMOTION_ARC",
    "NER_ENTITY_TIMELINE",
    "NER_TOP_ENTITIES",
    "SHAPES_STORY_ARC",
    "TIME_POSITIONS_PANELS",
    "coreference_chain_positions",
    "date_annotator_timeline",
    "date_annotator_type_counts",
    "dispersion_frequency_vs_concentration",
    "gender_annotator_mentions_over_time",
    "kwic_hit_positions",
    "lexicon_series_heatmap",
    "lexicon_series_rate",
    "narrative_character_positions",
    "narrative_emotion_arc",
    "ner_entity_timeline",
    "ner_top_entities",
    "shapes_story_arc",
]

DOCUMENT = "Document"
DOCUMENT_ID = "Document ID"
DATE = "Date"
YEAR = "Year"
SENTENCE_ID = "Sentence ID"

_MARK_CAP = 4000
#: Past this share of documents an entity is a form of address, not a subject.
_UBIQUITOUS = 0.75


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _leading_4digits(text: object) -> int | None:
    """The first 4-digit run in *text*, e.g. "1945-07" -> 1945, "1930s" -> 1930."""
    match = re.match(r"^\D*(\d{4})", str(text))
    return int(match.group(1)) if match else None


# =====================================================================
# lexicon_series -- per-1,000-token rate of each term/category over time
# =====================================================================
#
# core/analysis/lexicon_series.py::lexicon_series() already produces exactly
# one row per (Facet, Category) pair, including facets with zero hits -- see
# its own docstring ("Zero is a measurement"). So the explicit-zero half of
# the ngram_viewer rule is done before this file ever sees the table; what is
# left to this builder is turning a Facet string into a position (a year or
# decade axis, when the run used one) or a category axis (a heatmap, which
# works whatever the run's --by was).

FACET = "Facet"
CATEGORY = "Category"
PER_1000 = "Per 1000"
OCCURRENCES = "Occurrences"
LEX_DOCUMENTS = "Documents"
LEX_TOKENS = "Tokens"

_LEXICON_REQUIRES = (FACET, CATEGORY, PER_1000, OCCURRENCES, LEX_DOCUMENTS, LEX_TOKENS)


def lexicon_series_rate(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """One line per category, rate per 1,000 tokens, over a year/decade facet.

    Only meaningful when the run grouped by year or decade -- a facet of
    ``by=document`` or ``by=pattern`` names are not points on a time axis, and
    this panel refuses rather than plotting them on a numbered line as if
    they were.
    """
    by = str(provenance.settings.get("by", "year")).lower()
    if by not in ("year", "decade"):
        return Result.failure(
            Diagnostic.error(
                "PANEL_NOT_A_TIME_AXIS",
                f"this run grouped documents by {by!r}, not year or decade, so its facets are not points on a "
                "time axis; use the heatmap panel instead, which reads any grouping.",
            )
        )
    working = frame.copy()
    working["_year"] = working[FACET].map(_leading_4digits)
    working[PER_1000] = pd.to_numeric(working[PER_1000], errors="coerce")
    bad = working["_year"].isna() | working[PER_1000].isna()
    dropped = int(bad.sum())
    working = working.loc[~bad].copy()
    diagnostics: list[Diagnostic] = []
    if dropped:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_BAD_NUMERIC",
                f"{dropped} row(s) had an unparseable facet or rate and were left out.",
                dropped=dropped,
            )
        )
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no dated facet rows to plot"), *diagnostics)

    categories = tuple(sorted(working[CATEGORY].astype(str).unique()))
    marks: list[PanelMark] = []
    for _, row in working.sort_values([CATEGORY, "_year"], kind="stable").iterrows():
        category = str(row[CATEGORY])
        year = int(row["_year"])
        rate = float(row[PER_1000])
        occurrences = int(row[OCCURRENCES]) if _is_finite(row[OCCURRENCES]) else 0
        docs = int(row[LEX_DOCUMENTS]) if _is_finite(row[LEX_DOCUMENTS]) else 0
        tokens = int(row[LEX_TOKENS]) if _is_finite(row[LEX_TOKENS]) else 0
        marks.append(
            PanelMark(
                key=f"{category}␟{row[FACET]}",
                label=category,
                x=float(year),
                y=rate,
                group=category,
                evidence=Evidence(
                    scope="rows",
                    filters=((FACET, str(row[FACET])), (CATEGORY, category)),
                    count=occurrences,
                    describe=(
                        f"{category}, {row[FACET]}: {rate:.3f} per 1,000 tokens ({occurrences} occurrence(s) "
                        f"across {docs} document(s), {tokens} tokens counted)"
                    ),
                ),
            )
        )
    prepared = PreparedPanel(
        panel=LEXICON_SERIES_RATE.name,
        shape="line_series",
        title="Lexicon category rate over time",
        subtitle=f"{len(categories)} categor{'y' if len(categories) == 1 else 'ies'} · grouped by {by}",
        marks=tuple(marks),
        x_label="Decade" if by == "decade" else "Year",
        y_label="Occurrences per 1,000 tokens",
        provenance=provenance,
        data=working.drop(columns=["_year"]),
        groups=categories,
        line_gap=10.0 if by == "decade" else 1.0,
        notes=(
            "A zero point is a real measurement: the facet had dated documents and the category's words never "
            "occurred in them. A gap in the line is a facet with no documents at all, which is different and "
            "is not drawn as zero.",
            "Rates are per 1,000 tokens actually counted in that facet, so a rate is comparable across facets "
            "of very different total length.",
            "A category matched as a phrase list, not a single word; see the run's own lexicon text for which "
            "surface forms count toward it.",
        ),
    )
    return Result.success(prepared, *diagnostics)


def lexicon_series_heatmap(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Facet x category grid of the per-1,000-token rate, for any --by."""
    by = str(provenance.settings.get("by", "year")).lower()
    working = frame.copy()
    working[PER_1000] = pd.to_numeric(working[PER_1000], errors="coerce")
    bad = working[PER_1000].isna()
    dropped = int(bad.sum())
    working = working.loc[~bad].copy()
    diagnostics: list[Diagnostic] = []
    if dropped:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_BAD_NUMERIC", f"{dropped} row(s) had a non-numeric rate and were left out.", dropped=dropped
            )
        )
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no rate rows to plot"), *diagnostics)

    # Chronological order when the facets look like years/decades; otherwise
    # whatever order the run produced (already alphabetical from the engine).
    facets = sorted(
        working[FACET].astype(str).unique(), key=lambda f: (_leading_4digits(f) is None, _leading_4digits(f) or 0, f)
    )
    categories = sorted(working[CATEGORY].astype(str).unique())
    facet_index = {f: i for i, f in enumerate(facets)}
    category_index = {c: i for i, c in enumerate(categories)}

    marks: list[PanelMark] = []
    for _, row in working.iterrows():
        facet, category = str(row[FACET]), str(row[CATEGORY])
        rate = float(row[PER_1000])
        occurrences = int(row[OCCURRENCES]) if _is_finite(row[OCCURRENCES]) else 0
        marks.append(
            PanelMark(
                key=f"{facet}␟{category}",
                label=f"{facet} · {category}",
                x=float(facet_index[facet]),
                y=float(category_index[category]),
                value=rate,
                evidence=Evidence(
                    scope="rows",
                    filters=((FACET, facet), (CATEGORY, category)),
                    count=occurrences,
                    describe=f"{facet}, {category}: {rate:.3f} per 1,000 tokens ({occurrences} occurrence(s))",
                ),
            )
        )
    prepared = PreparedPanel(
        panel=LEXICON_SERIES_HEATMAP.name,
        shape="heatmap",
        title="Lexicon category rate by facet",
        subtitle=f"{len(facets)} facet(s) ({by}) x {len(categories)} categor{'y' if len(categories) == 1 else 'ies'}",
        marks=tuple(marks),
        x_label=by.capitalize(),
        y_label="Category",
        provenance=provenance,
        data=working,
        x_categories=tuple(facets),
        y_categories=tuple(categories),
        color_scale="sequential",
        value_label="Rate",
        notes=(
            "Colour is the rate per 1,000 tokens counted in that facet, not a raw count -- a facet with a "
            "handful of dated documents can still read dark if the category's words were common in them.",
            "Works for any grouping the run used (year, decade, document, or a name pattern); the rate axis "
            "panel only reads a year or decade run.",
        ),
    )
    return Result.success(prepared, *diagnostics)


LEXICON_SERIES_RATE = PanelDefinition(
    name="lexicon_series_rate",
    title="Lexicon category rate over time",
    question="How often does each word group appear per 1,000 tokens, year by year?",
    tool="lexicon_series",
    shape="line_series",
    summary="One line per lexicon category, rate per 1,000 tokens across a year or decade facet.",
    requires=_LEXICON_REQUIRES,
    params=(),
    build=lexicon_series_rate,
    notes=(
        "Requires a run grouped by year or decade; a run grouped by document or by a name pattern has no time "
        "axis to draw and this panel refuses -- use the heatmap panel instead.",
    ),
)

LEXICON_SERIES_HEATMAP = PanelDefinition(
    name="lexicon_series_heatmap",
    title="Lexicon category rate by facet",
    question="Which word group is strongest in which facet of the corpus?",
    tool="lexicon_series",
    shape="heatmap",
    summary="Facet x category grid of the per-1,000-token rate, for any grouping.",
    requires=_LEXICON_REQUIRES,
    params=(),
    build=lexicon_series_heatmap,
    notes=("Reads whatever facet the run used (year, decade, document name, or a regex group).",),
)


# =====================================================================
# date_annotator -- extracted dates: when the speech was given vs. what it
# mentions, plus what kind of date expression it found
# =====================================================================

SURFACE = "Surface"
NORMALIZED = "Normalized"
DATE_TYPE = "Type"
DATE_ID = "Date ID"


def date_annotator_timeline(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Speech date (x) against the year each extracted date expression names (y).

    A point on the diagonal is a speech mentioning roughly its own year; far
    below it is a speech reaching back into the past, far above is (rare) a
    forward-looking reference. ``Date ID`` is the table's own primary key
    (the schema audit found it unique across every row), so it alone is
    enough evidence to select exactly the one mention behind a point.
    """
    label_top = int(params["label-top"])
    working = frame.copy()
    working["_speech_year"] = working[DATE].map(decimal_year)
    working["_mentioned_year"] = working[NORMALIZED].map(_leading_4digits)
    bad = working["_speech_year"].isna() | working["_mentioned_year"].isna()
    dropped = int(bad.sum())
    working = working.loc[~bad].copy()
    diagnostics: list[Diagnostic] = []
    if dropped:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_BAD_NUMERIC",
                f"{dropped} row(s) had no readable speech date or extracted year and were left out.",
                dropped=dropped,
            )
        )
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no dated mentions to plot"), *diagnostics)

    working["_gap"] = (working["_mentioned_year"] - working["_speech_year"]).abs()
    # Label only mentions that are plausibly dates. On the real corpus the
    # widest gaps were bare numbers read as years ("1500", "1140"); a real
    # historical anchor (1776, 1789) recurs across speeches, and a full date
    # or month-year is a date by construction.
    recurring = working.groupby("_mentioned_year")[DOCUMENT_ID].transform("nunique") >= 2
    plausible = working[(working[DATE_TYPE] != "year") | recurring] if DATE_TYPE in working.columns else working
    # One label per year named: 1776 in six speeches is one anchor, and six
    # "1776" labels stacked on one row read as noise.
    widest = plausible.sort_values("_gap", ascending=False, kind="stable").drop_duplicates("_mentioned_year")
    top = widest.head(label_top) if label_top else working.iloc[0:0]
    labelled = set(top[DATE_ID].astype(str))

    if len(working) > _MARK_CAP:
        working = working.nlargest(_MARK_CAP, "_gap")
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_TOO_MANY_MARKS",
                f"kept the {_MARK_CAP} widest-gap mentions of {len(working)} for a readable scatter.",
                drawn=_MARK_CAP,
            )
        )

    marks: list[PanelMark] = []
    for _, row in working.iterrows():
        date_id = str(row[DATE_ID])
        speech_year = float(row["_speech_year"])
        mentioned_year = float(row["_mentioned_year"])
        marks.append(
            PanelMark(
                key=date_id,
                label=str(row[SURFACE]),
                x=speech_year,
                y=mentioned_year,
                labelled=date_id in labelled,
                evidence=Evidence(
                    scope="rows",
                    filters=((DATE_ID, date_id),),
                    count=1,
                    describe=(
                        f"{row[DOCUMENT]} ({row[DATE]}) mentions {row[SURFACE]!r} ({row[DATE_TYPE]}), "
                        f"{int(mentioned_year) - int(speech_year):+d} years from the speech date"
                    ),
                ),
            )
        )
    prepared = PreparedPanel(
        panel=DATE_ANNOTATOR_TIMELINE.name,
        shape="scatter_labelled",
        title="Dates mentioned, against the date spoken",
        subtitle=f"{len(marks)} extracted date expression(s); labelling the {len(labelled)} widest gaps",
        marks=tuple(marks),
        x_label="Speech date",
        y_label="Year the extracted date expression names",
        provenance=provenance,
        data=working.drop(columns=["_speech_year", "_mentioned_year", "_gap"]),
        annotations=(
            Annotation(
                kind="note",
                value=0.0,
                label="on the diagonal",
                note=(
                    "A point at x = y mentions its own speech year (a decade name is placed at its first year). "
                    "Points well below the diagonal reach into the past; the corpus has almost none above it."
                ),
            ),
        ),
        notes=(
            "A decade expression ('the 1930s') is placed at its first year, and a month/year expression at "
            "that year -- both lose precision the Surface text still carries.",
            "This is what the date-extraction pattern matched in running text, not a claim that every "
            "reference to a year is a claim about that specific year; check Surface for context.",
        ),
    )
    return Result.success(prepared, *diagnostics)


def date_annotator_type_counts(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Ranked counts of each extracted-date Type, with document coverage."""
    _ = params
    working = frame.copy()
    total_docs = working[DOCUMENT].astype(str).nunique()
    grouped = working.groupby(DATE_TYPE, sort=False).agg(_count=(DATE_TYPE, "count"), _docs=(DOCUMENT, "nunique"))
    if grouped.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no extracted dates to rank"))
    grouped = grouped.sort_values(["_count", "_docs"], ascending=[False, False], kind="stable")

    marks: list[PanelMark] = []
    for rank, (date_type, row) in enumerate(grouped.iterrows()):
        count, docs = int(row["_count"]), int(row["_docs"])
        marks.append(
            PanelMark(
                key=str(date_type),
                label=str(date_type),
                x=float(count),
                y=float(rank),
                evidence=Evidence(
                    scope="rows",
                    filters=((DATE_TYPE, str(date_type)),),
                    count=count,
                    describe=f"{date_type}: {count} extracted date(s) in {docs} of {total_docs} document(s)",
                ),
            )
        )
    prepared = PreparedPanel(
        panel=DATE_ANNOTATOR_TYPE_COUNTS.name,
        shape="ranked_bars",
        title="Extracted date expressions by type",
        subtitle=f"{len(marks)} type(s) across {total_docs} document(s)",
        marks=tuple(marks),
        x_label="Extracted date expressions",
        y_label="Date type, most common first",
        provenance=provenance,
        data=working,
        notes=(
            "A 'year' surface form (a bare '1934') is by far the easiest pattern to match and usually "
            "dominates; 'date', 'month_year' and 'decade' need more specific surface text to be recognised.",
        ),
    )
    return Result.success(prepared)


DATE_ANNOTATOR_TIMELINE = PanelDefinition(
    name="date_annotator_timeline",
    title="Dates mentioned vs. date spoken",
    question="When a speech mentions a date, how far into the past (or future) is it reaching?",
    tool="date_annotator",
    shape="scatter_labelled",
    summary="Speech date against the year each extracted date expression names.",
    requires=(DOCUMENT, DATE, SURFACE, NORMALIZED, DATE_TYPE, DATE_ID),
    params=(
        PanelParam(
            name="label-top",
            type="int",
            default=15,
            minimum=0,
            maximum=200,
            label="Points to label",
            help="How many of the widest speech/mentioned-year gaps get their surface text written on the plot.",
        ),
    ),
    build=date_annotator_timeline,
    notes=(
        "A decade or month/year expression is placed at its first year, losing the precision Surface still "
        "carries; read the hover for the exact matched text.",
    ),
)

DATE_ANNOTATOR_TYPE_COUNTS = PanelDefinition(
    name="date_annotator_type_counts",
    title="Extracted date expressions by type",
    question="What kinds of date expressions does this corpus actually use?",
    tool="date_annotator",
    shape="ranked_bars",
    summary="Ranked counts of each date Type (year, date, month_year, decade), with document coverage.",
    requires=(DOCUMENT, DATE_TYPE),
    params=(),
    build=date_annotator_type_counts,
    notes=("A bare year is the easiest pattern to match and usually dominates the ranking.",),
)


# =====================================================================
# gender_annotator -- mentions by predicted category, per year, unknown shown
# =====================================================================
#
# core/analysis/gender_annotator.py::annotate_names() outputs one row per
# (document, distinct name), labelled male/female/both/unknown -- never a
# name silently dropped for being unresolved. The profiler's _with_dates step
# (core/profiler/executor.py) then adds Date/Year beside Document ID before
# this table is ever published, because it has a Document ID column and no
# date columns of its own; this builder relies on that having already
# happened, matching every other per-document result in the suite.

NAME = "Name"
GENDER = "Gender"
MENTIONS = "Mentions"
_GENDER_ORDER: tuple[str, ...] = ("male", "female", "both", "unknown")


def gender_annotator_mentions_over_time(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Stacked mentions by predicted gender per year; unknown always a band."""
    normalize = bool(params["normalize"])
    working = frame.copy()
    working[YEAR] = pd.to_numeric(working[YEAR], errors="coerce")
    working[MENTIONS] = pd.to_numeric(working[MENTIONS], errors="coerce")
    bad = working[YEAR].isna() | working[MENTIONS].isna()
    dropped = int(bad.sum())
    working = working.loc[~bad].copy()
    working[YEAR] = working[YEAR].astype(int)
    working[GENDER] = working[GENDER].where(working[GENDER].isin(_GENDER_ORDER), "unknown")
    diagnostics: list[Diagnostic] = []
    if dropped:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_UNDATED_ROWS",
                f"{dropped} name row(s) had no dated Year and were left out.",
                dropped=dropped,
            )
        )
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no dated name rows to plot"), *diagnostics)

    totals = working.groupby(YEAR)[MENTIONS].sum()
    per_cell = working.groupby([YEAR, GENDER])[MENTIONS].sum()
    years = sorted(totals.index)

    marks: list[PanelMark] = []
    for year in years:
        total = float(totals.loc[year])
        for gender in _GENDER_ORDER:
            count = float(per_cell.get((year, gender), 0.0))
            value = (count / total) if normalize and total else count
            marks.append(
                PanelMark(
                    key=f"{year}␟{gender}",
                    label=gender,
                    x=float(year),
                    y=value,
                    group=gender,
                    size=count,
                    evidence=Evidence(
                        scope="rows",
                        filters=((YEAR, str(year)), (GENDER, gender)),
                        count=int(count),
                        describe=(
                            f"{year}: {int(count)} of {int(total)} name mention(s) predicted {gender}"
                            + (f" ({value:.0%})" if normalize and total else "")
                        ),
                    ),
                )
            )
    prepared = PreparedPanel(
        panel=GENDER_ANNOTATOR_MENTIONS_OVER_TIME.name,
        shape="stream",
        title="Predicted gender of named mentions, by year",
        subtitle=f"{len(years)} dated year(s) · {working[DOCUMENT_ID].nunique()} document(s)",
        marks=tuple(marks),
        x_label="Year",
        y_label="Share of name mentions" if normalize else "Name mentions",
        provenance=provenance,
        data=working,
        groups=_GENDER_ORDER,
        notes=(
            "'unknown' is names the chosen dictionary did not recognise, shown as its own band in every year "
            "rather than being dropped -- a corpus can be mostly unknown to a given dictionary, and hiding "
            "that would misstate how much this run actually classified.",
            "'both' is a name present in the dictionary's male AND female lists; it is not a third gender "
            "category the dictionary predicts, it is the dictionary contradicting itself on that name.",
            "This is name-list lookup on names NER (or PROPN) found in the text, not a claim about any real "
            "person's gender; a name shared by many people (or spelled unusually) can be misclassified.",
            "Counts are name mentions (repeat uses of one name in a document all count), not distinct people.",
        ),
    )
    return Result.success(prepared, *diagnostics)


GENDER_ANNOTATOR_MENTIONS_OVER_TIME = PanelDefinition(
    name="gender_annotator_mentions_over_time",
    title="Predicted gender of named mentions, by year",
    question="How does the predicted gender mix of named people mentioned change across the corpus?",
    tool="gender_annotator",
    shape="stream",
    summary="Stacked mentions by predicted gender category per dated year, with unknown always shown.",
    requires=(DOCUMENT_ID, YEAR, GENDER, MENTIONS),
    params=(
        PanelParam(
            name="normalize",
            type="bool",
            default=True,
            label="Show as a share",
            help="When on, each year's bands sum to 1 (a share of that year's mentions). When off, raw counts.",
        ),
    ),
    build=gender_annotator_mentions_over_time,
    notes=(
        "Built and tested on engine-shaped fixtures only: the NLTK 'names' corpus this machine's default "
        "dictionary needs is not downloaded here, so gender_annotator produces no real output to check this "
        "panel against on this machine. Unverified against a real run.",
    ),
)


# =====================================================================
# narrative -- emotion across relative position in the document
# shapes    -- normalised story-shape lines, same binning approach
# =====================================================================
#
# Both emotion_arc.csv and story_shape.csv are one row per scored sentence,
# with no Document filename column (only a numeric Document ID) -- confirmed
# against the real 87-speech run. A document is therefore labelled by its
# Date, which every row does carry.


def _pick_documents(ids_and_dates: Sequence[tuple[str, str | None]], spec: str) -> tuple[list[str], list[str]]:
    """Which Document IDs to draw: named by *spec*, or first/middle/last by date.

    *spec* tokens match a Document ID, a 4-digit year, or an exact Date
    string. Returns (chosen ids in spec/date order, unmatched tokens) so the
    caller can warn about a token nobody has.
    """
    ordered_by_date = sorted(dict.fromkeys(ids_and_dates), key=lambda pair: (pair[1] is None, pair[1] or ""))
    if not spec.strip():
        ids = [doc_id for doc_id, _ in ordered_by_date]
        if len(ids) <= 3:
            return ids, []
        chosen = [ids[0], ids[len(ids) // 2], ids[-1]]
        return list(dict.fromkeys(chosen)), []
    by_id = {doc_id: when for doc_id, when in ordered_by_date}
    chosen, unmatched = [], []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        found = None
        if token in by_id:
            found = token
        else:
            for doc_id, when in ordered_by_date:
                if when == token or (when and when[:4] == token and len(token) == 4):
                    found = doc_id
                    break
        if found is None:
            unmatched.append(token)
        elif found not in chosen:
            chosen.append(found)
    return chosen, unmatched


def _bin_by_relative_position(
    rows: pd.DataFrame,
    *,
    id_column: str,
    order_column: str,
    value_column: str,
    bins: int,
) -> pd.DataFrame:
    """Per document, the mean *value_column* in each of *bins* equal position bins.

    Position is the row's rank within its own document (by *order_column*),
    scaled to [0, 1]; a document with one scored row sits at 0.5. An empty
    bin is left out of the result entirely -- a gap, not a zero, because a
    mean over nothing is not a measurement (unlike a count, which is honestly
    zero when nothing was found).
    """
    out_rows: list[dict[str, Any]] = []
    for doc_id, group in rows.groupby(id_column, sort=False):
        ordered = group.sort_values(order_column, kind="stable").reset_index(drop=True)
        n = len(ordered)
        position = pd.Series(ordered.index / (n - 1) if n > 1 else [0.5] * n, index=ordered.index, dtype=float)
        ordered = ordered.assign(_position=position, _bin=(position * bins).clip(upper=bins - 1).astype(int))
        for bin_index, bin_group in ordered.groupby("_bin"):
            out_rows.append(
                {
                    id_column: doc_id,
                    "Bin": int(bin_index),
                    "Bin center": (int(bin_index) + 0.5) / bins,
                    "Mean": float(bin_group[value_column].mean()),
                    "N": len(bin_group),
                }
            )
    return pd.DataFrame(out_rows, columns=[id_column, "Bin", "Bin center", "Mean", "N"])


def _document_label(doc_id: str, when: str | None) -> str:
    return f"{when} (doc {doc_id})" if when else f"doc {doc_id}"


def narrative_emotion_arc(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Sentiment across relative position, one line per chosen document, plus
    a corpus median shown as unconnected points.

    Each selected document's line connects real, ordered sentences within
    ONE document, so joining them is honest continuity, not an invented
    trend across unrelated units. The corpus median is different: at each
    position bin it is a statistic over many heterogeneous documents, and
    per the house rule against averages standing in for spread, it is drawn
    as points only -- never smoothed into a line that would read as if the
    corpus itself had one continuous emotional arc.
    """
    bins = int(params["bins"])
    documents_spec = str(params["documents"])
    COMPOUND = "Compound"

    working = frame.copy()
    working[SENTENCE_ID] = pd.to_numeric(working[SENTENCE_ID], errors="coerce")
    working[COMPOUND] = pd.to_numeric(working[COMPOUND], errors="coerce")
    working[DOCUMENT_ID] = working[DOCUMENT_ID].astype(str)
    valid = working[SENTENCE_ID].notna() & working[COMPOUND].notna()
    dropped = int((~valid).sum())
    working = working.loc[valid].copy()
    diagnostics: list[Diagnostic] = []
    if dropped:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_BAD_NUMERIC",
                f"{dropped} sentence row(s) had no usable score and were left out.",
                dropped=dropped,
            )
        )
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no scored sentences to bin"), *diagnostics)

    when_by_id = working.drop_duplicates(DOCUMENT_ID).set_index(DOCUMENT_ID)[DATE].astype(str).to_dict()
    ids_and_dates = [(doc_id, when_by_id.get(doc_id)) for doc_id in working[DOCUMENT_ID].unique()]
    chosen, unmatched = _pick_documents(ids_and_dates, documents_spec)
    if unmatched:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_DOCUMENTS_NOT_FOUND",
                f"{len(unmatched)} document token(s) matched no document and were skipped: {', '.join(unmatched)}",
            )
        )
    if not chosen:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", "no requested document matched this table"), *diagnostics
        )

    subset = working[working[DOCUMENT_ID].isin(chosen)]
    binned = _bin_by_relative_position(
        subset, id_column=DOCUMENT_ID, order_column=SENTENCE_ID, value_column=COMPOUND, bins=bins
    )
    if binned.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no bin had a scored sentence"), *diagnostics)

    labels = {doc_id: _document_label(doc_id, when_by_id.get(doc_id)) for doc_id in chosen}
    groups = [labels[doc_id] for doc_id in chosen]

    marks: list[PanelMark] = []
    for _, row in binned.iterrows():
        doc_id = str(row[DOCUMENT_ID])
        bin_index = int(row["Bin"])
        marks.append(
            PanelMark(
                key=f"{doc_id}␟{bin_index}",
                label=labels[doc_id],
                x=float(row["Bin center"]),
                y=float(row["Mean"]),
                group=labels[doc_id],
                evidence=Evidence(
                    scope="rows",
                    filters=(
                        (DOCUMENT_ID, doc_id),
                        (SENTENCE_ID, str(subset[subset[DOCUMENT_ID] == doc_id][SENTENCE_ID].min())),
                    )
                    if False
                    else ((DOCUMENT_ID, doc_id),),
                    count=int(row["N"]),
                    describe=(
                        f"{labels[doc_id]}, position {row['Bin center']:.0%} through the speech: mean compound "
                        f"{row['Mean']:.3f} over {int(row['N'])} sentence(s)"
                    ),
                ),
            )
        )

    # Corpus median: across ALL documents in the table (not just the chosen
    # ones), the median of each document's own per-bin mean -- points only.
    all_binned = _bin_by_relative_position(
        working, id_column=DOCUMENT_ID, order_column=SENTENCE_ID, value_column=COMPOUND, bins=bins
    )
    median_group = "Median (all documents)"
    median_marks: list[PanelMark] = []
    if not all_binned.empty:
        for bin_index, bin_group in all_binned.groupby("Bin"):
            median = float(bin_group["Mean"].median())
            n_docs = int(bin_group[DOCUMENT_ID].nunique())
            median_marks.append(
                PanelMark(
                    key=f"median␟{bin_index}",
                    label=median_group,
                    x=(int(bin_index) + 0.5) / bins,
                    y=median,
                    group=median_group,
                    size=float(n_docs),
                    evidence=Evidence(
                        scope="rows",
                        filters=((DOCUMENT_ID, str(bin_group[DOCUMENT_ID].iloc[0])),),
                        count=n_docs,
                        describe=(
                            f"position {(int(bin_index) + 0.5) / bins:.0%} through the speech: median of "
                            f"{n_docs} documents' own mean compound is {median:.3f}"
                        ),
                    ),
                )
            )
        groups = [*groups, median_group]
        marks = [*marks, *median_marks]

    prepared = PreparedPanel(
        panel=NARRATIVE_EMOTION_ARC.name,
        shape="line_series",
        title="Sentiment across the speech, by relative position",
        subtitle=f"{len(chosen)} document(s) · {bins} bins",
        marks=tuple(marks),
        x_label="Relative position in the speech (0 = opening, 1 = close)",
        y_label="Mean VADER compound",
        provenance=provenance,
        data=pd.concat([binned, all_binned]).drop_duplicates().reset_index(drop=True),
        groups=tuple(groups),
        points_only=(median_group,) if median_marks else (),
        line_gap=1.5 / bins,
        notes=(
            "An empty bin is a gap: unlike a match count, a mean sentiment over zero sentences is not a real "
            "zero and is never invented to keep a line unbroken.",
            "Each chosen document's line is a real sequence within one document, safe to connect. The corpus "
            "median is a statistic over many different documents at each position and is shown as points only, "
            "never joined into a line, so it cannot be mistaken for one continuous arc.",
            "VADER's compound score is a lexicon-based signed estimate in [-1, +1], not a calibrated measure "
            "of emotional intensity.",
        ),
    )
    return Result.success(prepared, *diagnostics)


def shapes_story_arc(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Noun or verb ratio across relative position, same binning as the emotion arc."""
    bins = int(params["bins"])
    documents_spec = str(params["documents"])
    measure = str(params["measure"])
    column = "Noun Ratio" if measure == "noun" else "Verb Ratio"

    working = frame.copy()
    working[SENTENCE_ID] = pd.to_numeric(working[SENTENCE_ID], errors="coerce")
    working[column] = pd.to_numeric(working[column], errors="coerce")
    working[DOCUMENT_ID] = working[DOCUMENT_ID].astype(str)
    valid = working[SENTENCE_ID].notna() & working[column].notna()
    dropped = int((~valid).sum())
    working = working.loc[valid].copy()
    diagnostics: list[Diagnostic] = []
    if dropped:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_BAD_NUMERIC",
                f"{dropped} sentence row(s) had no usable {column} and were left out.",
                dropped=dropped,
            )
        )
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no scored sentences to bin"), *diagnostics)

    when_by_id = working.drop_duplicates(DOCUMENT_ID).set_index(DOCUMENT_ID)[DATE].astype(str).to_dict()
    ids_and_dates = [(doc_id, when_by_id.get(doc_id)) for doc_id in working[DOCUMENT_ID].unique()]
    chosen, unmatched = _pick_documents(ids_and_dates, documents_spec)
    if unmatched:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_DOCUMENTS_NOT_FOUND",
                f"{len(unmatched)} document token(s) matched no document and were skipped: {', '.join(unmatched)}",
            )
        )
    if not chosen:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", "no requested document matched this table"), *diagnostics
        )

    subset = working[working[DOCUMENT_ID].isin(chosen)]
    binned = _bin_by_relative_position(
        subset, id_column=DOCUMENT_ID, order_column=SENTENCE_ID, value_column=column, bins=bins
    )
    if binned.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no bin had a scored sentence"), *diagnostics)

    labels = {doc_id: _document_label(doc_id, when_by_id.get(doc_id)) for doc_id in chosen}
    marks: list[PanelMark] = []
    for _, row in binned.iterrows():
        doc_id = str(row[DOCUMENT_ID])
        bin_index = int(row["Bin"])
        marks.append(
            PanelMark(
                key=f"{doc_id}␟{bin_index}",
                label=labels[doc_id],
                x=float(row["Bin center"]),
                y=float(row["Mean"]),
                group=labels[doc_id],
                evidence=Evidence(
                    scope="rows",
                    filters=((DOCUMENT_ID, doc_id),),
                    count=int(row["N"]),
                    describe=(
                        f"{labels[doc_id]}, position {row['Bin center']:.0%} through the speech: mean {column} "
                        f"{row['Mean']:.3f} over {int(row['N'])} sentence(s)"
                    ),
                ),
            )
        )
    prepared = PreparedPanel(
        panel=SHAPES_STORY_ARC.name,
        shape="line_series",
        title=f"Story shape: {column.lower()} across the speech",
        subtitle=f"{len(chosen)} document(s) · {bins} bins",
        marks=tuple(marks),
        x_label="Relative position in the speech (0 = opening, 1 = close)",
        y_label=f"Mean {column}",
        provenance=provenance,
        data=binned,
        groups=tuple(labels[doc_id] for doc_id in chosen),
        line_gap=1.5 / bins,
        notes=(
            "An empty bin is a gap, not a zero: a ratio over zero scored sentences is not a measurement.",
            "Noun Ratio and Verb Ratio are shares of a sentence's tokens, not counts, so they are already "
            "comparable across sentences and documents of different lengths.",
            "Each line is one document's own shape; comparing across documents of very different lengths still "
            "means comparing different numbers of sentences per bin, shown in the hover's sentence count.",
        ),
    )
    return Result.success(prepared, *diagnostics)


_DOCUMENTS_PARAM_HELP = (
    "Comma-separated Document IDs, 4-digit years, or exact Date strings naming which documents to draw. "
    "Empty picks the first, middle and last document by date."
)

NARRATIVE_EMOTION_ARC = PanelDefinition(
    name="narrative_emotion_arc",
    title="Sentiment across the speech, by relative position",
    question="Does a speech's tone follow a shape -- does it open one way and close another?",
    tool="narrative",
    shape="line_series",
    summary="Binned mean sentiment over relative sentence position, one line per chosen document, plus a corpus median.",
    requires=(DOCUMENT_ID, DATE, SENTENCE_ID, "Compound"),
    params=(
        PanelParam(
            name="documents",
            type="str",
            default="",
            label="Documents",
            help=_DOCUMENTS_PARAM_HELP,
        ),
        PanelParam(
            name="bins",
            type="int",
            default=10,
            minimum=3,
            maximum=50,
            label="Position bins",
            help="How many equal-width bins to divide each document's sentence sequence into.",
        ),
    ),
    build=narrative_emotion_arc,
    notes=(
        "Documents lines are real per-document sequences; the corpus median is a cross-document statistic and "
        "is drawn as points only, never a connected line.",
    ),
)

NARRATIVE_CHARACTER_POSITIONS_REQUIRES = ("Character", "Mentions", DOCUMENT_ID, DOCUMENT, DATE, YEAR)


def narrative_character_positions(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Top characters (rows) x the documents they appear in, placed by date.

    A tick is one document's worth of mentions of one character, placed at
    that document's decimal year -- readable as "which characters recur
    across the corpus, and when do they show up".
    """
    top_n = int(params["top-n"])
    CHARACTER = "Character"
    working = frame.copy()
    working[MENTIONS] = pd.to_numeric(working[MENTIONS], errors="coerce").fillna(0)
    working["_year"] = working[DATE].map(decimal_year)
    bad = working["_year"].isna()
    dropped = int(bad.sum())
    working = working.loc[~bad].copy()
    diagnostics: list[Diagnostic] = []
    if dropped:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_UNDATED_ROWS", f"{dropped} row(s) had no readable Date and were left out.", dropped=dropped
            )
        )
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no dated character mentions to plot"), *diagnostics)

    totals = working.groupby(CHARACTER)[MENTIONS].sum().sort_values(ascending=False)
    top_characters = list(totals.head(top_n).index)
    if not top_characters:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no character rows to rank"), *diagnostics)
    if len(totals) > top_n:
        diagnostics.append(
            Diagnostic.info(
                "PANEL_CHARACTERS_CAPPED",
                f"showing the {top_n} most-mentioned of {len(totals)} extracted characters.",
                drawn=top_n,
                available=len(totals),
            )
        )
    y_categories = tuple(top_characters)
    row_index = {name: i for i, name in enumerate(y_categories)}
    subset = working[working[CHARACTER].isin(top_characters)]

    marks: list[PanelMark] = []
    for _, row in subset.iterrows():
        character = str(row[CHARACTER])
        marks.append(
            PanelMark(
                key=f"{character}␟{row[DOCUMENT_ID]}",
                label=character,
                x=float(row["_year"]),
                y=float(row_index[character]),
                evidence=Evidence(
                    scope="rows",
                    filters=((CHARACTER, character), (DOCUMENT_ID, str(row[DOCUMENT_ID]))),
                    count=int(row[MENTIONS]),
                    describe=f"{character} in {row[DOCUMENT]} ({row[DATE]}): {int(row[MENTIONS])} mention(s)",
                ),
            )
        )
    prepared = PreparedPanel(
        panel=NARRATIVE_CHARACTER_POSITIONS.name,
        shape="positions",
        title="Named characters, by the documents they appear in",
        subtitle=f"{len(y_categories)} character(s), ranked by total mentions",
        marks=tuple(marks),
        x_label="Speech date",
        y_label="Character (most-mentioned first)",
        provenance=provenance,
        data=subset.drop(columns=["_year"]),
        y_categories=y_categories,
        notes=(
            "'Character' is a NER PERSON span within one document, extracted heuristically; it can include "
            "titles, misparsed fragments, or pronouns picked up as proper nouns -- read the hover before "
            "treating a name as a real recurring figure.",
            "A tick is one document containing at least one mention; it does not show where within the "
            "document the mentions fall (see character_arcs.csv in the table for that detail).",
        ),
    )
    return Result.success(prepared, *diagnostics)


NARRATIVE_CHARACTER_POSITIONS = PanelDefinition(
    name="narrative_character_positions",
    title="Named characters, by document",
    question="Which named figures recur across the corpus, and in which speeches?",
    tool="narrative",
    shape="positions",
    summary="Top characters by total mentions, one row each, ticked at the dates of the documents they appear in.",
    requires=NARRATIVE_CHARACTER_POSITIONS_REQUIRES,
    params=(
        PanelParam(
            name="top-n",
            type="int",
            default=20,
            minimum=1,
            maximum=100,
            label="Characters to show",
            help="How many of the most-mentioned characters to draw as rows.",
        ),
    ),
    build=narrative_character_positions,
    notes=("Character extraction is heuristic NER; treat unfamiliar or odd-looking names with suspicion.",),
)

SHAPES_STORY_ARC = PanelDefinition(
    name="shapes_story_arc",
    title="Story shape across the speech",
    question="Does a speech's noun-vs-verb balance follow a shape across its length?",
    tool="shapes",
    shape="line_series",
    summary="Binned mean Noun Ratio or Verb Ratio over relative sentence position, one line per chosen document.",
    requires=(DOCUMENT_ID, DATE, SENTENCE_ID, "Noun Ratio", "Verb Ratio"),
    params=(
        PanelParam(name="documents", type="str", default="", label="Documents", help=_DOCUMENTS_PARAM_HELP),
        PanelParam(
            name="bins",
            type="int",
            default=10,
            minimum=3,
            maximum=50,
            label="Position bins",
            help="How many equal-width bins to divide each document's sentence sequence into.",
        ),
        PanelParam(
            name="measure",
            type="choice",
            default="noun",
            choices=("noun", "verb"),
            label="Measure",
            help="Which ratio to plot: the share of a sentence's tokens that are nouns, or that are verbs.",
        ),
    ),
    build=shapes_story_arc,
    notes=("An empty bin is a gap, not a zero: a ratio over zero scored sentences is not a measurement.",),
)


# =====================================================================
# dispersion -- frequency against how evenly a word is spread (Gries DP)
# =====================================================================

TERM = "Term"
FREQUENCY = "Frequency"
GRIES_DP = "Gries DP"
RANGE = "Range"


def dispersion_frequency_vs_concentration(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Scatter of log10(frequency) against Gries DP, labelling frequent-but-concentrated words.

    Gries DP runs 0 (spread perfectly evenly across the corpus, weighted for
    part size) to 1 (occurring in only a slice of it); a word can be frequent
    two ways -- genuinely common throughout, or a handful of documents using
    it heavily -- and only this axis tells them apart. The published
    dispersion table has no per-occurrence offsets, so a true positions plot
    of where each hit falls cannot be built from it; this is the figure the
    table can support.
    """
    label_top = int(params["label-top"])
    working = frame.copy()
    working[FREQUENCY] = pd.to_numeric(working[FREQUENCY], errors="coerce")
    working[GRIES_DP] = pd.to_numeric(working[GRIES_DP], errors="coerce")
    placeable = working[FREQUENCY].map(lambda v: _is_finite(v) and v > 0) & working[GRIES_DP].map(_is_finite)
    dropped = int((~placeable).sum())
    working = working.loc[placeable].copy()
    diagnostics: list[Diagnostic] = []
    if dropped:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_BAD_NUMERIC",
                f"{dropped} term(s) had a non-numeric frequency or DP and were left out.",
                dropped=dropped,
            )
        )
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no terms to plot"), *diagnostics)

    on_lemmas = counted_on_lemmas(provenance.settings)
    median_frequency = float(working[FREQUENCY].median())
    frequent = working[working[FREQUENCY] >= median_frequency]
    # "he", "your", "during" can be concentrated too (speeches addressed to
    # one person), but a label is a claim that the word is worth a look.
    frequent = frequent[~frequent[TERM].astype(str).map(is_function_word)]
    top = frequent.nlargest(label_top, GRIES_DP) if label_top else frequent.iloc[0:0]
    labelled = set(top[TERM].astype(str))

    marks: list[PanelMark] = []
    for _, row in working.iterrows():
        term = str(row[TERM])
        frequency = int(row[FREQUENCY])
        dp = float(row[GRIES_DP])
        marks.append(
            PanelMark(
                key=term,
                label=term,
                x=math.log10(frequency),
                y=dp,
                size=float(frequency),
                labelled=term in labelled,
                evidence=Evidence(
                    scope="terms",
                    filters=((TERM, term),),
                    count=frequency,
                    phrase=term,
                    lemma=on_lemmas,
                    describe=f"{term}: {frequency} occurrences in {int(row[RANGE])} document(s) -- Gries DP {dp:.3f}",
                ),
            )
        )
    prepared = PreparedPanel(
        panel=DISPERSION_FREQUENCY_VS_CONCENTRATION.name,
        shape="scatter_labelled",
        title="Frequency against dispersion",
        subtitle=f"{len(marks)} term(s); labelling the {len(labelled)} most concentrated among frequent terms",
        marks=tuple(marks),
        x_label="Frequency (log scale)",
        x_log10=True,
        y_label="Gries DP (0 = spread evenly, 1 = concentrated)",
        provenance=provenance,
        data=working,
        notes=(
            "A word can be frequent because it is common everywhere, or because a handful of documents use it "
            "heavily -- Gries DP is the number that tells these apart; frequency alone cannot.",
            "Labelled terms are frequent (at or above this table's median frequency) AND among the most "
            "concentrated (highest DP) of those -- the words worth checking are not spread the way their raw "
            "count would suggest.",
            "The x axis is log10: two units right is a hundred times more occurrences, and a term with zero "
            "occurrences has no position on it.",
        ),
    )
    return Result.success(prepared, *diagnostics)


DISPERSION_FREQUENCY_VS_CONCENTRATION = PanelDefinition(
    name="dispersion_frequency_vs_concentration",
    title="Frequency against dispersion",
    question="Which frequent words are actually spread across the corpus, and which are concentrated in a few documents?",
    tool="dispersion",
    shape="scatter_labelled",
    summary="Log frequency against Gries DP, one point per term, labelling frequent-but-concentrated words.",
    requires=(TERM, FREQUENCY, GRIES_DP, RANGE),
    params=(
        PanelParam(
            name="label-top",
            type="int",
            default=15,
            minimum=0,
            maximum=100,
            label="Terms to label",
            help="How many frequent-but-concentrated terms get their text written on the plot.",
        ),
    ),
    build=dispersion_frequency_vs_concentration,
    notes=(
        "The published dispersion table has no per-occurrence offsets, so a true positions plot of where each "
        "hit falls cannot be built from it; this scatter is what the table supports.",
    ),
)


# =====================================================================
# ner -- ranked entities by tag, and chosen entities' mentions over time
# =====================================================================

ENTITY = "Entity"
NER_TAG = "NER Tag"
NER_COUNT = "Count"


def ner_top_entities(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Ranked entities of one NER tag, with document coverage."""
    tag = str(params["tag"]).strip().upper()
    top_n = int(params["top-n"])
    working = frame.copy()
    working[NER_TAG] = working[NER_TAG].astype(str).str.upper()
    working[NER_COUNT] = pd.to_numeric(working[NER_COUNT], errors="coerce").fillna(0)
    matched = working[working[NER_TAG] == tag]
    if matched.empty:
        available = ", ".join(sorted(working[NER_TAG].unique())[:15])
        return Result.failure(
            Diagnostic.error(
                "PANEL_NO_DATA", f"no entity has NER tag {tag!r} in this table; tags present include: {available}"
            )
        )
    total_docs = matched[DOCUMENT_ID].astype(str).nunique()
    grouped = matched.groupby(ENTITY, sort=False).agg(_count=(NER_COUNT, "sum"), _docs=(DOCUMENT_ID, "nunique"))
    grouped = grouped.sort_values(["_count", "_docs"], ascending=[False, False], kind="stable").head(top_n)

    marks: list[PanelMark] = []
    for rank, (entity, row) in enumerate(grouped.iterrows()):
        count, docs = int(row["_count"]), int(row["_docs"])
        marks.append(
            PanelMark(
                key=str(entity),
                label=str(entity),
                x=float(count),
                y=float(rank),
                evidence=Evidence(
                    scope="rows",
                    filters=((ENTITY, str(entity)), (NER_TAG, tag)),
                    count=count,
                    describe=f"{entity} ({tag}): {count} mention(s) in {docs} of {total_docs} tagged document(s)",
                ),
            )
        )
    prepared = PreparedPanel(
        panel=NER_TOP_ENTITIES.name,
        shape="ranked_bars",
        title=f"Most-mentioned {tag} entities",
        subtitle=f"top {len(marks)} of {matched[ENTITY].nunique()} {tag} entities · {total_docs} document(s) tagged",
        marks=tuple(marks),
        x_label="Mentions",
        y_label=f"{tag} entity, most mentioned first",
        provenance=provenance,
        data=matched,
        notes=(
            "Coverage (N of M documents) matters as much as the count: an entity mentioned often in one "
            "document can outrank one mentioned once each across many.",
            "Entity text is the NER model's own span; the same real-world entity can appear under slightly "
            "different surface strings ('U.S.' vs 'United States') and be counted separately.",
        ),
    )
    return Result.success(prepared)


def ner_entity_timeline(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Chosen entities' mentions per speech-year; explicit zeros within the corpus's own dated span."""
    tag = str(params["tag"]).strip().upper()
    entities_spec = str(params["entities"]).strip()
    default_n = int(params["top-n"])

    working = frame.copy()
    working[NER_TAG] = working[NER_TAG].astype(str).str.upper()
    working[YEAR] = pd.to_numeric(working[YEAR], errors="coerce")
    working[NER_COUNT] = pd.to_numeric(working[NER_COUNT], errors="coerce").fillna(0)
    dated = working[working[YEAR].notna()].copy()
    if dated.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no dated rows in this table"))
    dated[YEAR] = dated[YEAR].astype(int)
    # Every year the corpus itself has documents in, from ANY entity's rows --
    # not just the chosen ones -- so a chosen entity's zero year (documents
    # existed, it just was not mentioned) can be told apart from a gap (no
    # documents that year at all), the rule panels_ngram_viewer.py set.
    corpus_years = sorted(dated[YEAR].unique())

    matched = dated[dated[NER_TAG] == tag]
    if matched.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"no entity has NER tag {tag!r} in this table"))

    diagnostics: list[Diagnostic] = []
    if entities_spec:
        entities = [e.strip() for e in entities_spec.split(",") if e.strip()]
        missing = [e for e in entities if e not in set(matched[ENTITY].astype(str))]
        if missing:
            diagnostics.append(
                Diagnostic.warning("PANEL_ENTITIES_NOT_FOUND", f"not found with tag {tag}: {', '.join(missing)}")
            )
        entities = [e for e in entities if e not in missing]
    else:
        coverage = matched.groupby(ENTITY, sort=False)[DOCUMENT_ID].nunique().sort_values(ascending=False)
        # An entity in nearly every document ("Speaker", from "Mr. Speaker")
        # is a form of address; its line is flat and says nothing about
        # change. Default to entities that come and go.
        documents = max(1, int(dated[DOCUMENT_ID].nunique()))
        varying = coverage[coverage / documents <= _UBIQUITOUS]
        entities = list((varying if not varying.empty else coverage).head(default_n).index)
    if not entities:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no requested entity matched this tag"), *diagnostics)

    per_cell = matched.groupby([ENTITY, YEAR])[NER_COUNT].sum()
    marks: list[PanelMark] = []
    for entity in entities:
        for year in corpus_years:
            count = float(per_cell.get((entity, year), 0.0))
            marks.append(
                PanelMark(
                    key=f"{entity}␟{year}",
                    label=entity,
                    x=float(year),
                    y=count,
                    group=entity,
                    evidence=Evidence(
                        scope="rows",
                        filters=((ENTITY, entity), (YEAR, str(year))),
                        count=int(count),
                        describe=f"{entity}, {year}: {int(count)} mention(s)"
                        + ("" if count else " (dated documents existed this year; none mentioned it)"),
                    ),
                )
            )
    prepared = PreparedPanel(
        panel=NER_ENTITY_TIMELINE.name,
        shape="line_series",
        title=f"{tag} entity mentions over time",
        subtitle=f"{len(entities)} {tag} entities across {len(corpus_years)} dated year(s)",
        marks=tuple(marks),
        x_label="Year",
        y_label="Mentions (raw count)",
        provenance=provenance,
        data=matched,
        groups=tuple(entities),
        notes=(
            "Raw counts, not a rate: this table carries no per-year token totals to normalise against, so a "
            "taller point can mean a longer speech that year rather than more attention to that entity.",
            "A zero point is a year with dated documents in which this entity was never mentioned; a gap is a "
            "year with no dated documents in the corpus at all.",
        ),
    )
    return Result.success(prepared, *diagnostics)


NER_TOP_ENTITIES = PanelDefinition(
    name="ner_top_entities",
    title="Most-mentioned entities by type",
    question="Which named entities of a given type dominate this corpus, and how widely?",
    tool="ner",
    shape="ranked_bars",
    summary="Ranked entities filtered to one NER tag, with mentions and document coverage.",
    requires=(ENTITY, NER_TAG, NER_COUNT, DOCUMENT_ID),
    params=(
        PanelParam(
            name="tag",
            type="str",
            default="PERSON",
            label="NER tag",
            help="Which NER tag to rank, e.g. PERSON, GPE, ORG, NORP.",
        ),
        PanelParam(
            name="top-n",
            type="int",
            default=25,
            minimum=1,
            maximum=200,
            label="Entities to show",
            help="How many of the most-mentioned entities of this tag to draw.",
        ),
    ),
    build=ner_top_entities,
    notes=("Different surface spellings of one real entity are counted as separate rows.",),
)

NER_ENTITY_TIMELINE = PanelDefinition(
    name="ner_entity_timeline",
    title="Entity mentions over time",
    question="Has attention to specific named entities risen or fallen across the corpus?",
    tool="ner",
    shape="line_series",
    summary="Chosen entities' mentions per dated year, one line each, raw counts with explicit no-mention zeros.",
    requires=(ENTITY, NER_TAG, NER_COUNT, YEAR),
    params=(
        PanelParam(
            name="tag",
            type="str",
            default="GPE",
            label="NER tag",
            help="Which NER tag the chosen entities belong to. Places (GPE) are the default: which countries a "
            "speech names changes with its decade.",
        ),
        PanelParam(
            name="entities",
            type="str",
            default="",
            label="Entities",
            help="Comma-separated entity names to plot. Empty picks the top 'Entities to show' by document coverage.",
        ),
        PanelParam(
            name="top-n",
            type="int",
            default=5,
            minimum=1,
            maximum=25,
            label="Entities to show",
            help="How many entities to auto-select by document coverage when 'entities' is left empty.",
        ),
    ),
    build=ner_entity_timeline,
    notes=("Raw mention counts; this table has no per-year token totals to normalise against.",),
)


# =====================================================================
# kwic -- where in each document its hits fall
# =====================================================================

HIT = "Hit"
LEFT_CONTEXT = "Left Context"
RIGHT_CONTEXT = "Right Context"


def kwic_hit_positions(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Documents with hits (rows, by date) x each hit's raw sentence position.

    The kwic table only carries the sentence each hit fell in, not how many
    sentences the document has, so a hit cannot be placed at a true 0-1
    fraction through the document (unlike the dispersion plot's corpus-wide
    token offset, or the narrative/shapes bins, which bin every scored
    sentence and so know each document's own length). ``x`` is the raw
    Sentence ID instead, stated as such in the axis label.
    """
    working = frame.copy()
    working[SENTENCE_ID] = pd.to_numeric(working[SENTENCE_ID], errors="coerce")
    bad = working[SENTENCE_ID].isna() | working[DATE].isna()
    dropped = int(bad.sum())
    working = working.loc[~bad].copy()
    diagnostics: list[Diagnostic] = []
    if dropped:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_BAD_NUMERIC",
                f"{dropped} hit(s) had no usable sentence id or date and were left out.",
                dropped=dropped,
            )
        )
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no hits to plot"), *diagnostics)

    if len(working) > _MARK_CAP:
        working = working.sort_values(DATE, kind="stable").head(_MARK_CAP)
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_TOO_MANY_MARKS", f"kept the first {_MARK_CAP} hits by date for a readable plot.", drawn=_MARK_CAP
            )
        )

    labels = document_labels(working[DOCUMENT].astype(str).unique())
    ordered_docs = working.drop_duplicates(DOCUMENT).sort_values(DATE, kind="stable")[DOCUMENT].astype(str).tolist()
    y_categories = tuple(labels[doc] for doc in ordered_docs)
    row_index = {doc: i for i, doc in enumerate(ordered_docs)}

    marks: list[PanelMark] = []
    for position, (_, row) in enumerate(working.iterrows()):
        doc = str(row[DOCUMENT])
        sent_id = int(row[SENTENCE_ID])
        hit = str(row[HIT])
        marks.append(
            PanelMark(
                key=f"{doc}␟{sent_id}␟{position}",
                label=hit,
                x=float(sent_id),
                y=float(row_index[doc]),
                evidence=Evidence(
                    scope="rows",
                    filters=((DOCUMENT, doc), (SENTENCE_ID, str(sent_id)), (HIT, hit)),
                    count=1,
                    describe=f"{labels[doc]}, sentence {sent_id}: …{row[LEFT_CONTEXT]} [{hit}] {row[RIGHT_CONTEXT]}…",
                ),
            )
        )
    prepared = PreparedPanel(
        panel=KWIC_HIT_POSITIONS.name,
        shape="positions",
        title="Where each hit falls, by document",
        subtitle=f"{len(marks)} hit(s) across {len(y_categories)} document(s), ordered by date",
        marks=tuple(marks),
        x_label="Sentence number in the document (raw, not a 0-1 fraction)",
        y_label="Document (earliest first)",
        provenance=provenance,
        data=working,
        y_categories=y_categories,
        notes=(
            "x is the raw sentence number the hit fell in, not a position fraction: this table does not carry "
            "each document's total sentence count, so a true 0-1 position (as the dispersion or narrative "
            "panels use) cannot be computed from it. A long document's hits will sit further right for that "
            "reason alone, not because the term occurs 'later' in a comparable sense.",
            "Documents are ordered top-to-bottom by speech date, earliest first.",
        ),
    )
    return Result.success(prepared, *diagnostics)


KWIC_HIT_POSITIONS = PanelDefinition(
    name="kwic_hit_positions",
    title="Where each hit falls, by document",
    question="Within each document that uses this term, where do its uses cluster?",
    tool="kwic",
    shape="positions",
    summary="Documents with a hit as rows (by date), ticked at each hit's raw sentence position.",
    requires=(DOCUMENT, DATE, SENTENCE_ID, HIT, LEFT_CONTEXT, RIGHT_CONTEXT),
    params=(),
    build=kwic_hit_positions,
    notes=(
        "x is a raw sentence number, not a normalised position: the table does not carry each document's total "
        "sentence count.",
    ),
)


# =====================================================================
# coreference -- chain mentions as positions per document (optional)
# =====================================================================

CLUSTER_ID = "Cluster ID"
MENTION = "Mention"
LEMMA = "Lemma"


def coreference_chain_positions(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """One chosen lemma's coreference chain: documents (rows) x raw sentence position.

    The engine (``core/analysis/coreference.py``) is a deliberately simple
    same-lemma clustering stub, not a real coreference resolver -- its own
    docstring says so. This panel is honest about drawing that stub's output,
    not a stronger claim about anaphora resolution.
    """
    lemma_param = str(params["lemma"]).strip().lower()
    working = frame.copy()
    working[SENTENCE_ID] = pd.to_numeric(working[SENTENCE_ID], errors="coerce")
    working[LEMMA] = working[LEMMA].astype(str).str.lower()
    bad = working[SENTENCE_ID].isna()
    working = working.loc[~bad].copy()
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no usable coreference rows"))

    diagnostics: list[Diagnostic] = []
    if lemma_param:
        lemma = lemma_param
        if lemma not in set(working[LEMMA]):
            return Result.failure(
                Diagnostic.error("PANEL_NO_DATA", f"lemma {lemma_param!r} has no coreference chain in this table")
            )
    else:
        # The broadest chain: the lemma whose mentions span the most documents.
        coverage = working.groupby(LEMMA)[DOCUMENT_ID].nunique().sort_values(ascending=False)
        lemma = str(coverage.index[0])
        diagnostics.append(
            Diagnostic.info("PANEL_LEMMA_AUTOSELECTED", f"no lemma given; showing the widest chain, {lemma!r}.")
        )

    chain = working[working[LEMMA] == lemma]
    has_dates = DATE in chain.columns and chain[DATE].notna().any()
    if has_dates:
        ordered_docs = (
            chain.drop_duplicates(DOCUMENT_ID).sort_values(DATE, kind="stable")[DOCUMENT_ID].astype(str).tolist()
        )
    else:
        ordered_docs = sorted(chain[DOCUMENT_ID].astype(str).unique())
    # Labels are keyed by document *id*: document_labels names files, and a
    # lookup by id into a name-keyed map fell back to the id on every row
    # ("1", "10", "11" on the real corpus).
    if DOCUMENT in chain.columns:
        name_of = dict(zip(chain[DOCUMENT_ID].astype(str), chain[DOCUMENT].astype(str), strict=True))
        by_name = document_labels(name_of[d] for d in ordered_docs)
        labels = {d: by_name[name_of[d]] for d in ordered_docs}
    else:
        labels = {d: f"Document {d}" for d in ordered_docs}
    y_categories = tuple(labels[d] for d in ordered_docs)
    row_index = {doc: i for i, doc in enumerate(ordered_docs)}

    marks: list[PanelMark] = []
    for position, (_, row) in enumerate(chain.iterrows()):
        doc_id = str(row[DOCUMENT_ID])
        sent_id = int(row[SENTENCE_ID])
        mention = str(row[MENTION])
        marks.append(
            PanelMark(
                key=f"{doc_id}␟{sent_id}␟{position}",
                label=mention,
                x=float(sent_id),
                y=float(row_index[doc_id]),
                evidence=Evidence(
                    scope="rows",
                    filters=((DOCUMENT_ID, doc_id), (LEMMA, lemma), (SENTENCE_ID, str(sent_id))),
                    count=1,
                    describe=f"{labels.get(doc_id, doc_id)}, sentence {sent_id}: {mention!r} (chain {row[CLUSTER_ID]})",
                ),
            )
        )
    prepared = PreparedPanel(
        panel=COREFERENCE_CHAIN_POSITIONS.name,
        shape="positions",
        title=f"Coreference chain for '{lemma}', by document",
        subtitle=f"{len(marks)} mention(s) across {len(y_categories)} document(s)",
        marks=tuple(marks),
        x_label="Sentence number in the document (raw)",
        y_label="Document",
        provenance=provenance,
        data=chain,
        y_categories=y_categories,
        notes=(
            "This engine clusters mentions by identical lowercased lemma only -- it is a baseline, not real "
            "coreference resolution, and cannot tell 'the bank' (river) from 'the bank' (finance).",
            "x is a raw sentence number, not a normalised position; this table does not carry each document's "
            "total sentence count.",
        ),
    )
    return Result.success(prepared, *diagnostics)


COREFERENCE_CHAIN_POSITIONS = PanelDefinition(
    name="coreference_chain_positions",
    title="Coreference chain, by document",
    question="Where in the corpus does one recurring noun's mention chain actually fall?",
    tool="coreference",
    shape="positions",
    summary="One lemma's mention chain as ticks (raw sentence position) across the documents it appears in.",
    requires=(CLUSTER_ID, MENTION, LEMMA, SENTENCE_ID, DOCUMENT_ID, DOCUMENT),
    params=(
        PanelParam(
            name="lemma",
            type="str",
            default="",
            label="Lemma",
            help="Which noun lemma's chain to draw. Empty auto-selects the chain spanning the most documents.",
        ),
    ),
    build=coreference_chain_positions,
    notes=("The engine is a same-lemma clustering baseline, not a real coreference resolver.",),
)


TIME_POSITIONS_PANELS: tuple[PanelDefinition, ...] = (
    LEXICON_SERIES_RATE,
    LEXICON_SERIES_HEATMAP,
    DATE_ANNOTATOR_TIMELINE,
    DATE_ANNOTATOR_TYPE_COUNTS,
    GENDER_ANNOTATOR_MENTIONS_OVER_TIME,
    NARRATIVE_EMOTION_ARC,
    NARRATIVE_CHARACTER_POSITIONS,
    SHAPES_STORY_ARC,
    DISPERSION_FREQUENCY_VS_CONCENTRATION,
    NER_TOP_ENTITIES,
    NER_ENTITY_TIMELINE,
    KWIC_HIT_POSITIONS,
    COREFERENCE_CHAIN_POSITIONS,
)
