"""Speech-level views of the verb-analysis output.

The document-measure panels can show passive voice and obligation as separate
trends. These panels put the categories back together: how each speech's verbs
divide among voice and modality, and whether passive voice and obligation
language tend to appear together in the same speech.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

import numpy as np
import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import decimal_year, document_labels, group_of
from core.viz.panelspec import Annotation, Evidence, PanelDefinition, PanelMark, PanelParam, PreparedPanel, Provenance

__all__ = ["VERB_PROFILE_PANELS"]

DOC_ID = "Document ID"
DOC = "Document"
VOICE = "Voice"
MODALITY = "Modality"

_GROUP_BY = PanelParam(
    name="group-by",
    type="choice",
    default="decade",
    choices=("decade", "speaker", "none"),
    label="Colour speeches by",
    help="Group points by decade or speaker parsed from the document name.",
)
_MIN_VERBS = PanelParam(
    name="minimum-verbs",
    type="int",
    default=5,
    minimum=1,
    maximum=10_000,
    label="Minimum verbs in a speech",
    help="Hide very short samples whose percentages are especially noisy.",
)

_VOICE_LABELS = {"active": "Active", "passive": "Passive", "unknown": "Unknown voice"}
_MODALITY_LABELS = {
    "ability": "Ability",
    "possibility": "Possibility",
    "obligation": "Obligation",
    "none": "No modality",
    "unknown": "Unknown modality",
}


def _year_from_name(name: str) -> float | None:
    """A year from the suite's date-prefixed file names; no date means none."""
    match = re.match(r"^(\d{4})(?:[-_]|$)", str(name).strip())
    return decimal_year(match.group(1)) if match else None


def _documents(frame: pd.DataFrame) -> pd.DataFrame:
    """Return stable, document-level verb counts from the per-verb table."""
    working = frame.copy()
    working[DOC_ID] = working[DOC_ID].astype(str).str.strip()
    working[DOC] = working[DOC].astype(str).str.strip()
    working[VOICE] = working[VOICE].fillna("unknown").astype(str).str.strip().str.casefold()
    working[MODALITY] = working[MODALITY].fillna("unknown").astype(str).str.strip().str.casefold()
    working.loc[~working[VOICE].isin(_VOICE_LABELS), VOICE] = "unknown"
    working.loc[~working[MODALITY].isin(_MODALITY_LABELS), MODALITY] = "unknown"
    working = working[(working[DOC_ID] != "") & (working[DOC] != "")]
    if working.empty:
        columns = [
            DOC_ID,
            DOC,
            "Year",
            "Verbs",
            *(f"voice:{key}" for key in _VOICE_LABELS),
            *(f"modality:{key}" for key in _MODALITY_LABELS),
        ]
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for (doc_id, name), group in working.groupby([DOC_ID, DOC], sort=False, dropna=False):
        year = _year_from_name(str(name))
        row: dict[str, object] = {DOC_ID: str(doc_id), DOC: str(name), "Year": year, "Verbs": len(group)}
        for category in _VOICE_LABELS:
            row[f"voice:{category}"] = int((group[VOICE] == category).sum())
        for category in _MODALITY_LABELS:
            row[f"modality:{category}"] = int((group[MODALITY] == category).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Year", DOC_ID], na_position="last", kind="stable").reset_index(drop=True)


def _composition(  # noqa: PLR0913 - panel metadata is explicit and keyword-only
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
    *,
    definition: PanelDefinition,
    field: str,
    labels: Mapping[str, str],
) -> Result[PreparedPanel]:
    del params
    docs = _documents(frame)
    if docs.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no verbs have a document name and ID"))
    category_order = tuple(labels)
    doc_names = docs[DOC].astype(str).tolist()
    short_labels = document_labels(doc_names)

    marks: list[PanelMark] = []
    table_rows: list[dict[str, object]] = []
    for row_index, row in docs.iterrows():
        total = int(row["Verbs"])
        if total <= 0:
            continue
        start = 0.0
        for category in category_order:
            count = int(row[f"{field}:{category}"])
            if count <= 0:
                continue
            share = count / total
            display_category = labels[category]
            short_label = short_labels[str(row[DOC])]
            table_rows.append(
                {
                    DOC_ID: str(row[DOC_ID]),
                    DOC: str(row[DOC]),
                    "Year": row["Year"],
                    "Category": display_category,
                    "Count": count,
                    "Share": share,
                    "Verbs": total,
                }
            )
            marks.append(
                PanelMark(
                    key=f"{row[DOC_ID]}:{field}:{category}",
                    label=short_label,
                    x=start,
                    y=float(row_index),
                    size=share,
                    group=display_category,
                    evidence=Evidence(
                        scope="rows",
                        filters=((DOC_ID, str(row[DOC_ID])), ("Category", display_category)),
                        count=count,
                        describe=(f"{short_label}: {share:.1%} {display_category.lower()} ({count} of {total} verbs)"),
                    ),
                )
            )
            start += share

    prepared = PreparedPanel(
        panel=definition.name,
        shape="ribbon",
        title=definition.title,
        subtitle=f"{len(docs)} speeches, ordered by year; each band totals 100% of that speech's verbs",
        marks=tuple(marks),
        x_label=f"Share of {field} categories in each speech",
        y_label="Speech",
        provenance=provenance,
        data=pd.DataFrame(table_rows),
        groups=tuple(labels[category] for category in category_order),
        height=max(500, min(2_400, 24 * len(docs) + 100)),
        notes=(
            *definition.notes,
            "The table beside the figure reports the count and share for each speech/category.",
        ),
    )
    return Result.success(prepared)


def _voice_mix(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    return _composition(
        frame,
        params,
        provenance,
        definition=VERB_VOICE_MIX,
        field="voice",
        labels=_VOICE_LABELS,
    )


def _modality_mix(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    return _composition(
        frame,
        params,
        provenance,
        definition=VERB_MODALITY_MIX,
        field="modality",
        labels=_MODALITY_LABELS,
    )


def _agency_commitment(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    docs = _documents(frame)
    minimum = int(params.get("minimum-verbs", 5))
    excluded = docs[docs["Verbs"] < minimum]
    docs = docs[docs["Verbs"] >= minimum].copy().reset_index(drop=True)
    if docs.empty:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", f"no speeches contain at least {minimum} verbs; lower the minimum")
        )

    names = docs[DOC].astype(str).tolist()
    short_labels = document_labels(names)
    grouping = str(params.get("group-by", "decade"))
    rows: list[dict[str, object]] = []
    for _, row in docs.iterrows():
        total = int(row["Verbs"])
        passive = int(row["voice:passive"])
        obligation = int(row["modality:obligation"])
        year = row["Year"]
        rows.append(
            {
                DOC_ID: str(row[DOC_ID]),
                DOC: str(row[DOC]),
                "Year": year,
                "Verbs": total,
                "Passive %": 100.0 * passive / total,
                "Obligation %": 100.0 * obligation / total,
                "Group": group_of(str(row[DOC]), str(int(year)) if pd.notna(year) else None, grouping),
            }
        )
    result = pd.DataFrame(rows).sort_values(["Year", DOC_ID], na_position="last", kind="stable").reset_index(drop=True)
    groups = tuple(dict.fromkeys(result["Group"].astype(str)))

    # Label only the most extreme points, keeping a dense corpus readable.
    extremes = {
        int(result["Passive %"].idxmin()),
        int(result["Passive %"].idxmax()),
        int(result["Obligation %"].idxmin()),
        int(result["Obligation %"].idxmax()),
    }
    marks = tuple(
        PanelMark(
            key=str(row[DOC_ID]),
            label=short_labels[str(row[DOC])],
            x=float(row["Passive %"]),
            y=float(row["Obligation %"]),
            size=float(row["Verbs"]),
            group=str(row["Group"]),
            labelled=index in extremes,
            evidence=Evidence(
                scope="rows",
                filters=((DOC_ID, str(row[DOC_ID])),),
                count=int(row["Verbs"]),
                describe=(
                    f"{short_labels[str(row[DOC])]}: {row['Passive %']:.1f}% passive, "
                    f"{row['Obligation %']:.1f}% obligation language ({int(row['Verbs'])} verbs)"
                ),
            ),
        )
        for index, (_, row) in enumerate(result.iterrows())
    )
    passive_median = float(np.median(result["Passive %"].to_numpy(dtype=float)))
    obligation_median = float(np.median(result["Obligation %"].to_numpy(dtype=float)))
    prepared = PreparedPanel(
        panel=VERB_AGENCY_COMMITMENT.name,
        shape="scatter_labelled",
        title=VERB_AGENCY_COMMITMENT.title,
        subtitle=f"{len(result)} speeches; bubble area reflects the number of analyzed verbs",
        marks=marks,
        x_label="Passive verbs (% of all verbs)",
        y_label="Obligation-modality verbs (% of all verbs)",
        provenance=provenance,
        data=result,
        groups=groups,
        annotations=(
            Annotation(
                kind="vline",
                value=passive_median,
                label=f"corpus median {passive_median:.1f}% passive",
                note="The median divides this corpus for visual comparison; it is not a significance threshold.",
            ),
            Annotation(
                kind="hline",
                value=obligation_median,
                label=f"corpus median {obligation_median:.1f}% obligation",
                note="The median divides this corpus for visual comparison; it is not a significance threshold.",
            ),
        ),
        notes=VERB_AGENCY_COMMITMENT.notes,
    )
    diagnostics = (
        [
            Diagnostic.info(
                "PANEL_SHORT_SPEECHES_HIDDEN",
                f"{len(excluded)} speech(es) with fewer than {minimum} analyzed verbs were hidden.",
                excluded=len(excluded),
                minimum=minimum,
            )
        ]
        if not excluded.empty
        else []
    )
    return Result.success(prepared, *diagnostics)


_VERB_INPUT = (DOC_ID, DOC, VOICE, MODALITY)

VERB_VOICE_MIX = PanelDefinition(
    name="verb_analysis_voice_mix",
    title="Voice mix within each speech",
    question="How does the share of active and passive verbs vary from speech to speech?",
    tool="verb_analysis",
    shape="ribbon",
    summary="One 100% band per speech, split into active, passive, and unknown voice.",
    requires=_VERB_INPUT,
    params=(),
    build=_voice_mix,
    notes=(
        "Each band is one speech and totals 100% of its analyzed verbs; the speeches are ordered by the year in "
        "their file name. Click a segment to show its speech/category count and share in the table.",
        "The parser assigns voice from dependency labels and a small passive-verb fallback. Unknown voice stays "
        "visible instead of being counted as active.",
    ),
)

VERB_MODALITY_MIX = PanelDefinition(
    name="verb_analysis_modality_mix",
    title="Modality mix within each speech",
    question="How often does each speech use verbs of ability, possibility, or obligation?",
    tool="verb_analysis",
    shape="ribbon",
    summary="One 100% band per speech, split by the modality assigned to each verb.",
    requires=_VERB_INPUT,
    params=(),
    build=_modality_mix,
    notes=(
        "Each band is one speech and totals 100% of its analyzed verbs; the speeches are ordered by the year in "
        "their file name. Click a segment to show its speech/category count and share in the table.",
        "The categories come from the suite's local modal-verb rules. 'No modality' means no listed modal was "
        "detected near the verb; it does not mean the sentence has no persuasive force.",
    ),
)

VERB_AGENCY_COMMITMENT = PanelDefinition(
    name="verb_analysis_agency_commitment",
    title="Passive voice and obligation by speech",
    question="Do speeches with more passive verbs also use more obligation language?",
    tool="verb_analysis",
    shape="scatter_labelled",
    summary="Each speech is a point; compare its passive-verb share with its obligation-modality share.",
    requires=_VERB_INPUT,
    params=(_GROUP_BY, _MIN_VERBS),
    build=_agency_commitment,
    notes=(
        "Each dot is one speech. Both percentages use all analyzed verbs in that speech as the denominator; bubble "
        "area reflects the number of verb rows, and colour is the selected grouping.",
        "The median lines are visual guides for this corpus, not statistical cutoffs. A position shows association "
        "between two parser-derived measures, not that one causes the other.",
    ),
)

VERB_PROFILE_PANELS = (VERB_VOICE_MIX, VERB_MODALITY_MIX, VERB_AGENCY_COMMITMENT)
