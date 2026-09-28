"""Figures over a comparison run (docs/internal/PLAN_0.5.0.md 3.7).

Each is an ordinary panel over one of the ``contrast`` tool's tables, so it
gets Explore, saved views and publication like every other figure.

The headline figure is the per-group one: "per president, did the two
settings differ?" is a row per president with one bar per side, not a single
pooled number that one long-serving president can dominate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panelspec import Annotation, Evidence, PanelDefinition, PanelMark, PanelParam, PreparedPanel, Provenance

__all__ = ["CONTRAST_PANELS"]

TOOL = "contrast"
SIDE = "Side"
GROUP = "Group"
DOC_ID = "Document ID"
DOC = "Document"
#: The per-document measures ``contrast_measures.csv`` can hold (core/contrast/measures.py).
MEASURE_COLUMNS = ("Avg Sentence Length", "Avg Word Length", "Flesch Reading Ease", "Flesch-Kincaid Grade", "MTLD")
_ALL = ("All documents", "All groups")

_SIDE_NOTES = (
    "Each point is one document. Sides that come from different collections also differ in setting "
    "(read the comparison's notes): a gap here may be the setting rather than the speaker.",
    "Documents of very different lengths are compared as documents, one point each; the measures are "
    "per-document averages, so a long speech counts once.",
)


def _side_distribution(
    frame: pd.DataFrame, measure: str, by_group: bool, definition: PanelDefinition, provenance: Provenance
) -> Result[PreparedPanel]:
    working = frame.copy()
    working[measure] = pd.to_numeric(working[measure], errors="coerce")
    working = working.dropna(subset=[measure])
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"no compared document has a {measure} value"))
    sides = list(dict.fromkeys(working[SIDE].astype(str)))
    if by_group and working[GROUP].nunique() > 1:
        working["_row"] = working[GROUP].astype(str) + " · " + working[SIDE].astype(str)
        order = list(dict.fromkeys(f"{g} · {s}" for g in dict.fromkeys(working[GROUP].astype(str)) for s in sides))
        order = [row for row in order if row in set(working["_row"])]
    else:
        working["_row"] = working[SIDE].astype(str)
        order = sides
    index = {row: position for position, row in enumerate(order)}
    marks = tuple(
        PanelMark(
            key=f"doc:{row[DOC_ID]}",
            label=str(row[DOC]),
            x=float(row[measure]),
            y=float(index[row["_row"]]),
            group=str(row[SIDE]),
            evidence=Evidence(
                scope="rows",
                filters=((DOC_ID, str(row[DOC_ID])),),
                count=1,
                describe=f"{row[DOC]} ({row[SIDE]}, {row[GROUP]}): {measure} {float(row[measure]):.2f}",
            ),
        )
        for _, row in working.iterrows()
    )
    return Result.success(
        PreparedPanel(
            panel=definition.name,
            shape="distribution",
            title=f"{definition.title}: {measure}",
            subtitle=f"{len(working)} documents on {len(sides)} sides"
            + (" · by group" if len(order) > len(sides) else ""),
            marks=marks,
            x_label=measure,
            y_label="",
            provenance=provenance,
            data=working.drop(columns=["_row"]).reset_index(drop=True),
            groups=tuple(sides),
            y_categories=tuple(order),
            notes=definition.notes,
        )
    )


def _measures(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    measure = str(params["measure"])
    if measure not in frame.columns:
        present = [column for column in MEASURE_COLUMNS if column in frame.columns]
        return Result.failure(
            Diagnostic.error(
                "PANEL_NO_MEASURE", f"this comparison has no {measure}; it has {', '.join(present) or 'none'}"
            )
        )
    return _side_distribution(frame, measure, bool(params["by-group"]), CONTRAST_MEASURES, provenance)


def _tone(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    return _side_distribution(frame, "Compound", bool(params["by-group"]), CONTRAST_TONE, provenance)


def _effects(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    _ = params
    working = frame[frame[GROUP].isin(_ALL)].copy()
    working["Cliff's delta"] = pd.to_numeric(working["Cliff's delta"], errors="coerce")
    working = working.dropna(subset=["Cliff's delta"])
    if working.empty:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", "no measure had enough documents on both sides to compute an effect size")
        )
    working = working.assign(_abs=working["Cliff's delta"].abs()).sort_values("_abs", ascending=False, kind="stable")
    pairs = sorted({(str(a), str(b)) for a, b in zip(working["Side A"], working["Side B"], strict=True)})
    marks = []
    for rank, (_, row) in enumerate(working.iterrows()):
        label = str(row["Measure"]) if len(pairs) == 1 else f"{row['Measure']} ({row['Side A']} vs {row['Side B']})"
        marks.append(
            PanelMark(
                key=f"{row['Measure']}|{row['Side A']}|{row['Side B']}",
                label=label,
                x=float(row["Cliff's delta"]),
                y=float(rank),
                group=str(row["Side A"] if row["Cliff's delta"] > 0 else row["Side B"]),
                evidence=Evidence(
                    scope="rows", filters=(("Measure", str(row["Measure"])),), count=1, describe=str(row["Reading"])
                ),
            )
        )
    side_a, side_b = pairs[0]
    return Result.success(
        PreparedPanel(
            panel=CONTRAST_EFFECTS.name,
            shape="ranked_bars",
            title=CONTRAST_EFFECTS.title,
            subtitle=f"right = higher for {side_a}, left = higher for {side_b}"
            if len(pairs) == 1
            else "each pair of sides",
            marks=tuple(marks),
            x_label="Cliff's delta (-1 to 1)",
            y_label="",
            provenance=provenance,
            data=working.drop(columns=["_abs"]).reset_index(drop=True),
            groups=tuple(dict.fromkeys(mark.group for mark in marks)),
            annotations=(
                Annotation(
                    kind="vline",
                    value=0.0,
                    label="no difference",
                    note="0: a document from either side is as likely to score higher.",
                ),
                Annotation(
                    kind="vline",
                    value=0.474,
                    label="large",
                    note="Beyond ±0.474 the difference is large (Romano et al. bands).",
                ),
                Annotation(
                    kind="vline",
                    value=-0.474,
                    label="large",
                    note="Beyond ±0.474 the difference is large (Romano et al. bands).",
                ),
            ),
            notes=CONTRAST_EFFECTS.notes,
        )
    )


def _words(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    per_side = int(params["per-side"])
    rank_by = "G2" if params["rank-by"] == "evidence" else "Log Ratio"
    working = frame.copy()
    for column in ("Log Ratio", "G2", "Per 10k", "Other per 10k"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=["Log Ratio", "G2"])
    if bool(params["significant-only"]):
        working = working[pd.to_numeric(working["p-value"], errors="coerce") < 0.05]
    chosen = (
        working.sort_values([SIDE, rank_by, "Word"], ascending=[True, False, True], kind="stable")
        .groupby(SIDE)
        .head(per_side)
    )
    if chosen.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no distinctive words left; turn off significant-only"))
    sides = list(dict.fromkeys(frame[SIDE].astype(str)))
    marks = []
    rank = 0
    for side in sides:
        for _, row in chosen[chosen[SIDE] == side].iterrows():
            word = str(row["Word"])
            marks.append(
                PanelMark(
                    key=f"{side}|{word}",
                    label=word,
                    x=float(row["Log Ratio"]),
                    y=float(rank),
                    group=side,
                    evidence=Evidence(
                        scope="terms",
                        filters=(("Word", word),),
                        count=int(row["Count"]),
                        phrase=word,
                        lemma=True,
                        describe=(
                            f"{word}: {float(row['Per 10k']):.1f} per 10,000 words in {side}, "
                            f"{float(row['Other per 10k']):.1f} in {row['Compared with']} "
                            f"(log ratio {float(row['Log Ratio']):+.2f}, G2 {float(row['G2']):.1f})"
                        ),
                    ),
                )
            )
            rank += 1
    return Result.success(
        PreparedPanel(
            panel=CONTRAST_WORDS.name,
            shape="ranked_bars",
            title=CONTRAST_WORDS.title,
            subtitle=f"top {per_side} per side by {'evidence (G2)' if rank_by == 'G2' else 'effect (log ratio)'}",
            marks=tuple(marks),
            x_label="Log ratio: doublings more often on this side",
            y_label="",
            provenance=provenance,
            data=chosen.reset_index(drop=True),
            groups=tuple(sides),
            notes=CONTRAST_WORDS.notes,
        )
    )


def _other_word_groups(rates: list[str], drawn: str) -> str:
    """The word groups this figure is not showing, so the reader knows what to type in "Word group"."""
    others = [rate.removesuffix(" per 10k") for rate in rates if rate != drawn]
    return f" · also counted: {', '.join(others)}" if others else ""


def _focus(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    rates = [column for column in frame.columns if str(column).endswith(" per 10k")]
    wanted = str(params["word-group"]).strip()
    column = f"{wanted} per 10k" if wanted else (rates[0] if rates else "")
    if column not in frame.columns:
        names = ", ".join(rate.removesuffix(" per 10k") for rate in rates)
        return Result.failure(
            Diagnostic.error("PANEL_NO_GROUP", f"no word group '{wanted}'; this comparison counted {names}")
        )
    name = column.removesuffix(" per 10k")
    working = frame.copy()
    working[column] = pd.to_numeric(working[column], errors="coerce")
    sides = list(dict.fromkeys(working[SIDE].astype(str)))
    wide = working.pivot_table(index=GROUP, columns=SIDE, values=column, aggfunc="first")
    if len(sides) >= 2:
        wide = wide.assign(_gap=(wide[sides[0]] - wide[sides[1]]).abs()).sort_values(
            "_gap", ascending=False, kind="stable"
        )
    marks = []
    for rank, (group, row) in enumerate(wide.iterrows()):
        for side in sides:
            value = row.get(side)
            if value is None or pd.isna(value):
                continue
            count_column = f"{name} count"
            here = working[(working[GROUP] == group) & (working[SIDE] == side)]
            count = int(here[count_column].iloc[0]) if count_column in here and len(here) else 0
            words = int(here["Words counted"].iloc[0]) if len(here) else 0
            marks.append(
                PanelMark(
                    key=f"{group}|{side}",
                    label=str(group),
                    x=float(value),
                    y=float(rank),
                    group=side,
                    evidence=Evidence(
                        scope="rows",
                        filters=((GROUP, str(group)), (SIDE, side)),
                        count=count,
                        describe=f"{group}, {side}: {count} uses in {words:,} words = {float(value):.2f} per 10,000",
                    ),
                )
            )
    if not marks:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"no group has a {name} rate"))
    return Result.success(
        PreparedPanel(
            panel=CONTRAST_FOCUS.name,
            shape="ranked_bars",
            title=f"{name}: per group, side by side",
            subtitle=(
                f"{len(wide)} group(s), largest gap between {sides[0]} and {sides[1]} first" if len(sides) >= 2 else ""
            )
            + _other_word_groups(rates, column),
            marks=tuple(marks),
            x_label=f"{name} per 10,000 words",
            y_label="",
            provenance=provenance,
            data=working.reset_index(drop=True),
            groups=tuple(sides),
            notes=CONTRAST_FOCUS.notes,
        )
    )


def _topics(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    _ = params
    shares = [column for column in frame.columns if str(column).endswith(" share")]
    if len(shares) < 2:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "the topic table has fewer than two sides"))
    sides = [column.removesuffix(" share") for column in shares]
    working = frame.copy()
    working["_gap"] = (pd.to_numeric(working[shares[0]]) - pd.to_numeric(working[shares[1]])).abs()
    working = working.sort_values("_gap", ascending=False, kind="stable")
    marks = []
    for rank, (_, row) in enumerate(working.iterrows()):
        label = f"Topic {row['Topic']}: {str(row['Keywords'])[:48]}"
        for side, column in zip(sides, shares, strict=True):
            value = float(row[column]) * 100
            marks.append(
                PanelMark(
                    key=f"{row['Topic']}|{side}",
                    label=label,
                    x=value,
                    y=float(rank),
                    group=side,
                    evidence=Evidence(
                        scope="rows",
                        filters=(("Topic", str(row["Topic"])),),
                        count=1,
                        describe=f"Topic {row['Topic']} ({row['Keywords']}): {value:.1f}% of {side}'s text",
                    ),
                )
            )
    return Result.success(
        PreparedPanel(
            panel=CONTRAST_TOPICS.name,
            shape="ranked_bars",
            title=CONTRAST_TOPICS.title,
            subtitle="one topic model over every side; largest difference first",
            marks=tuple(marks),
            x_label="Share of the side's text (%)",
            y_label="",
            provenance=provenance,
            data=working.drop(columns=["_gap"]).reset_index(drop=True),
            groups=tuple(sides),
            notes=CONTRAST_TOPICS.notes,
        )
    )


_BY_GROUP = PanelParam(
    name="by-group",
    type="bool",
    default=False,
    label="One row per group",
    help="Split each side by the comparison's groups (the speaker, the decade), so like is shown beside like.",
)

CONTRAST_MEASURES = PanelDefinition(
    name="contrast_measures",
    title="Side by side",
    question="Do the sides differ in how they are written: sentence length, reading ease, vocabulary?",
    tool=TOOL,
    shape="distribution",
    summary="One point per document, one row per side, for a chosen measure.",
    # A measure column, so this panel is offered on the measures table and not
    # on the tone table, which has the same Side/Group/Document columns.
    requires=(SIDE, GROUP, DOC_ID, DOC, "Avg Sentence Length"),
    params=(
        PanelParam(
            name="measure",
            type="choice",
            default="Avg Sentence Length",
            choices=MEASURE_COLUMNS,
            label="Measure",
            help="Which per-document measure to compare.",
        ),
        _BY_GROUP,
    ),
    build=_measures,
    notes=_SIDE_NOTES,
)

CONTRAST_TONE = PanelDefinition(
    name="contrast_tone",
    title="Tone, side by side",
    question="Is one side more positive in tone than the other?",
    tool=TOOL,
    shape="distribution",
    summary="VADER's overall tone (compound, -1 to 1) per document, one row per side.",
    requires=(SIDE, GROUP, DOC_ID, DOC, "Compound"),
    params=(_BY_GROUP,),
    build=_tone,
    notes=(
        "VADER scores words from a fixed list; it reads 'we will not secure the border' and 'we will secure "
        "the border' as nearly the same tone. Read the most extreme documents before trusting a difference.",
        *_SIDE_NOTES,
    ),
)

CONTRAST_EFFECTS = PanelDefinition(
    name="contrast_effects",
    title="Where the sides differ most",
    question="On which measures do the sides differ most, and in which direction?",
    tool=TOOL,
    shape="ranked_bars",
    summary="Cliff's delta per measure: how often a document from one side scores above one from the other.",
    requires=("Measure", GROUP, "Side A", "Side B", "Cliff's delta", "Reading"),
    params=(),
    build=_effects,
    notes=(
        "Cliff's delta is the chance a document from the first side scores higher than one from the second, "
        "minus the reverse: 0 is no difference, ±1 means every document of one side is above every document "
        "of the other.",
        "An effect size says how big a difference is, not why. Two settings (an address to Congress, an "
        "inaugural) differ in purpose and audience before anyone's style comes into it.",
    ),
)

CONTRAST_WORDS = PanelDefinition(
    name="contrast_words",
    title="Words that set each side apart",
    question="Which words does each side use far more often than the other?",
    tool=TOOL,
    shape="ranked_bars",
    summary="The most distinctive words of each side, by log ratio, with rates per 10,000 words.",
    requires=(SIDE, "Word", "Log Ratio", "G2", "Per 10k", "Other per 10k", "Compared with"),
    params=(
        PanelParam(
            name="per-side",
            type="int",
            default=15,
            minimum=3,
            maximum=60,
            label="Words per side",
            help="How many words to show for each side.",
        ),
        PanelParam(
            name="rank-by",
            type="choice",
            default="evidence",
            choices=("evidence", "effect"),
            label="Rank by",
            help="'evidence' ranks by G2 (most reliably different), 'effect' by log ratio (most different).",
        ),
        PanelParam(
            name="significant-only",
            type="bool",
            default=True,
            label="Only p < 0.05",
            help="Leave out words whose difference could easily be chance.",
        ),
    ),
    build=_words,
    notes=(
        "Each bar is how many doublings more often a word is used on its side, counted per 10,000 words so a "
        "longer side does not win by length.",
        "A distinctive word is a reason to read it in context on both sides, not a finding: 'oath' marks "
        "inaugurals because of the ceremony, not a change of mind.",
    ),
)

CONTRAST_FOCUS = PanelDefinition(
    name="contrast_focus",
    title="Focus words, per group, side by side",
    question="Per president (or decade), does each side use the focus words more or less?",
    tool=TOOL,
    shape="ranked_bars",
    summary="The focus rate of each group on each side: a row per group, a bar per side.",
    requires=(GROUP, SIDE, "Documents", "Words counted"),
    params=(
        PanelParam(
            name="word-group",
            type="str",
            default="",
            label="Word group",
            help="Which focus word group to draw; empty draws the first.",
        ),
    ),
    build=_focus,
    notes=(
        "Rates are per 10,000 words of each group's documents on each side, pooled: a group with one short "
        "inaugural and ten long addresses is compared rate to rate, not count to count.",
        "A group with one document on a side rests on one speech. Open the documents before reading a gap.",
    ),
)

CONTRAST_TOPICS = PanelDefinition(
    name="contrast_topics",
    title="Topics each side spends its words on",
    question="Which themes does one side spend far more of its text on?",
    tool=TOOL,
    shape="ranked_bars",
    summary="Each topic's share of each side's text, from one topic model fitted over every side.",
    requires=("Topic", "Keywords", "Leans towards"),
    params=(),
    build=_topics,
    notes=(
        "One model is fitted over all sides together, so topic 3 means the same thing on each; two separate "
        "models could not be compared topic by topic.",
        "A paragraph's words count towards the topic that dominates it, so mixed paragraphs are simplified.",
    ),
)

CONTRAST_PANELS: tuple[PanelDefinition, ...] = (
    CONTRAST_FOCUS,
    CONTRAST_EFFECTS,
    CONTRAST_WORDS,
    CONTRAST_MEASURES,
    CONTRAST_TONE,
    CONTRAST_TOPICS,
)
