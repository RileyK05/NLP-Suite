"""Figures for the model tools: story-shape clustering and components, BERT
topics, and topic stability across seeds.

What each figure is for, on the real 87-speech corpus:

* ``shape_hc`` -- a **dendrogram** (which speeches the clustering joined
  first, and at what distance), and **when each cluster's speeches are
  from**, because a cluster that is one era is a finding about time and a
  cluster that spans ninety years is a finding about form.
* ``shape_svd`` / ``shape_nmf`` -- how much each component explains, the
  **loadings drawn along the speech** (one small multiple per measure, x =
  position through the speech), so a component reads as "long openings" or
  "noun-heavy endings" rather than as a list of feature codes, and the
  documents placed on the components.
* ``bert_topics`` -- each topic's words, and when its documents are from.
* ``lda_stability`` -- which reference topic came back under each seed.

The story-shape tables carry the shape matrix's features as
``<Measure> <position>`` columns ("Noun Ratio 05" is the noun ratio in the
fifth of 32 slices through the speech); the loadings are parsed from those
names rather than listed, so a run resampled to other than 32 slices draws
the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import decimal_year, document_labels, group_of
from core.viz.panelspec import (
    Evidence,
    PanelDefinition,
    PanelEdge,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
)

__all__ = ["MODELS_PANELS"]

DOC_ID = "Document ID"
DOC = "Document"
DATE = "Date"
_FEATURE = re.compile(r"^(?P<measure>.+?) (?P<position>\d+)$")
#: Rows of the table a dendrogram publishes: one per node, leaf or merge.
NODE = "Node"


def _label_map(names: pd.Series) -> dict[str, str]:
    return document_labels(names.astype(str))


# ------------------------------------------------------------ shape_hc --


def _dendrogram(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    """Leaves down the left in the tree's own order; each merge at its height.

    Standard linkage ids: below n is a document (0-based, in the table's
    order), n and above is the merge numbered id - n + 1. Leaves sit at
    x = 0 and every merge at its height over the tallest, so reading right
    is reading "joined later, at a greater distance". The elbow links are
    parent -> child, drawn down then across, which is the dendrogram shape.
    """
    merges = frame.sort_values("Merge").reset_index(drop=True)
    n = len(merges) + 1
    if n < 3:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "a dendrogram needs at least three documents"))
    leaf_name: dict[int, str] = {}
    for _, row in merges.iterrows():
        for side in ("Left", "Right"):
            node = int(row[f"{side} id"])
            if node < n:
                leaf_name[node] = str(row[side])
    children = {n + index: (int(row["Left id"]), int(row["Right id"])) for index, row in merges.iterrows()}
    height = {n + index: float(row["Height"]) for index, row in merges.iterrows()}
    tallest = max(height.values()) or 1.0

    order: list[int] = []
    stack = [n + len(merges) - 1]
    while stack:
        node = stack.pop()
        if node < n:
            order.append(node)
        else:
            left, right = children[node]
            stack.extend((right, left))
    row_of = {leaf: index for index, leaf in enumerate(order)}
    spread = max(1, len(order) - 1)

    y: dict[int, float] = {leaf: row_of[leaf] / spread for leaf in order}
    for node in sorted(children):
        left, right = children[node]
        y[node] = (y[left] + y[right]) / 2

    del params
    labels = document_labels(leaf_name[leaf] for leaf in order)

    marks: list[PanelMark] = []
    rows: list[dict[str, Any]] = []
    for leaf in order:
        name = leaf_name[leaf]
        key = f"doc:{leaf}"
        rows.append({NODE: key, "Kind": "document", DOC: name, "Height": 0.0})
        marks.append(
            PanelMark(
                key=key,
                label=labels[name],
                x=0.0,
                y=y[leaf],
                # Unsized: sized marks scale to the largest, which drew every
                # leaf at full size and merged the column into a bar.
                size=None,
                group="Document",
                labelled=True,
                evidence=Evidence(scope="rows", filters=((NODE, key),), count=1, describe=labels[name]),
            )
        )
    edges: list[PanelEdge] = []
    for node in sorted(children):
        key = f"merge:{node - n + 1}"
        left, right = children[node]
        members = _leaves(node, children, n)
        rows.append({NODE: key, "Kind": "merge", DOC: "", "Height": height[node]})
        marks.append(
            PanelMark(
                key=key,
                label="",
                x=height[node] / tallest,
                y=y[node],
                size=None,
                group="Merge",
                evidence=Evidence(
                    scope="rows",
                    filters=((NODE, key),),
                    count=len(members),
                    describe=f"Merge {node - n + 1}: {len(members)} speech(es) joined at distance {height[node]:.1f}",
                ),
            )
        )
        for child in (left, right):
            child_key = f"doc:{child}" if child < n else f"merge:{child - n + 1}"
            edges.append(
                PanelEdge(
                    key=f"{key}->{child_key}",
                    source=key,
                    target=child_key,
                    weight=1.0,
                    evidence=Evidence(
                        scope="rows",
                        filters=((NODE, key),),
                        count=1,
                        describe=f"Merge {node - n + 1} at distance {height[node]:.1f}",
                    ),
                )
            )

    prepared = PreparedPanel(
        panel=SHAPE_HC_DENDROGRAM.name,
        shape="network",
        title="Story shapes: which speeches cluster together",
        subtitle=f"{n} speeches; distance of each merge along the bottom (farther right = joined later)",
        marks=tuple(marks),
        x_label="Merge distance, scaled to the last merge",
        y_label="",
        provenance=provenance,
        data=pd.DataFrame(rows),
        groups=("Document", "Merge"),
        edges=tuple(edges),
        edge_style="elbow",
        height=max(500, 12 * n + 80),
        notes=SHAPE_HC_NOTES,
    )
    return Result.success(prepared)


def _leaves(node: int, children: dict[int, tuple[int, int]], n: int) -> list[int]:
    if node < n:
        return [node]
    left, right = children[node]
    return _leaves(left, children, n) + _leaves(right, children, n)


SHAPE_HC_NOTES = (
    "Read from left to right: speeches joined near the left are the most alike in story shape; the last "
    "merges, far right, split the corpus into its broadest groups.",
    "Story shape here is how length, noun ratio and verb ratio change through a speech (32 slices), not what "
    "the speech is about. Two speeches can share a shape and share no subject.",
    "The order of leaves down the page is the tree's own; it is not a ranking.",
)

SHAPE_HC_DENDROGRAM = PanelDefinition(
    name="shape_hc_dendrogram",
    title="Story-shape dendrogram",
    question="Which speeches are built the same way, and how early do they join?",
    tool="shape_hc",
    shape="network",
    summary="The clustering's merge tree, documents named, merges at their distance.",
    requires=("Merge", "Left", "Right", "Height", "Left id", "Right id"),
    params=(),
    build=_dendrogram,
    notes=SHAPE_HC_NOTES,
)


def _membership_years(  # noqa: PLR0913 - two model tools share this figure; each keyword is one real difference
    frame: pd.DataFrame,
    provenance: Provenance,
    *,
    definition: PanelDefinition,
    group_column: str,
    group_name: str,
    title: str,
    notes: tuple[str, ...],
) -> Result[PreparedPanel]:
    """One row per cluster or topic; every member document a point at its year.

    The question is whether a grouping the model found is an era. A box
    that covers ninety years says the grouping is about form; a narrow
    one says it is about time.
    """
    working = frame.copy()
    working["_year"] = working[DATE].map(decimal_year) if DATE in working.columns else None
    dated = working[working["_year"].notna()].copy()
    if dated.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no dated documents to place in time"))
    groups = sorted(dated[group_column].astype(str).unique(), key=lambda value: (len(value), value))
    categories = tuple(f"{group_name} {value}" for value in groups)
    row_of = {value: index for index, value in enumerate(groups)}
    labels = _label_map(dated[DOC])
    marks = tuple(
        PanelMark(
            key=f"{row[DOC_ID]}",
            label=labels[str(row[DOC])],
            x=float(row["_year"]),
            y=float(row_of[str(row[group_column])]),
            group="",
            evidence=Evidence(
                scope="rows",
                filters=((DOC_ID, str(row[DOC_ID])),),
                count=1,
                describe=f"{labels[str(row[DOC])]}: {group_name.lower()} {row[group_column]}",
            ),
        )
        for _, row in dated.iterrows()
    )
    prepared = PreparedPanel(
        panel=definition.name,
        shape="distribution",
        title=title,
        subtitle=f"{len(dated)} dated document(s) across {len(groups)} {group_name.lower()}(s)",
        marks=marks,
        x_label="Year of the speech",
        y_label=group_name,
        provenance=provenance,
        data=dated.drop(columns=["_year"]).reset_index(drop=True),
        y_categories=categories,
        notes=notes,
    )
    return Result.success(prepared)


def _hc_membership(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    del params
    return _membership_years(
        frame,
        provenance,
        definition=SHAPE_HC_ERAS,
        group_column="Cluster",
        group_name="Cluster",
        title="When each story-shape cluster's speeches are from",
        notes=SHAPE_HC_ERAS.notes,
    )


SHAPE_HC_ERAS = PanelDefinition(
    name="shape_hc_eras",
    title="Clusters through time",
    question="Is each cluster an era, or a form that recurs across the decades?",
    tool="shape_hc",
    shape="distribution",
    summary="Every speech placed at its date, one row per cluster.",
    requires=(DOC_ID, DOC, DATE, "Cluster"),
    params=(),
    build=_hc_membership,
    notes=(
        "Each point is a speech, at its date; the box spans the middle half of its cluster's dates.",
        "A narrow box means the cluster is an era; a box across the whole corpus means the shape is a "
        "form that recurs, whoever gave the speech.",
        "Undated speeches cannot be placed and are left out.",
    ),
)


# ------------------------------------------------------- svd and nmf --


def _explained(
    frame: pd.DataFrame, provenance: Provenance, *, definition: PanelDefinition, column: str, meaning: str
) -> Result[PreparedPanel]:
    working = frame.copy()
    working[column] = pd.to_numeric(working[column], errors="coerce")
    marks = tuple(
        PanelMark(
            key=f"component:{int(row['Component'])}",
            label=f"Component {int(row['Component'])}",
            x=float(row[column]),
            y=float(rank),
            evidence=Evidence(
                scope="rows",
                filters=(("Component", str(row["Component"])),),
                count=1,
                describe=f"Component {int(row['Component'])}: {float(row[column]):.1%} of the {meaning}",
            ),
        )
        for rank, (_, row) in enumerate(working.iterrows())
    )
    prepared = PreparedPanel(
        panel=definition.name,
        shape="ranked_bars",
        title=definition.title,
        subtitle=f"{len(marks)} component(s)",
        marks=marks,
        x_label=f"Share of the {meaning}",
        y_label="Component",
        provenance=provenance,
        data=working,
        notes=definition.notes,
    )
    return Result.success(prepared)


def _loadings(frame: pd.DataFrame, provenance: Provenance, *, definition: PanelDefinition) -> Result[PreparedPanel]:
    """Each measure its own small multiple, loadings along the speech."""
    parsed = frame["Feature"].astype(str).str.extract(_FEATURE)
    if parsed["measure"].isna().all():
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", "no loading features name a measure and a position ('Tokens 05')")
        )
    working = frame.assign(_measure=parsed["measure"], _position=pd.to_numeric(parsed["position"], errors="coerce"))
    working = working.dropna(subset=["_measure", "_position"])
    components = [column for column in frame.columns if column.startswith("Component ")]
    slices = float(working["_position"].max())
    facets = tuple(dict.fromkeys(working["_measure"]))
    marks: list[PanelMark] = []
    for _, row in working.iterrows():
        for component in components:
            value = float(row[component])
            marks.append(
                PanelMark(
                    key=f"{row['Feature']}:{component}",
                    label=str(row["Feature"]),
                    x=float(row["_position"]) / slices,
                    y=value,
                    group=component,
                    facet=str(row["_measure"]),
                    evidence=Evidence(
                        scope="rows",
                        filters=(("Feature", str(row["Feature"])),),
                        count=1,
                        describe=(
                            f"{component}: loading {value:.3f} on {row['_measure']} at slice "
                            f"{int(row['_position'])} of {int(slices)}"
                        ),
                    ),
                )
            )
    prepared = PreparedPanel(
        panel=definition.name,
        shape="small_multiples",
        title=definition.title,
        subtitle=f"{len(components)} component(s) across {len(facets)} measure(s), {int(slices)} slices each",
        marks=tuple(marks),
        x_label="Position through the speech (0 = opening, 1 = close)",
        y_label="Loading",
        provenance=provenance,
        data=working.drop(columns=["_measure", "_position"]).reset_index(drop=True),
        groups=tuple(components),
        facets=facets,
        line_gap=1.5 / slices,
        notes=definition.notes,
    )
    return Result.success(prepared)


def _scores(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, definition: PanelDefinition
) -> Result[PreparedPanel]:
    components = [column for column in frame.columns if column.startswith("Component ")]
    if len(components) < 2:
        return Result.failure(
            Diagnostic.error(
                "PANEL_NO_DATA", "a map of documents needs two components; re-run with n-components 2 or more"
            )
        )
    x_col, y_col = components[0], components[1]
    group_by = str(params.get("group-by", "decade"))
    labels = _label_map(frame[DOC])
    working = frame.copy()
    working["_group"] = [
        group_of(str(d), when, group_by)
        for d, when in zip(
            working[DOC], working[DATE] if DATE in working.columns else [None] * len(working), strict=True
        )
    ]
    groups = tuple(sorted(working["_group"].unique()))
    extreme = set(working.assign(_r=(working[x_col] ** 2 + working[y_col] ** 2)).nlargest(8, "_r")[DOC_ID].astype(str))
    marks = tuple(
        PanelMark(
            key=str(row[DOC_ID]),
            label=labels[str(row[DOC])],
            x=float(row[x_col]),
            y=float(row[y_col]),
            group=str(row["_group"]),
            labelled=str(row[DOC_ID]) in extreme,
            evidence=Evidence(
                scope="rows",
                filters=((DOC_ID, str(row[DOC_ID])),),
                count=1,
                describe=f"{labels[str(row[DOC])]}: {x_col} {row[x_col]:.2f}, {y_col} {row[y_col]:.2f}",
            ),
        )
        for _, row in working.iterrows()
    )
    prepared = PreparedPanel(
        panel=definition.name,
        shape="scatter_labelled",
        title=definition.title,
        subtitle=f"{len(marks)} document(s); the eight farthest from the origin are named",
        marks=marks,
        x_label=x_col,
        y_label=y_col,
        provenance=provenance,
        data=working.drop(columns=["_group"]).reset_index(drop=True),
        groups=groups,
        notes=definition.notes,
    )
    return Result.success(prepared)


def _composition(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    """NMF parts are additive, so each speech is a mix: a band of shares."""
    del params
    components = [column for column in frame.columns if column.startswith("Component ")]
    working = frame.copy()
    working["_year"] = working[DATE].map(decimal_year) if DATE in working.columns else None
    working = working.sort_values(["_year", DOC_ID], na_position="last", kind="stable").reset_index(drop=True)
    labels = _label_map(working[DOC])
    marks: list[PanelMark] = []
    for row_index, (_, row) in enumerate(working.iterrows()):
        values = [max(0.0, float(row[component])) for component in components]
        total = sum(values)
        if total <= 0:
            continue
        start = 0.0
        for component, value in zip(components, values, strict=True):
            share = value / total
            marks.append(
                PanelMark(
                    key=f"{row[DOC_ID]}:{component}",
                    label=labels[str(row[DOC])],
                    x=start,
                    y=float(row_index),
                    size=share,
                    group=component,
                    evidence=Evidence(
                        scope="rows",
                        filters=((DOC_ID, str(row[DOC_ID])),),
                        count=1,
                        describe=f"{labels[str(row[DOC])]}: {share:.0%} {component}",
                    ),
                )
            )
            start += share
    prepared = PreparedPanel(
        panel=SHAPE_NMF_COMPOSITION.name,
        shape="ribbon",
        title=SHAPE_NMF_COMPOSITION.title,
        subtitle=f"{len(working)} speech(es), in date order",
        marks=tuple(marks),
        x_label="Share of the speech's shape",
        y_label="Speech",
        provenance=provenance,
        data=working.drop(columns=["_year"]),
        groups=tuple(components),
        height=max(500, 12 * len(working) + 80),
        notes=SHAPE_NMF_COMPOSITION.notes,
    )
    return Result.success(prepared)


_GROUP_BY = PanelParam(
    name="group-by",
    type="choice",
    default="decade",
    choices=("decade", "speaker", "year", "none"),
    label="Colour documents by",
    help="Colour each document by its decade or by its speaker (read from the file name).",
)
_LOADING_NOTE = (
    "Each small multiple is one measure through the speech, opening at the left and close at the right; a line "
    "far from zero is the part of the speech that component weights.",
    "The shape matrix is not standardised: token counts run in the tens and the noun and verb ratios below 1, "
    "so the components are driven mostly by sentence length. On the real 87-speech run component 1's largest "
    "token loading was about 100 times its largest ratio loading. Each small multiple has its own scale so "
    "the ratio panels stay readable, but compare shapes within a panel, not heights across panels.",
)


def _factory(method: str, *, explained_column: str, meaning: str, sign_note: str) -> tuple[PanelDefinition, ...]:
    tool = f"shape_{method}"
    upper = method.upper()
    explained = PanelDefinition(
        name=f"{tool}_explained",
        title=f"{upper}: how much each component explains",
        question="How many components does it take to describe these story shapes?",
        tool=tool,
        shape="ranked_bars",
        summary=f"Share of the {meaning} each component accounts for.",
        requires=("Component", explained_column),
        params=(),
        build=lambda f, p, prov: _explained(f, prov, definition=explained, column=explained_column, meaning=meaning),
        notes=(f"A component that explains little is not a pattern worth naming. {sign_note}",),
    )
    loadings = PanelDefinition(
        name=f"{tool}_loadings",
        title=f"{upper}: what each component weights, through the speech",
        question="Which part of a speech, and which measure, does each component pick out?",
        tool=tool,
        shape="small_multiples",
        summary="Loadings along the speech, one small multiple per measure.",
        requires=("Feature", "Component 1"),
        params=(),
        build=lambda f, p, prov: _loadings(f, prov, definition=loadings),
        notes=(*_LOADING_NOTE, sign_note),
    )
    scores = PanelDefinition(
        name=f"{tool}_documents",
        title=f"{upper}: speeches on the first two components",
        question="Which speeches sit at the extremes of each component?",
        tool=tool,
        shape="scatter_labelled",
        summary="Documents placed by their component scores, coloured by decade or speaker.",
        requires=(DOC_ID, DOC, "Component 1", "Component 2"),
        params=(_GROUP_BY,),
        build=lambda f, p, prov: _scores(f, p, prov, definition=scores),
        notes=(
            "Distance from the origin is how strongly a speech shows the components; the named points are the "
            f"extremes worth reading. {sign_note}",
        ),
    )
    return explained, loadings, scores


SHAPE_SVD_PANELS = _factory(
    "svd",
    explained_column="Explained variance",
    meaning="variance",
    sign_note="An SVD component's sign is arbitrary: a negative loading is as strong as a positive one.",
)
SHAPE_NMF_PANELS = _factory(
    "nmf",
    explained_column="Share of mass",
    meaning="shape's mass",
    sign_note="NMF parts are never negative, so a speech is a sum of them.",
)
SHAPE_NMF_COMPOSITION = PanelDefinition(
    name="shape_nmf_composition",
    title="NMF: each speech as a mix of parts",
    question="Which part dominates each speech, and does that change over time?",
    tool="shape_nmf",
    shape="ribbon",
    summary="Each speech's NMF scores as shares, one band per speech in date order.",
    requires=(DOC_ID, DOC, "Component 1", "Component 2"),
    params=(),
    build=_composition,
    notes=(
        "A band's colours are the speech's mix of parts, normalised to 1; the total size of a speech's shape "
        "is not shown here.",
    ),
)


# ---------------------------------------------------------- bert_topics --


def _topic_words(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    working = frame.copy()
    working["Weight"] = pd.to_numeric(working["Weight"], errors="coerce")
    topics = sorted(working["Topic"].astype(str).unique(), key=lambda value: (len(value), value))
    wanted = str(params.get("topic", "") or topics[0])
    if wanted not in topics:
        return Result.failure(
            Diagnostic.error("PANEL_TOPIC_NOT_FOUND", f"no topic {wanted!r}; this run has topics {', '.join(topics)}")
        )
    rows = working[working["Topic"].astype(str) == wanted].sort_values("Weight", ascending=False).reset_index(drop=True)
    marks = tuple(
        PanelMark(
            key=f"{wanted}:{row['Word']}",
            label=str(row["Word"]),
            x=float(row["Weight"]),
            y=float(rank),
            evidence=Evidence(
                scope="terms",
                filters=(("Topic", wanted), ("Word", str(row["Word"]))),
                count=1,
                describe=f"Topic {wanted}: {row['Word']} (weight {row['Weight']:.4f})",
                phrase=str(row["Word"]),
            ),
        )
        for rank, (_, row) in enumerate(rows.iterrows())
    )
    prepared = PreparedPanel(
        panel=BERT_TOPIC_WORDS.name,
        shape="ranked_bars",
        title=f"Topic {wanted}: its words",
        subtitle=f"{len(marks)} word(s), by c-TF-IDF weight; {len(topics)} topic(s) in this run",
        marks=marks,
        x_label="Weight in the topic",
        y_label="Word",
        provenance=provenance,
        data=rows,
        notes=BERT_TOPIC_WORDS.notes,
    )
    return Result.success(prepared)


BERT_TOPIC_WORDS = PanelDefinition(
    name="bert_topics_words",
    title="Topic words",
    question="Which words define each topic?",
    tool="bert_topics",
    shape="ranked_bars",
    summary="One topic's words, ranked by their weight in it.",
    requires=("Topic", "Word", "Weight"),
    params=(
        PanelParam(
            name="topic",
            type="str",
            default="",
            label="Topic",
            help="Which topic to show; empty shows the first.",
        ),
    ),
    build=_topic_words,
    notes=(
        "A word high in several topics ('year', 'world') is a sign the topics have not separated, not that the "
        "word matters.",
    ),
)


def _topic_eras(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    del params
    return _membership_years(
        frame,
        provenance,
        definition=BERT_TOPIC_ERAS,
        group_column="Topic",
        group_name="Topic",
        title="When each topic's documents are from",
        notes=BERT_TOPIC_ERAS.notes,
    )


BERT_TOPIC_ERAS = PanelDefinition(
    name="bert_topics_eras",
    title="Topics through time",
    question="Is each topic tied to an era?",
    tool="bert_topics",
    shape="distribution",
    summary="Every document at its date, one row per assigned topic.",
    requires=(DOC_ID, DOC, DATE, "Topic", "Distance"),
    params=(),
    build=_topic_eras,
    notes=(
        "Each point is a document at its date, in the row of the one topic it was assigned; a document that "
        "mixes topics is shown under its closest one only.",
        "A topic whose documents all sit in one decade is a subject of its time; one spread across the corpus "
        "is a standing subject.",
    ),
)


# --------------------------------------------------------- lda_stability --


def _stability_matches(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    del params
    working = frame.copy()
    working["Jaccard"] = pd.to_numeric(working["Jaccard"], errors="coerce")
    topics = sorted(working["Topic"].astype(str).unique(), key=lambda value: (len(value), value))
    seeds = sorted(working["Seed"].astype(str).unique(), key=lambda value: (len(value), value))
    marks = tuple(
        PanelMark(
            key=f"{row['Topic']}:{row['Seed']}",
            label=f"Topic {row['Topic']} / seed {row['Seed']}",
            x=float(seeds.index(str(row["Seed"]))),
            y=float(topics.index(str(row["Topic"]))),
            value=float(row["Jaccard"]),
            evidence=Evidence(
                scope="rows",
                filters=(("Topic", str(row["Topic"])), ("Seed", str(row["Seed"]))),
                count=1,
                describe=(
                    f"Topic {row['Topic']} under seed {row['Seed']}: matched topic {row['Matched topic']}, "
                    f"Jaccard {row['Jaccard']:.2f}; shared: {row['Shared words']}"
                ),
            ),
        )
        for _, row in working.iterrows()
    )
    prepared = PreparedPanel(
        panel=LDA_STABILITY_MATCHES.name,
        shape="heatmap",
        title="Topic matches across seeds",
        subtitle=f"{len(topics)} reference topic(s) x {len(seeds)} other seed(s); darker = the topic came back",
        marks=marks,
        x_label="Seed",
        y_label="Reference topic",
        provenance=provenance,
        data=working,
        x_categories=tuple(f"seed {seed}" for seed in seeds),
        y_categories=tuple(f"Topic {topic}" for topic in topics),
        color_scale="sequential",
        value_label="Topic-word similarity of the best match",
        notes=LDA_STABILITY_MATCHES.notes,
    )
    return Result.success(prepared)


LDA_STABILITY_MATCHES = PanelDefinition(
    name="lda_stability_matches",
    title="Topic matches across seeds",
    question="Under which seeds does each topic come back, and how closely?",
    tool="lda_stability",
    shape="heatmap",
    summary="Reference topics against seeds, coloured by the Jaccard overlap of their top words.",
    requires=("Topic", "Seed", "Matched topic", "Jaccard", "Shared words"),
    params=(),
    build=_stability_matches,
    notes=(
        "Each cell is the best match for a reference topic among one other seed's topics (Hungarian matching on "
        "top-word overlap). A pale row is a topic the model does not reliably find.",
        "Stability is about the model, not the corpus: a topic that comes back under every seed is one this "
        "corpus supports; one that does not is an artefact of one run.",
    ),
)


MODELS_PANELS: tuple[PanelDefinition, ...] = (
    SHAPE_HC_DENDROGRAM,
    SHAPE_HC_ERAS,
    *SHAPE_SVD_PANELS,
    *SHAPE_NMF_PANELS,
    SHAPE_NMF_COMPOSITION,
    BERT_TOPIC_WORDS,
    BERT_TOPIC_ERAS,
    LDA_STABILITY_MATCHES,
)
