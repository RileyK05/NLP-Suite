"""Figures for doc_embeddings: documents by meaning.

The similarity heatmap and neighbour network are Document similarity's own
figures read from a table in the same shape (``doc_pairs.csv``), retitled
and re-noted for what the numbers are here: the cosine of two sentence-model
vectors, not shared vocabulary. The map and the search results are drawn
from this tool's own tables.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import document_labels
from core.viz.panels_pairs import (
    BAND,
    DOC_A,
    DOC_B,
    DOC_SIMILARITY_NEIGHBOURS,
    SIMILARITY,
    doc_similarity_heatmap,
    doc_similarity_neighbours,
)
from core.viz.panelspec import Evidence, PanelDefinition, PanelMark, PanelParam, PreparedPanel, Provenance

__all__ = ["MEANING_PANELS"]

_HEATMAP_NOTES: tuple[str, ...] = (
    "Similarity is relative to this corpus: the cosine of the two documents' sentence-model vectors after removing "
    "the average document, as a percentage. 0 is as alike as a typical pair here; positive is more alike, negative "
    "less. (Uncentred, a sentence model scores nearly every pair of speeches of one genre above 95%; that raw "
    "cosine is in the table's Cosine column.)",
    "Compare with Document similarity (TF-IDF) on the same documents: a pair close here but far there shares "
    "meaning without sharing words.",
    "The diagonal is left blank: comparing a document to itself is not a measured similarity.",
)
_NEIGHBOUR_NOTES: tuple[str, ...] = (
    "An edge joins a document to each of its k nearest by meaning (relative similarity of their sentence-model "
    "vectors, the corpus average removed); "
    "a document chosen by many others collects more than k edges.",
    "Position on the page comes from a force layout over the drawn edges; distance there is not measured.",
)
_MAP_NOTES: tuple[str, ...] = (
    "Each point is a document (or sentence) placed by t-SNE from its sentence-model vector; with fewer than "
    "eight points the map is the first two principal components instead.",
    "Neighbourhoods are meaningful, distances across the map are not: t-SNE keeps near things near and says "
    "little about how far apart two groups are.",
    "Colour is the k-means cluster, with k chosen by silhouette in the original vector space, not on the map.",
)
_SEARCH_NOTES: tuple[str, ...] = (
    "Sentences are ranked by the cosine between their vector and the query's; the best match may share no word "
    "with the query, because the comparison is by meaning.",
    "Qwen3 Embedding reads the query with its search instruction; Granite reads query and sentences alike.",
)

X, Y, CLUSTER, DOCUMENT = "X", "Y", "Cluster", "Document"
SENTENCE, COSINE, RANK = "Sentence", "Cosine", "Rank"
_LABEL_MAX = 60


def _heatmap(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    built = doc_similarity_heatmap(frame, params, provenance)
    if built.value is None:
        return built
    prepared = replace(
        built.value,
        panel=DOC_EMBEDDINGS_HEATMAP.name,
        title="Documents by meaning",
        value_label="Similarity, relative to the corpus (%)",
        color_scale="diverging",
        notes=_HEATMAP_NOTES,
    )
    return Result.success(prepared, *built.diagnostics)


def _neighbours(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    built = doc_similarity_neighbours(frame, params, provenance)
    if built.value is None:
        return built
    prepared = replace(
        built.value,
        panel=DOC_EMBEDDINGS_NEIGHBOURS.name,
        title="Nearest documents by meaning",
        notes=_NEIGHBOUR_NOTES,
    )
    return Result.success(prepared, *built.diagnostics)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _map(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    for column in (DOCUMENT, X, Y, CLUSTER):
        if column not in frame.columns:
            return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"the map table has no {column!r}"))
    working = frame[frame[X].map(_finite) & frame[Y].map(_finite)].copy()
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no mapped point has usable coordinates"))
    working[X] = working[X].astype(float)
    working[Y] = working[Y].astype(float)
    label_count = int(params["label-count"])
    # Label the points farthest from their cluster's centre first: the
    # outliers and the edges of each neighbourhood are what a reader asks about.
    centres = working.groupby(CLUSTER)[[X, Y]].transform("mean")
    spread = ((working[X] - centres[X]) ** 2 + (working[Y] - centres[Y]) ** 2).rank(ascending=False, method="first")
    labelled = set(spread[spread <= label_count].index) if len(working) > label_count else set(working.index)
    groups = tuple(sorted(working[CLUSTER].astype(str).unique(), key=lambda g: (len(g), g)))
    marks = tuple(
        PanelMark(
            key=f"point-{index}",
            label=str(row[DOCUMENT]),
            x=float(row[X]),
            y=float(row[Y]),
            group=str(row[CLUSTER]),
            labelled=index in labelled,
            evidence=Evidence(
                scope="rows",
                filters=((DOCUMENT, str(row[DOCUMENT])),),
                count=1,
                describe=f"{row[DOCUMENT]}: {row[CLUSTER]}",
            ),
        )
        for index, row in working.iterrows()
    )
    prepared = PreparedPanel(
        panel=DOC_EMBEDDINGS_MAP.name,
        shape="scatter_labelled",
        title="Map of the corpus by meaning",
        subtitle=f"{len(marks)} point(s) in {len(groups)} cluster(s) · {len(labelled)} labelled",
        marks=marks,
        x_label="Map dimension 1",
        y_label="Map dimension 2",
        provenance=provenance,
        data=working.reset_index(drop=True),
        groups=groups,
        notes=_MAP_NOTES,
    )
    return Result.success(prepared)


def _search(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    for column in (DOCUMENT, SENTENCE, COSINE):
        if column not in frame.columns:
            return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"the search table has no {column!r}"))
    working = frame[frame[COSINE].map(_finite)].copy()
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no search result has a usable score"))
    working[COSINE] = working[COSINE].astype(float)
    drawn = working.sort_values(COSINE, ascending=False, kind="stable").head(int(params["top-n"]))
    names = document_labels(drawn[DOCUMENT].astype(str))
    marks = []
    for rank, (_, row) in enumerate(drawn.iterrows()):
        text = " ".join(str(row[SENTENCE]).split())
        document = str(row[DOCUMENT])
        name = names.get(document, document)
        # The label fits the figure contract (60 characters): the snippet gets
        # what the speaker's name leaves; the whole sentence is on hover.
        room = max(16, _LABEL_MAX - len(name) - 5)  # " — “" and "”"
        snippet = text if len(text) <= room else text[: room - 1].rstrip() + "…"
        marks.append(
            PanelMark(
                key=f"hit-{rank}",
                label=f"{name} — “{snippet}”",
                x=float(row[COSINE]),
                y=float(rank),
                evidence=Evidence(
                    scope="rows",
                    filters=((DOCUMENT, document), (SENTENCE, str(row[SENTENCE]))),
                    count=1,
                    describe=f"{document}: {text} (cosine {float(row[COSINE]):.3f})",
                ),
            )
        )
    prepared = PreparedPanel(
        panel=DOC_EMBEDDINGS_SEARCH.name,
        shape="ranked_bars",
        title="Sentences closest in meaning to the query",
        subtitle=f"top {len(marks)} of {len(working)} · ranked by cosine similarity",
        marks=tuple(marks),
        x_label="Cosine similarity to the query",
        y_label="Sentence",
        provenance=provenance,
        data=drawn.reset_index(drop=True),
        notes=_SEARCH_NOTES,
    )
    return Result.success(prepared)


DOC_EMBEDDINGS_HEATMAP = PanelDefinition(
    name="doc_embeddings_heatmap",
    title="Documents by meaning (heatmap)",
    question="Which documents mean most alike, whatever words they use?",
    tool="doc_embeddings",
    shape="heatmap",
    summary="Document x document cosine similarity of sentence-model vectors, ordered by date.",
    requires=(DOC_A, DOC_B, SIMILARITY, BAND),
    params=(),
    build=_heatmap,
    notes=_HEATMAP_NOTES,
)

DOC_EMBEDDINGS_NEIGHBOURS = PanelDefinition(
    name="doc_embeddings_neighbours",
    title="Nearest documents by meaning",
    question="Which documents sit together by meaning, and do eras or speakers stand out?",
    tool="doc_embeddings",
    shape="network",
    summary="Each document linked to its nearest by meaning, grouped by decade or speaker.",
    requires=(DOC_A, DOC_B, SIMILARITY),
    params=DOC_SIMILARITY_NEIGHBOURS.params,
    build=_neighbours,
    notes=_NEIGHBOUR_NOTES,
)

DOC_EMBEDDINGS_MAP = PanelDefinition(
    name="doc_embeddings_map",
    title="Map of the corpus by meaning",
    question="Which documents form neighbourhoods of meaning, and which stand apart?",
    tool="doc_embeddings",
    shape="scatter_labelled",
    summary="Documents placed by their sentence-model vectors, coloured by k-means cluster.",
    requires=(DOCUMENT, X, Y, CLUSTER),
    params=(
        PanelParam(
            name="label-count",
            type="int",
            default=20,
            minimum=0,
            maximum=80,
            label="Labels",
            help="How many points get their name written beside them (the farthest from their cluster's centre first).",
        ),
    ),
    build=_map,
    notes=_MAP_NOTES,
)

DOC_EMBEDDINGS_SEARCH = PanelDefinition(
    name="doc_embeddings_search",
    title="Semantic search results",
    question="Which sentences in the corpus are closest in meaning to my query?",
    tool="doc_embeddings",
    shape="ranked_bars",
    summary="The sentences nearest the query in meaning, best first.",
    requires=(DOCUMENT, SENTENCE, COSINE, RANK),
    params=(
        PanelParam(
            name="top-n",
            type="int",
            default=15,
            minimum=1,
            maximum=40,
            label="Results shown",
            help="How many of the closest sentences to draw.",
        ),
    ),
    build=_search,
    notes=_SEARCH_NOTES,
)

MEANING_PANELS: tuple[PanelDefinition, ...] = (
    DOC_EMBEDDINGS_HEATMAP,
    DOC_EMBEDDINGS_NEIGHBOURS,
    DOC_EMBEDDINGS_MAP,
    DOC_EMBEDDINGS_SEARCH,
)
