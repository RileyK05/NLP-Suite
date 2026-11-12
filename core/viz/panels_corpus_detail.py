"""Questions specific to topic fits, entity geography, and concordances."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import document_labels
from core.viz.panelspec import Evidence, PanelDefinition, PanelMark, PanelParam, PreparedPanel, Provenance

_DOC = "Document"
_DOC_ID = "Document ID"
_TOP_N = PanelParam(
    name="top_n",
    type="int",
    default=15,
    minimum=3,
    maximum=40,
    label="Rows shown",
    help="Show the most frequent rows first.",
)
_ENTITIES = PanelParam(
    name="locations",
    type="int",
    default=14,
    minimum=3,
    maximum=30,
    label="Places shown",
    help="Keep the places with the most mentions across the corpus.",
)
_DOCUMENTS = PanelParam(
    name="documents",
    type="int",
    default=14,
    minimum=3,
    maximum=40,
    label="Documents shown",
    help="Keep documents with the most mentions of these places.",
)
_LOCATION_NOTE = "Counts come from the location entities found by the parser. Empty cells have no recorded mentions."


def _places_in_documents(frame: pd.DataFrame, params: Mapping[str, Any], source: Provenance) -> Result[PreparedPanel]:
    data = frame.dropna(subset=[_DOC_ID, _DOC, "Location"]).copy()
    data[_DOC_ID] = data[_DOC_ID].astype(str)
    data["Count"] = pd.to_numeric(data["Count"], errors="coerce")
    data = data[data["Count"].between(0, float("inf"), inclusive="neither")]
    if data.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_LOCATIONS", "No documents contain countable locations."))
    places = tuple(data.groupby("Location")["Count"].sum().nlargest(params["locations"]).index)
    data = data[data["Location"].isin(places)]
    docs = tuple(data.groupby(_DOC_ID)["Count"].sum().nlargest(params["documents"]).index)
    data = data[data[_DOC_ID].isin(docs)]
    doc_names = data.drop_duplicates(_DOC_ID).set_index(_DOC_ID)[_DOC].astype(str)
    labels = document_labels(doc_names)
    short = pd.Series({doc: labels[doc_names[doc]] for doc in docs})
    repeated = short.duplicated(keep=False)
    unique = tuple(f"{short[doc]} ({doc})" if repeated[doc] else short[doc] for doc in docs)
    grouped = data.groupby([_DOC_ID, "Location"])["Count"].agg(["sum", "size"])
    marks = []
    for y, doc in enumerate(docs):
        for x, place in enumerate(places):
            if (doc, place) not in grouped.index:
                continue
            count, rows = grouped.loc[(doc, place)]
            marks.append(
                PanelMark(
                    key=f"{y}:{x}",
                    label=f"{unique[y]} · {place}",
                    x=float(x),
                    y=float(y),
                    value=float(count),
                    evidence=Evidence(
                        "rows",
                        ((_DOC_ID, doc), ("Location", str(place))),
                        int(rows),
                        f"{unique[y]} · {place} · {count:,g} mentions",
                    ),
                )
            )
    return Result.success(
        PreparedPanel(
            panel="ner_places_by_document",
            title="Places by document",
            shape="heatmap",
            marks=tuple(marks),
            x_label="Recognized place",
            y_label="Document",
            x_categories=places,
            y_categories=unique,
            color_scale="sequential",
            value_label="Mentions",
            data=data,
            provenance=source,
            subtitle=f"{len(docs)} documents · {len(places)} places · {data['Count'].sum():,g} mentions shown",
            notes=(_LOCATION_NOTE,),
        )
    )


def _topic_fit(tool: str, title: str, columns: tuple[str, str], axis: str, note: str) -> PanelDefinition:
    group, value = columns
    name = f"{tool}_topic_fit"
    requires: tuple[str, ...] = (_DOC_ID, _DOC, group, value)
    if tool == "lda_gensim":
        requires += ("Topic keywords",)

    def build(frame: pd.DataFrame, _params: Mapping[str, Any], source: Provenance) -> Result[PreparedPanel]:
        data = frame.dropna(subset=[_DOC_ID, _DOC]).astype({_DOC_ID: str, _DOC: str})
        data[group] = pd.to_numeric(data[group], errors="coerce")
        data[value] = pd.to_numeric(data[value], errors="coerce")
        valid_topic = data[group].between(0, float("inf"), inclusive="left") & data[group].mod(1).eq(0)
        valid_value = (
            data[value].between(0, 1)
            if value == "Contribution"
            else data[value].between(
                0,
                float("inf"),
                inclusive="left",
            )
        )
        data = data[valid_topic & valid_value]
        if data.empty:
            return Result.failure(
                Diagnostic.error("PANEL_NO_TOPIC_FIT", "No documents have usable topic-fit measurements.")
            )
        topics = sorted(data[group].unique())
        labels = document_labels(data[_DOC].astype(str))
        marks = tuple(
            PanelMark(
                key=f"{y}:{i}",
                label=labels[doc],
                x=float(measure),
                y=float(y),
                evidence=Evidence(
                    "documents",
                    ((_DOC_ID, identifier),),
                    1,
                    f"{labels[doc]} · Topic {topic:g} · {axis} {measure:.3f}",
                ),
            )
            for y, topic in enumerate(topics)
            for i, (identifier, doc, measure) in enumerate(
                data.loc[data[group].eq(topic), [_DOC_ID, _DOC, value]].itertuples(index=False, name=None)
            )
        )
        return Result.success(
            PreparedPanel(
                panel=name,
                title=title,
                shape="distribution",
                marks=marks,
                x_label=axis,
                y_label="Topic",
                y_categories=tuple(f"Topic {topic:g}" for topic in topics),
                data=data,
                provenance=source,
                subtitle=f"{len(data):,} of {len(frame):,} documents · {len(topics)} topics",
                notes=(note,),
            )
        )

    return PanelDefinition(
        name=name,
        title=title,
        tool=tool,
        shape="distribution",
        requires=requires,
        params=(),
        build=build,
        notes=(note,),
        summary="One point per document, with the spread of fit values shown separately for each assigned topic.",
        question=(
            "Which topics dominate their assigned documents, and which documents have more mixed memberships?"
            if tool == "lda_gensim"
            else "Which clusters contain the most peripheral documents, and which documents are they?"
        ),
    )


_NEIGHBOR_NOTE = "Counts cover the immediately adjacent token on each side of each returned hit, with case folded."


def _kwic_neighbors(frame: pd.DataFrame, params: Mapping[str, Any], source: Provenance) -> Result[PreparedPanel]:
    sides = ("Before the hit", "After the hit")
    adjacent = []
    for column, side in zip(("Left Context", "Right Context"), sides, strict=True):
        data = frame.copy()
        tokens = data[column].fillna("").astype(str).str.split()
        data["Neighbor"] = tokens.str[-1 if column == "Left Context" else 0].str.casefold()
        data["Side"] = side
        adjacent.append(data[data["Neighbor"].notna()])
    data = pd.concat(adjacent, ignore_index=True)
    if data.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_CONTEXT", "The hits have no neighboring tokens."))
    neighbors = tuple(data["Neighbor"].value_counts().head(params["top_n"]).index)
    data = data[data["Neighbor"].isin(neighbors)]
    counts = data.groupby(["Neighbor", "Side"]).size()
    marks = tuple(
        PanelMark(
            key=f"{y}:{side}",
            label=word,
            x=float(count),
            y=float(y),
            group=side,
            evidence=Evidence(
                "rows",
                (("Neighbor", word), ("Side", side)),
                int(count),
                f"{word} · {count:,} times {side.lower()}",
            ),
        )
        for y, word in enumerate(neighbors)
        for side in sides
        if (count := counts.get((word, side), 0))
    )
    return Result.success(
        PreparedPanel(
            panel="kwic_neighbor_balance",
            title="Words beside the search hit",
            shape="ranked_bars",
            marks=marks,
            x_label="Occurrences",
            y_label="",
            data=data,
            provenance=source,
            groups=sides,
            notes=(_NEIGHBOR_NOTE,),
            subtitle=f"{len(neighbors)} neighbors · {len(data):,} adjacent occurrences from {len(frame):,} hits",
            height=max(400, 130 + 32 * len(neighbors)),
        )
    )


CORPUS_DETAIL_PANELS = (
    PanelDefinition(
        name="ner_places_by_document",
        title="Places by document",
        tool="ner",
        shape="heatmap",
        requires=("Location", _DOC, _DOC_ID, "Count"),
        params=(_ENTITIES, _DOCUMENTS),
        build=_places_in_documents,
        summary="Place mentions by document, with counts and source rows.",
        notes=(_LOCATION_NOTE,),
        question="Which documents mention the same places, and where are those mentions concentrated?",
    ),
    _topic_fit(
        "lda_gensim",
        "Gensim LDA · Dominant topic strength",
        ("Dominant topic", "Contribution"),
        "Largest topic share",
        "Each point shows the share of the fitted mixture assigned to a document's strongest topic.",
    ),
    _topic_fit(
        "topic_clusters_bert",
        "BERT topics · Distance to cluster center",
        ("Topic", "Distance"),
        "Euclidean distance",
        "Larger values indicate documents farther from their assigned cluster center in embedding space.",
    ),
    PanelDefinition(
        name="kwic_neighbor_balance",
        title="Words beside the search hit",
        tool="kwic",
        shape="ranked_bars",
        requires=("Hit", "Left Context", "Right Context", _DOC_ID, "Sentence ID"),
        params=(_TOP_N,),
        build=_kwic_neighbors,
        summary=_NEIGHBOR_NOTE,
        notes=(_NEIGHBOR_NOTE,),
        question="Which words appear most often immediately before and after the matched term?",
    ),
)
