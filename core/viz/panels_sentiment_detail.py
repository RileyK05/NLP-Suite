"""Sentence sentiment profiles and the coverage behind lexicon scores."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd

from core.analysis.sentiment_neural import sentiment_bucket
from core.result import Diagnostic, Result
from core.viz.panel_helpers import document_labels
from core.viz.panelspec import (
    Annotation,
    Evidence,
    PanelDefinition,
    PanelMark,
    PanelParam,
    PanelShape,
    PreparedPanel,
    Provenance,
)

DOC_ID = "Document ID"
DOC = "Document"
SENT_ID = "Sentence ID"
SCORE = "Compound"
BUCKET = "Polarity class"
SECTION = "Position in document"
CLASSES = ("Negative", "Neutral", "Positive")
_DOCS = PanelParam(
    name="documents",
    type="int",
    default=12,
    minimum=1,
    maximum=40,
    label="Documents shown",
    help="Documents with the most scored sentences come first.",
)
_FIND = PanelParam(
    name="document",
    type="str",
    default="",
    label="Focus on a document",
    help="Enter a full document name or ID. Leave blank to compare documents.",
)
_SECTIONS = PanelParam(
    name="sections",
    type="choice",
    default="10",
    choices=("5", "10", "20"),
    label="Sections per document",
    help="Split each document's sentence positions into equal intervals.",
)


def _document_labels(data: pd.DataFrame, ids: tuple[str, ...]) -> dict[str, str]:
    names = data.drop_duplicates(DOC_ID).set_index(DOC_ID)[DOC].astype(str)
    short = document_labels(names)
    labels = {key: short[names[key]] for key in ids}
    counts = pd.Series(list(labels.values())).value_counts()
    return {key: f"{label} ({key})" if counts[label] > 1 else label for key, label in labels.items()}


def _selected_documents(data: pd.DataFrame, params: Mapping[str, Any]) -> tuple[str, ...]:
    focus = params["document"].strip().casefold()
    if focus:
        data = data[data[DOC_ID].str.casefold().eq(focus) | data[DOC].astype(str).str.casefold().eq(focus)]
    return tuple(data[DOC_ID].value_counts().head(params["documents"]).index)


def _composition(
    data: pd.DataFrame,
    docs: tuple[str, ...],
    labels: dict[str, str],
) -> tuple[PanelMark, ...]:
    data[BUCKET] = data["Label"].astype(str).map(sentiment_bucket).str.capitalize()
    counts = data.groupby([DOC_ID, BUCKET]).size()
    totals = data[DOC_ID].value_counts()
    marks = []
    for y, doc in enumerate(docs):
        for x, bucket in enumerate(CLASSES):
            count = int(counts.get((doc, bucket), 0))
            total = int(totals[doc])
            share = 100 * count / total
            marks.append(
                PanelMark(
                    key=f"{y}:{x}",
                    label=f"{labels[doc]} · {bucket}",
                    x=float(x),
                    y=float(y),
                    value=share,
                    evidence=Evidence(
                        "rows",
                        ((DOC_ID, doc), (BUCKET, bucket)),
                        count,
                        f"{labels[doc]} · {bucket} · {share:.1f}% ({count:,} of {total:,} scored sentences)",
                    ),
                )
            )
    return tuple(marks)


def _progression(
    data: pd.DataFrame,
    docs: tuple[str, ...],
    labels: dict[str, str],
    sections: tuple[str, ...],
) -> tuple[PanelMark, ...]:
    positions = pd.to_numeric(data[SENT_ID])
    by_document = positions.groupby(data[DOC_ID])
    first = by_document.transform("min")
    span = by_document.transform("max") - first
    fraction = (positions - first).div(span.where(span > 0)).fillna(0.5)
    section_index = (fraction * len(sections)).astype(int).clip(upper=len(sections) - 1)
    data[SECTION] = section_index.map(dict(enumerate(sections)))
    grouped = data.groupby([DOC_ID, SECTION])[SCORE].agg(["mean", "size"])
    marks = []
    for y, doc in enumerate(docs):
        for x, section in enumerate(sections):
            if (doc, section) not in grouped.index:
                continue
            value, size = grouped.loc[(doc, section)]
            marks.append(
                PanelMark(
                    key=f"{y}:{x}",
                    label=f"{labels[doc]} · {section}",
                    x=float(x),
                    y=float(y),
                    value=float(value),
                    evidence=Evidence(
                        "rows",
                        ((DOC_ID, doc), (SECTION, section)),
                        int(size),
                        f"{labels[doc]} · {section} · mean {value:+.3f} across {int(size):,} sentences",
                    ),
                )
            )
    return tuple(marks)


def _distribution(data: pd.DataFrame, docs: tuple[str, ...], labels: dict[str, str]) -> tuple[PanelMark, ...]:
    marks = []
    for y, doc in enumerate(docs):
        rows = data[data[DOC_ID] == doc]
        for index, (sent, score, text) in enumerate(
            rows[[SENT_ID, SCORE, "Sentence"]].itertuples(index=False, name=None)
        ):
            marks.append(
                PanelMark(
                    key=f"{y}:{index}",
                    label=f"{labels[doc]} · sentence {sent}",
                    x=float(score),
                    y=float(y),
                    evidence=Evidence(
                        "sentences",
                        ((DOC_ID, doc), (SENT_ID, str(sent))),
                        1,
                        f"{labels[doc]} · sentence {sent} · {score:+.3f} · {str(text)[:120]}",
                    ),
                )
            )
    return tuple(marks)


def _sentence_panel(
    tool: str,
    model: str,
    view: Literal["mix", "progression", "distribution"],
) -> PanelDefinition:
    names = {
        "mix": ("Sentence sentiment mix", "What share of each document's scored sentences falls into each class?"),
        "progression": ("Sentiment through each document", "Where does tone shift between the beginning and end?"),
        "distribution": ("Sentence score spread", "Which documents contain a wide range of sentence scores?"),
    }
    title, question = names[view]
    title = f"{model} · {title}"
    name = f"{tool}_sentence_{view}"
    shape: PanelShape = "distribution" if view == "distribution" else "heatmap"
    note = (
        "The share is calculated within each document. Categories follow the same label grouping as the sentiment summaries."
        if view == "mix"
        else "Position is relative to the first and last scored sentence. Blank sections have no scored sentences; each colored cell is a mean."
        if view == "progression"
        else "Each point is a scored sentence. The box shows the middle half and the line through it marks the median."
    )
    notes: tuple[str, ...] = (note, f"Compare scores within {model} and the same settings.")
    if tool == "polarity_distilbert_sst2":
        notes += ("DistilBERT SST-2 predicts positive or negative; its label set has no neutral class.",)
    required = (DOC_ID, DOC, SENT_ID, "Sentence", "Label" if view == "mix" else SCORE)
    options = (_DOCS, _FIND, _SECTIONS) if view == "progression" else (_DOCS, _FIND)

    def build(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
        data = frame.dropna(subset=[DOC_ID, DOC, SENT_ID]).copy()
        data[DOC_ID] = data[DOC_ID].astype(str)
        if view == "mix":
            data = data[data["Label"].notna() & data["Label"].astype(str).str.strip().ne("")].copy()
        else:
            data[SCORE] = pd.to_numeric(data[SCORE], errors="coerce")
            data = data[data[SCORE].between(-1, 1)].copy()
        if view == "progression":
            data = data[pd.to_numeric(data[SENT_ID], errors="coerce").between(0, float("inf"), inclusive="left")]
        docs = _selected_documents(data, params)
        if not docs:
            return Result.failure(
                Diagnostic.error("PANEL_NO_SENTIMENT_ROWS", "No scored sentences match this document selection.")
            )
        total_docs = data[DOC_ID].nunique()
        data = data[data[DOC_ID].isin(docs)].copy()
        labels = _document_labels(data, docs)
        sections: tuple[str, ...] = ()
        scale, measure, domain = "", "", None
        if view == "mix":
            sections = CLASSES
            scale, measure, domain = "sequential", "Share (%)", (0.0, 100.0)
            marks = _composition(data, docs, labels)
            x_label = "Polarity class"
        elif view == "progression":
            count = int(params["sections"])
            sections = tuple(f"{100 * i // count}-{100 * (i + 1) // count}%" for i in range(count))
            scale, measure, domain = "diverging", "Mean score", (-1.0, 1.0)
            marks = _progression(data, docs, labels, sections)
            x_label = "Relative sentence position"
        else:
            marks = _distribution(data, docs, labels)
            x_label = "Sentence compound score"
        return Result.success(
            PreparedPanel(
                panel=name,
                shape=shape,
                title=title,
                marks=marks,
                data=data,
                provenance=provenance,
                notes=notes,
                x_label=x_label,
                y_label="Document",
                x_categories=sections,
                y_categories=tuple(labels.values()),
                color_scale=scale,
                value_label=measure,
                value_domain=domain,
                subtitle=f"{len(docs):,} of {total_docs:,} documents · {len(data):,} scored sentences shown",
                height=max(310, 185 + 27 * len(docs)),
            )
        )

    return PanelDefinition(
        name=name,
        tool=tool,
        title=title,
        shape=shape,
        question=question,
        summary=note,
        requires=required,
        params=options,
        build=build,
        notes=notes,
    )


def _lexicon_coverage(tool: str, model: str, score: str, center: float) -> PanelDefinition:
    name = f"{tool}_lexicon_coverage"
    title = f"{model} · Lexicon coverage"
    notes = (
        "Coverage is rated tokens divided by input tokens. Documents with no rated words have no score in this view.",
    )

    def build(frame: pd.DataFrame, _params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
        data = frame.dropna(subset=[DOC_ID, DOC]).copy()
        data[DOC_ID] = data[DOC_ID].astype(str)
        for column in ("Tokens", "Hits", score):
            data[column] = pd.to_numeric(data[column], errors="coerce")
        valid = (
            data["Hits"].gt(0)
            & data["Hits"].le(data["Tokens"])
            & data["Tokens"].lt(float("inf"))
            & data[score].between(-float("inf"), float("inf"), inclusive="neither")
        )
        data = data[valid].copy()
        if data.empty:
            return Result.failure(
                Diagnostic.error("PANEL_NO_LEXICON_HITS", "No documents have valid ratings and matched words.")
            )
        data["Coverage (%)"] = 100 * data["Hits"] / data["Tokens"]
        labels = _document_labels(data, tuple(data[DOC_ID].unique()))
        label_ids = set(data.nsmallest(4, "Coverage (%)")[DOC_ID])
        marks = tuple(
            PanelMark(
                key=str(index),
                label=labels[doc],
                x=coverage,
                y=value,
                size=float(hits),
                labelled=doc in label_ids,
                evidence=Evidence(
                    "documents",
                    ((DOC_ID, doc),),
                    1,
                    f"{labels[doc]} · {coverage:.1f}% covered ({hits:,g} of {tokens:,g} tokens) · {score} {value:.3f}",
                ),
            )
            for index, (doc, hits, tokens, value, coverage) in enumerate(
                data[[DOC_ID, "Hits", "Tokens", score, "Coverage (%)"]].itertuples(index=False, name=None)
            )
        )
        return Result.success(
            PreparedPanel(
                panel=name,
                title=title,
                shape="scatter_labelled",
                marks=marks,
                data=data,
                provenance=provenance,
                x_label="Tokens rated by the lexicon (%)",
                y_label=score,
                notes=notes,
                annotations=(Annotation("hline", center, "Neutral score"),),
                subtitle=f"{len(data):,} of {len(frame):,} documents plotted · Point area shows rated tokens",
            )
        )

    return PanelDefinition(
        name=name,
        tool=tool,
        shape="scatter_labelled",
        title=title,
        question="Which document scores rely on the smallest share of the document's words?",
        summary="Document scores against the percentage of tokens rated by the lexicon.",
        requires=(DOC_ID, DOC, "Tokens", "Hits", score),
        params=(),
        build=build,
        notes=notes,
    )


SENTIMENT_DETAIL_PANELS = (
    *(
        _sentence_panel(tool, model, view)
        for tool, model in (
            ("polarity_vader", "VADER"),
            ("polarity_distilbert_sst2", "DistilBERT"),
            ("polarity_spacy", "spaCy"),
            ("polarity_stanza", "Stanza"),
            ("polarity_corenlp", "CoreNLP"),
        )
        for view in ("mix", "progression", "distribution")
    ),
    _lexicon_coverage("valence_anew", "ANEW", "Valence", 5.0),
    _lexicon_coverage("happiness_hedonometer", "Hedonometer", "Happiness", 5.0),
    _lexicon_coverage("polarity_sentiwordnet", "SentiWordNet", "Net", 0.0),
)
