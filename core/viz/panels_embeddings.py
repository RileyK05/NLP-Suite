"""Ranked nearest-neighbour panels for the two public Word2Vec tools.

Both tools publish ``Word/Neighbor/Cosine`` rows. The cosine value is the
original-space similarity computed by the embedding model; this panel shows
that value directly and does not infer cosine similarity from a 2-D t-SNE
projection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Any

import numpy as np
import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import FUNCTION_WORDS
from core.viz.panelspec import Evidence, PanelDefinition, PanelMark, PanelParam, PreparedPanel, Provenance

__all__ = [
    "WORD2VEC_BERT_NEIGHBOURS",
    "WORD2VEC_BERT_TSNE",
    "WORD2VEC_BERT_VECTOR_QUERY",
    "WORD2VEC_GENSIM_NEIGHBOURS",
    "WORD2VEC_GENSIM_TSNE",
    "WORD2VEC_GENSIM_VECTOR_QUERY",
    "nearest_neighbours",
    "query_saved_vectors",
    "tsne_projection",
]

WORD = "Word"
NEIGHBOR = "Neighbor"
COSINE = "Cosine"
GROUP = "Group"
_OTHER_WORDS = "(other words)"
_GROUP_NOTE = (
    "Colour is a meaning group: k-means over the words' original vectors (not the map), each group named by its "
    "three words nearest its centre. Function words and rare words belong to no group and are grey."
)
_TOP_N = PanelParam(
    name="top-n",
    type="int",
    default=20,
    minimum=1,
    maximum=100,
    label="Neighbours to show",
    help="Limit the ranked rows shown from this neighbours.csv table.",
)
_VECTOR_QUERY = PanelParam(
    name="query",
    type="str",
    default="",
    label="Query word",
    help="Enter a word from vectors.csv to calculate its nearest neighbours without retraining.",
)
_VECTOR_TOP_N = PanelParam(
    name="top-n",
    type="int",
    default=20,
    minimum=1,
    maximum=100,
    label="Neighbours to show",
    help="Limit the ranked neighbours calculated from this saved vectors.csv table.",
)
_NOTES = (
    "Bar length is cosine similarity in the original embedding space, read directly from neighbours.csv. It is not a distance.",
    "The query and neighbours are word types. Similarity does not establish that two words are interchangeable in every context.",
    "A t-SNE map is a separate 2-D projection; distances between points on that map do not equal these cosine values.",
)
_TSNE_NOTES = (
    "Each point is one word type from tsne.csv, positioned by its saved 2-D t-SNE coordinates.",
    "The projection is an exploratory view of higher-dimensional vectors; it can distort global distances and structure, and layouts can change with the run settings.",
    "Distances on this map are not cosine similarities. Cosine scores in neighbours.csv are computed in the original embedding space.",
)
_LABEL_COUNT = PanelParam(
    name="label-count",
    type="int",
    default=25,
    minimum=0,
    maximum=100,
    label="Words to label",
    help="Label up to this many words: the most frequent shown, placed so no two labels overlap.",
)
_MAP_WORDS = PanelParam(
    name="words",
    type="int",
    default=400,
    minimum=20,
    maximum=10000,
    label="Words to map",
    help="Show this many of the corpus's most frequent words. Thousands of points make one unreadable cloud.",
)
_MAP_HIDE_FUNCTION_WORDS = PanelParam(
    name="hide-function-words",
    type="bool",
    default=True,
    label="Hide function words",
    help="Leave out the, of, and ... : the most frequent words, and the least telling on a map of meaning.",
)


def _gensim(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    return nearest_neighbours(frame, params, provenance, panel=WORD2VEC_GENSIM_NEIGHBOURS)


def _bert(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    return nearest_neighbours(frame, params, provenance, panel=WORD2VEC_BERT_NEIGHBOURS)


def _gensim_tsne(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    return tsne_projection(frame, params, provenance, panel=WORD2VEC_GENSIM_TSNE)


def _bert_tsne(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    return tsne_projection(frame, params, provenance, panel=WORD2VEC_BERT_TSNE)


def _gensim_vectors(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    return query_saved_vectors(frame, params, provenance, panel=WORD2VEC_GENSIM_VECTOR_QUERY)


def _bert_vectors(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    return query_saved_vectors(frame, params, provenance, panel=WORD2VEC_BERT_VECTOR_QUERY)


def query_saved_vectors(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
    *,
    panel: PanelDefinition | None = None,
) -> Result[PreparedPanel]:
    """Calculate cosine neighbours from a saved ``Word/Count/Vector`` table.

    This is intentionally separate from training: changing ``query`` in the
    panel controls must not trigger a costly Gensim or transformer rerun.
    The table values are the rounded vectors the run actually published.
    """
    if panel is None:
        return Result.failure(
            Diagnostic.error("PANEL_BUILD_FAILED", "query_saved_vectors needs a tool panel definition")
        )
    requested_query = str(params["query"]).strip()
    query = requested_query.casefold()
    top_n = int(params["top-n"])

    words = frame[WORD].astype(str).str.strip()
    folded = words.str.casefold()
    duplicated = folded.duplicated(keep=False)
    if duplicated.any():
        examples = sorted(set(words.loc[duplicated].tolist()))[:5]
        return Result.failure(
            Diagnostic.error(
                "PANEL_AMBIGUOUS_VOCABULARY",
                f"case-insensitive duplicate words make vector lookup ambiguous: {examples}",
                words=examples,
            )
        )

    parsed: list[np.ndarray | None] = []
    bad_rows = 0
    dimensions: set[int] = set()
    for raw in frame["Vector"]:
        try:
            vector = np.asarray([float(value.strip()) for value in str(raw).split(",")], dtype=float)
        except (TypeError, ValueError):
            vector = np.asarray([], dtype=float)
        if vector.size == 0 or not np.isfinite(vector).all():
            parsed.append(None)
            bad_rows += 1
        else:
            parsed.append(vector)
            dimensions.add(int(vector.size))
    if len(dimensions) > 1:
        common_dimension = max(
            dimensions, key=lambda size: (sum(v is not None and v.size == size for v in parsed), -size)
        )
        bad_rows += sum(vector is not None and vector.size != common_dimension for vector in parsed)
        parsed = [vector if vector is not None and vector.size == common_dimension else None for vector in parsed]
    elif dimensions:
        common_dimension = next(iter(dimensions))
    else:
        return Result.failure(Diagnostic.error("PANEL_NO_VECTORS", "vectors.csv has no usable numeric vectors"))

    if not query:
        counts = pd.to_numeric(frame["Count"], errors="coerce")
        usable_positions = [
            position
            for position, vector in enumerate(parsed)
            if vector is not None and _is_finite(counts.iloc[position]) and words.iloc[position]
        ]
        if not usable_positions:
            return Result.failure(
                Diagnostic.error(
                    "PANEL_QUERY_REQUIRED",
                    "enter a word from vectors.csv; no usable Count/vector pair is available for a default query",
                )
            )
        default_position = min(
            usable_positions,
            key=lambda position: (-float(counts.iloc[position]), folded.iloc[position], words.iloc[position]),
        )
        query = folded.iloc[default_position]

    lookup = {key: index for index, key in enumerate(folded.tolist()) if parsed[index] is not None}
    query_index = lookup.get(query)
    if query_index is None:
        return Result.failure(
            Diagnostic.error("PANEL_QUERY_NOT_FOUND", f"{params['query']!r} is not in this run's usable vocabulary")
        )
    query_vector = parsed[query_index]
    if query_vector is None or query_vector.size != common_dimension:
        return Result.failure(
            Diagnostic.error("PANEL_BAD_QUERY_VECTOR", f"the saved vector for {params['query']!r} is not usable")
        )
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm == 0:
        return Result.failure(Diagnostic.error("PANEL_ZERO_VECTOR", f"{params['query']!r} has a zero vector"))

    rows: list[dict[str, Any]] = []
    for index, other in enumerate(parsed):
        if other is None or index == query_index:
            continue
        norm = float(np.linalg.norm(other))
        score = 0.0 if norm == 0 else float(np.clip(np.dot(query_vector, other) / (query_norm * norm), -1.0, 1.0))
        rows.append({"Word": str(words.iloc[query_index]), "Neighbor": str(words.iloc[index]), "Cosine": score})
    rows.sort(key=lambda row: (-row["Cosine"], row["Neighbor"].casefold(), row["Neighbor"]))
    if not rows:
        return Result.failure(Diagnostic.error("PANEL_NO_NEIGHBOURS", "the vocabulary has no other usable vectors"))
    drawn = pd.DataFrame(rows[:top_n], columns=[WORD, NEIGHBOR, COSINE])
    query_label = str(words.iloc[query_index])
    if not requested_query:
        provenance = replace(provenance, params={**provenance.params, "query": query_label})
    marks: list[PanelMark] = []
    for rank, row in enumerate(drawn.to_dict("records")):
        neighbor = str(row[NEIGHBOR])
        score = float(row[COSINE])
        marks.append(
            PanelMark(
                key=f"{query_label}:{neighbor}",
                label=neighbor,
                x=score,
                y=float(rank),
                evidence=Evidence(
                    scope="terms",
                    filters=((WORD, query_label), (NEIGHBOR, neighbor)),
                    count=1,
                    describe=f"{neighbor} is neighbour {rank + 1} of {query_label}; cosine similarity {score:.4f}, recalculated from this run's saved vectors",
                    phrase=neighbor,
                    lemma=str(provenance.settings.get("field", "lemma")) == "lemma",
                ),
            )
        )
    diagnostics: tuple[Diagnostic, ...] = ()
    if bad_rows:
        diagnostics = (
            Diagnostic.warning(
                "PANEL_BAD_VECTORS",
                f"{bad_rows} vector row(s) were malformed, non-finite, or had a different dimension and were omitted.",
                dropped=bad_rows,
            ),
        )
    return Result.success(
        PreparedPanel(
            panel=panel.name,
            shape="ranked_bars",
            title=f"Words nearest to {query_label}",
            subtitle=(
                f"{len(drawn)} neighbour(s) · cosine from saved {common_dimension}-dimensional vectors"
                + (" · defaulted to the most frequent word" if not requested_query else "")
            ),
            marks=tuple(marks),
            x_label="Cosine similarity in the saved vector space",
            y_label="Neighbour rank",
            provenance=provenance,
            data=drawn,
            notes=panel.notes,
        ),
        *diagnostics,
    )


def tsne_projection(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
    *,
    panel: PanelDefinition | None = None,
) -> Result[PreparedPanel]:
    """Prepare t-SNE points with row-resolving evidence and sparse labels."""
    if panel is None:
        return Result.failure(Diagnostic.error("PANEL_BUILD_FAILED", "tsne_projection needs a tool panel definition"))
    working = frame.copy()
    working["X"] = pd.to_numeric(working["X"], errors="coerce")
    working["Y"] = pd.to_numeric(working["Y"], errors="coerce")
    finite = working["X"].map(_is_finite) & working["Y"].map(_is_finite)
    dropped = int((~finite).sum())
    working = working.loc[finite].copy()
    diagnostic = (
        Diagnostic.warning(
            "PANEL_BAD_NUMERIC",
            f"{dropped} row(s) had non-numeric or non-finite t-SNE coordinates and were left out.",
            dropped=dropped,
        )
        if dropped
        else None
    )
    if working.empty:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", "no t-SNE rows have usable X and Y coordinates"),
            *((diagnostic,) if diagnostic else ()),
        )

    # Word + both saved coordinates identify an actual source row, even if a
    # table contains repeated word labels. Identical triples are ambiguous.
    evidence_columns = (WORD, "X", "Y")
    if working.duplicated(list(evidence_columns)).any():
        return Result.failure(
            Diagnostic.error(
                "PANEL_AMBIGUOUS_EVIDENCE", "duplicate Word/X/Y rows cannot be linked to a unique t-SNE source row"
            ),
            *((diagnostic,) if diagnostic else ()),
        )

    label_count = int(params["label-count"])
    diagnostics: list[Diagnostic] = [diagnostic] if diagnostic else []
    counted = "Count" in working.columns and pd.to_numeric(working["Count"], errors="coerce").notna().any()
    total_words = len(working)
    if counted:
        # The map shows the words the corpus uses most: 7,478 points drew one
        # cloud in which no point or label could be told apart.
        working["Count"] = pd.to_numeric(working["Count"], errors="coerce").fillna(0)
        if bool(params.get("hide-function-words", True)):
            is_function = working[WORD].astype(str).str.lower().isin(FUNCTION_WORDS)
            hidden = int(is_function.sum())
            working = working[~is_function]
            if hidden:
                diagnostics.append(
                    Diagnostic.info(
                        "PANEL_FUNCTION_WORDS_HIDDEN",
                        f"{hidden} function word(s) were left off the map. Turn off hide-function-words to see them.",
                        removed=hidden,
                    )
                )
        limit = int(params.get("words", 400))
        working = working.sort_values(["Count", WORD], ascending=[False, True], kind="stable").head(limit)
        if len(working) < total_words:
            diagnostics.append(
                Diagnostic.info(
                    "PANEL_WORDS_CAPPED",
                    f"Showing the {len(working)} most frequent of {total_words} mapped words; raise 'words' to see more.",
                    shown=len(working),
                    total=total_words,
                )
            )
        working = working.reset_index(drop=True)
    else:
        diagnostics.append(
            Diagnostic.info(
                "PANEL_NO_FREQUENCY",
                "This run's tsne.csv has no Count column (runs before this release), so every word is drawn and "
                "labels go to words spread across the map. Re-run the tool to map its most frequent words.",
            )
        )
    coords = working[["X", "Y"]].to_numpy(dtype=float)
    selected: list[int] = []
    if counted:
        # Rows are already most-frequent first: label the top of the list.
        selected = list(range(min(label_count, len(working))))
    elif label_count:
        # No frequency to go by: coordinate extremes first, then repeatedly
        # the point farthest from those already chosen, which spreads a few
        # labels over the map without implying those words matter more.
        extrema = [
            int(coords[:, 0].argmin()),
            int(coords[:, 0].argmax()),
            int(coords[:, 1].argmin()),
            int(coords[:, 1].argmax()),
        ]
        for index in extrema:
            if index not in selected:
                selected.append(index)
                if len(selected) >= label_count:
                    break
        # Each point's squared distance to its nearest chosen label, kept up
        # to date as labels are added: the same choice as recomputing every
        # distance each round, in linear time per label (7,478 words took 6 s).
        words = working[WORD].astype(str).tolist()
        nearest = np.full(len(coords), np.inf)
        for index in selected:
            nearest = np.minimum(nearest, ((coords - coords[index]) ** 2).sum(axis=1))
        taken = np.zeros(len(coords), dtype=bool)
        taken[selected] = True
        while len(selected) < min(label_count, len(working)):
            open_distances = np.where(taken, -np.inf, nearest)
            best = open_distances.max()
            ties = np.flatnonzero(open_distances == best)
            candidate = int(max(ties, key=lambda i: words[int(i)]))
            selected.append(candidate)
            taken[candidate] = True
            nearest = np.minimum(nearest, ((coords - coords[candidate]) ** 2).sum(axis=1))
    selected_set = set(selected)
    # Runs from this release name each word's meaning group: the map's
    # regions get names, which is most of what a t-SNE map can honestly say.
    grouped = GROUP in working.columns and working[GROUP].fillna("").astype(str).str.strip().ne("").any()
    group_of = working[GROUP].fillna("").astype(str).str.strip() if grouped else None
    groups: tuple[str, ...] = ()
    if group_of is not None:
        named = group_of[group_of != ""].value_counts()
        groups = (*sorted(named.index, key=lambda name: (-int(named[name]), name)), _OTHER_WORDS)

    marks: list[PanelMark] = []
    for index, (_, row) in enumerate(working.iterrows()):
        word, x, y = str(row[WORD]), float(row["X"]), float(row["Y"])
        filters = tuple((column, str(row[column])) for column in evidence_columns)
        frequency = int(row["Count"]) if counted else None
        group = (str(group_of.iloc[index]) or _OTHER_WORDS) if group_of is not None else ""
        marks.append(
            PanelMark(
                key=f"point-{index}:{word}",
                label=word,
                x=x,
                y=y,
                size=float(frequency) if frequency is not None else None,
                labelled=index in selected_set,
                group=group,
                evidence=Evidence(
                    scope="rows",
                    filters=filters,
                    count=1,
                    describe=(
                        f"{word}: {frequency:,} occurrence(s), at t-SNE ({x:g}, {y:g})"
                        if frequency is not None
                        else f"{word} at saved t-SNE coordinates ({x:g}, {y:g})"
                    ),
                ),
            )
        )
    shown = f"the {len(marks)} most frequent of {total_words} words" if counted else f"{len(marks)} word type(s)"
    prepared = PreparedPanel(
        panel=panel.name,
        shape="scatter_labelled",
        title="Word-vector t-SNE projection",
        subtitle=(
            f"{shown} · point size is frequency · {len(selected_set)} labelled"
            if counted
            else f"{shown} · {len(selected_set)} labels"
        )
        + (" · colour: meaning group, named by its central words" if groups else ""),
        marks=tuple(marks),
        x_label="t-SNE dimension 1",
        y_label="t-SNE dimension 2",
        provenance=provenance,
        data=working.reset_index(drop=True),
        groups=groups,
        notes=(*_TSNE_NOTES, _GROUP_NOTE) if groups else _TSNE_NOTES,
    )
    return Result.success(prepared, *diagnostics)


def nearest_neighbours(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
    *,
    panel: PanelDefinition | None = None,
) -> Result[PreparedPanel]:
    """Prepare ranked bars whose evidence filters identify their source row."""
    definition = panel
    if definition is None:
        return Result.failure(
            Diagnostic.error("PANEL_BUILD_FAILED", "nearest_neighbours needs a tool panel definition")
        )
    top_n = int(params["top-n"])
    working = frame.copy()
    working[COSINE] = pd.to_numeric(working[COSINE], errors="coerce")
    finite = working[COSINE].map(_is_finite)
    dropped = int((~finite).sum())
    working = working[finite].copy()
    if dropped:
        diagnostic = Diagnostic.warning(
            "PANEL_BAD_NUMERIC",
            f"{dropped} row(s) had a non-numeric or non-finite cosine score and were left out.",
            dropped=dropped,
        )
    else:
        diagnostic = None
    if working.empty:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", "no neighbours have a usable cosine score"),
            *((diagnostic,) if diagnostic else ()),
        )

    # Preserve all useful source columns in the adjacent data table while
    # using only the published score to order and size bars.
    ordered = working.assign(_neighbor=working[NEIGHBOR].astype(str)).sort_values(
        [COSINE, "_neighbor"], ascending=[False, True], kind="stable"
    )
    drawn = ordered.head(top_n).drop(columns=["_neighbor"]).copy()
    marks: list[PanelMark] = []
    query = str(drawn.iloc[0][WORD])
    for rank, (_, row) in enumerate(drawn.iterrows()):
        query, neighbor, score = str(row[WORD]), str(row[NEIGHBOR]), float(row[COSINE])
        # Query + neighbor are the natural row key in both public tables.
        # Evidence filters are text by contract, so avoid formatting a float
        # as text and asking a caller to compare that text to a numeric cell.
        filters = ((WORD, query), (NEIGHBOR, neighbor))
        marks.append(
            PanelMark(
                key=f"{query}:{neighbor}",
                label=neighbor,
                x=score,
                y=float(rank),
                evidence=Evidence(
                    scope="rows",
                    filters=filters,
                    count=1,
                    describe=f"{neighbor} is neighbour {rank + 1} of {query}; cosine similarity {score:.4f}",
                ),
            )
        )
    query = str(drawn.iloc[0][WORD])
    prepared = PreparedPanel(
        panel=definition.name,
        shape="ranked_bars",
        title=f"Words nearest to {query}",
        subtitle=f"{len(drawn)} neighbour(s) · ranked by cosine similarity",
        marks=tuple(marks),
        x_label="Cosine similarity in the original embedding space",
        y_label="Neighbour rank",
        provenance=provenance,
        data=drawn.reset_index(drop=True),
        notes=_NOTES,
    )
    return Result.success(prepared, *((diagnostic,) if diagnostic else ()))


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


WORD2VEC_GENSIM_NEIGHBOURS = PanelDefinition(
    name="word2vec_gensim_neighbours",
    title="Word2Vec nearest neighbours (Gensim)",
    question="Which words does the model place closest to this one?",
    tool="word2vec_gensim",
    shape="ranked_bars",
    summary="Words nearest to the query, ranked by the model's original-space cosine similarity.",
    requires=(WORD, NEIGHBOR, COSINE),
    params=(_TOP_N,),
    build=_gensim,
    notes=_NOTES,
)

WORD2VEC_BERT_NEIGHBOURS = PanelDefinition(
    name="word2vec_bert_neighbours",
    title="Word2Vec nearest neighbours (BERT)",
    question="Which words does the model place closest to this one?",
    tool="word2vec_bert",
    shape="ranked_bars",
    summary="Words nearest to the query, ranked by the model's original-space cosine similarity.",
    requires=(WORD, NEIGHBOR, COSINE),
    params=(_TOP_N,),
    build=_bert,
    notes=_NOTES,
)

WORD2VEC_GENSIM_VECTOR_QUERY = PanelDefinition(
    name="word2vec_gensim_saved_vectors",
    title="Explore saved Word2Vec vectors (Gensim)",
    question="Which words are nearest to a word I choose, without retraining?",
    tool="word2vec_gensim",
    shape="ranked_bars",
    summary="Recalculate a query word's nearest neighbours from this run's saved vectors.csv values.",
    requires=(WORD, "Count", "Vector"),
    params=(_VECTOR_QUERY, _VECTOR_TOP_N),
    build=_gensim_vectors,
    notes=(
        "Cosine similarity is recalculated from the rounded vectors saved in this run's vectors.csv; changing the query does not retrain the model.",
        "The query and neighbours are word types. Similarity means the vectors point in similar directions, not that the words are interchangeable in every passage.",
        "The displayed cosine uses both saved vectors. Click a neighbour to read occurrences of that word in the corpus.",
        "If Query word is left empty, the panel starts with the most frequent word in the saved vocabulary.",
    ),
)

WORD2VEC_BERT_VECTOR_QUERY = PanelDefinition(
    name="word2vec_bert_saved_vectors",
    title="Explore saved BERT word vectors",
    question="Which words are nearest to a word I choose, without rerunning BERT?",
    tool="word2vec_bert",
    shape="ranked_bars",
    summary="Recalculate a query word's nearest neighbours from this run's saved vectors.csv values.",
    requires=(WORD, "Count", "Vector"),
    params=(_VECTOR_QUERY, _VECTOR_TOP_N),
    build=_bert_vectors,
    notes=(
        "Cosine similarity is recalculated from the rounded vectors saved in this run's vectors.csv; changing the query does not rerun BERT.",
        "Each saved row is a word-type vector averaged over its contexts. This panel compares saved type vectors, not token-by-token contextual representations.",
        "The displayed cosine uses both saved vectors. Click a neighbour to read occurrences of that word in the corpus.",
        "If Query word is left empty, the panel starts with the most frequent word in the saved vocabulary.",
    ),
)

WORD2VEC_GENSIM_TSNE = PanelDefinition(
    name="word2vec_gensim_tsne",
    title="Word-vector t-SNE projection (Gensim)",
    question="Which words does the model treat as neighbours overall?",
    tool="word2vec_gensim",
    shape="scatter_labelled",
    summary="Exploratory 2-D t-SNE map of the corpus's most frequent words (Gensim vectors).",
    requires=(WORD, "X", "Y"),
    params=(_MAP_WORDS, _LABEL_COUNT, _MAP_HIDE_FUNCTION_WORDS),
    build=_gensim_tsne,
    notes=_TSNE_NOTES,
)

WORD2VEC_BERT_TSNE = PanelDefinition(
    name="word2vec_bert_tsne",
    title="Word-vector t-SNE projection (BERT)",
    question="Which words does the model treat as neighbours overall?",
    tool="word2vec_bert",
    shape="scatter_labelled",
    summary="Exploratory 2-D t-SNE map of the corpus's most frequent words (BERT vectors).",
    requires=(WORD, "X", "Y"),
    params=(_MAP_WORDS, _LABEL_COUNT, _MAP_HIDE_FUNCTION_WORDS),
    build=_bert_tsne,
    notes=_TSNE_NOTES,
)
