"""Word-pair and document-pair panels: co-occurrence, similarity, comparison.

Six tools share one shape of problem here: their output is a table of pairs
(word-word, document-document) rather than a table of one thing per row, and
a pair table breaks the usual "x by y" chart in the same way every time --
neither column of the pair names anything on its own.

``ngram_cooccurrence``'s ``cooccurrences.csv`` is the sharpest example: one
row per (pair, document), ``Word 1`` always the alphabetically-first word of
an *unordered* pair, so a bar chart of "largest Count by Word 1" ranks "of"
and "be" -- the two words most likely to sort first, not the two words that
matter (this is FIGURE_RECIPES.md's own headline complaint). The fix used
throughout this module is the same one: aggregate the pair over whatever it
was measured per (documents, in this table) *before* drawing anything, and
compute PPMI from the table's own marginals rather than plotting a raw count.

``doc_similarity`` and ``svo_compare`` share a second shape: a symmetric
document x document table (each unordered pair appearing once). Both get a
heatmap ordered by date (parsed from the file name -- neither table carries
a ``Date`` column) and a k-nearest-neighbour network.

Evidence for a pair mark resolves against a *mirrored* long-format table
(``Word``/``Document``/``Entity`` and ``Partner``/``Other``, two rows per
unordered pair or triple) rather than the tool's own wide table. This is
deliberate: an ``Evidence`` filter is a plain AND of (column, value) pairs,
and "every row touching word W" is not expressible as an AND filter over a
table where W can sit in either of two columns. The mirror makes it one:
filtering ``Word == W`` selects exactly W's rows, whichever side of the
original pair it started on. See ``_mirror`` and its call sites.

Pure data-shape functions of (frame, params, provenance): no plotly, no
filesystem, no corpus access.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import (
    communities,
    community_order,
    decimal_year,
    document_labels,
    force_layout,
    group_of,
    is_function_word,
)
from core.viz.panelspec import (
    Evidence,
    PanelDefinition,
    PanelEdge,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
)

__all__ = [
    "COLLOCATIONS_NETWORK",
    "DOC_DUPLICATES_GRAPH",
    "DOC_SIMILARITY_HEATMAP",
    "DOC_SIMILARITY_NEIGHBOURS",
    "KNOWLEDGE_GRAPH_NETWORK",
    "NGRAM_COOCCURRENCE_ASSOCIATION",
    "NGRAM_COOCCURRENCE_NETWORK",
    "NGRAM_COOCCURRENCE_PARTNERS",
    "PAIRS_PANELS",
    "SVO_COMPARE_HEATMAP",
    "SVO_COMPARE_NEIGHBOURS",
    "collocations_network",
    "doc_duplicates_graph",
    "doc_similarity_heatmap",
    "doc_similarity_neighbours",
    "knowledge_graph_network",
    "ngram_cooccurrence_association",
    "ngram_cooccurrence_network",
    "ngram_cooccurrence_partners",
    "svo_compare_heatmap",
    "svo_compare_neighbours",
]

# ngram_cooccurrence / cooccurrences.csv -- core/analysis/ngram_cooccurrence.py.
WORD_1 = "Word 1"
WORD_2 = "Word 2"
COUNT = "Count"
DOC_ID = "Document ID"

# doc_similarity / doc_pairs.csv and duplicates.csv, doc_duplicates / fuzzy_pairs.csv
# -- core/analysis/doc_similarity.py, core/analysis/doc_duplicates.py. All three
# share this exact column set.
DOC_A = "Document A"
DOC_B = "Document B"
SIMILARITY = "Similarity"
BAND = "Band"

# svo_compare / svo_compare.csv -- core/analysis/svo_compare.py. No Document
# name or Date column exists here: only the bare Document ID survives from
# clause_svo's grouping, so these panels cannot offer date order or speaker
# grouping the way doc_similarity's can.
SVO_DOC_A = "Doc A"
SVO_DOC_B = "Doc B"
COMMON_TRIPLES = "Common Triples"
SUBJECT_JACCARD = "Subject Jaccard"
VERB_JACCARD = "Verb Jaccard"
OBJECT_JACCARD = "Object Jaccard"
TRIPLE_JACCARD = "Triple Jaccard"

# collocations / collocations.csv -- core/analysis/collocations.py, spelled
# exactly as panels_collocations.py reads them (not imported from there: that
# module is off limits to edit and its constants are private to its file).
COLL_WORD_1 = "Word 1"
COLL_WORD_2 = "Word 2"
COOCCURRENCES = "Co-occurrences"
PMI = "PMI"
T_SCORE = "T-Score"
G2 = "G2 (log-likelihood)"
LOG_DICE = "Log Dice"

# knowledge_graph / knowledge_graph.csv -- core/analysis/knowledge_graph.py.
SUBJECT = "Subject"
PREDICATE = "Predicate"
OBJECT = "Object"
SOURCE = "Source"

_MARK_CAP = 1500
_MAX_HEATMAP_DOCS = 150

# date_speaker_kind.ext -- just the date prefix, matching panel_helpers'
# private _NAMED pattern closely enough for ordering (that pattern is not
# exported, and these tables' pair columns are file names, not a Document
# column panel_helpers.dated() could read).
_DATE_PREFIX = re.compile(r"^(\d{4}(?:-\d{2}-\d{2})?)_")


def _doc_date(name: str) -> str | None:
    """The leading date substring of a file name, or None if it has none."""
    match = _DATE_PREFIX.match(str(name).strip())
    return match.group(1) if match else None


def _document_order(names: list[str]) -> tuple[list[str], bool]:
    """Documents in date order when every name is dated, else alphabetical.

    Mixing the two (some documents sorted by date, undated ones tacked on)
    would silently misplace whichever documents happen to lack a parseable
    prefix; an all-or-nothing rule is at least honest about which axis is
    actually in effect, and the caller states which one fired.
    """
    dates = {name: _doc_date(name) for name in names}
    if all(value is not None for value in dates.values()):
        ordered = sorted(names, key=lambda n: (decimal_year(dates[n]), n))
        return ordered, True
    return sorted(names), False


def _numeric_or_text(value: str) -> tuple[int, object]:
    """Sort key: numeric document IDs in numeric order, else text order.

    svo_compare's ``Doc A``/``Doc B`` are bare Document IDs (``"1"``..``"87"``
    in this corpus) with no file name to parse a date from; sorting them as
    text would put "10" before "2".
    """
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def _hub_pairs(  # noqa: PLR0913 - keyword-only; one rule for both word networks
    pairs: pd.DataFrame, *, a: str, b: str, mass: str, strength: str, words: int, links: int
) -> tuple[pd.DataFrame, int]:
    """The pairs among the *words* most connected words, each word keeping
    its *links* strongest partners; and how many words qualified.

    Hubs first, then the links among them. Taking the strongest pairs alone
    drew forty disconnected dyads on the real corpus: the strongest pairs are
    rare or fixed phrases, and those share no words. A network is only a
    network if its nodes are the words that connect.
    """
    totals = (
        pd.concat([pairs[[a, mass]].rename(columns={a: "Word"}), pairs[[b, mass]].rename(columns={b: "Word"})])
        .groupby("Word", sort=False)[mass]
        .sum()
    )
    hubs = set(totals.nlargest(words).index)
    among = pairs[pairs[a].isin(hubs) & pairs[b].isin(hubs)]
    ranked = _mirror(among, a, b, (strength,)).sort_values([strength, "Partner"], ascending=[False, True])
    strongest = ranked.groupby("Word", sort=False).head(links)
    chosen = {frozenset((str(x), str(y))) for x, y in zip(strongest["Word"], strongest["Partner"], strict=True)}
    keep = [frozenset((str(x), str(y))) in chosen for x, y in zip(among[a], among[b], strict=True)]
    return among[keep], len(totals)


def _svo_labels(frame: pd.DataFrame, ids: list[str]) -> dict[str, str]:
    """Axis labels for svo_compare's document ids: "1934 Roosevelt" when the
    run carries the names (``Document A``/``Document B``, appended by the
    executor), "Doc 7" for an older run that does not."""
    if "Document A" not in frame.columns or "Document B" not in frame.columns:
        return {name: f"Doc {name}" for name in ids}
    names = dict(zip(frame[SVO_DOC_A].astype(str), frame["Document A"].astype(str), strict=True))
    names.update(zip(frame[SVO_DOC_B].astype(str), frame["Document B"].astype(str), strict=True))
    short = document_labels(names.get(name, name) for name in ids)
    return {name: short[names.get(name, name)] for name in ids}


def _mirror(frame: pd.DataFrame, a: str, b: str, keep: tuple[str, ...]) -> pd.DataFrame:
    """A symmetric pair table as two directed rows, ``a``/``b`` renamed to
    ``Word``/``Partner``.

    See the module docstring: this is what makes "every row touching X"
    expressible as a single equality filter, for both directions of an
    unordered pair, against one published table.
    """
    columns = ["Word", "Partner", *keep]
    left = frame.rename(columns={a: "Word", b: "Partner"})[columns]
    right = frame.rename(columns={b: "Word", a: "Partner"})[columns]
    return pd.concat([left, right], ignore_index=True)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------- ngram_cooccurrence --


def _pair_marginals(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, float]:
    """Aggregate cooccurrences.csv to one row per pair, with PPMI.

    Vectorised throughout -- this table is 1,036,073 rows on the real
    corpus, and looping it would blow the panel's 2-second budget many times
    over. ``Total`` sums ``Count`` (co-occurring sentences) over every
    document; ``Documents`` counts how many documents contributed. PPMI uses
    this table's OWN marginals: P(w) = (sum of Total over every pair
    containing w) / (sum of Total over every pair), the standard
    distributional-semantics definition. A table produced with the tool's
    own ``--min-count`` above 1 has already dropped rare (pair, document)
    observations before this function ever sees them, which understates
    every marginal a little; callers surface that with
    ``PANEL_TRUNCATED_MARGINALS``, this function does not judge it.
    """
    working = frame.copy()
    working[WORD_1] = working[WORD_1].astype(str)
    working[WORD_2] = working[WORD_2].astype(str)
    working[COUNT] = pd.to_numeric(working[COUNT], errors="coerce").fillna(0)
    total_docs = int(working[DOC_ID].nunique()) if DOC_ID in working.columns else 0

    grouped = (
        working.groupby([WORD_1, WORD_2], sort=False)
        .agg(Total=(COUNT, "sum"), Documents=(DOC_ID, "nunique"))
        .reset_index()
    )
    marginal = (
        pd.concat(
            [
                grouped[[WORD_1, "Total"]].rename(columns={WORD_1: "Word"}),
                grouped[[WORD_2, "Total"]].rename(columns={WORD_2: "Word"}),
            ]
        )
        .groupby("Word", sort=False)["Total"]
        .sum()
    )
    grand_total = float(grouped["Total"].sum())
    p1 = grouped[WORD_1].map(marginal).to_numpy(dtype=float) / grand_total
    p2 = grouped[WORD_2].map(marginal).to_numpy(dtype=float) / grand_total
    p_pair = grouped["Total"].to_numpy(dtype=float) / grand_total
    # p1, p2 and p_pair are all strictly positive here: every pair has
    # Total >= 1, and a word's marginal is at least the Total of the one
    # pair being evaluated, so log2 never sees zero or a negative ratio.
    grouped["PPMI"] = np.maximum(0.0, np.log2(p_pair / (p1 * p2)))
    return grouped, total_docs, float(working[COUNT].min()) if len(working) else 1.0


def ngram_cooccurrence_association(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance
) -> Result[PreparedPanel]:
    """One point per pair: log10(co-occurring sentences) against PPMI."""
    min_total = int(params["min-total"])
    hide_function_words = bool(params["hide-function-words"])
    label_top = int(params["label"])

    for column in (WORD_1, WORD_2, COUNT, DOC_ID):
        if column not in frame.columns:
            return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"the co-occurrence table has no {column!r}"))

    diagnostics: list[Diagnostic] = []
    pairs, total_docs, smallest_count = _pair_marginals(frame)
    if smallest_count > 1:
        # This table's own --min-count (not this panel's --min-total) already
        # dropped every (pair, document) observation below that floor, which
        # truncates the marginals PPMI is computed from -- a word that only
        # ever co-occurred once with anything now looks like it never
        # co-occurs at all. Not fatal, just worth saying next to every PPMI.
        diagnostics.append(
            Diagnostic.info(
                "PANEL_TRUNCATED_MARGINALS",
                f"the smallest Count in this table is {smallest_count:g}, not 1: the co-occurrence run's own "
                "--min-count filtered out rarer observations before this panel saw them, which understates "
                "every word's marginal and so every PPMI value here.",
                smallest_count=smallest_count,
            )
        )

    if hide_function_words:
        before = len(pairs)
        pairs = pairs[~pairs[WORD_1].map(is_function_word) & ~pairs[WORD_2].map(is_function_word)]
        removed = before - len(pairs)
        if removed:
            # Justified against the real corpus: at min-total=10 on the real
            # 1,036,073-row table, function words ("of", "the", "be", "and")
            # dominate both the highest-total and highest-PPMI ends of an
            # unfiltered plot -- they co-occur with everything, so PPMI
            # rewards them for touching thousands of partners. Hiding a pair
            # if EITHER word is a function word (not just both) removes that
            # cluster; requiring both would leave "of freedom" on the plot.
            diagnostics.append(
                Diagnostic.info(
                    "PANEL_FUNCTION_WORDS_HIDDEN",
                    f"{removed} pair(s) with a function word ('of', 'the', 'be', ...) on either side were left "
                    "out. Set hide-function-words=false to see them.",
                    removed=removed,
                )
            )

    if min_total:
        before = len(pairs)
        pairs = pairs[pairs["Total"] >= min_total]
        removed = before - len(pairs)
        if removed:
            diagnostics.append(
                Diagnostic.info(
                    "PANEL_FREQUENCY_FILTERED",
                    f"{removed} pair(s) co-occurring in fewer than {min_total} sentence(s) in total were left "
                    "out; a pair seen once or twice is mostly noise.",
                    removed=removed,
                    minimum=min_total,
                )
            )

    if pairs.empty:
        return Result[PreparedPanel].failure(
            Diagnostic.error(
                "PANEL_NO_DATA", "no pairs left to plot after filtering; lower min-total or hide-function-words"
            ),
            *diagnostics,
        )

    if len(pairs) > _MARK_CAP:
        available = len(pairs)
        pairs = pairs.nlargest(_MARK_CAP, "Total")
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_TOO_MANY_MARKS",
                f"{available} pairs is more than a readable scatter holds; drew the {_MARK_CAP} with the most "
                "co-occurring sentences. Raise min-total to choose differently.",
                drawn=_MARK_CAP,
                available=available,
            )
        )

    # Label the pairs strong on both axes. Labelling the top by PPMI alone put
    # all fifteen labels in the rare-pair corner (PPMI's known bias), piled on
    # each other; PPMI times log evidence names "social security" and
    # "united states" instead.
    ranked = pairs.assign(_score=pairs["PPMI"] * np.log10(pairs["Total"].clip(lower=1))).sort_values(
        ["_score", WORD_1, WORD_2], ascending=[False, True, True]
    )
    labelled = (
        set(zip(ranked.head(label_top)[WORD_1], ranked.head(label_top)[WORD_2], strict=True)) if label_top else set()
    )

    marks: list[PanelMark] = []
    for record in pairs.to_dict("records"):
        w1, w2 = str(record[WORD_1]), str(record[WORD_2])
        total, documents, ppmi = float(record["Total"]), int(record["Documents"]), float(record["PPMI"])
        marks.append(
            PanelMark(
                key=f"{w1}::{w2}",
                label=f"{w1} · {w2}",
                x=math.log10(total),
                y=ppmi,
                size=float(documents),
                labelled=(w1, w2) in labelled,
                evidence=Evidence(
                    scope="terms",
                    filters=((WORD_1, w1), (WORD_2, w2)),
                    count=int(total),
                    describe=(
                        f"{w1} {w2}: {int(total)} co-occurring sentences in {documents} of {total_docs} speeches "
                        f"-- PPMI {ppmi:.2f}"
                    ),
                ),
            )
        )

    prepared = PreparedPanel(
        panel=NGRAM_COOCCURRENCE_ASSOCIATION.name,
        shape="scatter_labelled",
        title="Co-occurrence association: PPMI against evidence",
        subtitle=f"{len(marks)} pairs -- naming the {label_top} strongest on both association and evidence",
        marks=tuple(marks),
        x_label="Co-occurring sentences, pooled over documents (log scale)",
        x_log10=True,
        y_label="PPMI (from this table's own word marginals)",
        provenance=provenance,
        data=pairs.reset_index(drop=True),
        notes=_ASSOCIATION_NOTES,
    )
    return Result.success(prepared, *diagnostics)


def ngram_cooccurrence_partners(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance
) -> Result[PreparedPanel]:
    """Ranked bars: one word's partners, by PPMI, bar length = evidence."""
    requested = str(params["word"]).strip()
    top_n = int(params["top-n"])
    for column in (WORD_1, WORD_2, COUNT, DOC_ID):
        if column not in frame.columns:
            return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"the co-occurrence table has no {column!r}"))

    pairs, total_docs, _smallest = _pair_marginals(frame)
    if pairs.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "the co-occurrence table has no pairs to draw"))

    if requested:
        word = requested.casefold()
    else:
        # Default: the content word (function words excluded) carrying the
        # most total co-occurrence mass -- the word a reader would reach for
        # first, not "of" or "the".
        mass = (
            pd.concat(
                [
                    pairs[[WORD_1, "Total"]].rename(columns={WORD_1: "Word"}),
                    pairs[[WORD_2, "Total"]].rename(columns={WORD_2: "Word"}),
                ]
            )
            .groupby("Word", sort=False)["Total"]
            .sum()
            .sort_values(ascending=False)
        )
        content = mass[~mass.index.map(is_function_word)]
        word = str((content if not content.empty else mass).index[0]).casefold()

    matches = pairs[(pairs[WORD_1].str.casefold() == word) | (pairs[WORD_2].str.casefold() == word)]
    if matches.empty:
        available = sorted({*pairs[WORD_1], *pairs[WORD_2]})
        suggestions = ", ".join(available[:8])
        return Result.failure(
            Diagnostic.error(
                "PANEL_WORD_NOT_FOUND",
                f"{requested or word!r} does not co-occur with anything in this table. Some words that do: "
                f"{suggestions}.",
                requested=requested or word,
            )
        )

    matches = matches.assign(
        Partner=np.where(matches[WORD_1].str.casefold() == word, matches[WORD_2], matches[WORD_1])
    ).sort_values(["PPMI", "Partner"], ascending=[False, True])
    diagnostics: list[Diagnostic] = []
    # PPMI's known bias: a partner seen once with the word scores highest of
    # all. On the real corpus "year"'s top partners were one-off names
    # ("lanphier", "rowan") until this floor was added.
    min_total = int(params.get("min-total", 10))
    floor = matches["Total"] >= min_total
    if bool(params.get("hide-function-words", True)):
        floor &= ~matches["Partner"].map(is_function_word)
    if not floor.any():
        return Result.failure(
            Diagnostic.error(
                "PANEL_NO_DATA",
                f"{word!r} has no partner sharing at least {min_total} sentences with it; lower min-total.",
            )
        )
    if not floor.all():
        diagnostics.append(
            Diagnostic.info(
                "PANEL_FREQUENCY_FILTERED",
                f"{int((~floor).sum())} partner(s) sharing fewer than {min_total} sentences with {word!r}, or "
                "that are function words, were left out.",
                removed=int((~floor).sum()),
            )
        )
    matches = matches[floor]
    drawn = matches.head(top_n)

    if len(matches) > top_n:
        diagnostics.append(
            Diagnostic.info(
                "PANEL_TOO_MANY_MARKS",
                f"{word!r} has {len(matches)} partners; showing the top {top_n} by PPMI. Raise top-n to see more.",
                available=len(matches),
                drawn=top_n,
            )
        )

    marks: list[PanelMark] = []
    for rank, record in enumerate(drawn.to_dict("records")):
        w1, w2 = str(record[WORD_1]), str(record[WORD_2])
        partner, total, documents, ppmi = (
            str(record["Partner"]),
            float(record["Total"]),
            int(record["Documents"]),
            float(record["PPMI"]),
        )
        marks.append(
            PanelMark(
                key=f"{w1}::{w2}",
                label=partner,
                x=total,
                y=float(rank),
                evidence=Evidence(
                    scope="terms",
                    filters=((WORD_1, w1), (WORD_2, w2)),
                    count=int(total),
                    describe=(
                        f"{partner} co-occurs with {word} in {int(total)} sentences across {documents} of "
                        f"{total_docs} speeches -- PPMI {ppmi:.2f}"
                    ),
                ),
            )
        )

    prepared = PreparedPanel(
        panel=NGRAM_COOCCURRENCE_PARTNERS.name,
        shape="ranked_bars",
        title=f'Partners of "{word}"',
        subtitle=f"{len(drawn)} of {len(matches)} partner(s) -- ranked by PPMI",
        marks=tuple(marks),
        x_label="Co-occurring sentences (bar length)",
        y_label="Partner rank, by PPMI",
        provenance=provenance,
        data=drawn.reset_index(drop=True),
        notes=_PARTNERS_NOTES,
    )
    return Result.success(prepared, *diagnostics)


def ngram_cooccurrence_network(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance
) -> Result[PreparedPanel]:
    """The strongest co-occurrence pairs as a word network."""
    top_n = int(params["top-n"])
    min_total = int(params["min-total"])
    edge_weight = str(params["edge-weight"])
    for column in (WORD_1, WORD_2, COUNT, DOC_ID):
        if column not in frame.columns:
            return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"the co-occurrence table has no {column!r}"))

    pairs, _total_docs, _smallest = _pair_marginals(frame)
    before = len(pairs)
    # Function words are hidden unconditionally here, not behind a toggle:
    # a network is a small, curated picture by design (top-n edges), and "of"
    # or "the" would sit at the centre of it touching almost every other node
    # by force alone, crowding out every real relationship the figure exists
    # to show. The scatter (association panel) keeps the toggle because it
    # draws everything and a reader can filter by eye; a network cannot.
    pairs = pairs[~pairs[WORD_1].map(is_function_word) & ~pairs[WORD_2].map(is_function_word)]
    pairs = pairs[pairs["Total"] >= min_total]
    diagnostics: list[Diagnostic] = []
    if before - len(pairs):
        diagnostics.append(
            Diagnostic.info(
                "PANEL_FUNCTION_WORDS_HIDDEN",
                f"{before - len(pairs)} pair(s) with a function word or below min-total={min_total} were left "
                "out before ranking.",
                removed=before - len(pairs),
            )
        )
    if pairs.empty:
        return Result[PreparedPanel].failure(
            Diagnostic.error("PANEL_NO_DATA", "no pairs left after removing function words and rare pairs"),
            *diagnostics,
        )

    links = int(params.get("links", 3))
    top, qualified = _hub_pairs(pairs, a=WORD_1, b=WORD_2, mass="Total", strength="PPMI", words=top_n, links=links)
    if top.empty:
        return Result[PreparedPanel].failure(
            Diagnostic.error("PANEL_NO_DATA", "the most connected words share no pairs above min-total; lower it"),
            *diagnostics,
        )
    if qualified > top_n:
        diagnostics.append(
            Diagnostic.info(
                "PANEL_TOO_MANY_MARKS",
                f"{qualified} words qualified; drew the {top_n} with the most co-occurring sentences, each with "
                f"up to {links} link(s). Raise top-n to see more.",
                available=qualified,
                drawn=top_n,
            )
        )

    mirrored = _mirror(top, WORD_1, WORD_2, ("Total", "Documents", "PPMI"))
    nodes = sorted({*top[WORD_1], *top[WORD_2]})
    edges_raw = [
        (
            str(record[WORD_1]),
            str(record[WORD_2]),
            float(record["PPMI"] if edge_weight == "ppmi" else record["Total"]),
            float(record["Total"]),
            int(record["Documents"]),
            float(record["PPMI"]),
        )
        for record in top.to_dict("records")
    ]
    layout = force_layout(nodes, [(w1, w2, weight) for w1, w2, weight, *_ in edges_raw])
    mass = mirrored.groupby("Word", sort=False)["Total"].sum()
    community = communities(nodes, [(w1, w2, weight) for w1, w2, weight, *_ in edges_raw], mass.to_dict())

    marks: list[PanelMark] = []
    for word in nodes:
        x, y = layout[word]
        own = mirrored[mirrored["Word"] == word]
        marks.append(
            PanelMark(
                key=word,
                label=word,
                x=x,
                y=y,
                size=float(mass.get(word, 0.0)),
                labelled=True,
                group=community[word],
                evidence=Evidence(
                    scope="terms",
                    filters=(("Word", word),),
                    count=len(own),
                    describe=f"{word}: {len(own)} partner(s) drawn, {int(mass.get(word, 0.0))} co-occurring sentences",
                ),
            )
        )
    edges: list[PanelEdge] = []
    for w1, w2, weight, total, documents, ppmi in edges_raw:
        edges.append(
            PanelEdge(
                key=f"{w1}::{w2}",
                source=w1,
                target=w2,
                weight=weight,
                evidence=Evidence(
                    scope="terms",
                    filters=(("Word", w1), ("Partner", w2)),
                    count=int(total),
                    describe=f"{w1} & {w2}: {int(total)} co-occurring sentences, {documents} document(s), PPMI {ppmi:.2f}",
                ),
            )
        )

    prepared = PreparedPanel(
        panel=NGRAM_COOCCURRENCE_NETWORK.name,
        shape="network",
        edges=tuple(edges),
        title="Co-occurrence network",
        subtitle=f"{len(nodes)} most connected words, {len(edges)} links -- each word's strongest partners among them, edge weight = {edge_weight}",
        marks=tuple(marks),
        x_label="",
        y_label="",
        provenance=provenance,
        data=mirrored.reset_index(drop=True),
        groups=community_order(community, mass.to_dict()),
        notes=(*_NETWORK_NOTES, _COMMUNITY_NOTE),
    )
    return Result.success(prepared, *diagnostics)


_COMMUNITY_NOTE = (
    "Colours are communities found by weighted label propagation over the drawn links -- words more linked to "
    "each other than to the rest -- each named by its two most frequent words. A community is a pattern in "
    "this drawing, not a topic the corpus declares."
)

_ASSOCIATION_NOTES: tuple[str, ...] = (
    "Word 1 and Word 2 are only the alphabetically-first and -second words of an unordered pair; neither column "
    "means anything read alone, which is why this figure is built on the PAIR, aggregated over every document.",
    "PPMI's marginals come from this table only: P(w) is this word's share of every pair total IN THIS TABLE, "
    "not its share of every word in the corpus. A table produced with --min-count above 1 has already dropped "
    "rare observations, which understates every marginal a little (PANEL_TRUNCATED_MARGINALS says when this "
    "table's floor is above 1).",
    "Total is a raw count pooled across documents; it is not divided by document length or corpus size, so it "
    "mixes 'this pair is common' with 'this corpus has a lot of sentences'.",
    "A window co-occurrence is not a phrase: the two words may never sit adjacent to each other, so this figure "
    "offers no text lookup for a pair, only the row count in the published table.",
    "Hiding function words removes a pair if EITHER word is one ('of freedom' is hidden, not just 'of the'); "
    "on the real State of the Union table this is what it takes to clear 'of'/'the'/'be' off the top of the plot.",
)
_PARTNERS_NOTES: tuple[str, ...] = (
    "Bars are ranked by PPMI but sized by evidence (co-occurring sentences), on purpose: a pair with high PPMI "
    "and one sentence behind it is a different claim from one with high PPMI and a hundred.",
    *_ASSOCIATION_NOTES[:2],
)
_NETWORK_NOTES: tuple[str, ...] = (
    "Function words are always hidden here, not just by default: a network draws a small, curated set of edges, "
    "and a function word would sit in the middle touching nearly everything, crowding out real relationships.",
    "Node position comes from a force-directed layout over the drawn edges only; distance on the page is not a "
    "measured quantity, only 'strongly connected things pull together'.",
    *_ASSOCIATION_NOTES[1:4],
)

NGRAM_COOCCURRENCE_ASSOCIATION = PanelDefinition(
    name="ngram_cooccurrence_association",
    title="Co-occurrence association",
    question="Which word pairs co-occur most, and how strongly are they associated once frequency is accounted for?",
    tool="ngram_cooccurrence",
    shape="scatter_labelled",
    summary="PPMI against pooled co-occurring-sentence evidence, one point per unordered word pair.",
    requires=(WORD_1, WORD_2, COUNT, DOC_ID),
    params=(
        PanelParam(
            name="min-total",
            type="int",
            default=10,
            minimum=1,
            maximum=1_000_000,
            label="Minimum total sentences",
            help="Leave out pairs co-occurring fewer than this many sentences in total across the corpus.",
        ),
        PanelParam(
            name="hide-function-words",
            type="bool",
            default=True,
            label="Hide function-word pairs",
            help="Drop a pair if either word is a function word ('of', 'the', 'be', ...).",
        ),
        PanelParam(
            name="label",
            type="int",
            default=15,
            minimum=0,
            maximum=100,
            label="Pairs to label",
            help="How many pairs get their text written on the plot, ranked by PPMI times log evidence (strong and well attested). 0 labels none.",
        ),
    ),
    build=ngram_cooccurrence_association,
    notes=_ASSOCIATION_NOTES,
)

NGRAM_COOCCURRENCE_PARTNERS = PanelDefinition(
    name="ngram_cooccurrence_partners",
    title="Co-occurrence partners",
    question="Which words co-occur with this one, and how strongly?",
    tool="ngram_cooccurrence",
    shape="ranked_bars",
    summary="One word's co-occurrence partners, ranked by PPMI.",
    requires=(WORD_1, WORD_2, COUNT, DOC_ID),
    params=(
        PanelParam(
            name="word",
            type="str",
            default="",
            label="Focus word",
            help="The word whose partners to show. Left blank, it is the content word carrying the most total co-occurrence mass.",
        ),
        PanelParam(
            name="top-n",
            type="int",
            default=25,
            minimum=1,
            maximum=200,
            label="Partners to show",
            help="How many partners to draw, ranked by PPMI.",
        ),
        PanelParam(
            name="min-total",
            type="int",
            default=10,
            minimum=1,
            maximum=1_000_000,
            label="Minimum total sentences",
            help="Leave out partners sharing fewer than this many sentences with the word. PPMI rewards rare "
            "pairs, so without a floor the top partners are words seen once.",
        ),
        PanelParam(
            name="hide-function-words",
            type="bool",
            default=True,
            label="Hide function-word partners",
            help="Drop partners that are function words ('of', 'the', 'be', ...).",
        ),
    ),
    build=ngram_cooccurrence_partners,
    notes=_PARTNERS_NOTES,
)

NGRAM_COOCCURRENCE_NETWORK = PanelDefinition(
    name="ngram_cooccurrence_network",
    title="Co-occurrence network",
    question="Which words cluster together across the corpus?",
    tool="ngram_cooccurrence",
    shape="network",
    summary="The corpus's most connected content words, each linked to its strongest partners among them.",
    requires=(WORD_1, WORD_2, COUNT, DOC_ID),
    params=(
        PanelParam(
            name="top-n",
            type="int",
            default=30,
            minimum=5,
            maximum=120,
            label="Words to draw",
            help="How many content words to draw: those with the most co-occurring sentences, after removing "
            "function words and pairs below min-total.",
        ),
        PanelParam(
            name="links",
            type="int",
            default=3,
            minimum=1,
            maximum=10,
            label="Links per word",
            help="Each word keeps its strongest links (by PPMI) to the other drawn words.",
        ),
        PanelParam(
            name="min-total",
            type="int",
            default=10,
            minimum=1,
            maximum=1_000_000,
            label="Minimum total sentences",
            help="Leave out pairs co-occurring fewer than this many sentences in total before ranking.",
        ),
        PanelParam(
            name="edge-weight",
            type="choice",
            default="ppmi",
            choices=("ppmi", "count"),
            label="Edge weight",
            help="What an edge's thickness and layout pull represent: 'ppmi' (association strength) or 'count' (co-occurring sentences).",
        ),
    ),
    build=ngram_cooccurrence_network,
    notes=_NETWORK_NOTES,
)


# -------------------------------------------------------------- doc_similarity --


def _heatmap_marks(  # noqa: PLR0913, PLR0917 - both pair heatmaps share this; each argument is one real difference
    working: pd.DataFrame,
    a_col: str,
    b_col: str,
    value_col: str,
    ordered: list[str],
    labels: dict[str, str],
    describe: Any,
) -> list[PanelMark]:
    """Off-diagonal cells of a symmetric document x document heatmap.

    Looks up each cell's ORIGINAL row (whichever of A/B order it was stored
    in) so evidence filters can name that row's own columns and resolve to
    exactly it -- the mirrored-cell (j, i) reads the same underlying row as
    (i, j), which is correct: they are one measured relationship.
    """
    lookup: dict[frozenset[str], pd.Series] = {}
    for _, row in working.iterrows():
        lookup[frozenset((str(row[a_col]), str(row[b_col])))] = row
    index = {name: position for position, name in enumerate(ordered)}
    marks: list[PanelMark] = []
    for row_name in ordered:
        for col_name in ordered:
            if row_name == col_name:
                continue
            row = lookup.get(frozenset((row_name, col_name)))
            if row is None:
                continue
            value = float(row[value_col])
            marks.append(
                PanelMark(
                    key=f"{index[row_name]}:{index[col_name]}",
                    label=f"{labels[row_name]} / {labels[col_name]}",
                    x=float(index[col_name]),
                    y=float(index[row_name]),
                    value=value,
                    evidence=Evidence(
                        scope="documents",
                        filters=((a_col, str(row[a_col])), (b_col, str(row[b_col]))),
                        count=1,
                        describe=describe(labels[row_name], labels[col_name], row),
                    ),
                )
            )
    return marks


def doc_similarity_heatmap(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance
) -> Result[PreparedPanel]:
    """Document x document TF-IDF cosine similarity, ordered by date."""
    _ = params
    for column in (DOC_A, DOC_B, SIMILARITY, BAND):
        if column not in frame.columns:
            return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"the similarity table has no {column!r}"))
    working = frame.copy()
    working[DOC_A] = working[DOC_A].astype(str)
    working[DOC_B] = working[DOC_B].astype(str)
    working[SIMILARITY] = pd.to_numeric(working[SIMILARITY], errors="coerce")
    working = working[working[SIMILARITY].notna()]
    names = sorted({*working[DOC_A], *working[DOC_B]})
    if len(names) < 2:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "fewer than 2 documents have a similarity score"))

    diagnostics: list[Diagnostic] = []
    ordered, dated = _document_order(names)
    if len(ordered) > _MAX_HEATMAP_DOCS:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_TOO_MANY_DOCS",
                f"{len(ordered)} documents is more than a readable heatmap holds; showing the first "
                f"{_MAX_HEATMAP_DOCS} in {'date' if dated else 'name'} order.",
                drawn=_MAX_HEATMAP_DOCS,
                available=len(ordered),
            )
        )
        ordered = ordered[:_MAX_HEATMAP_DOCS]
    kept = set(ordered)
    working = working[working[DOC_A].isin(kept) & working[DOC_B].isin(kept)]
    labels = document_labels(ordered)
    categories = tuple(labels[name] for name in ordered)

    def describe(row_label: str, col_label: str, row: pd.Series) -> str:
        return f"{row_label} vs {col_label}: {float(row[SIMILARITY]):.1f}% similarity ({row[BAND]})"

    marks = _heatmap_marks(working, DOC_A, DOC_B, SIMILARITY, ordered, labels, describe)
    prepared = PreparedPanel(
        panel=DOC_SIMILARITY_HEATMAP.name,
        shape="heatmap",
        title="Document similarity",
        subtitle=f"{len(ordered)} documents, ordered by {'date' if dated else 'name'}; diagonal left blank",
        marks=tuple(marks),
        x_label="Document",
        y_label="Document",
        provenance=provenance,
        data=working.reset_index(drop=True),
        color_scale="sequential",
        value_label="Similarity (%)",
        x_categories=categories,
        y_categories=categories,
        notes=_SIMILARITY_HEATMAP_NOTES,
    )
    return Result.success(prepared, *diagnostics)


def _neighbours_network(  # noqa: PLR0913 - shared by the similarity and SVO networks; keyword-only
    working: pd.DataFrame,
    *,
    panel_name: str,
    a_col: str,
    b_col: str,
    value_col: str,
    k: int,
    labels: dict[str, str],
    node_group: dict[str, str],
    groups: tuple[str, ...],
    title: str,
    value_note: str,
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Shared k-nearest-neighbour network builder for a symmetric pair table.

    Used by both ``doc_similarity_neighbours`` and ``svo_compare_neighbours``:
    same shape of problem (an unordered pair table, draw each item's
    strongest few links), different column names and, for svo_compare, no
    grouping (no dates or speakers survive into that table).
    """
    mirrored = _mirror(working, a_col, b_col, (value_col,))
    names = sorted({*working[a_col], *working[b_col]})
    edges_seen: dict[frozenset[str], tuple[str, str, float]] = {}
    for name, group_rows in mirrored.groupby("Word", sort=False):
        top = group_rows.sort_values([value_col, "Partner"], ascending=[False, True]).head(k)
        for record in top.to_dict("records"):
            other = str(record["Partner"])
            key = frozenset((name, other))
            value = float(record[value_col])
            existing = edges_seen.get(key)
            if existing is None or value > existing[2]:
                edges_seen[key] = (str(name), other, value)

    if not edges_seen:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"no neighbour links at k={k}"))

    layout = force_layout(names, list(edges_seen.values()))
    marks: list[PanelMark] = []
    for name in names:
        x, y = layout[name]
        own = mirrored[mirrored["Word"] == name]
        if own.empty:
            nearest = "n/a"
        else:
            best = own.sort_values(value_col, ascending=False).iloc[0]
            nearest = f"{labels.get(str(best['Partner']), best['Partner'])} ({float(best[value_col]):.1f})"
        marks.append(
            PanelMark(
                key=name,
                label=labels.get(name, name),
                x=x,
                y=y,
                size=float(1 + sum(1 for a, b, _ in edges_seen.values() if name in (a, b))),
                group=node_group.get(name, ""),
                evidence=Evidence(
                    scope="documents",
                    filters=(("Word", name),),
                    count=len(own),
                    describe=f"{labels.get(name, name)}: compared with {len(own)} other document(s); nearest is {nearest}",
                ),
            )
        )
    edges: list[PanelEdge] = []
    for a, b, value in edges_seen.values():
        edges.append(
            PanelEdge(
                key=f"{a}::{b}",
                source=a,
                target=b,
                weight=value,
                evidence=Evidence(
                    scope="documents",
                    filters=(("Word", a), ("Partner", b)),
                    count=1,
                    describe=f"{labels.get(a, a)} & {labels.get(b, b)}: {value:.1f} {value_note}",
                ),
            )
        )

    prepared = PreparedPanel(
        panel=panel_name,
        shape="network",
        edges=tuple(edges),
        title=title,
        subtitle=f"{len(names)} documents, top {k} neighbour(s) each -- {len(edges)} unique links",
        marks=tuple(marks),
        x_label="",
        y_label="",
        provenance=provenance,
        data=mirrored.reset_index(drop=True),
        groups=groups,
        notes=_NEIGHBOURS_NOTES,
    )
    return Result.success(prepared)


def doc_similarity_neighbours(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance
) -> Result[PreparedPanel]:
    """Each document linked to its top-k most similar, grouped by era or speaker."""
    k = int(params["k"])
    group_by = str(params["group"])
    for column in (DOC_A, DOC_B, SIMILARITY):
        if column not in frame.columns:
            return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"the similarity table has no {column!r}"))
    working = frame.copy()
    working[DOC_A] = working[DOC_A].astype(str)
    working[DOC_B] = working[DOC_B].astype(str)
    working[SIMILARITY] = pd.to_numeric(working[SIMILARITY], errors="coerce")
    working = working[working[SIMILARITY].notna()]
    names = sorted({*working[DOC_A], *working[DOC_B]})
    if len(names) < 2:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "fewer than 2 documents have a similarity score"))

    labels = document_labels(names)
    if group_by == "none":
        node_group = {name: "All documents" for name in names}
        groups: tuple[str, ...] = ("All documents",)
    else:
        node_group = {name: group_of(name, _doc_date(name), group_by) for name in names}
        groups = _group_draw_order(set(node_group.values()))

    result = _neighbours_network(
        working,
        panel_name=DOC_SIMILARITY_NEIGHBOURS.name,
        a_col=DOC_A,
        b_col=DOC_B,
        value_col=SIMILARITY,
        k=k,
        labels=labels,
        node_group=node_group,
        groups=groups,
        title=f"Nearest {k} neighbour(s) per document",
        value_note="% similarity",
        provenance=provenance,
    )
    return result


def _group_draw_order(values: set[str]) -> tuple[str, ...]:
    """Chronological (or alphabetical) order, with an 'unparsed' bucket last."""

    def key(group: str) -> tuple[int, str]:
        if group in ("(undated)", "(speaker not in file name)"):
            return (1, group)
        return (0, group)

    return tuple(sorted(values, key=key))


_SIMILARITY_HEATMAP_NOTES: tuple[str, ...] = (
    "The diagonal is left blank, not filled with 1.0: comparing a document to itself is not a measured "
    "similarity, and drawing it would imply otherwise.",
    "Similarity is TF-IDF cosine similarity (0-100%) over the whole document, ignoring word order -- two "
    "speeches sharing the same stock phrases and topics score high even if their arguments differ.",
    "Documents are ordered by the date in their file name when every document in the run has one; otherwise "
    "by name. The order used is stated in the subtitle.",
)
_NEIGHBOURS_NOTES: tuple[str, ...] = (
    "An edge is drawn if EITHER endpoint counts the other among its top-k, so a document can end up with more "
    "than k edges when others chose it back.",
    "Node position comes from a force-directed layout over the drawn edges only; distance on the page is not a "
    "measured quantity.",
)

DOC_SIMILARITY_HEATMAP = PanelDefinition(
    name="doc_similarity_heatmap",
    title="Document similarity heatmap",
    question="Which documents read most alike?",
    tool="doc_similarity",
    shape="heatmap",
    summary="Document x document TF-IDF cosine similarity, ordered by date.",
    requires=(DOC_A, DOC_B, SIMILARITY, BAND),
    params=(),
    build=doc_similarity_heatmap,
    notes=_SIMILARITY_HEATMAP_NOTES,
)

DOC_SIMILARITY_NEIGHBOURS = PanelDefinition(
    name="doc_similarity_neighbours",
    title="Document similarity neighbours",
    question="Which documents cluster together, and do eras or speakers stand out?",
    tool="doc_similarity",
    shape="network",
    summary="Each document linked to its top-k most similar, grouped by decade or speaker.",
    requires=(DOC_A, DOC_B, SIMILARITY),
    params=(
        PanelParam(
            name="k",
            type="int",
            default=2,
            minimum=1,
            maximum=10,
            label="Neighbours per document",
            help="How many of each document's most-similar documents get an edge.",
        ),
        PanelParam(
            name="group",
            type="choice",
            default="decade",
            choices=("decade", "speaker", "none"),
            label="Group nodes by",
            help="Colour documents by the decade or speaker parsed from their file name, or leave them ungrouped.",
        ),
    ),
    build=doc_similarity_neighbours,
    notes=_NEIGHBOURS_NOTES,
)


# ----------------------------------------------------------------- svo_compare --

_SVO_MEASURES: dict[str, str] = {
    "subject": SUBJECT_JACCARD,
    "verb": VERB_JACCARD,
    "object": OBJECT_JACCARD,
    "triple": TRIPLE_JACCARD,
}
_SVO_MEASURE_LABEL: dict[str, str] = {
    "subject": "Subject Jaccard",
    "verb": "Verb Jaccard",
    "object": "Object Jaccard",
    "triple": "Triple Jaccard",
}


def svo_compare_heatmap(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance
) -> Result[PreparedPanel]:
    """Document x document SVO-overlap heatmap (param: which Jaccard)."""
    measure = str(params["measure"])
    column = _SVO_MEASURES[measure]
    for needed in (SVO_DOC_A, SVO_DOC_B, column, COMMON_TRIPLES):
        if needed not in frame.columns:
            return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"the svo_compare table has no {needed!r}"))
    working = frame.copy()
    working[SVO_DOC_A] = working[SVO_DOC_A].astype(str)
    working[SVO_DOC_B] = working[SVO_DOC_B].astype(str)
    working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working[working[column].notna()]
    names = sorted({*working[SVO_DOC_A], *working[SVO_DOC_B]}, key=_numeric_or_text)
    if len(names) < 2:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "fewer than 2 documents have a score"))

    diagnostics: list[Diagnostic] = []
    if len(names) > _MAX_HEATMAP_DOCS:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_TOO_MANY_DOCS",
                f"{len(names)} documents is more than a readable heatmap holds; showing the first {_MAX_HEATMAP_DOCS}.",
                drawn=_MAX_HEATMAP_DOCS,
                available=len(names),
            )
        )
        names = names[:_MAX_HEATMAP_DOCS]
    kept = set(names)
    working = working[working[SVO_DOC_A].isin(kept) & working[SVO_DOC_B].isin(kept)]
    labels = _svo_labels(working, names)
    categories = tuple(labels[name] for name in names)

    def describe(row_label: str, col_label: str, row: pd.Series) -> str:
        return (
            f"{row_label} vs {col_label}: {float(row[column]):.3f} {_SVO_MEASURE_LABEL[measure]} "
            f"({int(row[COMMON_TRIPLES])} common triples)"
        )

    marks = _heatmap_marks(working, SVO_DOC_A, SVO_DOC_B, column, names, labels, describe)
    prepared = PreparedPanel(
        panel=SVO_COMPARE_HEATMAP.name,
        shape="heatmap",
        title="SVO overlap between documents",
        subtitle=f"{len(names)} documents -- {_SVO_MEASURE_LABEL[measure]}; diagonal left blank",
        marks=tuple(marks),
        x_label="Document",
        y_label="Document",
        provenance=provenance,
        data=working.reset_index(drop=True),
        color_scale="sequential",
        value_label=_SVO_MEASURE_LABEL[measure],
        x_categories=categories,
        y_categories=categories,
        notes=_SVO_HEATMAP_NOTES,
    )
    return Result.success(prepared, *diagnostics)


def svo_compare_neighbours(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance
) -> Result[PreparedPanel]:
    """Each document linked to its top-k most SVO-similar documents."""
    k = int(params["k"])
    measure = str(params["measure"])
    column = _SVO_MEASURES[measure]
    for needed in (SVO_DOC_A, SVO_DOC_B, column):
        if needed not in frame.columns:
            return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"the svo_compare table has no {needed!r}"))
    working = frame.copy()
    working[SVO_DOC_A] = working[SVO_DOC_A].astype(str)
    working[SVO_DOC_B] = working[SVO_DOC_B].astype(str)
    working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working[working[column].notna()]
    names = sorted({*working[SVO_DOC_A], *working[SVO_DOC_B]}, key=_numeric_or_text)
    if len(names) < 2:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "fewer than 2 documents have a score"))
    labels = _svo_labels(working, names)

    result = _neighbours_network(
        working,
        panel_name=SVO_COMPARE_NEIGHBOURS.name,
        a_col=SVO_DOC_A,
        b_col=SVO_DOC_B,
        value_col=column,
        k=k,
        labels=labels,
        node_group={},
        groups=(),
        title=f"Nearest {k} SVO-similar document(s) each",
        value_note=_SVO_MEASURE_LABEL[measure],
        provenance=provenance,
    )
    return result


_SVO_HEATMAP_NOTES: tuple[str, ...] = (
    "svo_compare carries no document name or date, only the bare Document ID this run assigned; labels here are "
    "'Doc <id>' and documents are ordered numerically, not by date.",
    "Jaccard overlap counts distinct subjects, verbs, objects or full triples shared between two documents' SVO "
    "sets; it says nothing about how often each was used, only whether it appears at all in both.",
    "The diagonal is left blank: comparing a document to itself is not a measured overlap.",
)

SVO_COMPARE_HEATMAP = PanelDefinition(
    name="svo_compare_heatmap",
    title="SVO overlap heatmap",
    question="Which documents share the most subject-verb-object structure?",
    tool="svo_compare",
    shape="heatmap",
    summary="Document x document Jaccard overlap of SVO triples (subject, verb, object, or full triple).",
    requires=(SVO_DOC_A, SVO_DOC_B, COMMON_TRIPLES, SUBJECT_JACCARD, VERB_JACCARD, OBJECT_JACCARD, TRIPLE_JACCARD),
    params=(
        PanelParam(
            name="measure",
            type="choice",
            default="triple",
            choices=("subject", "verb", "object", "triple"),
            label="Overlap measure",
            help="Which Jaccard overlap to plot: shared subjects, verbs, objects, or full (subject, verb, object) triples.",
        ),
    ),
    build=svo_compare_heatmap,
    notes=_SVO_HEATMAP_NOTES,
)

SVO_COMPARE_NEIGHBOURS = PanelDefinition(
    name="svo_compare_neighbours",
    title="SVO overlap neighbours",
    question="Which documents share the most SVO structure with each other?",
    tool="svo_compare",
    shape="network",
    summary="Each document linked to its top-k most SVO-similar documents.",
    requires=(SVO_DOC_A, SVO_DOC_B, SUBJECT_JACCARD, VERB_JACCARD, OBJECT_JACCARD, TRIPLE_JACCARD),
    params=(
        PanelParam(
            name="k",
            type="int",
            default=2,
            minimum=1,
            maximum=10,
            label="Neighbours per document",
            help="How many of each document's most-overlapping documents get an edge.",
        ),
        PanelParam(
            name="measure",
            type="choice",
            default="triple",
            choices=("subject", "verb", "object", "triple"),
            label="Overlap measure",
            help="Which Jaccard overlap ranks and weighs the edges.",
        ),
    ),
    build=svo_compare_neighbours,
    notes=_SVO_HEATMAP_NOTES,
)


# --------------------------------------------------------------- collocations --

_COLLOCATION_MEASURES: dict[str, str] = {"pmi": PMI, "log-dice": LOG_DICE, "t-score": T_SCORE, "g2": G2}


def collocations_network(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance
) -> Result[PreparedPanel]:
    """The strongest collocations as a word network (companion to the scatter)."""
    measure = str(params["measure"])
    top_n = int(params["top-n"])
    min_count = int(params["min-count"])
    edge_weight = str(params["edge-weight"])
    column = _COLLOCATION_MEASURES[measure]
    for needed in (COLL_WORD_1, COLL_WORD_2, COOCCURRENCES, column):
        if needed not in frame.columns:
            return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"the collocations table has no {needed!r}"))

    working = frame.copy()
    working[COLL_WORD_1] = working[COLL_WORD_1].astype(str)
    working[COLL_WORD_2] = working[COLL_WORD_2].astype(str)
    working[COOCCURRENCES] = pd.to_numeric(working[COOCCURRENCES], errors="coerce")
    working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working[working[COOCCURRENCES].notna() & working[column].map(_finite)]

    diagnostics: list[Diagnostic] = []
    if min_count:
        before = len(working)
        working = working[working[COOCCURRENCES] >= min_count]
        if before - len(working):
            diagnostics.append(
                Diagnostic.info(
                    "PANEL_FREQUENCY_FILTERED",
                    f"{before - len(working)} pair(s) below min-count={min_count} were left out.",
                    removed=before - len(working),
                )
            )
    if working.empty:
        return Result[PreparedPanel].failure(
            Diagnostic.error("PANEL_NO_DATA", "no pairs left after min-count filtering"), *diagnostics
        )

    if bool(params.get("hide-function-words", True)):
        # The collocations table keeps function words, and as hubs they took
        # over the real corpus's network ("the", "of", "in", "be" at the
        # centre of everything); "united states" survives the filter.
        before = len(working)
        working = working[~working[COLL_WORD_1].map(is_function_word) & ~working[COLL_WORD_2].map(is_function_word)]
        if before - len(working):
            diagnostics.append(
                Diagnostic.info(
                    "PANEL_FUNCTION_WORDS_HIDDEN",
                    f"{before - len(working)} collocation(s) with a function word on either side were left out.",
                    removed=before - len(working),
                )
            )
    links = int(params.get("links", 3))
    top, qualified = _hub_pairs(
        working, a=COLL_WORD_1, b=COLL_WORD_2, mass=COOCCURRENCES, strength=column, words=top_n, links=links
    )
    if top.empty:
        return Result[PreparedPanel].failure(
            Diagnostic.error("PANEL_NO_DATA", "the most connected words share no collocations; raise top-n"),
            *diagnostics,
        )
    if qualified > top_n:
        diagnostics.append(
            Diagnostic.info(
                "PANEL_TOO_MANY_MARKS",
                f"{qualified} words qualified; drew the {top_n} in the most collocations, each with up to "
                f"{links} link(s) by {measure}. Raise top-n to see more.",
                available=qualified,
                drawn=top_n,
            )
        )

    mirrored = _mirror(top, COLL_WORD_1, COLL_WORD_2, (COOCCURRENCES, column))
    nodes = sorted({*top[COLL_WORD_1], *top[COLL_WORD_2]})
    edges_raw = [
        (
            str(record[COLL_WORD_1]),
            str(record[COLL_WORD_2]),
            float(record[column] if edge_weight == "measure" else record[COOCCURRENCES]),
            int(record[COOCCURRENCES]),
            float(record[column]),
        )
        for record in top.to_dict("records")
    ]
    layout = force_layout(nodes, [(w1, w2, weight) for w1, w2, weight, *_ in edges_raw])
    mass = mirrored.groupby("Word", sort=False)[COOCCURRENCES].sum()
    community = communities(nodes, [(w1, w2, weight) for w1, w2, weight, *_ in edges_raw], mass.to_dict())

    marks: list[PanelMark] = []
    for word in nodes:
        x, y = layout[word]
        own = mirrored[mirrored["Word"] == word]
        marks.append(
            PanelMark(
                key=word,
                label=word,
                x=x,
                y=y,
                size=float(mass.get(word, 0.0)),
                labelled=True,
                group=community[word],
                evidence=Evidence(
                    scope="terms",
                    filters=(("Word", word),),
                    count=len(own),
                    describe=f"{word}: {len(own)} partner(s) drawn, {int(mass.get(word, 0.0))} co-occurrences",
                ),
            )
        )
    edges: list[PanelEdge] = []
    for w1, w2, weight, cooccurrences, value in edges_raw:
        edges.append(
            PanelEdge(
                key=f"{w1}::{w2}",
                source=w1,
                target=w2,
                weight=weight,
                evidence=Evidence(
                    scope="terms",
                    filters=(("Word", w1), ("Partner", w2)),
                    count=cooccurrences,
                    describe=f"{w1} & {w2}: {cooccurrences} co-occurrences, {measure} {value:.2f}",
                ),
            )
        )

    prepared = PreparedPanel(
        panel=COLLOCATIONS_NETWORK.name,
        shape="network",
        edges=tuple(edges),
        title="Collocation network",
        subtitle=f"{len(nodes)} most connected words, {len(edges)} links by {measure}, edge weight = {edge_weight}",
        marks=tuple(marks),
        x_label="",
        y_label="",
        provenance=provenance,
        data=mirrored.reset_index(drop=True),
        groups=community_order(community, mass.to_dict()),
        notes=(*_COLLOCATIONS_NETWORK_NOTES, _COMMUNITY_NOTE),
    )
    return Result.success(prepared, *diagnostics)


_COLLOCATIONS_NETWORK_NOTES: tuple[str, ...] = (
    "Node position comes from a force-directed layout over the drawn edges only; distance on the page is not a "
    "measured quantity.",
    "PMI rewards rare pairs: a low min-count with measure=pmi can fill the network with pairs seen only a "
    "couple of times. Raise min-count, or use log-dice, to draw well-attested pairs instead.",
    "This is a companion to the collocation-strength scatter (collocation_strength): that figure shows every "
    "pair and lets association trade off against frequency; this one curates the strongest few into a picture "
    "of which words cluster together.",
)

COLLOCATIONS_NETWORK = PanelDefinition(
    name="collocations_network",
    title="Collocation network",
    question="Which words cluster together by collocation strength?",
    tool="collocations",
    shape="network",
    summary="The strongest collocations as a word network, links weighted by association or by count.",
    requires=(COLL_WORD_1, COLL_WORD_2, COOCCURRENCES, PMI, T_SCORE, G2, LOG_DICE),
    params=(
        PanelParam(
            name="measure",
            type="choice",
            default="log-dice",
            choices=("pmi", "log-dice", "t-score", "g2"),
            label="Association measure",
            help="Which measure ranks the strongest pairs and, if edge-weight=measure, sets edge weight.",
        ),
        PanelParam(
            name="top-n",
            type="int",
            default=30,
            minimum=5,
            maximum=300,
            label="Words to draw",
            help="How many words to draw: those in the most collocations.",
        ),
        PanelParam(
            name="links",
            type="int",
            default=3,
            minimum=1,
            maximum=10,
            label="Links per word",
            help="Each word keeps its strongest collocations (by the chosen measure) with the other drawn words.",
        ),
        PanelParam(
            name="hide-function-words",
            type="bool",
            default=True,
            label="Hide function words",
            help="Drop collocations with a function word on either side ('of the', 'we must').",
        ),
        PanelParam(
            name="min-count",
            type="int",
            default=2,
            minimum=1,
            maximum=1_000_000,
            label="Minimum co-occurrences",
            help="Leave out pairs occurring fewer than this many times before ranking.",
        ),
        PanelParam(
            name="edge-weight",
            type="choice",
            default="measure",
            choices=("measure", "count"),
            label="Edge weight",
            help="What an edge's thickness and layout pull represent: the chosen association 'measure' or raw 'count'.",
        ),
    ),
    build=collocations_network,
    notes=_COLLOCATIONS_NETWORK_NOTES,
)


# ------------------------------------------------------------- doc_duplicates --


def doc_duplicates_graph(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance
) -> Result[PreparedPanel]:
    """Fuzzy-matched near-duplicate documents as a graph; refuses politely when empty."""
    _ = params
    for column in (DOC_A, DOC_B, SIMILARITY):
        if column not in frame.columns:
            return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"the fuzzy-match table has no {column!r}"))
    working = frame.copy()
    if working.empty:
        return Result.failure(
            Diagnostic.error(
                "PANEL_NO_DUPLICATES",
                "no near-duplicate document pairs were found at this run's fuzzy-match threshold -- nothing to "
                "draw. Lower --threshold on the doc_duplicates run to look for looser matches.",
            )
        )
    working[DOC_A] = working[DOC_A].astype(str)
    working[DOC_B] = working[DOC_B].astype(str)
    working[SIMILARITY] = pd.to_numeric(working[SIMILARITY], errors="coerce")
    working = working[working[SIMILARITY].notna()]
    if working.empty:
        return Result.failure(
            Diagnostic.error(
                "PANEL_NO_DUPLICATES",
                "no near-duplicate document pairs were found at this run's fuzzy-match threshold -- nothing to "
                "draw. Lower --threshold on the doc_duplicates run to look for looser matches.",
            )
        )

    names = sorted({*working[DOC_A], *working[DOC_B]})
    labels = document_labels(names)
    mirrored = _mirror(working, DOC_A, DOC_B, (SIMILARITY,))
    degree: dict[str, int] = dict.fromkeys(names, 0)
    edges_raw: list[tuple[str, str, float]] = []
    for record in working.to_dict("records"):
        a, b, sim = str(record[DOC_A]), str(record[DOC_B]), float(record[SIMILARITY])
        degree[a] += 1
        degree[b] += 1
        edges_raw.append((a, b, sim))
    layout = force_layout(names, edges_raw)

    marks: list[PanelMark] = []
    for name in names:
        x, y = layout[name]
        marks.append(
            PanelMark(
                key=name,
                label=labels[name],
                x=x,
                y=y,
                size=float(1 + degree[name]),
                evidence=Evidence(
                    scope="documents",
                    filters=(("Word", name),),
                    count=degree[name],
                    describe=f"{labels[name]}: {degree[name]} near-duplicate match(es)",
                ),
            )
        )
    edges: list[PanelEdge] = []
    for a, b, sim in edges_raw:
        edges.append(
            PanelEdge(
                key=f"{a}::{b}",
                source=a,
                target=b,
                weight=sim,
                evidence=Evidence(
                    scope="documents",
                    filters=(("Word", a), ("Partner", b)),
                    count=1,
                    describe=f"{labels[a]} & {labels[b]}: {sim:.1f}% fuzzy match",
                ),
            )
        )

    prepared = PreparedPanel(
        panel=DOC_DUPLICATES_GRAPH.name,
        shape="network",
        edges=tuple(edges),
        title="Near-duplicate documents",
        subtitle=f"{len(names)} document(s), {len(edges)} fuzzy match(es)",
        marks=tuple(marks),
        x_label="",
        y_label="",
        provenance=provenance,
        data=mirrored.reset_index(drop=True),
        notes=_DUPLICATES_NOTES,
    )
    return Result.success(prepared)


_DUPLICATES_NOTES: tuple[str, ...] = (
    "An edge means the fuzzy-match threshold this run used was met; a lower threshold on the doc_duplicates run "
    "would draw more edges (or none, at the real corpus's default threshold, which is why this panel refuses "
    "rather than drawing an empty graph).",
    "Fuzzy matching compares normalised text (case, whitespace, punctuation collapsed); it will not catch two "
    "documents that share content but differ in wording.",
)

DOC_DUPLICATES_GRAPH = PanelDefinition(
    name="doc_duplicates_graph",
    title="Near-duplicate document graph",
    question="Which documents are likely near-duplicates of each other?",
    tool="doc_duplicates",
    shape="network",
    summary="Documents linked by a fuzzy-match near-duplicate pair.",
    requires=(DOC_A, DOC_B, SIMILARITY),
    params=(),
    build=doc_duplicates_graph,
    notes=_DUPLICATES_NOTES,
)


# ------------------------------------------------------------- knowledge_graph --

_MIN_KG_EDGES = 2


def knowledge_graph_network(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance
) -> Result[PreparedPanel]:
    """Subject-predicate-object triples as an entity network; refuses when sparse."""
    _ = params
    for column in (SUBJECT, PREDICATE, OBJECT, SOURCE):
        if column not in frame.columns:
            return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"the knowledge-graph table has no {column!r}"))
    working = frame.copy()
    for column in (SUBJECT, PREDICATE, OBJECT, SOURCE):
        working[column] = working[column].astype(str)
    working = working[working[SUBJECT].str.strip().ne("") & working[OBJECT].str.strip().ne("")]

    if len(working) < _MIN_KG_EDGES:
        return Result.failure(
            Diagnostic.error(
                "PANEL_TOO_FEW_TRIPLES",
                f"only {len(working)} triple(s) matched an entity in the offline stub knowledge base, which "
                "recognises a small fixed set of names (Paris, London, Obama, Einstein, Google, ...) -- too few "
                "to draw a network. Run knowledge_graph with --source dbpedia for live lookups outside batch "
                "mode, or check that the corpus mentions any of the stub's known entities.",
                triples=len(working),
            )
        )

    mirrored = pd.concat(
        [
            working.assign(Entity=working[SUBJECT], Role="subject", Other=working[OBJECT])[
                ["Entity", "Role", "Other", PREDICATE, SOURCE]
            ],
            working.assign(Entity=working[OBJECT], Role="object", Other=working[SUBJECT])[
                ["Entity", "Role", "Other", PREDICATE, SOURCE]
            ],
        ],
        ignore_index=True,
    )
    nodes = sorted({*working[SUBJECT], *working[OBJECT]})
    degree: dict[str, int] = dict.fromkeys(nodes, 0)
    edges_raw: list[tuple[str, str, str, str]] = []
    for record in working.to_dict("records"):
        subject, predicate, obj, source = (
            str(record[SUBJECT]),
            str(record[PREDICATE]),
            str(record[OBJECT]),
            str(record[SOURCE]),
        )
        if subject == obj:
            continue  # PanelEdge forbids a self-loop; the stub KB never emits one, but stay defensive
        degree[subject] += 1
        degree[obj] += 1
        edges_raw.append((subject, predicate, obj, source))
    layout = force_layout(nodes, [(s, o, 1.0) for s, _p, o, _src in edges_raw])

    marks: list[PanelMark] = []
    for name in nodes:
        x, y = layout[name]
        marks.append(
            PanelMark(
                key=name,
                label=name,
                x=x,
                y=y,
                size=float(1 + degree[name]),
                evidence=Evidence(
                    scope="terms",
                    filters=(("Entity", name),),
                    count=degree[name],
                    describe=f"{name}: {degree[name]} triple(s) in this graph",
                ),
            )
        )
    edges: list[PanelEdge] = []
    seen_keys: set[str] = set()
    for subject, predicate, obj, source in edges_raw:
        key = f"{subject}::{predicate}::{obj}"
        suffix = 0
        while key in seen_keys:
            suffix += 1
            key = f"{subject}::{predicate}::{obj}#{suffix}"
        seen_keys.add(key)
        edges.append(
            PanelEdge(
                key=key,
                source=subject,
                target=obj,
                weight=1.0,
                label=predicate,
                evidence=Evidence(
                    scope="terms",
                    filters=(("Entity", subject), ("Role", "subject"), ("Other", obj), (PREDICATE, predicate)),
                    count=1,
                    describe=f"{subject} {predicate} {obj} (source: {source})",
                ),
            )
        )

    prepared = PreparedPanel(
        panel=KNOWLEDGE_GRAPH_NETWORK.name,
        shape="network",
        edges=tuple(edges),
        title="Knowledge graph",
        subtitle=f"{len(nodes)} entities, {len(edges)} triple(s)",
        marks=tuple(marks),
        x_label="",
        y_label="",
        provenance=provenance,
        data=mirrored.reset_index(drop=True),
        notes=_KG_NOTES,
    )
    return Result.success(prepared)


_KG_NOTES: tuple[str, ...] = (
    "This offline build recognises only a small fixed set of entity names (an offline stub, not a live DBpedia "
    "lookup); most corpora will have few or no matches, and PANEL_TOO_FEW_TRIPLES explains why the panel "
    "refused rather than drawing an empty or near-empty graph.",
    "An edge's label is the predicate (e.g. 'isA', 'country'); direction runs subject to object, but the panel "
    "does not draw an arrowhead -- read the hover for which end is which.",
    "Node position comes from a force-directed layout over the drawn edges only; distance on the page is not a "
    "measured quantity.",
)

KNOWLEDGE_GRAPH_NETWORK = PanelDefinition(
    name="knowledge_graph_network",
    title="Knowledge graph",
    question="What entities and relations did the knowledge-graph lookup find in this corpus?",
    tool="knowledge_graph",
    shape="network",
    summary="Subject-predicate-object triples as an entity network.",
    requires=(SUBJECT, PREDICATE, OBJECT, SOURCE),
    params=(),
    build=knowledge_graph_network,
    notes=_KG_NOTES,
)


PAIRS_PANELS: tuple[PanelDefinition, ...] = (
    NGRAM_COOCCURRENCE_ASSOCIATION,
    NGRAM_COOCCURRENCE_PARTNERS,
    NGRAM_COOCCURRENCE_NETWORK,
    DOC_SIMILARITY_HEATMAP,
    DOC_SIMILARITY_NEIGHBOURS,
    SVO_COMPARE_HEATMAP,
    SVO_COMPARE_NEIGHBOURS,
    COLLOCATIONS_NETWORK,
    DOC_DUPLICATES_GRAPH,
    KNOWLEDGE_GRAPH_NETWORK,
)
