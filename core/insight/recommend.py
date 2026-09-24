"""Which charts are worth drawing from this table -- and which are not.

A tool hands back a CSV with a dozen columns, and the interface offers every
column as x against every column as y. Most of those pairings are useless and
a few are actively misleading: plot a corpus's size against its own size and
you get a flawless diagonal line that looks like a discovery. The chart is
real, the correlation is 1.0, and it means nothing, because the two axes are
the same quantity.

So this module does two jobs.

:func:`degenerate_reasons` answers "is this pairing capable of telling me
anything?" *before* it is drawn, and says why not in words the analyst can
act on. It catches the identity case, perfectly collinear columns, columns
that are one value, and category axes where every row is its own category.

:func:`recommend_charts` proposes a short ranked list of pairings that do
answer something, each carrying the question it answers. Recommendations are
built from column roles (see :mod:`core.insight.profile`) plus per-tool
knowledge for the tools whose output shape we know. Every recommendation is a
real :class:`~core.viz.chartspec.ChartSpec`, so "recommended" and "runnable"
cannot drift apart.

Nothing here decides anything for the analyst. It orders the options and
refuses to stay quiet about the ones that cannot work.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from core.insight.profile import ColumnProfile, ColumnRole, profile_frame
from core.viz.chartspec import ChartSpec

__all__ = ["Recommendation", "degenerate_reasons", "recommend_charts"]

# Above this absolute Pearson/Spearman correlation two columns carry the same
# information for charting purposes. Not a significance test: a deliberately
# blunt threshold for "this line is the definition, not a finding".
_COLLINEAR = 0.999
# Rows sampled when testing collinearity, so the check stays cheap on a large
# artifact. Deterministic: the head, never a random sample.
_COLLINEAR_SAMPLE = 5000
# Default bar-chart cut, so a recommendation on a long table stays readable.
_DEFAULT_TOP_N = 20
# Generic recommendations stop here; past four the list stops being a shortlist.
_MAX_GENERIC = 4


@dataclass(frozen=True, slots=True)
class Recommendation:
    """One chart worth drawing, and the question it answers."""

    spec: ChartSpec
    question: str
    why: str
    rank: int = 0

    def describe(self) -> str:
        """One line for a CLI list or a UI caption."""
        axes = f"{self.spec.x} vs {self.spec.y}" if self.spec.kind != "histogram" else f"{self.spec.x}"
        return f"{self.spec.kind}: {axes} -- {self.question}"


def _numeric_pair(frame: pd.DataFrame, left: str, right: str) -> tuple[pd.Series, pd.Series] | None:
    """Aligned numeric values for two columns, or None if not comparable."""
    if left not in frame.columns or right not in frame.columns:
        return None
    subset = frame[[left, right]].head(_COLLINEAR_SAMPLE).dropna()
    if subset.empty:
        return None
    a, b = subset[left], subset[right]
    if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
        return None
    return a, b


def degenerate_reasons(frame: pd.DataFrame, x: str, y: str) -> list[str]:
    """Why charting *x* against *y* cannot be informative. Empty means fine.

    Each reason is written to be read by the person about to draw the chart,
    and says what the chart would show instead of only that it is bad.
    """
    reasons: list[str] = []
    if x == y:
        return [
            f"{x!r} is on both axes, so the chart can only be a straight diagonal line. "
            "That line is the definition of the axes, not a result."
        ]
    profiles = profile_frame(frame)
    for axis, name in (("x", x), ("y", y)):
        column = profiles.get(name)
        if column is None:
            reasons.append(f"{name!r} is not a column in this table")
            continue
        if column.role is ColumnRole.EMPTY:
            reasons.append(f"{name!r} ({axis}) has no values at all")
        elif column.role is ColumnRole.CONSTANT:
            reasons.append(
                f"{name!r} ({axis}) has one value in every row, so it cannot vary and nothing can be compared"
            )
        elif column.role is ColumnRole.TEXT:
            reasons.append(f"{name!r} ({axis}) is free text, which has no natural order or scale to plot against")

    # A *numeric* identifier on x plots against row position, which carries no
    # meaning. A *text* identifier is a label, and "top N items by score" is a
    # legitimate ranking chart -- refusing it would rule out the most useful
    # chart the collocations, tfidf and readability tables have.
    x_profile = profiles.get(x)
    if (
        x_profile is not None
        and x_profile.role is ColumnRole.IDENTIFIER
        and len(frame) > 1
        and pd.api.types.is_numeric_dtype(frame[x])
    ):
        reasons.append(
            f"{x!r} (x) is a row identifier, so the chart would plot values against row position "
            "rather than against anything meaningful"
        )

    pair = _numeric_pair(frame, x, y)
    if pair is not None and not reasons:
        a, b = pair
        if a.nunique() > 1 and b.nunique() > 1:
            pearson = abs(float(a.corr(b)))
            spearman = abs(float(a.corr(b, method="spearman")))
            if pearson >= _COLLINEAR:
                reasons.append(
                    f"{x!r} and {y!r} are {pearson:.4f} correlated -- they carry the same information, "
                    "so the chart shows the relationship between a quantity and itself"
                )
            elif spearman >= _COLLINEAR:
                reasons.append(
                    f"{x!r} and {y!r} rank every row identically (rank correlation {spearman:.4f}), "
                    "so the chart can only rise monotonically and tells you nothing you did not already know"
                )
    return reasons


def _bar(x: str, y: str, question: str, why: str, rank: int) -> Recommendation:
    return Recommendation(
        spec=ChartSpec(
            kind="bar",
            x=x,
            y=y,
            agg="sum",
            top_n=_DEFAULT_TOP_N,
            horizontal=True,
            title=f"{y} by {x}",
        ),
        question=question,
        why=why,
        rank=rank,
    )


def _histogram(x: str, question: str, why: str, rank: int) -> Recommendation:
    return Recommendation(
        spec=ChartSpec(kind="histogram", x=x, y=x, title=f"Distribution of {x}"),
        question=question,
        why=why,
        rank=rank,
    )


def _scatter(x: str, y: str, question: str, why: str, rank: int) -> Recommendation:
    return Recommendation(
        spec=ChartSpec(kind="scatter", x=x, y=y, title=f"{y} against {x}"),
        question=question,
        why=why,
        rank=rank,
    )


def _line(x: str, y: str, question: str, why: str, rank: int) -> Recommendation:
    """A timeline. No ``top_n``, and averaged rather than summed.

    Both defaults matter here in a way they do not for a bar chart. A top-N
    cut keeps the twenty largest *categories*, so applied to a timeline it
    deletes the quiet years and draws a line between what is left as though
    nothing happened in between. And where two documents share a date, their
    sum is a number about how many documents that day had; their mean is a
    number about the day.
    """
    return Recommendation(
        spec=ChartSpec(kind="line", x=x, y=y, agg="mean", top_n=None, title=f"{y} over {x}"),
        question=question,
        why=why,
        rank=rank,
    )


# Per-tool recommendations. Keyed by tool name; each entry names the columns
# to use and the question the chart answers. Kept declarative so adding a tool
# is a data change, and so a wrong column name fails visibly in tests rather
# than producing a chart of nothing.
_TOOL_CHARTS: dict[str, tuple[tuple[str, str, str, str, str], ...]] = {
    # (kind, x, y, question, why)
    "collocations": (
        (
            "bar",
            "Pair",
            "G2 (log-likelihood)",
            "Which word pairs are most strongly associated?",
            "G2 is the measure to rank by on sparse data; the top bars are the corpus's real fixed phrases.",
        ),
        (
            "scatter",
            "Co-occurrences",
            "PMI",
            "Is a high PMI backed by enough evidence?",
            "PMI rewards rare pairs, so the interesting region is high PMI with a co-occurrence count "
            "far from the left edge. Points hugging the left are the measure's known failure mode.",
        ),
    ),
    "tfidf": (
        (
            "bar",
            "Term",
            "TF-IDF",
            "Which terms distinguish this document from the rest?",
            "The top-weighted terms are the candidate label for the document.",
        ),
    ),
    "dispersion": (
        (
            "scatter",
            "Frequency",
            "Gries DP",
            "Which frequent words are concentrated rather than spread?",
            "The top-right corner is the trap frequency alone hides: words that look like core "
            "vocabulary but live in one document.",
        ),
        (
            "bar",
            "Term",
            "Adjusted Frequency",
            "What is the defensible core vocabulary?",
            "Adjusted frequency discounts raw frequency by how clumped the word is.",
        ),
    ),
    "keyness": (
        (
            "bar",
            "Word",
            "G2 (log-likelihood)",
            "Which words most distinguish the two groups?",
            "Check the Log Ratio sign to see which group each word belongs to.",
        ),
    ),
    "readability": (
        (
            "line",
            "Date",
            "Flesch Reading Ease",
            "Has this writing got easier or harder to read over time?",
            "Flesch runs the opposite way to the grade measures: a rising line is prose getting simpler.",
        ),
        (
            "bar",
            "Document",
            "Flesch Reading Ease",
            "Which documents are hardest to read?",
            "Flesch runs the opposite way to the grade measures: lower is harder.",
        ),
    ),
    "lexical_diversity": (
        (
            "line",
            "Date",
            "MTLD",
            "Has the vocabulary got richer or narrower over time?",
            "MTLD is the length-independent measure here, so a trend in it is not just a trend in how long the documents are.",
        ),
        (
            "scatter",
            "Total Tokens",
            "MTLD",
            "Is measured diversity just a function of document length?",
            "MTLD is designed to be length-independent; a strong upward trend here means the "
            "documents are too short for the measure to behave.",
        ),
        (
            "scatter",
            "Total Tokens",
            "TTR",
            "How much of this TTR column is just document length?",
            "TTR falls as a document gets longer, by construction. A clear downward curve here "
            "means the column is ranking documents by length wearing a different name.",
        ),
    ),
    "corpus_statistics": (
        (
            "line",
            "Date",
            "Yule K",
            "Has vocabulary repetition changed over time?",
            "Yule's K is far less length-sensitive than TTR, so it is the safer of the two to compare documents from different decades on.",
        ),
        (
            "scatter",
            "Tokens",
            "TTR",
            "How much of this TTR column is just document length?",
            "Type-token ratio falls mechanically as a text gets longer, so this scatter is the "
            "check to run before reading anything into the TTR ranking.",
        ),
        (
            "bar",
            "Document",
            "Yule K",
            "Which documents repeat their vocabulary most?",
            "Yule's K is far less length-sensitive than TTR, so it is the safer of the two "
            "columns to compare documents on.",
        ),
    ),
    "text_statistics": (
        (
            "line",
            "Date",
            "Avg Sentence Length",
            "Have the sentences got longer or shorter over time?",
            "Sentence length drives every readability score, so this line usually explains them.",
        ),
        (
            "scatter",
            "Tokens",
            "Avg Sentence Length",
            "Do longer documents also use longer sentences?",
            "Length and sentence style are separate choices; if they track each other closely "
            "here, one of the two is standing in for the other in anything downstream.",
        ),
        (
            "bar",
            "Document",
            "Avg Sentence Length",
            "Which documents are built from the longest sentences?",
            "Sentence length is the single biggest driver of every readability score, so this "
            "is usually the chart behind a readability result.",
        ),
    ),
    "sentence_complexity": (
        (
            "line",
            "Date",
            "Mean Dependency Distance",
            "Has the sentence structure got more or less involved over time?",
            "Dependency distance grows with sentence length, so read this beside the sentence-length line rather than on its own.",
        ),
        (
            "scatter",
            "Tokens",
            "Mean Dependency Distance",
            "Is the measured complexity anything more than sentence length?",
            "Dependency distance grows with sentence length almost automatically. The sentences "
            "worth looking at are the short ones sitting high on this chart.",
        ),
        (
            "bar",
            "Document",
            "Subordinate Clauses",
            "Which documents lean hardest on subordination?",
            "Summed per document, subordinate clauses separate reported and argued prose from "
            "narrative more cleanly than any single score.",
        ),
    ),
    "ner": (
        (
            "bar",
            "Entity",
            "Count",
            "Who and what does this corpus talk about most?",
            "The top of this chart is the corpus's cast list, and reading it is the fastest way "
            "to catch a tagger that has misread a recurring name.",
        ),
    ),
    "lda_gensim": (
        (
            "bar",
            "Word",
            "Weight",
            "Which words carry the model's topics?",
            "A word appearing near the top for several topics is a sign the topics have not "
            "separated, rather than a sign the word matters.",
        ),
    ),
    "sentiment_vader_anew": (
        (
            "line",
            "Date",
            "Compound",
            "Has the tone of these documents changed over time?",
            "Document-level compound against the calendar. This is the question a corpus spanning years exists to answer, and the one a ranking of documents against each other cannot.",
        ),
        (
            "scatter",
            "Hits",
            "Compound",
            "Is each sentiment score backed by any lexicon evidence?",
            "Hits counts the words VADER actually recognised. Points at zero hits scored exactly "
            "neutral because nothing was found, not because the text is balanced.",
        ),
        (
            "bar",
            "Document",
            "Compound",
            "Which documents read most positive or negative overall?",
            "Document-level compound is a mean over sentences, so a document near zero may be "
            "calm or may be equal parts of both.",
        ),
    ),
    "doc_similarity": (
        (
            "histogram",
            "Similarity",
            "Similarity",
            "How similar are documents in this corpus, typically?",
            "The shape matters more than any single pair: a bump up near 100% is duplicates, and "
            "a single tall block in the middle means shared vocabulary, not shared content.",
        ),
    ),
    "ngrams": (
        (
            "bar",
            "N-gram",
            "Frequency in Corpus",
            "Which fixed phrases does this corpus repeat?",
            "Ranked across the whole corpus rather than per document, so a phrase that is common "
            "everywhere outranks one that is heavy in a single file.",
        ),
    ),
    "doc_duplicates": (
        (
            "bar",
            "Members",
            "Size",
            "How large is each duplicate group?",
            "Group size is how many copies of one text the corpus holds, and every one of them "
            "is currently counted separately in every other result.",
        ),
    ),
}


def recommend_charts(
    frame: pd.DataFrame,
    *,
    tool: str = "",
    limit: int = 5,
) -> list[Recommendation]:
    """Charts worth drawing from *frame*, best first.

    Tool-specific recommendations come first when the tool is known and its
    columns are present; generic role-based ones fill the rest. Every
    recommendation is checked against :func:`degenerate_reasons`, so a
    suggestion can never be one of the pairings this module exists to warn
    about.
    """
    if frame.empty or limit < 1:
        return []
    profiles = profile_frame(frame)
    out: list[Recommendation] = []

    for kind, x, y, question, why in _TOOL_CHARTS.get(tool, ()):
        if x not in frame.columns or y not in frame.columns:
            continue
        rank = len(out)
        if kind == "histogram":
            # A histogram has one axis, so the x-against-y check does not apply
            # to it -- x == y is how a histogram is spelled, not a degenerate
            # pairing. What does disqualify it is a column with nothing to bin.
            column = profiles.get(x)
            if column is None or column.role in (ColumnRole.EMPTY, ColumnRole.CONSTANT, ColumnRole.TEXT):
                continue
            out.append(_histogram(x, question, why, rank))
            continue
        if degenerate_reasons(frame, x, y):
            continue
        if kind == "scatter":
            out.append(_scatter(x, y, question, why, rank))
        elif kind == "line":
            out.append(_line(x, y, question, why, rank))
        else:
            out.append(_bar(x, y, question, why, rank))

    out.extend(_generic_recommendations(frame, profiles, taken={(r.spec.x, r.spec.y) for r in out}))
    return [replace(recommendation, rank=index) for index, recommendation in enumerate(out[:limit])]


def _timeline_recommendation(
    frame: pd.DataFrame,
    dates: list[ColumnProfile],
    measures: list[ColumnProfile],
    *,
    taken: set[tuple[str, str]],
    out: list[Recommendation],
) -> None:
    """A date column and a measure is a timeline: the one chart that answers
    "when did this happen", which no label-magnitude pair can."""
    for date_column in dates[:1]:
        # One timeline. When the tool's own table already named the measure
        # worth following, a second line over the same axis under a vaguer
        # question ("How does Neg change over Date?") spends a slot in a
        # four-item shortlist restating the first.
        if any(x == date_column.name for x, _ in taken):
            return
        # A table of items per document (word pairs per speech) is not a
        # series. Its mean per date is a number about the table's cut-offs
        # and the speeches' lengths, not about any item in it.
        if _pooled_items(frame, date_column.name):
            return
        for measure in measures[:1]:
            if (date_column.name, measure.name) in taken:
                continue
            if degenerate_reasons(frame, date_column.name, measure.name):
                continue
            out.append(
                _line(
                    date_column.name,
                    measure.name,
                    f"How does {measure.name} change over {date_column.name}?",
                    "A timeline shows the order of change, which a bar chart of totals cannot. "
                    "Read the shape of the line, not a single step: one document is one point, "
                    "and one point is not a trend.",
                    rank=len(out),
                )
            )


#: Columns naming the document a row came from. Rows sharing a date and a
#: document are one observation; rows sharing a date across many *items* are
#: a pool. Mirrors ``DOCUMENT_COLUMN`` in ``desktop/src/chartLayout.ts``.
_DOCUMENT_COLUMNS = frozenset({"document", "document id", "doc", "file", "filename", "file name"})
#: Past this many distinct items per date on average, a date is a pool.
_POOLED = 1.5


def _pooled_items(frame: pd.DataFrame, date_column: str) -> str | None:
    """The label column whose many values each date mixes together, if any."""
    profiles = profile_frame(frame)
    for name, profile in profiles.items():
        if name == date_column or name.casefold() in _DOCUMENT_COLUMNS:
            continue
        if profile.role not in (ColumnRole.CATEGORICAL, ColumnRole.IDENTIFIER):
            continue
        if float(frame.groupby(date_column)[name].nunique().mean()) > _POOLED:
            return name
    return None


def _label_recommendations(
    frame: pd.DataFrame,
    categoricals: list[ColumnProfile],
    measures: list[ColumnProfile],
    *,
    taken: set[tuple[str, str]],
    out: list[Recommendation],
) -> None:
    """A label and a magnitude is the workhorse chart."""
    for category in categoricals[:1]:
        for measure in measures[:2]:
            if (category.name, measure.name) in taken:
                continue
            if degenerate_reasons(frame, category.name, measure.name):
                continue
            out.append(
                _bar(
                    category.name,
                    measure.name,
                    f"Which {category.name} values have the largest {measure.name}?",
                    "A labelled magnitude comparison, cut to the top rows so the axis stays readable.",
                    rank=len(out),
                )
            )


def _generic_recommendations(
    frame: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    *,
    taken: set[tuple[str, str]],
) -> list[Recommendation]:
    """Role-based fallbacks for any table, including ones with no tool."""
    dates = [p for p in profiles.values() if p.role is ColumnRole.DATE]
    categoricals = [p for p in profiles.values() if p.role is ColumnRole.CATEGORICAL]
    counts = [p for p in profiles.values() if p.role is ColumnRole.COUNT]
    scores = [p for p in profiles.values() if p.role in (ColumnRole.SCORE, ColumnRole.PROPORTION)]
    out: list[Recommendation] = []

    # Scores first here, counts first below. Over time, a raw count mostly
    # tracks document length -- "sentences per speech, 1934 to 2024" is a
    # chart of how long speeches got -- whereas a score is a property of the
    # writing itself. Against a label, the reverse: a count is the magnitude
    # a bar chart is for.
    _timeline_recommendation(frame, dates, scores + counts, taken=taken, out=out)
    _label_recommendations(frame, categoricals, counts + scores, taken=taken, out=out)

    # The shape of a measure is worth seeing before comparing it -- unless the
    # tool already asked for that exact histogram, in which case offering it
    # again under a vaguer question wastes a slot in a five-item shortlist.
    for measure in (scores + counts)[:1]:
        if (measure.name, measure.name) in taken:
            continue
        out.append(
            _histogram(
                measure.name,
                f"How is {measure.name} distributed?",
                "Distribution first: a mean over a bimodal or heavily skewed column describes nothing real.",
                rank=len(out),
            )
        )

    # Two independent measures may genuinely relate.
    numeric = counts + scores
    for index, left in enumerate(numeric):
        for right in numeric[index + 1 :]:
            if (left.name, right.name) in taken or len(out) >= _MAX_GENERIC:
                continue
            if degenerate_reasons(frame, left.name, right.name):
                continue
            out.append(
                _scatter(
                    left.name,
                    right.name,
                    f"Do {left.name} and {right.name} move together?",
                    "These two columns are not collinear, so the relationship is not built into the table.",
                    rank=len(out),
                )
            )
            break
        break
    return out
