"""Read a result table back in plain language.

A finished run hands the analyst a CSV and a filename. This writes the first
paragraph they would otherwise have to write themselves: how big the result
is, what dominates it, and which of that table's known traps it has actually
fallen into.

What it will not do is more important than what it will. It describes what is
in the table in front of it. It never infers causation, never calls anything
significant, never converts a difference into a claim about the world, and
never recommends a conclusion. Where a number is easy to misread, it says so
and names the specific check -- that is the useful part, and it is the part a
summary statistic cannot carry.

Structure, in the order an analyst needs it:

* ``headline`` -- the one-sentence shape of the result.
* ``observations`` -- what the table actually contains, in descending order
  of how much it should change your reading.
* ``cautions`` -- the traps this particular result has triggered, each with
  the concrete thing to do about it.

Tool-specific rules exist for the tools whose output shape and failure modes
are known; everything else gets the generic reading, which is still useful
because it is built on column roles rather than guesses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from core.insight.profile import ColumnProfile, ColumnRole, profile_frame

__all__ = ["Readout", "readout"]

# A single value covering this much of a column dominates the result, and the
# analyst should know before averaging anything.
_DOMINANCE_SHARE = 0.5
# Cut for "almost everything is in the tail" style observations.
_TOP_SHARE_SAMPLE = 10
# Beyond this an int cast loses precision, so format as a float instead.
_INT_FORMAT_LIMIT = 1e15
# Constant column names listed before eliding with "...".
_MAX_LISTED_NAMES = 4
# A column needs at least this many values before a shape claim is honest.
_MIN_VALUES_FOR_SHAPE = 2
# "Seen this few times" for the PMI low-count warning.
_LOW_COUNT = 3
# Gries DP at or above this is a concentrated word, not a spread one.
_CONCENTRATED_DP = 0.7
# Flesch range narrower than this means the documents are not really different.
_NARROW_FLESCH_RANGE = 10
# Share of listed TF-IDF terms appearing everywhere before it is worth saying.
_UBIQUITOUS_SHARE = 0.25
# Absolute rank correlation at which a measure is tracking document length.
_LENGTH_DEPENDENCE = 0.85
# Fewer documents than this and a correlation is not a claim worth making.
_MIN_DOCS_FOR_CORRELATION = 5
# Similarity percentage at which a document pair is probably the same text.
_NEAR_DUPLICATE = 90.0
# A minimum pairwise similarity above this means shared function words.
_HIGH_SIMILARITY_FLOOR = 50.0
# VADER's own neutral band (legacy thresholds, +/-0.05).
_VADER_NEUTRAL = 0.05
# Below this share of tokens matched, a lexicon average describes the lexicon.
_LOW_LEXICON_COVERAGE = 0.05
# MTLD needs roughly this much text before its value is interpretable.
_MTLD_MIN_TOKENS = 100
# Per-document averages below this token count are not worth comparing.
_SHORT_DOCUMENT_TOKENS = 100
# Share of n-grams occurring once before the n is too large for the corpus.
_HAPAX_SHARE = 0.6
# Share of topics a word must appear in before it marks unseparated topics.
_TOPIC_OVERLAP_SHARE = 0.5
# Spread across a topic's listed weights below which the topic has no core.
_FLAT_TOPIC_SHARE = 0.2
# Words used when quoting a topic back as a label.
_TOPIC_LABEL_WORDS = 5
# Values read out of a one-row result before the line gets unreadable.
_MAX_SINGLE_ROW_VALUES = 6


@dataclass(frozen=True, slots=True)
class Readout:
    """A plain-language reading of one result table."""

    headline: str
    observations: tuple[str, ...] = ()
    cautions: tuple[str, ...] = ()
    tool: str = ""

    def to_text(self) -> str:
        """Render as plain text for a CLI, a README or an artifact file."""
        lines = [self.headline]
        if self.observations:
            lines.append("")
            lines.append("What the table says:")
            lines.extend(f"  - {item}" for item in self.observations)
        if self.cautions:
            lines.append("")
            lines.append("Read with care:")
            lines.extend(f"  - {item}" for item in self.cautions)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render as Markdown for the desktop or a report."""
        lines = [self.headline, ""]
        if self.observations:
            lines.append("**What the table says**")
            lines.append("")
            lines.extend(f"- {item}" for item in self.observations)
            lines.append("")
        if self.cautions:
            lines.append("**Read with care**")
            lines.append("")
            lines.extend(f"- {item}" for item in self.cautions)
        return "\n".join(lines).rstrip() + "\n"


@dataclass(slots=True)
class _Draft:
    """Mutable accumulator while rules run."""

    observations: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    #: Columns whose repetition is built into the table's shape rather than
    #: being a property of the data. In a pairwise similarity table every
    #: document appears in exactly n-1 rows; in an n-gram table every phrase
    #: appears once per document it occurs in. Announcing "50% of rows share
    #: one Document A value" there is announcing arithmetic. A tool rule names
    #: these so the generic pass stays quiet about them.
    structural: set[str] = field(default_factory=set)


def _fmt(value: float) -> str:
    """Compact number formatting that does not imply false precision."""
    if value == int(value) and abs(value) < _INT_FORMAT_LIMIT:
        return f"{int(value):,}"
    return f"{value:,.4g}"


def _ident(value: object) -> str:
    """An identifier as written, not as pandas stored it.

    A column of whole-number ids becomes float64 the moment anything in the
    frame forces an upcast, and "the closest pair is 1.0 and 2.0" reads like a
    score rather than two document numbers.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _plural(count: float, singular: str, plural: str = "") -> str:
    """ "1 time" but "3 times" -- a readout people read should read correctly."""
    return singular if count == 1 else (plural or singular + "s")


def _top_row(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in frame.columns or frame.empty:
        return None
    ordered = frame.sort_values(column, ascending=False, kind="stable")
    return ordered.iloc[0]


def _single_row(frame: pd.DataFrame, profiles: dict[str, ColumnProfile], draft: _Draft) -> None:
    """A one-row result: read the row out, rather than describe its shape.

    Every column of a one-row table is constant and every value covers 100% of
    the rows, so the generic observations degenerate into arithmetic -- one
    summary table was told that "10 column(s) hold the same value in every
    row", which is true of any table with one row and informative about none of
    them. What the reader actually wants is the row.
    """
    # Roles are no help here: with one row every column profiles as CONSTANT,
    # including the numbers. The dtype is what still carries the distinction.
    del profiles
    row = frame.iloc[0]
    parts = [
        f"{name} {_fmt(float(row[name]))}"
        for name in frame.columns
        if pd.api.types.is_numeric_dtype(frame[name]) and pd.notna(row[name])
    ]
    if parts:
        draft.observations.append("A single row of results: " + ", ".join(parts[:_MAX_SINGLE_ROW_VALUES]) + ".")


def _time_span(frame: pd.DataFrame, profiles: dict[str, ColumnProfile], draft: _Draft) -> None:
    """What stretch of time this table covers, and where it is thin.

    Worth saying first because it changes what every other number in the
    table means. Ninety years of one document a year is a different object
    from ninety documents from one year, and the row count alone cannot tell
    the two apart.
    """
    dates = [p for p in profiles.values() if p.role is ColumnRole.DATE and p.name in frame.columns]
    if not dates:
        return
    column = dates[0]
    values = frame[column.name].dropna()
    if values.empty:
        return
    ordered = sorted(str(value) for value in values)
    draft.observations.insert(
        0,
        f"These {len(values):,} dated row(s) run from {ordered[0]} to {ordered[-1]}, "
        f"across {column.distinct:,} distinct {column.name.lower()} value(s).",
    )
    undated = len(frame) - len(values)
    if undated:
        draft.cautions.append(
            f"{undated:,} of {len(frame):,} row(s) have no {column.name}, so any chart over time "
            "silently leaves them out. They are in the table and in every total; they are not on the timeline."
        )


def _generic(frame: pd.DataFrame, draft: _Draft) -> None:
    """Reading that holds for any table, built from column roles."""
    profiles = profile_frame(frame)
    # Below two rows there is nothing to compare, and every "all rows share
    # this" claim is true by construction rather than by observation.
    if len(frame) < _MIN_VALUES_FOR_SHAPE:
        _single_row(frame, profiles, draft)
        return
    constants = [p.name for p in profiles.values() if p.role is ColumnRole.CONSTANT and p.name not in draft.structural]
    if constants:
        draft.observations.append(
            f"{len(constants)} column(s) hold the same value in every row ({', '.join(constants[:_MAX_LISTED_NAMES])}"
            f"{', ...' if len(constants) > _MAX_LISTED_NAMES else ''}); they describe the run, not differences between rows."
        )
    for profile in profiles.values():
        if profile.role is not ColumnRole.CATEGORICAL or profile.name not in frame.columns:
            continue
        if profile.name in draft.structural:
            continue
        counts = frame[profile.name].value_counts(normalize=True)
        if counts.empty:
            continue
        share = float(counts.iloc[0])
        if share >= _DOMINANCE_SHARE:
            draft.observations.append(
                f"{share:.0%} of rows share one {profile.name} value ({counts.index[0]!r}), "
                "so totals over this table are largely describing that one group."
            )
    # "This row holds n% of the total" needs a total, and only a magnitude that
    # accumulates has one. Summing a ratio is meaningless (70% of the sum of
    # every document's type-token ratio is not a fact about anything), and on a
    # signed score the positives and negatives cancel in the denominator, which
    # is how one run came to report a row holding "417% of all Compound".
    # ColumnRole already draws this line -- see core.insight.profile, which
    # says of PROPORTION: "averages meaningfully, sums do not".
    counts = [p for p in profiles.values() if p.role is ColumnRole.COUNT]
    for profile in counts[:3]:
        series = frame[profile.name].dropna()
        if len(series) < _MIN_VALUES_FOR_SHAPE or series.nunique() < _MIN_VALUES_FOR_SHAPE:
            continue
        if bool((series < 0).any()):
            continue
        top = float(series.nlargest(1).iloc[0])
        total = float(series.sum())
        if total > 0 and top / total >= _DOMINANCE_SHARE:
            draft.cautions.append(
                f"A single row holds {top / total:.0%} of all {profile.name}; a mean over this column "
                "is describing that one row more than the rest of the table."
            )
    if not draft.observations:
        _largest_row(frame, profiles, draft)
    _time_span(frame, profiles, draft)


def _largest_row(frame: pd.DataFrame, profiles: dict[str, ColumnProfile], draft: _Draft) -> None:
    """Name the biggest thing in the table, when nothing else had anything to say.

    A readout whose entire content is "240 row(s) and 5 column(s)" is an
    artifact nobody benefits from opening. Several tools produced exactly that:
    a word list, a co-occurrence table, a per-sentence arc. Each has an obvious
    first sentence available -- which row is the largest, on the column the
    table is really about -- and it needs no per-tool knowledge to find.

    A floor, not a default: it only speaks when the tool rule and the rest of
    the generic pass found nothing, so it can never crowd out a better reading.
    """
    measures = [p for p in profiles.values() if p.role is ColumnRole.COUNT] or [
        p for p in profiles.values() if p.role in (ColumnRole.SCORE, ColumnRole.PROPORTION)
    ]
    if not measures:
        return
    measure = measures[0].name
    series = pd.to_numeric(frame[measure], errors="coerce").dropna()
    if series.empty or series.nunique() < _MIN_VALUES_FOR_SHAPE:
        return

    labels = [
        p.name
        for p in profiles.values()
        if p.role in (ColumnRole.CATEGORICAL, ColumnRole.IDENTIFIER)
        and p.name in frame.columns
        and not pd.api.types.is_numeric_dtype(frame[p.name])
    ]
    if not labels:
        draft.observations.append(
            f"{measure} runs from {_fmt(float(series.min()))} to {_fmt(float(series.max()))} "
            f"across {len(series)} row(s)."
        )
        return

    label = labels[0]
    # Stable ordering on the label breaks ties the same way every run (R6).
    ordered = frame.loc[series.index].sort_values([measure, label], ascending=[False, True], kind="stable")
    top = ordered.iloc[0]
    draft.observations.append(
        f"The largest {measure} is {_fmt(float(top[measure]))}, at {label} {str(top[label])!r}. "
        f"The column runs from {_fmt(float(series.min()))} to {_fmt(float(series.max()))}."
    )


def _collocations(frame: pd.DataFrame, draft: _Draft) -> None:
    top = _top_row(frame, "G2 (log-likelihood)")
    if top is not None:
        draft.observations.append(
            f"Strongest association: {top['Pair']!r} "
            f"(G2 {_fmt(float(top['G2 (log-likelihood)']))}, seen {_fmt(float(top['Co-occurrences']))} "
            f"{_plural(float(top['Co-occurrences']), 'time')})."
        )
    if "PMI" in frame.columns and "Co-occurrences" in frame.columns and not frame.empty:
        by_pmi = frame.sort_values(["PMI", "Pair"], ascending=[False, True], kind="stable")
        head = by_pmi.head(_TOP_SHARE_SAMPLE)
        rare = int((head["Co-occurrences"] <= _LOW_COUNT).sum())
        if rare >= _TOP_SHARE_SAMPLE // 2:
            draft.cautions.append(
                f"{rare} of the top {len(head)} pairs by PMI occur 3 times or fewer. That is PMI behaving "
                "as designed, not a discovery: raise --min-count or rank by G2 instead."
            )
    if "Word 1" in frame.columns and not frame.empty:
        words = pd.concat([frame["Word 1"], frame["Word 2"]]).value_counts()
        if not words.empty and float(words.iloc[0]) / len(frame) >= _DOMINANCE_SHARE:
            draft.cautions.append(
                f"The word {words.index[0]!r} appears in {words.iloc[0]} of {len(frame)} pairs. "
                "A function word dominating the list usually means you want a stopword file."
            )


def _tfidf(frame: pd.DataFrame, draft: _Draft) -> None:
    if "Term" not in frame.columns:
        return
    draft.structural |= {"Term", "Document", "Document ID"}
    if "Document" in frame.columns and not frame.empty:
        documents = frame["Document"].nunique()
        draft.observations.append(f"Top terms listed for {documents} document(s).")
        first = frame[frame["Document"] == frame["Document"].iloc[0]].head(5)
        if not first.empty:
            terms = ", ".join(str(t) for t in first["Term"])
            draft.observations.append(f"{frame['Document'].iloc[0]} reads as: {terms}.")
    if "Document Frequency" in frame.columns and "Document" in frame.columns and not frame.empty:
        documents = max(frame["Document"].nunique(), 1)
        ubiquitous = frame[frame["Document Frequency"] >= documents]
        if documents > 1 and len(ubiquitous) >= len(frame) * _UBIQUITOUS_SHARE:
            draft.cautions.append(
                f"{len(ubiquitous)} of {len(frame)} listed terms appear in every document, so they are "
                "distinguishing nothing. Lower --max-df-ratio to drop them."
            )


def _dispersion(frame: pd.DataFrame, draft: _Draft) -> None:
    if "Gries DP" not in frame.columns or frame.empty:
        return
    clumped = frame[frame["Gries DP"] >= _CONCENTRATED_DP]
    draft.observations.append(
        f"{len(frame)} term(s) measured across {int(frame['Parts'].iloc[0])} parts; "
        f"{len(clumped)} have a DP of 0.7 or higher, meaning they are concentrated rather than spread."
    )
    if not clumped.empty:
        worst = clumped.sort_values(["Frequency", "Term"], ascending=[False, True], kind="stable").iloc[0]
        draft.observations.append(
            f"The most frequent concentrated word is {worst['Term']!r}: {_fmt(float(worst['Frequency']))} "
            f"occurrences but a range of only {int(worst['Range'])} part(s). Raw frequency would have "
            "ranked it as core vocabulary."
        )
    if "Range" in frame.columns:
        singletons = int((frame["Range"] <= 1).sum())
        if singletons >= len(frame) * _DOMINANCE_SHARE:
            draft.cautions.append(
                f"{singletons} of {len(frame)} terms occur in only one part, which usually means the "
                "corpus has too few parts for dispersion to say much. Try --parts chunk with more chunks."
            )


def _keyness(frame: pd.DataFrame, draft: _Draft) -> None:
    if "Log Ratio" not in frame.columns or frame.empty:
        return
    group_a = int((frame["Log Ratio"] > 0).sum())
    draft.observations.append(
        f"{len(frame)} word(s) scored: {group_a} over-represented in group A, {len(frame) - group_a} in group B."
    )
    top = _top_row(frame, "G2 (log-likelihood)")
    if top is not None and "Word" in frame.columns:
        side = "A" if float(top["Log Ratio"]) > 0 else "B"
        draft.observations.append(f"Most distinguishing word: {top['Word']!r}, belonging to group {side}.")


def _svo_compare(frame: pd.DataFrame, draft: _Draft) -> None:
    if "Triple Jaccard" not in frame.columns or frame.empty:
        return
    # Pairwise again: with n documents every one appears in n-1 of the rows,
    # so "50% of rows share one Doc A value" is the shape of the table rather
    # than anything about the documents.
    draft.structural |= {"Doc A", "Doc B"}
    overlap = pd.to_numeric(frame["Triple Jaccard"], errors="coerce")
    common = pd.to_numeric(frame.get("Common Triples", pd.Series(dtype=float)), errors="coerce")

    # A Jaccard of 1.0 between two empty sets is the definition doing its job,
    # not two documents saying the same thing. Ranking on it puts the pair that
    # extracted *nothing* at the top of the table, which is the opposite of
    # what the column is being read for.
    shared = frame[common > 0] if not common.empty else frame.iloc[0:0]
    if not shared.empty:
        best = shared.sort_values(["Triple Jaccard", "Doc A"], ascending=[False, True], kind="stable").iloc[0]
        draft.observations.append(
            f"{len(frame)} document pair(s) compared on who-does-what-to-whom. The closest pair that actually "
            f"shares a triple is {_ident(best['Doc A'])} and {_ident(best['Doc B'])} "
            f"(triple overlap {_fmt(float(best['Triple Jaccard']))})."
        )
    else:
        draft.observations.append(
            f"{len(frame)} document pair(s) compared on who-does-what-to-whom; none of them share a complete "
            "subject-verb-object triple."
        )
        if bool((overlap >= 1.0).any()):
            draft.cautions.append(
                "Some pairs score a triple overlap of 1.0 while sharing no triples at all: both documents "
                "yielded no triples, and Jaccard calls two empty sets identical. Those rows are a parser "
                "result, not a similarity, and sorting this column puts them on top."
            )
        draft.cautions.append(
            "With no shared triples the Jaccard columns are comparing subjects, verbs and objects in "
            "isolation. Overlap on subjects alone is not overlap in meaning."
        )


def _readability(frame: pd.DataFrame, draft: _Draft) -> None:
    column = "Flesch Reading Ease"
    if column not in frame.columns or frame.empty:
        return
    series = frame[column].dropna()
    if series.empty:
        return
    draft.observations.append(
        f"Flesch Reading Ease runs from {_fmt(float(series.min()))} to {_fmt(float(series.max()))} "
        f"across {len(series)} document(s); lower is harder, which is the opposite direction to the grade scores."
    )
    if float(series.max() - series.min()) < _NARROW_FLESCH_RANGE:
        draft.observations.append(
            "The documents are close together on this scale, so differences between them are unlikely to be meaningful."
        )


def _length_dependence(
    frame: pd.DataFrame,
    length_column: str,
    measure_column: str,
    draft: _Draft,
    *,
    because: str,
) -> None:
    """Warn when a measure is tracking document length rather than anything else.

    This is the single most common way a corpus result is over-read: a column
    that is mechanically a function of how long each document is gets compared
    across documents of very different lengths, and the ranking that comes out
    is a length ranking wearing another name. Rank correlation rather than
    Pearson, because the relationship is usually curved, not linear.
    """
    if length_column not in frame.columns or measure_column not in frame.columns:
        return
    subset = frame[[length_column, measure_column]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(subset) < _MIN_DOCS_FOR_CORRELATION:
        return
    if (
        subset[length_column].nunique() < _MIN_VALUES_FOR_SHAPE
        or subset[measure_column].nunique() < _MIN_VALUES_FOR_SHAPE
    ):
        return
    rho = subset[length_column].corr(subset[measure_column], method="spearman")
    if pd.isna(rho) or abs(float(rho)) < _LENGTH_DEPENDENCE:
        return
    direction = "rises" if float(rho) > 0 else "falls"
    draft.cautions.append(
        f"{measure_column} {direction} with {length_column} across these documents "
        f"(rank correlation {abs(float(rho)):.2f}), so the ranking is largely a length ranking. {because}"
    )


def _ner(frame: pd.DataFrame, draft: _Draft) -> None:
    name_column = "Entity" if "Entity" in frame.columns else "Location"
    if name_column not in frame.columns or "Count" not in frame.columns or frame.empty:
        return
    draft.structural |= {name_column, "Document", "Document ID"}
    top = frame.sort_values(["Count", name_column], ascending=[False, True], kind="stable").iloc[0]
    draft.observations.append(
        f"{frame[name_column].nunique()} distinct {name_column.lower()} name(s) found; the most mentioned is "
        f"{top[name_column]!r} ({_fmt(float(top['Count']))} {_plural(float(top['Count']), 'mention')})."
    )
    if "NER Tag" in frame.columns:
        tags = frame["NER Tag"].value_counts()
        if not tags.empty:
            listed = ", ".join(f"{tag} {count}" for tag, count in tags.head(_MAX_LISTED_NAMES).items())
            draft.observations.append(f"By tag: {listed}.")
        # One surface form under two tags is the tagger disagreeing with itself,
        # and it silently splits an entity's counts in two.
        ambiguous = frame.groupby(name_column)["NER Tag"].nunique()
        split = ambiguous[ambiguous > 1]
        if not split.empty:
            names = ", ".join(repr(str(n)) for n in split.index[:_MAX_LISTED_NAMES])
            draft.cautions.append(
                f"{len(split)} name(s) were tagged more than one way ({names}"
                f"{', ...' if len(split) > _MAX_LISTED_NAMES else ''}). Each one's mentions are split across "
                "rows, so its totals here are lower than its real count."
            )
    singles = int((frame["Count"] <= 1).sum())
    if singles >= len(frame) * _DOMINANCE_SHARE:
        draft.cautions.append(
            f"{singles} of {len(frame)} entities were seen exactly once. Single-mention entities are where "
            "a tagger's mistakes collect, so read the tail of this table before quoting a total."
        )


def _topic_model(frame: pd.DataFrame, draft: _Draft) -> None:
    if not {"Topic", "Word", "Weight"} <= set(frame.columns) or frame.empty:
        return
    draft.structural |= {"Topic", "Word"}
    topics = int(frame["Topic"].nunique())
    first = frame[frame["Topic"] == frame["Topic"].iloc[0]].sort_values("Weight", ascending=False, kind="stable")
    words = ", ".join(str(w) for w in first["Word"].head(_TOPIC_LABEL_WORDS))
    draft.observations.append(f"{topics} topic(s) fitted. Topic {frame['Topic'].iloc[0]} reads as: {words}.")

    # A word that is near the top of most topics is not a finding about that
    # word; it is the model failing to separate the topics at all.
    shared = frame.groupby("Word")["Topic"].nunique()
    everywhere = shared[shared >= max(2, int(topics * _TOPIC_OVERLAP_SHARE))]
    if topics > 1 and not everywhere.empty:
        names = ", ".join(repr(str(w)) for w in everywhere.index[:_MAX_LISTED_NAMES])
        draft.cautions.append(
            f"{len(everywhere)} word(s) appear in at least half the topics ({names}"
            f"{', ...' if len(everywhere) > _MAX_LISTED_NAMES else ''}). Topics sharing their top words have "
            "not separated: fit fewer topics, or remove these words as stopwords and refit."
        )
    if len(first) >= _MIN_VALUES_FOR_SHAPE:
        high, low = float(first["Weight"].iloc[0]), float(first["Weight"].iloc[-1])
        if high > 0 and (high - low) / high < _FLAT_TOPIC_SHARE:
            draft.cautions.append(
                "Within a topic the listed words carry nearly equal weight, which means the topic has no "
                "core vocabulary to name it by. Treat the label you would give it as your reading, not the model's."
            )


def _sentiment(frame: pd.DataFrame, draft: _Draft) -> None:
    draft.structural |= {"Document", "Document ID", "Sentence"}
    if "Compound" in frame.columns and not frame.empty:
        series = frame["Compound"].dropna()
        if not series.empty:
            unit = "sentence" if "Sentence ID" in frame.columns else "document"
            neutral = int(series.between(-_VADER_NEUTRAL, _VADER_NEUTRAL).sum())
            positive = int((series > _VADER_NEUTRAL).sum())
            draft.observations.append(
                f"{len(series)} {unit}(s) scored: {positive} positive, "
                f"{len(series) - positive - neutral} negative, {neutral} neutral, "
                f"on a compound scale running {_fmt(float(series.min()))} to {_fmt(float(series.max()))}."
            )
            if neutral >= len(series) * _DOMINANCE_SHARE:
                draft.cautions.append(
                    f"{neutral} of {len(series)} scores fall inside the neutral band. A neutral VADER score "
                    "means no sentiment word was matched as often as it means balanced tone."
                )
    # The coverage check that makes the neutral band readable.
    if "Hits" in frame.columns and not frame.empty:
        empty_rows = int((frame["Hits"] <= 0).sum())
        if empty_rows:
            draft.cautions.append(
                f"{empty_rows} of {len(frame)} rows contained no lexicon word at all, so their score is the "
                "default rather than a measurement. Exclude them before averaging."
            )
    if "Valence" in frame.columns and not frame.empty:
        valence = frame["Valence"].dropna()
        if not valence.empty:
            draft.observations.append(
                f"ANEW valence runs {_fmt(float(valence.min()))} to {_fmt(float(valence.max()))} across "
                f"{len(valence)} document(s), on the 1-9 scale the ratings were collected on."
            )
        if "Hits" in frame.columns and "Tokens" in frame.columns:
            covered = frame["Hits"].sum() / max(float(frame["Tokens"].sum()), 1.0)
            if covered < _LOW_LEXICON_COVERAGE:
                draft.cautions.append(
                    f"Only {covered:.1%} of tokens appear in the ANEW lexicon. These averages describe that "
                    "small matched slice, not the documents."
                )


def _doc_similarity(frame: pd.DataFrame, draft: _Draft) -> None:
    if "Similarity" not in frame.columns or frame.empty:
        return
    draft.structural |= {"Document A", "Document B", "Document ID A", "Document ID B"}
    series = frame["Similarity"].dropna()
    if series.empty:
        return
    top = frame.sort_values("Similarity", ascending=False, kind="stable").iloc[0]
    pair = f"{top.get('Document A', '?')} and {top.get('Document B', '?')}"
    draft.observations.append(
        f"{len(frame)} document pair(s) compared, from {_fmt(float(series.min()))}% to "
        f"{_fmt(float(series.max()))}% similar. The closest pair is {pair}."
    )
    near = frame[series >= _NEAR_DUPLICATE]
    if not near.empty:
        draft.cautions.append(
            f"{len(near)} pair(s) are {int(_NEAR_DUPLICATE)}% similar or more, which usually means the same "
            "text is in the corpus twice. Duplicates inflate every corpus-level count downstream, so check "
            "these before running anything else."
        )
    if float(series.min()) >= _HIGH_SIMILARITY_FLOOR:
        draft.cautions.append(
            f"Even the least similar pair scores {_fmt(float(series.min()))}%. A high floor across every pair "
            "is shared function words rather than shared content: remove stopwords and compare again."
        )


def _ngrams(frame: pd.DataFrame, draft: _Draft) -> None:
    if "N-gram" not in frame.columns or "Frequency in Corpus" not in frame.columns or frame.empty:
        return
    draft.structural |= {"N-gram", "Document", "Document ID"}
    corpus_totals = frame.drop_duplicates("N-gram").set_index("N-gram")["Frequency in Corpus"]
    top = corpus_totals.sort_values(ascending=False, kind="stable")
    draft.observations.append(
        f"{len(corpus_totals)} distinct n-gram(s) across the corpus; the most repeated is "
        f"{str(top.index[0])!r} ({_fmt(float(top.iloc[0]))} {_plural(float(top.iloc[0]), 'time')})."
    )
    hapax = int((corpus_totals <= 1).sum())
    if hapax >= len(corpus_totals) * _HAPAX_SHARE:
        draft.cautions.append(
            f"{hapax} of {len(corpus_totals)} n-grams occur exactly once. At that rate the n is too long for "
            "this corpus, and the list is mostly recording sentences rather than phrases."
        )
    if len(frame) > len(corpus_totals):
        draft.cautions.append(
            "This table has one row per n-gram per document, and Frequency in Corpus repeats the same corpus "
            "total on each of those rows. Summing that column double-counts; sum Frequency in Document instead."
        )


def _kwic(frame: pd.DataFrame, draft: _Draft) -> None:
    if "Hit" not in frame.columns or frame.empty:
        return
    # A concordance for one query term has one Hit value in every row, and its
    # context columns repeat wherever the corpus repeats a phrase. Reporting
    # either as a property of the data describes the query, not the corpus.
    draft.structural |= {"Hit", "Document", "Document ID", "Left Context", "Right Context", "Sentence ID"}
    forms = frame["Hit"].value_counts()
    documents = frame["Document"].nunique() if "Document" in frame.columns else 0
    draft.observations.append(
        f"{len(frame)} concordance line(s) over {documents} document(s), matching "
        f"{frame['Hit'].nunique()} distinct surface form(s); the most common is {str(forms.index[0])!r}."
    )
    draft.cautions.append(
        "A concordance is evidence to read, not a count to total: one sentence can produce several lines here, "
        "so the row count is not the number of sentences that matched."
    )


def _text_statistics(frame: pd.DataFrame, draft: _Draft) -> None:
    if "Tokens" not in frame.columns or frame.empty:
        return
    draft.observations.append(
        f"{len(frame)} document(s), {_fmt(float(frame['Tokens'].sum()))} tokens in total, "
        f"from {_fmt(float(frame['Tokens'].min()))} to {_fmt(float(frame['Tokens'].max()))} per document."
    )
    if "Avg Sentence Length" in frame.columns:
        series = frame["Avg Sentence Length"].dropna()
        if not series.empty:
            draft.observations.append(
                f"Average sentence length runs {_fmt(float(series.min()))} to {_fmt(float(series.max()))} words."
            )
    short = int((frame["Tokens"] < _SHORT_DOCUMENT_TOKENS).sum())
    if short:
        draft.cautions.append(
            f"{short} document(s) hold fewer than {_SHORT_DOCUMENT_TOKENS} tokens. Every per-document average "
            "in this table is unstable at that length, and the ratios more so than the counts."
        )
    _length_dependence(
        frame,
        "Tokens",
        "Types",
        draft,
        because="Compare vocabulary with a length-corrected measure -- run lexical_diversity for MTLD.",
    )


def _corpus_statistics(frame: pd.DataFrame, draft: _Draft) -> None:
    if "TTR" not in frame.columns or frame.empty:
        return
    series = frame["TTR"].dropna()
    if not series.empty:
        draft.observations.append(
            f"Type-token ratio runs {_fmt(float(series.min()))} to {_fmt(float(series.max()))} "
            f"across {len(series)} document(s)."
        )
    _length_dependence(
        frame,
        "Tokens",
        "TTR",
        draft,
        because="Rank on Yule K or Herdan instead, or use lexical_diversity, which reports MTLD.",
    )
    if "Yule K" in frame.columns and not frame.empty:
        top = frame.sort_values("Yule K", ascending=False, kind="stable").iloc[0]
        draft.observations.append(
            f"The most repetitive vocabulary by Yule K is {top.get('Document', '?')} "
            f"({_fmt(float(top['Yule K']))}); higher K means more repetition, which is the opposite "
            "direction to TTR."
        )


def _lexical_diversity(frame: pd.DataFrame, draft: _Draft) -> None:
    if "MTLD" not in frame.columns or frame.empty:
        return
    series = pd.to_numeric(frame["MTLD"], errors="coerce").dropna()
    if not series.empty:
        draft.observations.append(
            f"MTLD runs {_fmt(float(series.min()))} to {_fmt(float(series.max()))} across "
            f"{len(series)} document(s); higher means the text goes further before repeating itself."
        )
    if "Total Tokens" in frame.columns:
        short = int((frame["Total Tokens"] < _MTLD_MIN_TOKENS).sum())
        if short:
            draft.cautions.append(
                f"{short} document(s) are shorter than {_MTLD_MIN_TOKENS} tokens. MTLD needs roughly that much "
                "text before its value means anything, so those rows are not comparable with the rest."
            )
        _length_dependence(
            frame,
            "Total Tokens",
            "TTR",
            draft,
            because="That is exactly what MTLD in this same table is for; use it rather than the TTR column.",
        )


def _sentence_complexity(frame: pd.DataFrame, draft: _Draft) -> None:
    column = "Mean Dependency Distance"
    if column not in frame.columns or frame.empty:
        return
    draft.structural |= {"Document", "Document ID", "Sentence"}
    series = frame[column].dropna()
    if not series.empty:
        draft.observations.append(
            f"{len(frame)} sentence(s) parsed; mean dependency distance runs {_fmt(float(series.min()))} to "
            f"{_fmt(float(series.max()))}, and tree depth reaches {int(frame['Depth'].max())}."
        )
    _length_dependence(
        frame,
        "Tokens",
        column,
        draft,
        because="The sentences actually worth reading are the short ones that still score high.",
    )


def _doc_duplicates(frame: pd.DataFrame, draft: _Draft) -> None:
    if "Size" not in frame.columns or frame.empty:
        return
    copies = int(frame["Size"].sum())
    draft.observations.append(
        f"{len(frame)} duplicate group(s) covering {copies} file(s); the largest group holds "
        f"{int(frame['Size'].max())} copies of one text."
    )
    draft.cautions.append(
        f"Until these are resolved, {copies - len(frame)} file(s) are counted more than once in every other "
        "result computed over this corpus, including frequencies, similarities and every average."
    )


_RULES: dict[str, Callable[[pd.DataFrame, _Draft], None]] = {
    "collocations": _collocations,
    "tfidf": _tfidf,
    "dispersion": _dispersion,
    "keyness": _keyness,
    "readability": _readability,
    "ner": _ner,
    "lda_gensim": _topic_model,
    "topic_model": _topic_model,  # older runs under the pre-split name
    "sentiment_vader_anew": _sentiment,
    "doc_similarity": _doc_similarity,
    "ngrams": _ngrams,
    "kwic": _kwic,
    "text_statistics": _text_statistics,
    "corpus_statistics": _corpus_statistics,
    "lexical_diversity": _lexical_diversity,
    "sentence_complexity": _sentence_complexity,
    "doc_duplicates": _doc_duplicates,
    "svo_compare": _svo_compare,
}


def readout(frame: pd.DataFrame, *, tool: str = "") -> Readout:
    """A plain-language reading of *frame*, tailored to *tool* when known."""
    if frame.empty:
        # Threshold names differ per tool; the hint should name the knob this
        # tool actually has, not a generic "filters" hand-wave.
        hints = {
            "collocations": "Lower the threshold: minimum co-occurrences defaults to 3, so either lower it or add more text.",
            "dispersion": "Lower the threshold: the minimum frequency, or split the corpus into more parts.",
            "keyness": "Check that the group pattern matched documents in both groups.",
            "ngrams": "Lower the threshold: the minimum count, or shorten the sequence length.",
            "tfidf": "Adjust the threshold: the document-frequency cutoffs, or add more documents.",
        }
        hint = hints.get(
            tool, "Check the run's minimum-count and filter settings before concluding the corpus lacks the pattern."
        )
        return Readout(
            headline="The result is empty: no rows met the settings used for this run.",
            observations=(),
            cautions=(
                "An empty result is a real answer, but it is usually a threshold that was set too high. " + hint,
            ),
            tool=tool,
        )

    rows, columns = frame.shape
    headline = f"{rows:,} row(s) and {columns} column(s)"
    headline = f"{tool}: {headline}." if tool else f"{headline}."

    draft = _Draft()
    rule = _RULES.get(tool)
    if rule is not None:
        # A rule is a reading of a table shape it expects. The tool name can be
        # wrong -- inferred from a directory, typed by hand, or pointed at an
        # unrelated CSV -- so a rule that does not recognise what it was given
        # must yield the generic reading, never take the whole readout down.
        try:
            rule(frame, draft)
        except (KeyError, IndexError, TypeError, ValueError):
            draft.observations.append(
                f"This table does not have the shape {tool!r} normally produces, so only the "
                "generic reading below applies."
            )
    _generic(frame, draft)
    return Readout(
        headline=headline,
        observations=tuple(draft.observations),
        cautions=tuple(draft.cautions),
        tool=tool,
    )
