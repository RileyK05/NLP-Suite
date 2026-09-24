"""Topic prevalence over time — the history a topic model's own charts cannot show.

``core/analysis/lda.py`` fits topics and hands back a ``dominant`` frame: one
row per document, naming the topic that explains the largest share of it
(``Dominant topic``), how large that share was (``Contribution``), and the
words that define that topic (``Topic keywords``). The intertopic distance
map and the ranked-bars relevance chart both show the fitted model as a
snapshot. Neither shows the one thing this suite's own corpus invites: 87
State of the Union addresses running from 1934 to 2024 have a *history*, and
"did topic 3 grow or shrink" is a question no snapshot can answer.

This panel answers it by turning ``Document`` — a filename, not a date column
— into time. That is also its one delicate step, and the module is
deliberately conservative about it:

* A year is read only from the **start** of the filename (``YYYY`` or
  ``YYYY-MM-DD``). A regex that matched a year anywhere would treat
  ``report_1999_v2.txt`` as a 1999 document, which is a guess dressed as a
  fact — the "1999" there could be a version, a case number, anything.
* A document whose name carries no such prefix is **dropped, not guessed
  at**. Bucketing it under year 0 (or under "no date") would either wreck
  the x axis or silently invent a category nobody asked for, so it is left
  out and counted in a warning (``PANEL_NO_DATE``) instead.
* If nothing in the corpus has a parseable name, there is no plot to draw at
  all, and the failure (``PANEL_NO_DATES``) says so rather than returning an
  empty figure that looks like a bug.

The other design choice worth stating: a bucket is a value this builder
computes, not a column the source frame already has. ``PreparedPanel.data``
is the table published beside the figure — the CLI writes it out as
``panel_data.csv`` next to the HTML — so the honest place to put the derived
grouping is *in* that table, as a real ``Period`` column, rather than
smuggling it into a filter a reader cannot check. Evidence then reads
``[("Period", "1930"), ("Dominant topic", "3")]``, and a test can take those
filters straight back to ``panel.data`` and find exactly the documents the
band was drawn from — see ``tests/test_panels_lda_prevalence.py``.

Like ``panels_keyness.py``, this is a pure function of (frame, params,
provenance): no plotly import, no filesystem, no corpus access. The renderer
for ``stream`` (stacked bands over an ordered axis) does not exist yet, so
this panel cannot be registered or drawn through ``prepare_panel`` until that
lands elsewhere; what it can do today is state, precisely and testably, what
the figure would claim.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panelspec import (
    Evidence,
    PanelDefinition,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
)

__all__ = ["LDA_PREVALENCE", "lda_prevalence"]

DOCUMENT_ID = "Document ID"
DOCUMENT = "Document"
DOMINANT_TOPIC = "Dominant topic"
CONTRIBUTION = "Contribution"
TOPIC_KEYWORDS = "Topic keywords"
#: The derived column this builder adds to the published table. Not a column
#: ``core/analysis/lda.py`` produces — a bucket is a value this panel
#: computes, and it is added here (rather than kept as a private filter only
#: the builder understands) so a mark's evidence can point at a real column
#: in ``PreparedPanel.data`` instead of a computation a reader cannot see.
PERIOD = "Period"

#: A leading year, optionally followed by -MM-DD, and NOT followed by another
#: digit (so "19345" does not read as the year 1934 with a stray "5"). Real
#: names in this corpus: "1934-01-03_franklin d roosevelt_sotu.txt",
#: "2024-03-07_joseph r biden_sotu.txt". Anchored with ``^`` on purpose: a
#: year anywhere in the name is ambiguous ("report_1999_v2.txt" is a report,
#: not necessarily a 1999 one), and guessing would be worse than dropping.
_LEADING_DATE = re.compile(r"^(\d{4})(?:-\d{2}-\d{2})?(?!\d)")

#: How many dropped-document names to quote in a diagnostic. Enough to show
#: the pattern (or lack of one) in the corpus's naming; not the whole list.
_MAX_EXAMPLES = 3

#: Above this many keywords a mark's label would stop being readable; the
#: keyword string in the source frame is already the model's top ten words,
#: this trims it further to the ones worth putting on a hover label.
_LABEL_KEYWORDS = 3


def lda_prevalence(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Prepare the stream: share (or count) of documents per topic, over time.

    One :class:`PanelMark` per (time bucket, topic) cell that has at least
    one document in it. ``x`` is the bucket as a plain number (1930 for the
    1930s, or 1934 for that single year) so the renderer can lay bands out on
    an ordered numeric axis without knowing anything about calendars.
    """
    bucket = str(params["bucket"])
    measure = str(params["measure"])
    normalize = bool(params["normalize"])

    diagnostics: list[Diagnostic] = []
    working = frame.copy()

    # -- date extraction -----------------------------------------------
    years = working[DOCUMENT].astype(str).map(_leading_year)
    dated = years.notna()
    undated = int((~dated).sum())

    if undated == len(working):
        examples = working[DOCUMENT].astype(str).head(_MAX_EXAMPLES).tolist()
        shown = ", ".join(repr(name) for name in examples)
        return Result[PreparedPanel].failure(
            Diagnostic.error(
                "PANEL_NO_DATES",
                "no document name in this corpus starts with a parseable year, so topic prevalence over "
                "time cannot be plotted. This panel needs names of the form 'YYYY...' or 'YYYY-MM-DD...', "
                f"e.g. '1934-01-03_franklin d roosevelt_sotu.txt'. Got, for example: {shown}.",
                examples=examples,
            )
        )
    if undated:
        examples = working.loc[~dated, DOCUMENT].astype(str).head(_MAX_EXAMPLES).tolist()
        shown = ", ".join(repr(name) for name in examples)
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_NO_DATE",
                f"{undated} document(s) had no parseable year at the start of their name and were left out "
                f"of the plot rather than being guessed at or bucketed as year zero, e.g. {shown}.",
                dropped=undated,
                examples=examples,
            )
        )

    working = working.loc[dated].copy()
    years = years[dated]
    # A passage too short to score has no dominant topic; it is left out and
    # counted rather than crashing the cast to int.
    topic = pd.to_numeric(working[DOMINANT_TOPIC], errors="coerce")
    unscored = int(topic.isna().sum())
    if unscored:
        diagnostics.append(
            Diagnostic.info("PANEL_UNSCORED_ROWS", f"{unscored} row(s) had no dominant topic and were left out.")
        )
        working, years, topic = working[topic.notna()].copy(), years[topic.notna()], topic[topic.notna()]
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no row has a dominant topic"), *diagnostics)
    working[DOMINANT_TOPIC] = topic.astype(int)
    working[PERIOD] = [_bucket_of(int(year), bucket) for year in years]

    # -- numeric contribution, only when the measure needs it -----------
    # "documents" never reads Contribution, so a broken value there should
    # not remove a document from a count it plays no part in.
    if measure == "contribution":
        numeric = pd.to_numeric(working[CONTRIBUTION], errors="coerce")
        finite = numeric.map(_is_finite)
        bad = int((~finite).sum())
        if bad:
            diagnostics.append(
                Diagnostic.warning(
                    "PANEL_BAD_NUMERIC",
                    f"{bad} document(s) had a non-numeric or non-finite Contribution and were left out of "
                    "the contribution-weighted total.",
                    dropped=bad,
                )
            )
        working = working.loc[finite].copy()
        working[CONTRIBUTION] = numeric[finite]

    if working.empty:
        return Result[PreparedPanel].failure(
            *diagnostics,
            Diagnostic.error(
                "PANEL_NO_DATA",
                "no documents were left to plot once undated names and unusable Contribution values were "
                "removed; check the source frame or try measure='documents', which does not need Contribution.",
            ),
        )

    # -- how many documents landed in each bucket, for the thin-bucket check
    # and for "N of M addresses" wording on the marks --
    doc_counts_by_period: dict[int, int] = {
        int(period): int(count) for period, count in working.groupby(PERIOD)[DOCUMENT_ID].nunique().items()
    }
    thin = sorted(period for period, count in doc_counts_by_period.items() if count == 1)
    if thin:
        labels = ", ".join(_period_label(period, bucket) for period in thin)
        diagnostics.append(
            Diagnostic.info(
                "PANEL_THIN_BUCKET",
                f"{len(thin)} time bucket(s) hold only one document, so their 100% band is one speech, not "
                f"a trend: {labels}.",
                buckets=thin,
            )
        )

    # -- the aggregate a band's height actually is -----------------------
    if measure == "documents":
        grouped = (
            working.groupby([PERIOD, DOMINANT_TOPIC])
            .agg(_count=(DOCUMENT, "count"), _value=(DOCUMENT, "count"), _keywords=(TOPIC_KEYWORDS, "first"))
            .reset_index()
        )
    else:
        grouped = (
            working.groupby([PERIOD, DOMINANT_TOPIC])
            .agg(_count=(DOCUMENT, "count"), _value=(CONTRIBUTION, "sum"), _keywords=(TOPIC_KEYWORDS, "first"))
            .reset_index()
        )
    if normalize:
        period_totals = grouped.groupby(PERIOD)["_value"].transform("sum")
        grouped["_share"] = grouped["_value"] / period_totals
    else:
        grouped["_share"] = grouped["_value"]

    # Topic number ascending, not order-of-appearance: a colour must mean the
    # same topic on every run, and sorting by an arbitrary label the fit
    # assigned is the only order that is stable across re-runs and re-sorts.
    # Named by their words, in topic-number order (a colour must mean the
    # same topic on every run): "Topic 2: world, peace, nations".
    words = {int(t): _short_keywords(k) for t, k in zip(grouped[DOMINANT_TOPIC], grouped["_keywords"], strict=True)}
    name_of = {t: f"Topic {t}: {w}" if w else f"Topic {t}" for t, w in words.items()}
    groups = tuple(name_of[topic] for topic in sorted(name_of))

    marks: list[PanelMark] = []
    for _, row in grouped.sort_values([PERIOD, DOMINANT_TOPIC], kind="stable").iterrows():
        period = int(row[PERIOD])
        topic = int(row[DOMINANT_TOPIC])
        count = int(row["_count"])
        value = float(row["_share"])
        period_label = _period_label(period, bucket)
        doc_total = doc_counts_by_period.get(period, count)
        keywords = _short_keywords(row["_keywords"])
        value_desc = _value_desc(measure, normalize, value, count, doc_total)
        label = f"{period_label} · Topic {topic}"
        if keywords:
            label += f" ({keywords})"
        label += f" — {value_desc}"
        marks.append(
            PanelMark(
                key=f"{period}:{topic}",
                label=label,
                x=float(period),
                y=value,
                group=name_of[topic],
                size=float(count),
                evidence=Evidence(
                    scope="documents",
                    filters=((PERIOD, str(period)), (DOMINANT_TOPIC, str(topic))),
                    count=count,
                    describe=f"{count} document(s) dominant in topic {topic} during {period_label}: {value_desc}",
                ),
            )
        )

    x_label = "Decade" if bucket == "decade" else "Year"
    if measure == "documents":
        y_label = "Share of addresses (dominant topic)" if normalize else "Number of addresses (dominant topic)"
    else:
        y_label = "Share of contribution mass" if normalize else "Total contribution mass"

    n_docs = int(working[DOCUMENT_ID].nunique())
    n_periods = int(grouped[PERIOD].nunique())
    subtitle = f"{n_docs} addresses across {n_periods} {bucket}(s) · {len(groups)} topic(s)"

    prepared = PreparedPanel(
        panel=LDA_PREVALENCE.name,
        shape="stream",
        title="Topic prevalence over time",
        subtitle=subtitle,
        marks=tuple(marks),
        x_label=x_label,
        y_label=y_label,
        provenance=provenance,
        data=working,
        groups=groups,
        notes=_NOTES,
    )
    return Result.success(prepared, *diagnostics)


def _leading_year(document: str) -> int | None:
    """The year at the start of a document name, or ``None`` if there isn't one.

    Deliberately narrow: only a leading ``YYYY`` (optionally with
    ``-MM-DD``) counts. A year elsewhere in the name is a coincidence this
    function refuses to resolve, because a wrong date silently attached to a
    real document is worse than no date at all.
    """
    match = _LEADING_DATE.match(document)
    return int(match.group(1)) if match else None


def _bucket_of(year: int, bucket: str) -> int:
    """The time bucket a year falls in: itself, or the decade it opens."""
    return year if bucket == "year" else year - (year % 10)


def _period_label(period: int, bucket: str) -> str:
    """How a bucket reads to a person: "1934" or "1930s"."""
    return str(period) if bucket == "year" else f"{period}s"


def _is_finite(value: Any) -> bool:
    """True for a real, finite number. NaN and +/-inf cannot be summed honestly."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _short_keywords(keywords: Any) -> str:
    """The first few words of a topic's keyword string, for a mark's label.

    ``Topic keywords`` already holds the model's top ten words; a label with
    all ten is unreadable, so this keeps the first three -- enough to remind
    a reader what the topic is about without turning the legend into a wall
    of text.
    """
    text = str(keywords or "")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return ", ".join(parts[:_LABEL_KEYWORDS])


def _value_desc(measure: str, normalize: bool, value: float, count: int, doc_total: int) -> str:
    """One clause describing a mark's height, in the units it is actually in."""
    if measure == "documents":
        if normalize:
            return f"{value:.0%} of {doc_total} address(es) that period"
        return f"{count} of {doc_total} address(es) that period"
    if normalize:
        return f"{value:.0%} of that period's contribution mass"
    return f"{value:.2f} total contribution"


_NOTES: tuple[str, ...] = (
    "A document's dominant topic is a summary of a mixture, not the whole of it: a speech about several "
    "things is still counted once, under whichever topic had the largest share of it.",
    "Time buckets are not the same size -- some decades have more addresses than others, and this corpus "
    "does not have one per calendar year. Without normalize, a taller band can mean more speeches that "
    "period rather than more of that topic; a share only reads as a rate once normalize is on.",
    "A topic's rise or fall across the plot can reflect the model's vocabulary -- the words that happened "
    "to define that topic in this fit -- as much as it reflects history. Read a spike alongside the "
    "topic's keywords before calling it a trend.",
    "Topic numbers are arbitrary labels assigned during the fit, not a ranking or a timeline: topic 0 is "
    "not older, bigger or more important than topic 5, and a re-run of the model can renumber them entirely.",
    "A bucket built from a single address is not a trend. A 100% band from one speech says only that the "
    "one speech had a dominant topic; PANEL_THIN_BUCKET names which buckets these are.",
)


LDA_PREVALENCE = PanelDefinition(
    name="lda_prevalence",
    title="Topic prevalence over time",
    question="Which topics rise and fall across the years?",
    tool="lda_gensim",
    shape="stream",
    summary="How much each topic takes up in each period, by dominant documents or by contribution, as stacked bands.",
    requires=(DOCUMENT_ID, DOCUMENT, DOMINANT_TOPIC, CONTRIBUTION, TOPIC_KEYWORDS),
    params=(
        PanelParam(
            name="bucket",
            type="choice",
            default="decade",
            choices=("year", "decade"),
            label="Time grouping",
            help=(
                "How to group documents along the x axis. With roughly one State of the Union address per "
                "year, 'year' produces a very spiky picture; 'decade' is usually more readable."
            ),
        ),
        PanelParam(
            name="measure",
            type="choice",
            default="documents",
            choices=("documents", "contribution"),
            label="What a band measures",
            help=(
                "Whether a band's height is the COUNT of documents whose dominant topic it is, or the SUM "
                "of their Contribution (how strongly each document actually leaned into that topic). These "
                "are different quantities -- 'documents' counts speeches, 'contribution' weighs them -- and "
                "there is no blended default."
            ),
        ),
        PanelParam(
            name="normalize",
            type="bool",
            default=True,
            label="Show as a share",
            help=(
                "When on, each time bucket's bands sum to 1 (a share of that bucket). When off, bands show "
                "raw totals. A share hides how many documents a bucket had -- a 100% band from one speech "
                "looks identical to one built from twelve."
            ),
        ),
    ),
    build=lda_prevalence,
    notes=_NOTES,
)
