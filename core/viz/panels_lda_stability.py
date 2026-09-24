"""Topic stability -- does a topic come back when the model is refitted?

``core/analysis/topic_stability.py`` refits the same corpus at several random
seeds and matches each seed's topics back to a reference seed's by top-word
overlap (Jaccard). The result is a ``summary`` table, one row per reference
topic: how well it matched on average (``Mean Jaccard``) and at its worst
(``Min Jaccard``), across how many other seeds (``Seeds``), and whether it
clears a stability bar (``Stable``). Nothing else in the suite draws that
table, and without it a reader has no way to tell "the economy topic" (comes
back every time) from an artefact of one lucky initialization.

Two series, same 0..1 scale, one row per topic:

* **Worst seed** -- the topic's Min Jaccard. The cautionary number: the one
  seed where this topic looked least like the reference is what a reader
  needs before naming the topic, not the flattering average.
* **Mean overlap** -- the topic's Mean Jaccard. Always at or above Worst
  seed, so it draws as the lighter band the worst-case bar sits inside --
  the same foreground/background relationship ``panels_lda_relevance.py``
  uses for a topic's own count against the corpus total.

Rows are ordered by Min Jaccard descending, so the most reproducible topics
read first and the ones a reader should be suspicious of sink to the bottom
-- the opposite of ordering by topic number, which is an arbitrary label the
fit assigned and carries no information (see the notes below).

Pure function of (frame, params, provenance): no plotly, no filesystem, no
model access.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panelspec import (
    Annotation,
    Evidence,
    PanelDefinition,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
)

__all__ = ["LDA_STABILITY", "lda_stability"]

TOPIC = "Topic"
REFERENCE_WORDS = "Reference words"
MEAN_JACCARD = "Mean Jaccard"
MIN_JACCARD = "Min Jaccard"
SEEDS = "Seeds"
STABLE = "Stable"
#: The run's own stability cutoff. Column added to the summary so the panel
#: can draw a reference line at the threshold the run actually used, instead
#: of refusing to guess where the line was (the old behaviour).
STABLE_AT = "Stable at"

# Foreground first: the renderer draws the first group opaque and the second
# translucent behind it. Worst seed <= Mean overlap always (see
# core.analysis.topic_stability), so "Worst seed" is the smaller, solid bar
# sitting inside the lighter "Mean overlap" band -- the reading that matters
# most (how bad did it get) reads as the solid shape, the flattering average
# as the context around it.
_WORST_GROUP = "Worst seed"
_MEAN_GROUP = "Mean overlap"

#: How many of a topic's reference words label its row ("Topic 3: war, army,
#: battle"). More than a handful stops being a label and starts being the
#: whole word list, which the hover text already carries in full.
_LABEL_WORDS = 3

#: Above this share of topics failing Stable, the honest reading is "this
#: run's k or corpus size is the problem", not "these particular topics are
#: weak" -- see PANEL_MOSTLY_UNSTABLE below.
_MOSTLY_UNSTABLE_SHARE = 0.5


def lda_stability(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Prepare one bar pair per reference topic: mean overlap and worst seed."""
    top_n = int(params["top-n"])

    diagnostics: list[Diagnostic] = []
    working = frame.copy()

    # Both scores must be real, finite numbers to sit on an axis. A topic
    # whose Jaccard could not be computed is left out with a count, never
    # drawn at zero -- zero would claim the topic matched nothing, which is
    # a different (and much worse) finding than "unknown".
    for column in (MEAN_JACCARD, MIN_JACCARD):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    placeable = working[MEAN_JACCARD].apply(_is_finite) & working[MIN_JACCARD].apply(_is_finite)
    unplaceable = int((~placeable).sum())
    if unplaceable:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_BAD_NUMERIC",
                f"{unplaceable} topic(s) had a non-numeric or non-finite {MEAN_JACCARD}/{MIN_JACCARD} and "
                "were left out of the plot; the table beside it still shows them.",
                dropped=unplaceable,
            )
        )
    working = working[placeable]

    if working.empty:
        return Result[PreparedPanel].failure(
            Diagnostic.error(
                "PANEL_NO_DATA",
                "no topics left to plot once unusable Jaccard values were removed; check the stability run",
            ),
            *diagnostics,
        )

    working[SEEDS] = pd.to_numeric(working[SEEDS], errors="coerce").fillna(0).astype(int)
    working[STABLE] = working[STABLE].astype(bool)

    # The run's own cutoff, for the reference line. Optional on purpose: an
    # older summary.csv predates the column, and refusing to draw it at all
    # would throw away a correct table because one number is missing. When
    # it is absent the panel says so rather than guessing a threshold.
    stable_at: float | None = None
    if STABLE_AT in working.columns:
        thresholds = pd.to_numeric(working[STABLE_AT], errors="coerce").dropna()
        if not thresholds.empty:
            stable_at = float(thresholds.iloc[0])
            if thresholds.nunique() > 1:
                diagnostics.append(
                    Diagnostic.warning(
                        "PANEL_THRESHOLD_CONFLICT",
                        "rows of this summary disagree about the Stable cutoff; drew the first one "
                        f"({stable_at}). Regenerate the summary from the current engine.",
                    )
                )
        else:
            diagnostics.append(
                Diagnostic.warning(
                    "PANEL_THRESHOLD_MISSING",
                    "the summary carries no usable Stable-at value, so no threshold line is drawn.",
                )
            )
    else:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_THRESHOLD_MISSING",
                "this summary predates the Stable-at column; drew no threshold line.",
            )
        )
    # The published table keeps what it actually has: the threshold column
    # when present, the boolean and scores always. A hand-built or older
    # summary must not KeyError the panel for a column it never had.
    published_columns = (
        TOPIC,
        REFERENCE_WORDS,
        MEAN_JACCARD,
        MIN_JACCARD,
        SEEDS,
        STABLE,
        STABLE_AT,
    )

    unstable_share = float((~working[STABLE]).mean())
    if unstable_share > _MOSTLY_UNSTABLE_SHARE:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_MOSTLY_UNSTABLE",
                f"{int((~working[STABLE]).sum())} of {len(working)} topic(s) are not Stable -- more than half. "
                "That usually means this run asked for more topics than the corpus supports, or the corpus "
                "is too small for this many topics; try fewer topics or more documents rather than trusting "
                "individual topic names from this fit.",
                unstable=int((~working[STABLE]).sum()),
                total=len(working),
            )
        )

    # Most stable first: Min Jaccard descending, ties broken by Topic so a
    # redraw orders the same rows the same way.
    ranked = working.sort_values([MIN_JACCARD, TOPIC], ascending=[False, True], kind="stable")
    drawn = ranked.head(top_n)

    marks: list[PanelMark] = []
    for position, (_, row) in enumerate(drawn.iterrows()):
        topic = int(row[TOPIC])
        mean_jaccard = float(row[MEAN_JACCARD])
        min_jaccard = float(row[MIN_JACCARD])
        seeds = int(row[SEEDS])
        words = [w.strip() for w in str(row[REFERENCE_WORDS]).split(",") if w.strip()]
        label = f"Topic {topic}: {', '.join(words[:_LABEL_WORDS])}"
        describe = (
            f"Topic {topic} ({', '.join(words)}): mean overlap {mean_jaccard:.0%}, worst seed "
            f"{min_jaccard:.0%} across {seeds} seed(s)."
        )
        evidence = Evidence(
            scope="rows",
            filters=((TOPIC, str(topic)),),
            count=seeds,
            describe=describe,
        )
        marks.append(
            PanelMark(
                key=f"{topic}:worst",
                label=label,
                x=min_jaccard,
                y=float(position),
                group=_WORST_GROUP,
                evidence=evidence,
            )
        )
        marks.append(
            PanelMark(
                key=f"{topic}:mean",
                label=label,
                x=mean_jaccard,
                y=float(position),
                group=_MEAN_GROUP,
                evidence=evidence,
            )
        )

    prepared = PreparedPanel(
        panel=LDA_STABILITY.name,
        shape="ranked_bars",
        title="Topic stability across seeds",
        subtitle=f"{len(drawn)} of {len(working)} topic(s) shown, most reproducible first",
        marks=tuple(marks),
        x_label="Jaccard overlap with the reference seed's top words (0-1)",
        y_label="Topic, most stable first",
        provenance=provenance,
        data=drawn[[column for column in published_columns if column in drawn.columns]].reset_index(drop=True),
        # A vline at the run's own Stable-at threshold, when the summary
        # carries it: the line is where the run drew the line, stated as the
        # run stated it. When the column is missing (an older summary), the
        # threshold stays undrawn and the warning says why -- never guessed
        # from the Stable/not-Stable split, which could only bound it.
        annotations=(
            (
                Annotation(
                    kind="vline",
                    value=stable_at,
                    label=f"Stable at {stable_at:g}",
                    note=("the cutoff this run used: a topic is Stable when its worst-seed overlap reaches it"),
                ),
            )
            if stable_at is not None
            else ()
        ),
        groups=(_WORST_GROUP, _MEAN_GROUP),
        notes=_NOTES,
    )
    return Result.success(prepared, *diagnostics)


def _is_finite(value: Any) -> bool:
    """True for a real, finite number. NaN and +/-inf cannot be placed."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


_NOTES: tuple[str, ...] = (
    "Overlap is measured on each topic's top words only. Two topics can share every one of their top "
    "words and still differ further down their distributions -- this panel cannot see that tail.",
    "Stability says a topic is reproducible, not that it is meaningful. A topic that comes back at every "
    "seed can still be a mixture of unrelated words; naming it is still an interpretation, not a fact "
    "the model states.",
    "A small corpus produces unstable topics at almost any topic count. Low overlap here can be a fact "
    "about how few documents this run had, not about the number of topics asked for.",
    "Topic numbers are arbitrary and differ between seeds -- gensim does not promise that topic 3 means "
    "the same thing twice. That is why matching by word overlap is needed before topics can be compared "
    "at all, and it is why this panel's rows are never ordered by topic number.",
)


LDA_STABILITY = PanelDefinition(
    name="lda_stability",
    title="Topic stability across seeds",
    question="Which topics come back when the model is fitted again with a different seed?",
    tool="lda_stability",
    shape="ranked_bars",
    summary=(
        "One reference topic per row: mean and worst-seed word overlap (Jaccard) against topics matched "
        "from refits at other random seeds, most reproducible first."
    ),
    requires=(TOPIC, REFERENCE_WORDS, MEAN_JACCARD, MIN_JACCARD, SEEDS, STABLE),
    params=(
        PanelParam(
            name="top-n",
            type="int",
            default=20,
            minimum=1,
            maximum=100,
            label="Topics to show",
            help="How many reference topics to draw, taken from the most stable end of the ordering.",
        ),
    ),
    build=lda_stability,
    notes=_NOTES,
)
