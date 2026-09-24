"""Keyness volcano — effect size against strength of evidence.

The keyness table (``core/analysis/keyness.py``) scores every word with G2,
a log-likelihood statistic, and with Log Ratio, a base-2 effect size. Drawn
as a bar chart of G2 — which is what the generic chart path offers — the
figure answers "which words differ most reliably?" and hides the question a
reader actually has to settle: *by how much?*

The two numbers disagree, and the disagreement is the finding:

* **High G2, low Log Ratio.** A very frequent word whose small difference is
  measured precisely. Real, and usually uninteresting: ``the`` appearing 3%
  more often is not a discovery.
* **High Log Ratio, low G2.** A rare word whose large apparent difference
  rests on a handful of tokens. Striking, and not yet evidence.
* **High on both.** Words that are substantially more common in one group
  and measured well enough to believe. These are the ones worth reading in
  context, and on a bar chart of G2 they are indistinguishable from the
  first case.

A volcano plot puts effect size on x and evidence on y, so those three
regions are three places on the page. It is standard in corpus linguistics
for exactly this reason, and it costs nothing the table does not already
contain.

Reference lines are drawn with their reason attached (``Annotation.note``):
a line at G2 = 3.84 is meaningless to most readers, and "p < 0.05 at 1
degree of freedom" is not.

This is the reference implementation for :mod:`core.viz.panelspec` — the
shortest complete example of a panel builder, and the one to copy when
adding another. It is a pure function of (frame, params): no plotly, no
filesystem, no corpus access.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import FUNCTION_WORDS
from core.viz.panelspec import (
    Annotation,
    Evidence,
    PanelDefinition,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
    counted_on_lemmas,
)

__all__ = ["KEYNESS_VOLCANO", "SIGNIFICANCE_LEVELS", "keyness_volcano"]

WORD = "Word"
G2 = "G2 (log-likelihood)"
LOG_RATIO = "Log Ratio"
FREQ_A = "Freq Group A (pattern docs)"
FREQ_B = "Freq Group B (other docs)"
OVERREPRESENTED = "Overrepresented in"

#: Critical values of the chi-square distribution at 1 degree of freedom.
#: G2 is compared against these, which is the convention the keyness
#: literature uses and the reason ``p < 0.05`` is so often quoted as
#: ``G2 > 3.84``. The mapping is here rather than inline so the line, the
#: label and the note cannot drift apart.
SIGNIFICANCE_LEVELS: dict[str, float] = {
    "0.05": 3.84,
    "0.01": 6.63,
    "0.001": 10.83,
}

_MARK_CAP = 2000
# Above this many words, "all of them are significant" is a statement about
# the input having been pre-filtered rather than about the corpus.
_PRESELECTED_HINT = 20


def keyness_volcano(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Prepare the volcano: Log Ratio on x, G2 on y, one point per word."""
    label_top = int(params["label-top"])
    label_by = str(params["label-by"])
    min_frequency = int(params["min-frequency"])
    level = str(params["significance"])
    threshold = SIGNIFICANCE_LEVELS[level]

    diagnostics: list[Diagnostic] = []
    working = frame.copy()
    # Read the words back in the text the way the run counted them.
    on_lemmas = counted_on_lemmas(provenance.settings)

    # Numbers first: a word whose G2 or Log Ratio is not a finite number
    # cannot be placed on the page, and placing it at zero would invent a
    # finding. Drop with a count, never silently.
    for column in (G2, LOG_RATIO):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    for column in (FREQ_A, FREQ_B):
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0)
    placeable = working[G2].apply(_is_finite) & working[LOG_RATIO].apply(_is_finite)
    unplaceable = int((~placeable).sum())
    if unplaceable:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_BAD_NUMERIC",
                f"{unplaceable} row(s) had a non-numeric or infinite {G2}/{LOG_RATIO} and were left out "
                "of the plot; the table beside it still shows them.",
                dropped=unplaceable,
            )
        )
    working = working[placeable]

    working["_total"] = working[FREQ_A] + working[FREQ_B]
    if min_frequency:
        before = len(working)
        working = working[working["_total"] >= min_frequency]
        removed = before - len(working)
        if removed:
            diagnostics.append(
                Diagnostic.info(
                    "PANEL_FREQUENCY_FILTERED",
                    f"{removed} word(s) below the minimum combined frequency of {min_frequency} were left out.",
                    removed=removed,
                    minimum=min_frequency,
                )
            )

    if working.empty:
        return Result[PreparedPanel].failure(
            Diagnostic.error(
                "PANEL_NO_DATA",
                "no words left to plot after filtering; lower min-frequency or check the keyness run",
            ),
            *diagnostics,
        )

    if len(working) > _MARK_CAP:
        # One SVG element per word; a 50k-word keyness table would produce a
        # figure no browser draws in reasonable time. Keep the strongest
        # evidence, and say what was cut rather than trimming quietly.
        kept = working.nlargest(_MARK_CAP, G2)
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_TOO_MANY_MARKS",
                f"{len(working)} words is more than a readable volcano holds; drew the {_MARK_CAP} with the "
                "strongest evidence (highest G2). Raise min-frequency or lower the keyness top-n to choose "
                "differently.",
                drawn=_MARK_CAP,
                available=len(working),
            )
        )
        working = kept

    ranked = _label_order(working, label_by)
    # Labels go to content words: "of", "the" and "you" had the strongest
    # evidence on the real corpus and took the labels a reader wanted for
    # "expenditure" and "labor". They stay on the plot as points.
    content = ranked[~ranked[WORD].astype(str).str.lower().isin(FUNCTION_WORDS)]
    labelled = set(content.head(label_top)[WORD].astype(str)) if label_top else set()

    groups = tuple(sorted({str(value) for value in working[OVERREPRESENTED].fillna("")} - {""}))
    marks: list[PanelMark] = []
    for _, row in working.iterrows():
        word = str(row[WORD])
        freq_a, freq_b = int(row[FREQ_A]), int(row[FREQ_B])
        total = freq_a + freq_b
        g2_value, ratio = float(row[G2]), float(row[LOG_RATIO])
        marks.append(
            PanelMark(
                key=word,
                label=word,
                x=ratio,
                y=g2_value,
                group=str(row[OVERREPRESENTED] or ""),
                size=float(total),
                labelled=word in labelled,
                evidence=Evidence(
                    scope="terms",
                    filters=((WORD, word),),
                    count=total,
                    phrase=word,
                    lemma=on_lemmas,
                    describe=(
                        f"{word}: {freq_a} in group A, {freq_b} in group B — G2 {g2_value:.1f}, Log Ratio {ratio:+.2f}"
                    ),
                ),
            )
        )

    above = sum(1 for mark in marks if mark.y >= threshold)
    if marks and above == len(marks) and len(marks) > _PRESELECTED_HINT:
        # Every word clearing the line usually means the table arrived
        # pre-ranked -- `keyness --top-n` keeps the strongest N -- so the
        # y axis is truncated and "all significant" is an artifact of the
        # input, not a property of the corpus. Saying so is the difference
        # between a correct figure and an informative one.
        diagnostics.append(
            Diagnostic.info(
                "PANEL_ALL_ABOVE_THRESHOLD",
                f"every one of the {len(marks)} words plotted is above the p < {level} line. That usually means "
                "the keyness run was already cut to its strongest results (--top-n), so this plot shows the top "
                "of the distribution rather than all of it. Re-run keyness with --top-n 0 to see the whole shape.",
                words=len(marks),
                level=level,
            )
        )
    prepared = PreparedPanel(
        panel=KEYNESS_VOLCANO.name,
        shape="scatter_labelled",
        title="Keyness: effect size against evidence",
        subtitle=_subtitle(len(marks), above, level),
        marks=tuple(marks),
        x_label="Log Ratio  (how much more common, doubling per unit)",
        y_label="G2 log-likelihood  (how strong the evidence is)",
        provenance=provenance,
        data=working.drop(columns=["_total"]),
        annotations=(
            Annotation(
                kind="vline",
                value=0.0,
                label="no difference",
                note="A Log Ratio of 0 means the word is equally common in both groups. Each unit right or left is a doubling.",
            ),
            Annotation(
                kind="hline",
                value=threshold,
                label=f"p < {level}",
                note=(
                    f"G2 = {threshold} is the p < {level} threshold at 1 degree of freedom. "
                    "It is a conventional cut, not a decision: nothing about a word changes as it crosses."
                ),
            ),
        ),
        groups=groups,
        notes=_NOTES,
    )
    return Result.success(prepared, *diagnostics)


def _subtitle(total: int, above: int, level: str) -> str:
    """One line of orientation under the title.

    When every word clears the line, reporting the count would imply the
    corpus is uniformly significant; naming the truncation instead tells the
    reader what they are actually looking at.
    """
    side = "right of centre = more common in group A"
    if total and above == total:
        return f"{total} words, all above the p < {level} line (the table was already cut to the strongest) · {side}"
    return f"{total} words · {above} above the p < {level} line · {side}"


def _is_finite(value: Any) -> bool:
    """True for a real, finite number. NaN and +/-inf cannot be placed."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _label_order(frame: pd.DataFrame, label_by: str) -> pd.DataFrame:
    """Which words get their name written, in the order the reader asked for.

    Ranked by evidence (G2) or by effect size (|Log Ratio|). There is no
    composite default: the two orders answer different questions, and
    inventing a blend would put a number on the page that no column holds.
    Ties break on the word so a redraw labels the same points.
    """
    rank = frame[LOG_RATIO].abs() if label_by == "effect" else frame[G2]
    return frame.assign(_rank=rank).sort_values(["_rank", WORD], ascending=[False, True], kind="stable")


_NOTES: tuple[str, ...] = (
    "G2 measures how much evidence there is for a difference, not how big the difference is. "
    "On a large corpus a very small difference in a very frequent word can score high.",
    "Log Ratio measures how big the difference is: each unit is a doubling. It is unstable on rare "
    "words, which is why the keyness run adds smoothing before taking the ratio.",
    "Words in the upper left and upper right are both large in effect and well evidenced. Those are "
    "the ones worth reading in context before drawing a conclusion.",
    "Every point here is a word type, not a passage. A high score is a reason to go and read the "
    "concordance, not a result on its own.",
)


KEYNESS_VOLCANO = PanelDefinition(
    name="keyness_volcano",
    title="Keyness volcano",
    question="Which words set one group of documents apart, by how much, and on how much evidence?",
    tool="keyness",
    shape="scatter_labelled",
    summary="Effect size (Log Ratio) against strength of evidence (G2), one point per word.",
    requires=(WORD, G2, LOG_RATIO, FREQ_A, FREQ_B, OVERREPRESENTED),
    params=(
        PanelParam(
            name="label-top",
            type="int",
            default=20,
            minimum=0,
            maximum=200,
            label="Words to label",
            help="How many words get their name written on the plot. 0 labels none.",
        ),
        PanelParam(
            name="label-by",
            type="choice",
            default="evidence",
            choices=("evidence", "effect"),
            label="Label the strongest",
            help=(
                "Which words to name: 'evidence' picks the highest G2 (most reliably different), "
                "'effect' picks the largest Log Ratio (most different)."
            ),
        ),
        PanelParam(
            name="significance",
            type="choice",
            default="0.05",
            choices=("0.05", "0.01", "0.001"),
            label="Threshold line",
            help="Where to draw the significance line: p < 0.05, 0.01 or 0.001 at 1 degree of freedom.",
        ),
        PanelParam(
            name="min-frequency",
            type="int",
            default=0,
            minimum=0,
            maximum=1_000_000,
            label="Minimum total frequency",
            help=(
                "Leave out words occurring fewer than this many times across both groups. "
                "Rare words produce the most extreme Log Ratios and the least trustworthy ones."
            ),
        ),
    ),
    build=keyness_volcano,
    notes=_NOTES,
)
