"""Collocation strength -- association measure against frequency.

The collocations table (``core/analysis/collocations.py``) scores every word
pair with seven association measures computed from one contingency table.
They disagree with each other in ways that matter, and this panel is built
on that disagreement rather than around it -- it is the collocation
counterpart of the keyness volcano (``panels_keyness.py``), same argument,
different axes.

PMI (``log2(observed / expected)``) is the sharpest illustration. Because it
divides by the *expected* count, a pair whose words each occur only twice and
which always occur together scores as high as PMI can go: expected is tiny,
so observed/expected is huge, and the score tells a reader nothing about
whether the pair matters -- only that it is rare and self-consistent. Log
Dice, T-score and G2 do not share this bias (see each measure's help text
below and the engine's own docstring, which this file's claims are checked
against), which is why log-dice is the default y axis here rather than PMI.

Plotting co-occurrence count against the chosen measure puts three
regions in three different places on the page:

* **Strong and common** (upper right). Frequent enough to trust, and
  strongly associated by whichever measure was chosen. These are the pairs
  worth reading in context.
* **Strong but rare** (upper left). A handful of tokens driving an extreme
  score -- PMI's classic trap, and the reason ``PANEL_RARE_PAIRS_DOMINATE``
  exists: when the chosen measure is PMI and the labelled pairs are mostly
  rare ones, that is a property of the formula, not a finding about the
  corpus.
* **Common but weak** (lower right/middle). High-frequency pairs -- often
  function-word bigrams -- whose association is unremarkable once expected
  co-occurrence under independence is accounted for.

The x axis is co-occurrence count on a log10 scale, stated as such in the
axis label. Counts in a real collocation table span two or three orders of
magnitude (a handful of occurrences up to several thousand), and the scatter
renderer draws linear axes; plotting raw counts under a linear-looking label
would crowd every rare pair into the first few pixels and silently mislabel
the axis. A pair with zero co-occurrences has no log10 position and is
dropped with a diagnostic rather than placed at an invented x.

This is a pure function of (frame, params, provenance): no plotly, no
filesystem, no corpus access. Copied from ``panels_keyness.py``, the
reference implementation for :mod:`core.viz.panelspec`.
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
    counted_on_lemmas,
)

__all__ = ["COLLOCATION_STRENGTH", "collocation_strength"]

# Column names, spelled exactly as core/analysis/collocations.py's _COLUMNS
# produces them -- this module reads its output and nothing else.
WORD_1 = "Word 1"
WORD_2 = "Word 2"
PAIR = "Pair"
COOCCURRENCES = "Co-occurrences"
WORD_1_FREQ = "Word 1 Freq"
WORD_2_FREQ = "Word 2 Freq"
PMI = "PMI"
T_SCORE = "T-Score"
G2 = "G2 (log-likelihood)"
LOG_DICE = "Log Dice"

#: Which column backs the y axis for each choice of ``measure``.
_MEASURE_COLUMNS: dict[str, str] = {
    "pmi": PMI,
    "log-dice": LOG_DICE,
    "t-score": T_SCORE,
    "g2": G2,
}

_MEASURE_DISPLAY: dict[str, str] = {
    "pmi": "PMI",
    "log-dice": "Log Dice",
    "t-score": "T-score",
    "g2": "G2",
}

_Y_LABELS: dict[str, str] = {
    "pmi": "PMI  (log2 observed/expected -- rewards rare pairs)",
    "log-dice": "Log Dice  (14 = theoretical maximum overlap)",
    "t-score": "T-score  ((observed - expected) / sqrt(observed))",
    "g2": "G2 log-likelihood  (strength of evidence, not effect size)",
}

_MARK_CAP = 2000
# The critical value of the chi-square distribution at 1 degree of freedom,
# p < 0.05. Collocation G2 is the same four-cell Dunning statistic keyness
# uses (core/analysis/collocations.py says so explicitly: "computed by the
# same engine that backs keyness ... so the suite reports one number for one
# concept"), so the same conventional cut applies for the same reason.
_G2_SIGNIFICANCE_005 = 3.84
# Log Dice's maximum: the engine adds this offset (Sketch Engine's constant,
# core/analysis/collocations.py::_LOG_DICE_OFFSET) so total overlap scores it.
# Restated rather than imported because the engine's name is private;
# tests/test_panels_collocations.py pins the two equal.
_LOG_DICE_CEILING = 14.0


def collocation_strength(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Prepare the scatter: log10(co-occurrences) on x, the chosen measure on y."""
    measure = str(params["measure"])
    label_top = int(params["label-top"])
    min_count = int(params["min-count"])
    measure_column = _MEASURE_COLUMNS[measure]
    # Reading a pair in context means searching the text for it. An adjacent
    # pair is a phrase; a pair counted anywhere inside a token window is not,
    # and searching for "w1 w2" would find only its adjacent occurrences --
    # an undercount presented as the evidence. So windowed pairs get no lookup.
    adjacent = str(provenance.settings.get("span", "adjacent")).lower() == "adjacent"
    on_lemmas = counted_on_lemmas(provenance.settings)

    diagnostics: list[Diagnostic] = []
    working = frame.copy()

    working[COOCCURRENCES] = pd.to_numeric(working[COOCCURRENCES], errors="coerce")
    working[measure_column] = pd.to_numeric(working[measure_column], errors="coerce")
    for column in (WORD_1_FREQ, WORD_2_FREQ):
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0)

    # A pair cannot be placed unless its count is a positive, finite number
    # (log10 needs one -- zero and negative counts have no log10, and the
    # engine never emits them, but a hand-built or corrupted table might) and
    # its chosen measure is a finite number. Drop rather than invent a
    # position, same as the keyness volcano does for G2/Log Ratio.
    placeable = working[COOCCURRENCES].apply(_is_positive_finite) & working[measure_column].apply(_is_finite)
    unplaceable = int((~placeable).sum())
    if unplaceable:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_BAD_NUMERIC",
                f"{unplaceable} pair(s) had a non-numeric/non-finite {measure_column}, or a {COOCCURRENCES} "
                "count that cannot be placed on a log10 axis (zero, negative, or not a number), and were left "
                "out of the plot; the table beside it still shows them.",
                dropped=unplaceable,
            )
        )
    working = working[placeable]

    if min_count:
        before = len(working)
        working = working[working[COOCCURRENCES] >= min_count]
        removed = before - len(working)
        if removed:
            diagnostics.append(
                Diagnostic.info(
                    "PANEL_FREQUENCY_FILTERED",
                    f"{removed} pair(s) occurring fewer than {min_count} time(s) were left out. PMI on a pair "
                    "seen once or twice is mostly noise.",
                    removed=removed,
                    minimum=min_count,
                )
            )

    if working.empty:
        return Result[PreparedPanel].failure(
            Diagnostic.error(
                "PANEL_NO_DATA",
                "no pairs left to plot after filtering; lower min-count or check the collocations run",
            ),
            *diagnostics,
        )

    # Snapshot the co-occurrence distribution of the *whole* filtered table,
    # before it is possibly cut down to the strongest-by-measure 2000 below.
    # This is the baseline "rare pairs dominate" checks against. A fixed
    # absolute count (e.g. "<= 3") was tried and verified against the real
    # 87-address State of the Union corpus: with the collocations engine's
    # own default min_count=3, "3" is the table's floor, and PMI's top-15
    # labelled pairs there sat at 3-6 occurrences each -- rare by any
    # reading, but only 6 of 15 were literally "<= 3", so a fixed threshold
    # missed the exact bias it exists to catch. The median self-calibrates
    # to whatever min-count the run actually used, on any corpus.
    typical_cooccurrences = float(working[COOCCURRENCES].median())

    if len(working) > _MARK_CAP:
        # One SVG element per pair; a wide-window collocation run over a
        # large corpus can produce far more pairs than a browser draws
        # comfortably. Keep the strongest by the measure actually being
        # plotted -- not always G2 -- and say what was cut.
        kept = working.nlargest(_MARK_CAP, measure_column)
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_TOO_MANY_MARKS",
                f"{len(working)} pairs is more than a readable scatter holds; drew the {_MARK_CAP} with the "
                f"strongest {_MEASURE_DISPLAY[measure]}. Raise min-count to choose differently.",
                drawn=_MARK_CAP,
                available=len(working),
            )
        )
        working = kept

    top = _label_order(working, measure_column).head(label_top) if label_top else working.iloc[0:0]
    labelled = set(top[PAIR].astype(str))

    marks: list[PanelMark] = []
    for _, row in working.iterrows():
        word_1, word_2 = str(row[WORD_1]), str(row[WORD_2])
        pair = str(row[PAIR])
        cooccurrences = int(row[COOCCURRENCES])
        freq_1, freq_2 = int(row[WORD_1_FREQ]), int(row[WORD_2_FREQ])
        value = float(row[measure_column])
        marks.append(
            PanelMark(
                key=pair,
                label=pair,
                x=math.log10(cooccurrences),
                y=value,
                size=float(cooccurrences),
                labelled=pair in labelled,
                evidence=Evidence(
                    scope="terms",
                    filters=((WORD_1, word_1), (WORD_2, word_2)),
                    count=cooccurrences,
                    phrase=f"{word_1} {word_2}" if adjacent else "",
                    lemma=on_lemmas and adjacent,
                    describe=(
                        f"{pair}: {cooccurrences} co-occurrences (Word 1 {freq_1}x, Word 2 {freq_2}x) -- "
                        f"{_MEASURE_DISPLAY[measure]} {value:.2f}"
                    ),
                ),
            )
        )

    if measure == "pmi" and label_top:
        # The diagnostic that makes the panel informative rather than merely
        # correct: PMI's rare-pair bias is a property of the formula, and if
        # it dominates the labelled set that is worth saying plainly rather
        # than letting the plot imply these are the corpus's strongest pairs.
        # "Rare" is relative to this run's own table (median co-occurrence),
        # not a fixed count -- see the note above typical_cooccurrences.
        rare = int((top[COOCCURRENCES] <= typical_cooccurrences).sum())
        if rare > len(top) / 2:
            diagnostics.append(
                Diagnostic.warning(
                    "PANEL_RARE_PAIRS_DOMINATE",
                    f"{rare} of the {len(top)} labelled pairs occur at or below this table's median "
                    f"co-occurrence count ({typical_cooccurrences:.0f}). PMI rewards rare pairs most heavily -- "
                    "two words that each occur only a few times and always occur together score enormously -- "
                    "so a labelling this dominated by barely-attested pairs is PMI's known bias, not a finding "
                    "about the corpus. Try measure=log-dice, or raise min-count.",
                    rare=rare,
                    labelled=len(top),
                    median_cooccurrences=typical_cooccurrences,
                )
            )

    prepared = PreparedPanel(
        panel=COLLOCATION_STRENGTH.name,
        shape="scatter_labelled",
        title="Collocation strength: association against frequency",
        subtitle=f"{len(marks)} pairs -- y = {_MEASURE_DISPLAY[measure]} -- labelling the top {label_top}",
        marks=tuple(marks),
        x_label="Co-occurrences (log scale)",
        x_log10=True,
        y_label=_Y_LABELS[measure],
        provenance=provenance,
        data=working,
        annotations=_annotations_for(measure),
        groups=(),
        notes=_NOTES,
    )
    return Result.success(prepared, *diagnostics)


def _is_finite(value: Any) -> bool:
    """True for a real, finite number. NaN and +/-inf cannot be placed."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_positive_finite(value: Any) -> bool:
    """True for a real, finite, strictly positive number -- log10 needs one."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _label_order(frame: pd.DataFrame, measure_column: str) -> pd.DataFrame:
    """Rank pairs by the chosen measure, ties broken on ``Pair`` ascending.

    The tie-break is what lets a redraw label the same points: two pairs
    scoring identically on the measure would otherwise be ordered however
    the frame happened to arrive, which is not stable across a re-run.
    """
    return frame.assign(_rank=frame[measure_column]).sort_values(
        ["_rank", PAIR], ascending=[False, True], kind="stable"
    )


def _annotations_for(measure: str) -> tuple[Annotation, ...]:
    """A reference line only where the measure has one that means something.

    PMI and T-score are both zero exactly when observed equals expected --
    that follows directly from their formulas (``log2(observed/expected)``
    and ``(observed - expected) / sqrt(observed)`` respectively), so zero is
    a real, checkable fact about the pair: independence. G2 inherits the
    conventional chi-square significance cut from being the same statistic
    keyness plots. Log Dice has no independence baseline -- it is a
    set-overlap ratio (``14 + log2(2*observed / (freq1 + freq2))``) -- but it
    does have a principled ceiling: 14 is reached exactly when observed equals
    both word frequencies, i.e. the two words never occur apart. Real pairs
    sit there (on the State of the Union corpus: viet nam, roe wade), so the
    ceiling is drawn as a fact about the data rather than left for the reader
    to infer from a cluster of points at the top edge.
    """
    if measure == "pmi":
        return (
            Annotation(
                kind="hline",
                value=0.0,
                label="no association",
                note=(
                    "PMI = 0 means log2(observed / expected) = 0, i.e. the pair co-occurs exactly as often as "
                    "independence predicts. Each unit above is a doubling of the observed rate over the "
                    "expected one; each unit below is a halving."
                ),
            ),
        )
    if measure == "t-score":
        return (
            Annotation(
                kind="hline",
                value=0.0,
                label="observed = expected",
                note=(
                    "T-score = 0 means the observed co-occurrence count equals what independence predicts "
                    "((observed - expected) / sqrt(observed) = 0). Positive values occurred more than chance; "
                    "negative values occurred less."
                ),
            ),
        )
    if measure == "g2":
        return (
            Annotation(
                kind="hline",
                value=_G2_SIGNIFICANCE_005,
                label="p < 0.05",
                note=(
                    f"G2 = {_G2_SIGNIFICANCE_005} is the p < 0.05 threshold at 1 degree of freedom -- the same "
                    "chi-square statistic and the same conventional cut the keyness volcano uses. It is a "
                    "convention, not a decision: nothing about a pair changes as it crosses it."
                ),
            ),
        )
    return (
        Annotation(
            kind="hline",
            value=_LOG_DICE_CEILING,
            label="never apart",
            note=(
                f"Log Dice = {_LOG_DICE_CEILING:g} means every occurrence of each word is in this pair: the two "
                "never occur apart. Fixed names and set phrases sit on this line. It is a ceiling, not an "
                "independence baseline -- Log Dice has none."
            ),
        ),
    )


_NOTES: tuple[str, ...] = (
    "PMI rewards rare pairs: two words that each occur only a couple of times, always together, score at or "
    "near the top of the PMI scale. That is a property of dividing by a tiny expected count, not evidence that "
    "the pair matters -- use PMI with a raised min-count, and read PANEL_RARE_PAIRS_DOMINATE when it fires.",
    "Log-dice, t-score and G2 do not share PMI's rare-pair bias, which is why log-dice is the default measure: "
    "it is the most frequency-robust of the four offered here.",
    "The window (or adjacency) and min-count the collocations run itself used change every number in this "
    "table before the panel ever sees it. This panel draws the table exactly as it was computed; a wider "
    "window or a lower min-count on the collocations run moves every point.",
    "A strong collocation score is a reason to go and read the concordance for that pair, not a finding on "
    "its own. Every point here is a word-pair type, not a passage.",
    "The x axis is co-occurrence count on a log10 scale, not a linear count: two units right is one hundred "
    "times more co-occurrences, and a pair with zero co-occurrences has no position on it.",
)


COLLOCATION_STRENGTH = PanelDefinition(
    name="collocation_strength",
    title="Collocation strength",
    question="Which word pairs belong together, and how much evidence backs each?",
    tool="collocations",
    shape="scatter_labelled",
    summary="Association measure against co-occurrence frequency, one point per word pair.",
    requires=(WORD_1, WORD_2, PAIR, COOCCURRENCES, WORD_1_FREQ, WORD_2_FREQ, PMI, T_SCORE, G2, LOG_DICE),
    params=(
        PanelParam(
            name="measure",
            type="choice",
            default="log-dice",
            choices=("pmi", "log-dice", "t-score", "g2"),
            label="Association measure",
            help=(
                "Which association measure sets the y axis. 'pmi' rewards rare pairs most heavily -- a pair "
                "seen only a couple of times that always co-occurs can outscore a well-attested common one. "
                "'log-dice' scores overlap on a scale that does not grow with corpus size, making it the most "
                "frequency-robust of the four and the default. 't-score' favours frequent, well-attested pairs. "
                "'g2' is the log-likelihood statistic that behaves best on sparse data and measures strength "
                "of evidence, not size of effect."
            ),
        ),
        PanelParam(
            name="label-top",
            type="int",
            default=15,
            minimum=0,
            maximum=100,
            label="Pairs to label",
            help="How many pairs get their text written on the plot, ranked by the chosen measure. 0 labels none.",
        ),
        PanelParam(
            name="min-count",
            type="int",
            default=2,
            minimum=1,
            maximum=1_000_000,
            label="Minimum co-occurrences",
            help=(
                "Leave out pairs occurring fewer than this many times. PMI on a pair seen once is noise; this "
                "is the panel's own filter, on top of whatever --min-count the collocations run already applied."
            ),
        ),
    ),
    build=collocation_strength,
    notes=_NOTES,
)
