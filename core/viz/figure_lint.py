"""Checks that a prepared figure will read correctly, before anyone looks at it.

A figure can be *correct* -- every number traceable to its rows -- and still
unreadable: labelled "sentence 7", captioned with a measure the reader did not
pick, or carrying eighty labels nobody can separate. Those are the problems a
person finds by looking, and ``docs/FIGURE_QUALITY_PLAN.md`` lists the classes
they came in (C1-C8). This module finds the engine-side ones by machine, from
the :class:`PreparedPanel` alone, so a test can run it over every registered
panel with every choice of every parameter.

What it cannot see is geometry: whether two labels overprint depends on the
renderer's fonts and sizes. The static renderer measures that itself
(``core/viz/static/lint.py``), and the app's placement is tested in
``desktop/src/panelTicks.test.ts``.

Each finding is a :class:`Finding` with a stable ``code``, so a test can
assert on the kind of problem and a report can group them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from core.viz.panelspec import PanelDefinition, PreparedPanel

__all__ = [
    "MAX_LABELLED_MARKS",
    "MAX_LABEL_CHARS",
    "Finding",
    "lint_definition",
    "lint_prepared",
]

#: A category or mark label longer than this is cut by every renderer; the
#: builder should shorten it itself (a short document label, a snippet).
MAX_LABEL_CHARS = 60
#: More labelled marks than this cannot be separated on a 900px figure.
MAX_LABELLED_MARKS = 40
#: A row-banded figure with more categories than this needs its height sized
#: to them, or its rows are thinner than a line of text.
ROWS_PER_DEFAULT_HEIGHT = 40
#: Past this many points a scatter is a cloud: it needs a stated budget.
SCATTER_BUDGET = 1500

# A label that is only an identifier: "Doc 14", "sentence 7", "row 3",
# "58 · sentence 7", "#12". It tells the reader nothing about the mark.
_ID_ONLY = re.compile(
    r"^\s*(?:#?\d+\s*[·:,-]\s*)?(?:doc(?:ument)?|sentence|sent|row|id|item|segment|paragraph|para)\s*#?\d+\s*$",
    re.IGNORECASE,
)
# A bare number as a *row or column* label (a document axis labelled 1, 2,
# 3 ...) is an id too -- unless it is a year, which is a name for a row.
_BARE_NUMBER = re.compile(r"^\s*\d+\s*$")


def _id_like(label: str, *, category: bool) -> bool:
    if _ID_ONLY.match(label):
        return True
    if category and _BARE_NUMBER.match(label):
        return not 1000 <= int(label) <= 2500
    return False


@dataclass(frozen=True, slots=True)
class Finding:
    """One problem with one figure: a stable code and a sentence."""

    code: str
    panel: str
    message: str

    def __str__(self) -> str:
        return f"{self.code} [{self.panel}]: {self.message}"


def _mentions(text: str, word: str) -> bool:
    """Whether ``word`` appears in ``text`` as a whole word, any case."""
    if not word or not text:
        return False
    return re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text, re.IGNORECASE) is not None


def _text_of(prepared: PreparedPanel) -> str:
    return " \n".join((prepared.title, prepared.subtitle, prepared.x_label, prepared.y_label))


def _choice_text(value: str) -> list[str]:
    """The forms a choice value takes in prose: 'mtld', 'per-million' ->
    'per million', 'Per Million'. Values shorter than three characters are
    too common as words to check."""
    forms = {value, value.replace("-", " "), value.replace("_", " ")}
    return [form for form in forms if len(form) >= 3]


def lint_definition(definition: PanelDefinition) -> list[Finding]:
    """Problems in a panel's static text, which is shown before it is drawn.

    The one that shipped: a summary written from the default measure
    ("Each dated document's MTLD ...") shown whatever measure was picked.
    A summary may list choices ("grouped by decade or speaker"), but naming
    the default and no other describes one setting as if it were the figure.
    """
    findings: list[Finding] = []
    for param in definition.params:
        if param.type != "choice" or not isinstance(param.default, str):
            continue
        others = [choice for choice in param.choices if choice != param.default]
        default_named = any(_mentions(definition.summary, form) for form in _choice_text(param.default))
        other_named = any(any(_mentions(definition.summary, f) for f in _choice_text(c)) for c in others)
        if default_named and others and not other_named:
            findings.append(
                Finding(
                    "SUMMARY_NAMES_DEFAULT",
                    definition.name,
                    f"summary names the default {param.name} {param.default!r}, which the reader can change",
                )
            )
    if not definition.question.strip():
        findings.append(Finding("NO_QUESTION", definition.name, "the figure does not say what question it answers"))
    return findings


def lint_prepared(
    prepared: PreparedPanel,
    definition: PanelDefinition | None = None,
    params: dict[str, Any] | None = None,
    diagnostics: tuple[Any, ...] = (),
) -> list[Finding]:
    """Engine-side reading problems in one prepared figure.

    With ``definition`` and ``params``, also checks that the figure's text
    follows the parameters it was drawn with (C5): a choice other than the
    default must not leave the default's name in the title, subtitle or axis
    labels.
    """
    name = prepared.panel
    findings: list[Finding] = []

    # C8: labels that say what the mark is.
    shown = [mark.label for mark in prepared.marks if mark.labelled or prepared.shape == "ranked_bars"]
    categories = [*prepared.x_categories, *prepared.y_categories, *prepared.facets]
    id_like = sorted(
        {label for label in shown if label and _id_like(label, category=False)}
        | {label for label in categories if label and _id_like(label, category=True)}
    )
    if id_like:
        findings.append(
            Finding(
                "ID_ONLY_LABEL",
                name,
                f"{len(id_like)} label(s) are identifiers, not names: {', '.join(id_like[:4])}",
            )
        )
    long = sorted({label for label in (*shown, *categories) if len(label) > MAX_LABEL_CHARS})
    if long:
        findings.append(
            Finding(
                "LABEL_TOO_LONG",
                name,
                f"{len(long)} label(s) over {MAX_LABEL_CHARS} characters, e.g. {long[0][:80]!r}",
            )
        )

    # Duplicate labels drawn twice on one figure read as one thing twice.
    if prepared.shape in ("scatter_labelled", "network"):
        labelled = [mark.label for mark in prepared.marks if mark.labelled]
        duplicates = sorted({label for label in labelled if labelled.count(label) > 1})
        if duplicates:
            findings.append(Finding("DUPLICATE_LABEL", name, f"labelled twice: {', '.join(duplicates[:4])}"))
        # A dendrogram's leaves stand in one labelled column, a row apart:
        # they are read down a list, not picked out of a cloud.
        if len(labelled) > MAX_LABELLED_MARKS and prepared.edge_style != "elbow":
            findings.append(
                Finding(
                    "TOO_MANY_LABELS",
                    name,
                    f"{len(labelled)} labelled marks; at most {MAX_LABELLED_MARKS} can be read apart",
                )
            )
    if prepared.shape == "scatter_labelled" and len(prepared.marks) > SCATTER_BUDGET:
        findings.append(Finding("SCATTER_OVERPLOTTED", name, f"{len(prepared.marks)} points in one scatter"))

    # C6: rows thinner than a line of text.
    rows = len(prepared.y_categories)
    if prepared.shape in ("heatmap", "distribution", "positions") and rows > ROWS_PER_DEFAULT_HEIGHT:
        needed = 140 + rows * 12
        if prepared.height < needed:
            findings.append(
                Finding(
                    "ROWS_TOO_THIN",
                    name,
                    f"{rows} rows in {prepared.height}px; needs about {needed}px to label every row",
                )
            )
    if prepared.shape == "ranked_bars":
        bars = len({mark.label for mark in prepared.marks})
        if bars > 60:
            findings.append(Finding("TOO_MANY_BARS", name, f"{bars} bars in one ranking"))

    # Numbers the renderers cannot place.
    bad = [
        mark.key
        for mark in prepared.marks
        if not (math.isfinite(mark.x) and math.isfinite(mark.y))
        or (mark.value is not None and not math.isfinite(mark.value))
        or (mark.size is not None and (not math.isfinite(mark.size) or mark.size < 0))
    ]
    if bad:
        findings.append(Finding("NON_FINITE", name, f"{len(bad)} mark(s) with NaN or infinite geometry"))

    # C8: extremes of a signed measure that show one tail only -- unless the
    # builder said the data has no other tail (PANEL_ONE_SIDED).
    if not any(getattr(d, "code", "") == "PANEL_ONE_SIDED" for d in diagnostics):
        findings.extend(_one_tail(prepared))

    # C5: the text follows the parameters.
    if definition is not None and params is not None:
        text = _text_of(prepared)
        for param in definition.params:
            if param.type != "choice" or not isinstance(param.default, str):
                continue
            chosen = params.get(param.name, param.default)
            if chosen == param.default:
                continue
            stale = [form for form in _choice_text(param.default) if _mentions(text, form)]
            fresh = any(_mentions(text, form) for form in _choice_text(str(chosen)))
            if stale and not fresh:
                findings.append(
                    Finding(
                        "TEXT_NAMES_DEFAULT",
                        name,
                        f"{param.name}={chosen!r} but the figure's text still says {stale[0]!r}",
                    )
                )
    return findings


def _one_tail(prepared: PreparedPanel) -> list[Finding]:
    """A ranking titled as extremes whose bars all point one way.

    "Most extreme sentences" ranked by |compound| on a corpus of mostly
    positive speeches returned nineteen bars at +0.99: the question asked
    for both tails and the figure answered one.
    """
    if prepared.shape != "ranked_bars":
        return []
    wording = f"{prepared.title} {prepared.subtitle}".lower()
    if not any(word in wording for word in ("extrem", "most positive and", "most negative and")):
        return []
    values = [mark.x for mark in prepared.marks]
    if values and (all(v >= 0 for v in values) or all(v <= 0 for v in values)):
        return [
            Finding(
                "ONE_TAIL_ONLY",
                prepared.panel,
                "an 'extremes' ranking whose bars all have one sign; show the top of each tail",
            )
        ]
    # A lopsided split is the data (two negative speeches in 87), not the
    # ranking: only a figure with no bar at all on one side is flagged.
    return []
