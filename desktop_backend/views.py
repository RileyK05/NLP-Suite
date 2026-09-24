"""Saved chart views: a named way of looking at one result table, kept.

The workbench draws instantly and remembers nothing.  Close the panel, switch
project, or reload, and the axes, grouping and ordering someone arrived at are
gone.  That is fine for a glance and useless for research: the comparison you
want to show a colleague next week is the one you have to rebuild from memory.

A view is that arrangement written down -- which table, which settings, which
rows were drilled into -- so it reopens exactly as it was left.  Three things
this module is careful about:

* **A view is a draft, not a result.**  Saving one runs nothing, publishes
  nothing and records no provenance.  Publishing from a view is still a run,
  and still draws the whole file.  Keeping those apart is why experimenting
  cannot overwrite a previous result.

* **It stores what the source was, not just where it was.**  A run directory
  is immutable, but backups, restores and hand-edited workspaces are not, so a
  view carries the source file's SHA-256.  On reopening, a hash that no longer
  matches is reported rather than silently charted -- a view that quietly
  describes different numbers than the ones it was saved over is worse than a
  view that says it has gone stale.

* **It knows what the engine cannot reproduce.**  The preview and the
  published chart are two different renderers, and where they disagree this
  module names the disagreement out loud.  See :func:`publication_gaps`.

The settings model mirrors ``desktop/src/chartLayout.ts``'s ``Settings`` across
a language boundary, so ``tests/test_view_parity.py`` reads that file rather
than trusting this one to have kept up.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

MAX_VIEW_NAME = 120
MAX_VIEWS_PER_PROJECT = 500
# A drill-down is one mark's worth of rows. More predicates than a table has
# columns cannot narrow anything further and is a sign of a generated request.
MAX_FILTERS = 32

# The kinds the workbench draws live. The engine knows more (sunburst, treemap,
# violin, radar, waffle, calendar, pie), but a view is saved *from* the
# workbench, so a view can only hold a kind the workbench can reopen it in.
# Checked against desktop/src/chartLayout.ts by tests/test_view_parity.py.
LIVE_KINDS: tuple[str, ...] = (
    "bar",
    "line",
    "scatter",
    "bubble",
    "histogram",
    "box",
    "heatmap",
)

LiveKind = Literal["bar", "line", "scatter", "bubble", "histogram", "box", "heatmap"]
Agg = Literal["", "sum", "mean", "median", "count"]
SortOrder = Literal["table", "high", "low"]

# Kinds where the drawn order of x is the reader's choice rather than the
# data's. For the others -- a scatter, a histogram -- x is a number line and
# "largest first" means nothing, so no ordering promise is broken by publishing.
ORDERED_KINDS: frozenset[str] = frozenset({"bar", "line", "box", "heatmap"})


class RowFilter(BaseModel):
    """One column must equal one value.

    Deliberately the weakest predicate that expresses what the workbench can
    actually produce: clicking a mark means "the rows where x is this label,
    and the group column is this group".  Ranges, negation and free text are
    not here because nothing can create them yet, and a stored predicate the
    interface cannot round-trip is a trap for whoever adds the next one.
    """

    model_config = ConfigDict(extra="forbid")

    column: StrictStr = Field(min_length=1, max_length=200)
    equals: StrictStr = Field(max_length=500)


class ViewSettings(BaseModel):
    """What to draw. Field names match the desktop's ``Settings`` object.

    Validated here rather than trusted, because these arrive from the browser
    and are replayed later -- possibly after a restore, into a different build.
    A malformed saved view must fail when it is saved, where someone is looking,
    not when it is reopened months later.
    """

    model_config = ConfigDict(extra="forbid")

    kind: LiveKind
    # Empty is legal: a table with no usable column still saves, and the
    # workbench shows why instead of refusing to open the view at all.
    x: StrictStr = Field(default="", max_length=200)
    y: StrictStr = Field(default="", max_length=200)
    group: StrictStr = Field(default="", max_length=200)
    agg: Agg = "sum"
    top_n: int = Field(default=25, ge=0, le=10_000)
    sort: SortOrder = "high"
    bins: int = Field(default=12, ge=1, le=200)

    @field_validator("x", "y", "group")
    @classmethod
    def _no_surrounding_space(cls, value: str) -> str:
        """A column name is matched against a CSV header exactly.

        ``" Count"`` and ``"Count"`` are different keys to every lookup this
        goes through, and the difference is invisible in a text box.
        """
        if value != value.strip():
            raise ValueError("column names must not begin or end with spaces")
        return value


class ViewBody(BaseModel):
    """A view as the client sends it."""

    model_config = ConfigDict(extra="forbid")

    name: StrictStr = Field(min_length=1, max_length=MAX_VIEW_NAME)
    job: StrictStr = Field(min_length=1, max_length=64)
    index: int = Field(ge=0, le=10_000)
    settings: ViewSettings
    filters: list[RowFilter] = Field(default_factory=list, max_length=MAX_FILTERS)

    @field_validator("name")
    @classmethod
    def _named_something(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Give this view a name.")
        return value.strip()


def _ordering_note(sort: str) -> str:
    order = "largest first" if sort == "high" else "smallest first"
    return (
        f"Ordering: you are looking at categories {order}. A published chart "
        "arranges them in category order instead — the same categories, drawn "
        "in a different sequence."
    )


def chart_contract() -> dict[str, dict[str, str]]:
    """The preview/publication contract as data, for the interface to show.

    The desktop needs these sentences while someone is still arranging a chart,
    which is before anything has been saved and long before a request could be
    sent.  The obvious way to get them there is to write them again in
    TypeScript, and that is how ``/api/tools`` came to exist: the desktop kept
    its own copy of the tool names, both copies drifted, and nothing failed.

    So the words are shipped rather than restated.  This is a plain lookup --
    ``contract[kind][sort]`` is the note, or nothing -- so the interface applies
    no rule of its own and has nothing to keep in step.  Adding a gap here makes
    it appear in the desktop with no TypeScript change at all.
    """
    return {kind: {sort: _ordering_note(sort) for sort in ("high", "low")} for kind in sorted(ORDERED_KINDS)}


def publication_gaps(settings: ViewSettings) -> list[str]:
    """What a published chart will not reproduce about this preview.

    The workbench and ``core/viz`` are two renderers, and they do not agree on
    everything.  The honest options are to make them agree or to say where they
    do not; saying nothing is the one option that lets someone publish a chart
    believing it matches what they were looking at.

    Today there is exactly one gap, and it is about arrangement rather than
    about data.  ``prepare_chart_data`` orders the x axis ascending -- numeric
    and date axes by value, categories by name -- and has no field for anything
    else, so the workbench's "Largest first" cannot survive the trip.  Which
    categories appear *is* the same: both rank by total and keep the largest
    ``top_n``, so the published chart holds the same rows in a different order.

    Returned strings are shown to the reader, so they are sentences, not codes.
    """
    note = chart_contract().get(settings.kind, {}).get(settings.sort)
    return [note] if note else []


def describe_source(view: dict[str, Any], *, digest: str | None) -> dict[str, Any]:
    """Say plainly whether this view's source is still the file it was saved over.

    ``digest`` is the source's current SHA-256, or ``None`` when the file is no
    longer there.  Three states, each of which the interface renders differently:
    the source is intact, the source is gone, or the source is present but is
    not the file these settings were chosen against.
    """
    if digest is None:
        return {
            "state": "missing",
            "detail": (
                f"The table this view was saved from ({view['artifact_path']}) is no "
                "longer in the workspace. The view is kept, but there is nothing to draw."
            ),
        }
    if digest != view["source_sha256"]:
        return {
            "state": "changed",
            "detail": (
                f"{view['artifact_path']} is not the file this view was saved over. "
                "Its contents have changed, so the chart below may not be the one "
                "that was saved."
            ),
        }
    return {"state": "ok", "detail": ""}


__all__ = [
    "LIVE_KINDS",
    "MAX_FILTERS",
    "MAX_VIEWS_PER_PROJECT",
    "MAX_VIEW_NAME",
    "ORDERED_KINDS",
    "RowFilter",
    "ViewBody",
    "ViewSettings",
    "chart_contract",
    "describe_source",
    "publication_gaps",
]
