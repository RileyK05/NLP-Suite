"""A saved view is one record described in two languages. They must agree.

``desktop_backend/views.py`` validates what may be stored; ``desktop/src/
views.ts`` converts the workbench's settings into it and back. Neither knows
about the other, and a field added to one side alone fails in the quietest
possible way: pydantic's ``extra="forbid"`` rejects the request, the save
button appears not to work, and nothing in either test suite says why.

So these tests read the TypeScript rather than trusting it -- the same reason
``tests/test_chart_kind_parity.py`` exists, and the same failure it was written
after.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from desktop_backend.views import LIVE_KINDS, ORDERED_KINDS, ViewSettings, chart_contract

SOURCE = Path(__file__).resolve().parents[1] / "desktop/src"
VIEWS = SOURCE / "views.ts"
LAYOUT = SOURCE / "chartLayout.ts"


def ts(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def type_fields(source: str, name: str) -> set[str]:
    """The field names of an exported TypeScript object type."""
    start = source.index(f"export type {name} = {{")
    body = source[start : source.index("};", start)]
    return set(re.findall(r"^\s{2}(\w+)[?]?:", body, flags=re.MULTILINE))


def object_keys(source: str, name: str) -> set[str]:
    """The keys an arrow function's returned object literal sets."""
    start = source.index(f"export const {name} =")
    body = source[start : source.index("});", start)]
    return set(re.findall(r"^\s{2}(\w+):", body, flags=re.MULTILINE))


def test_the_readers_of_this_file_are_not_reading_an_empty_string() -> None:
    """A regex that silently matches nothing would pass every test below."""
    assert len(type_fields(ts(VIEWS), "ViewSettings")) >= 8
    assert len(type_fields(ts(LAYOUT), "Settings")) >= 8
    assert len(object_keys(ts(VIEWS), "toViewSettings")) >= 8


def test_the_stored_record_has_the_same_fields_on_both_sides() -> None:
    assert type_fields(ts(VIEWS), "ViewSettings") == set(ViewSettings.model_fields)


def test_every_setting_the_workbench_holds_is_stored() -> None:
    """A setting with no home in the record is a setting a saved view loses.

    Names differ between the two (``topN`` against ``top_n``); the count and
    the round trip are what has to hold.
    """
    workbench = type_fields(ts(LAYOUT), "Settings")
    stored = set(ViewSettings.model_fields)
    assert len(workbench) == len(stored), (sorted(workbench), sorted(stored))


def test_the_conversion_sets_every_stored_field() -> None:
    assert object_keys(ts(VIEWS), "toViewSettings") == set(ViewSettings.model_fields)


def test_the_conversion_back_sets_every_workbench_field() -> None:
    source = ts(VIEWS)
    start = source.index("export function fromViewSettings(")
    body = source[start : source.index("\n}", start)]
    restored = set(re.findall(r"^\s{4}(\w+):", body, flags=re.MULTILINE))
    assert restored == type_fields(ts(LAYOUT), "Settings")


def test_the_workbench_draws_every_kind_a_view_may_hold() -> None:
    """A view is reopened *in* the workbench. A kind it cannot draw would save
    and then never open -- the engine knows more kinds than this."""
    source = ts(LAYOUT)
    # Search for the close *from* the open: OKABE_ITO is also `as const` and
    # comes first, so an unanchored index would slice an empty string and make
    # this assertion pass against nothing.
    start = source.index("export const LIVE_KINDS = [")
    body = source[start : source.index("] as const;", start)]
    assert set(re.findall(r'"([a-z]+)"', body)) == set(LIVE_KINDS)


@pytest.mark.parametrize("kind", sorted(ORDERED_KINDS))
def test_every_kind_with_an_ordering_gap_is_one_the_workbench_draws(kind: str) -> None:
    """A note about a kind nobody can select would never be shown."""
    assert kind in LIVE_KINDS


def test_the_interface_looks_the_contract_up_by_kind_and_sort() -> None:
    """The desktop applies no rule of its own: it indexes the shipped table.

    If this ever becomes a condition written in TypeScript, the two copies can
    disagree about *when* a chart diverges while agreeing about the words.
    """
    source = ts(VIEWS)
    assert "contract[settings.kind]?.[settings.sort]" in source
    depth = {len(notes) for notes in chart_contract().values()}
    assert depth == {2}, "the served table is two levels deep, which is what that lookup assumes"


def test_the_drill_down_filters_match_what_the_workbench_filters_on() -> None:
    """``filtersFor`` stores a predicate; ``rowsBehind`` applies one. A view
    that reopens showing different rows than it was saved with is worse than
    one that does not reopen at all."""
    layout = ts(LAYOUT)
    start = layout.index("export function rowsBehind(")
    applied = layout[start : layout.index("\n}", start)]
    assert "settings.x" in applied and "settings.group" in applied

    views = ts(VIEWS)
    start = views.index("export function filtersFor(")
    stored = views[start : views.index("\n}", start)]
    assert "settings.x" in stored and "settings.group" in stored
    # Both skip the x predicate for a histogram, whose marks are bins rather
    # than values of a column.
    assert 'settings.kind !== "histogram"' in applied
    assert 'settings.kind !== "histogram"' in stored


def test_a_setting_the_workbench_no_longer_has_is_not_still_stored() -> None:
    """``logY`` was declared, defaulted in three places and read nowhere.

    It was removed rather than frozen into the storage schema, because a
    stored field that does nothing tells every later reader that it does.
    """
    assert "logY" not in ts(LAYOUT)
    assert "logY" not in ts(VIEWS)
    assert "log" not in set(ViewSettings.model_fields)
