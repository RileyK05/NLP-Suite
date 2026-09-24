"""Every tool the suite can run declares how its results are shown.

The finish line for ``docs/FIGURE_RECIPES.md``: a tool with no recipe falls
through to generic chart guesses, which is how "Count over Date" came to
average 9,407 unrelated word pairs. This test names every tool still in that
state.
"""

from __future__ import annotations

from core.profiler.registry import TOOL_REGISTRY
from core.viz.panels import PANELS
from core.viz.recipes import GENERIC_BUILDERS, TABLE_FIRST, recipe_for
from desktop_backend.tables import TABLE_TOOLS

EVERY_TOOL = sorted({spec.name for spec in TOOL_REGISTRY} | set(TABLE_TOOLS))


def test_every_tool_has_a_recipe() -> None:
    missing = [tool for tool in EVERY_TOOL if not recipe_for(tool).covered]
    assert not missing, f"{len(missing)} tool(s) have no figures and no table-first reason: {missing}"


def test_recipes_name_only_real_tools() -> None:
    """A recipe for a misspelt tool covers nothing and hides the real gap."""
    known = set(EVERY_TOOL)
    assert set(TABLE_FIRST) <= known, sorted(set(TABLE_FIRST) - known)
    assert set(GENERIC_BUILDERS) <= known, sorted(set(GENERIC_BUILDERS) - known)
    assert {panel.tool for panel in PANELS} <= known, sorted({panel.tool for panel in PANELS} - known)


def test_every_panel_asks_a_question() -> None:
    """The reading offers a tool's figures by the question each answers."""
    silent = [panel.name for panel in PANELS if not panel.question.strip()]
    assert not silent, f"panels with no question: {silent}"


def test_table_first_reasons_are_sentences() -> None:
    for tool, reason in TABLE_FIRST.items():
        assert reason.strip().endswith("."), tool
        assert len(reason) > 30, f"{tool}: say why, not just that"
