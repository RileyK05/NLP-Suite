"""Every column the decoder names must be a column the tool actually produces.

:mod:`core.insight.readout` and :mod:`core.insight.recommend` both hold
per-tool knowledge keyed by column name. That knowledge is written by hand,
against tables produced somewhere else, and nothing at runtime complains when
it drifts: a readout rule silently returns early, and a chart recommendation is
silently skipped. The result is a decoder that looks fine and says nothing.

That is not hypothetical. ``lexical_diversity`` was recommending a scatter of
``"Words"`` against ``"MTLD"`` when the column has always been called
``"Total Tokens"``, so the one recommendation that tool had never appeared.

So this module reads both sides statically -- the column names the rules use,
and the column names the analysis modules construct -- and fails when the first
is not contained in the second. Static rather than by running the tools,
because it has to cover tools whose models are not installed in every
environment, and a check that skips is not a check.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

from core.insight.readout import _RULES
from core.insight.recommend import _TOOL_CHARTS

ANALYSIS = Path(__file__).resolve().parent.parent / "core" / "analysis"
READOUT_SOURCE = Path(__file__).resolve().parent.parent / "core" / "insight" / "readout.py"
EXECUTOR_SOURCE = Path(__file__).resolve().parent.parent / "core" / "profiler" / "executor.py"

# Which analysis modules build a tool's output table. Usually the module named
# after the tool, but a tool that delegates its table to a shared statistics
# routine gets that module listed too -- keyness's G2 column is built inside
# stats_categorical, not keyness.
COLUMN_SOURCES: dict[str, tuple[str, ...]] = {
    "keyness": ("keyness", "stats_categorical"),
    "topic_model": ("topic_model", "lda"),
    "lda_gensim": ("lda", "topic_model"),
    "lda_mallet": ("mallet",),
}

# Calls whose first string argument names a column.
_COLUMN_ARG_CALLS = frozenset(
    {"get", "groupby", "drop_duplicates", "sort_values", "value_counts", "_top_row", "nunique"}
)


def _string_constants(node: ast.AST) -> set[str]:
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def declared_columns(tool: str) -> set[str]:
    """Every string literal in the modules that build *tool*'s output table.

    Deliberately over-broad: the assertion is containment, so a superset here
    can only cause a missed drift, never a false failure on a column that is
    genuinely constructed.
    """
    names = COLUMN_SOURCES.get(tool, (tool,))
    found: set[str] = set()
    for name in names:
        path = ANALYSIS / f"{name}.py"
        assert path.is_file(), f"{tool}: no analysis module at {path}"
        found |= _string_constants(ast.parse(path.read_text(encoding="utf-8")))
    return found


def executor_columns() -> set[str]:
    """Columns the executor joins onto a per-document frame after the tool.

    ``Date`` and ``Year`` are real columns of a real result, but no analysis
    module builds them: they come from the corpus, which is the only thing
    that knows when a document is from. Read out of the executor rather than
    written down here, so renaming the constant there still fails this guard
    rather than quietly exempting a column that no longer exists.
    """
    tree = ast.parse(EXECUTOR_SOURCE.read_text(encoding="utf-8"))
    wanted = {"DATE", "YEAR"}
    found = {
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets)
    }
    assert len(found) == len(wanted), f"executor no longer defines {sorted(wanted)} as plain string constants"
    return found


def _rule_functions() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(READOUT_SOURCE.read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _subscript_column(node: ast.Subscript) -> set[str]:
    """``frame["Similarity"]``."""
    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return {node.slice.value}
    return set()


def _set_columns(node: ast.Set) -> set[str]:
    """``{"Topic", "Word", "Weight"} <= set(frame.columns)``."""
    return {e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}


def _membership_column(node: ast.Compare) -> set[str]:
    """``"Sentence ID" in frame.columns``."""
    is_in = any(isinstance(op, ast.In) for op in node.ops)
    if is_in and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
        return {node.left.value}
    return set()


def _assigned_column(node: ast.Assign) -> set[str]:
    """``column = "Flesch Reading Ease"``, the name-it-once idiom."""
    if any(isinstance(target, ast.Name) and target.id.endswith("column") for target in node.targets):
        return _string_constants(node.value)
    return set()


def _call_columns(node: ast.Call) -> set[str]:
    """``frame.sort_values("Count")``, ``_length_dependence(frame, "Tokens", "TTR", ...)``."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if name == "_length_dependence":
        return {a.value for a in node.args[1:3] if isinstance(a, ast.Constant) and isinstance(a.value, str)}
    if name in _COLUMN_ARG_CALLS and node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return {first.value}
    return set()


def columns_used_by(function: ast.FunctionDef) -> set[str]:
    """Column names a readout rule reads, recovered from its source.

    Recognises the five shapes the rules are written in; each is handled by one
    of the small extractors above, so adding a sixth is adding a function
    rather than another branch.
    """
    found: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Subscript):
            found |= _subscript_column(node)
        elif isinstance(node, ast.Set):
            found |= _set_columns(node)
        elif isinstance(node, ast.Compare):
            found |= _membership_column(node)
        elif isinstance(node, ast.Assign):
            found |= _assigned_column(node)
        elif isinstance(node, ast.Call):
            found |= _call_columns(node)
    return found


class TestChartColumnsExist:
    @pytest.mark.parametrize("tool", sorted(_TOOL_CHARTS))
    def test_every_recommended_chart_names_real_columns(self, tool: str) -> None:
        declared = declared_columns(tool) | executor_columns()
        missing = []
        for kind, x, y, _question, _why in _TOOL_CHARTS[tool]:
            axes = {x} if kind == "histogram" else {x, y}
            missing.extend(sorted(column for column in axes if column not in declared))
        assert not missing, (
            f"{tool} recommends a chart of {missing}, which core/analysis never builds. "
            "The recommendation would be silently skipped at runtime."
        )

    @pytest.mark.parametrize("tool", sorted(_TOOL_CHARTS))
    def test_a_timeline_is_only_recommended_for_a_per_document_table(self, tool: str) -> None:
        """The executor dates a frame by joining on ``Document ID``.

        So a timeline recommended for a table that has no ``Document ID`` --
        a term list, a pair list -- names an axis that will never be there,
        and the recommendation is skipped in silence every time it is asked
        for. Charting the date column is the claim; carrying the join key is
        what makes the claim true.
        """
        timelines = [(x, y) for kind, x, y, _q, _w in _TOOL_CHARTS[tool] if x in executor_columns() or kind == "line"]
        if not timelines:
            pytest.skip(f"{tool} recommends no timeline")
        declared = declared_columns(tool)
        assert "Document ID" in declared, (
            f"{tool} recommends a timeline of {timelines}, but its table has no 'Document ID' for the "
            "executor to join the corpus's dates onto, so the date column never appears."
        )

    def test_a_wrong_column_name_is_caught(self) -> None:
        """The check itself, against a name no analysis module builds."""
        assert "Words" not in declared_columns("lexical_diversity")
        assert "Total Tokens" in declared_columns("lexical_diversity")


class TestReadoutColumnsExist:
    @pytest.mark.parametrize("tool", sorted(_RULES))
    def test_every_readout_rule_names_real_columns(self, tool: str) -> None:
        functions = _rule_functions()
        rule: Callable[..., None] = _RULES[tool]
        node = functions.get(rule.__name__)
        assert node is not None, f"{tool}: rule {rule.__name__} is not a module-level function"
        declared = declared_columns(tool)
        missing = sorted(column for column in columns_used_by(node) if column not in declared)
        assert not missing, (
            f"the {tool} readout rule reads {missing}, which core/analysis never builds. "
            "The rule would return early and the run would get only the generic reading."
        )

    def test_the_extractor_actually_finds_columns(self) -> None:
        """A silent extractor would make every test above pass vacuously."""
        functions = _rule_functions()
        for tool, rule in _RULES.items():
            node = functions[rule.__name__]
            assert columns_used_by(node), f"{tool}: extracted no column names, so its check proves nothing"
