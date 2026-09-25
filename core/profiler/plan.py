"""Profiler execution plan (FR-7.3) — pure selection/parameter model.

Validates a requested tool set against the declarative registry and orders
it by execution phase (stable within a phase). No execution, no parsing, no
I/O: the executor (a later child) consumes ``Plan``. Ordering rules beyond
phases (like the legacy's symbolic-after-SVO) gain an explicit table here
when the first real constraint lands — none exists among current tools.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from os import PathLike

from core.models.registry import get_model
from core.profiler.registry import TOOL_REGISTRY, ParamSpec, ToolSpec, get_tool
from core.result import Diagnostic, Result

__all__ = ["Plan", "PlannedTool", "build_plan", "validate_parameters"]


@dataclass(frozen=True, slots=True)
class PlannedTool:
    """One validated, ordered batch entry."""

    name: str
    params: dict[str, object]
    requires_parse: bool
    phase: int


@dataclass(frozen=True, slots=True)
class Plan:
    """An ordered, fully-defaulted batch."""

    tools: tuple[PlannedTool, ...]
    params: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def needs_parse(self) -> bool:
        return any(tool.requires_parse for tool in self.tools)


def _type_ok(kind: str, value: object) -> bool:
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "str":
        return isinstance(value, str)
    if kind == "path":
        return isinstance(value, (str, PathLike))
    return False


def _check_selection(order: list[str], given: Mapping[str, object]) -> Diagnostic | None:
    """Aggregate selection-shape errors (unknown, stray params, ineligible)."""
    if not order:
        return Diagnostic.error("PROFILER_EMPTY_SELECTION", "no tools selected")
    unknown = [name for name in order if get_tool(name) is None]
    if unknown:
        return Diagnostic.error("PROFILER_UNKNOWN_TOOL", f"unknown tool(s): {', '.join(unknown)}", tools=unknown)
    stray = [name for name in given if name not in order]
    if stray:
        return Diagnostic.error(
            "PROFILER_UNKNOWN_TOOL", f"params given for unselected tool(s): {', '.join(stray)}", tools=stray
        )
    ineligible = [name for name in order if not _spec_of(name).profiler_eligible]
    if ineligible:
        return Diagnostic.error(
            "PROFILER_NOT_ELIGIBLE",
            f"tool(s) cannot run in a batch: {', '.join(ineligible)}",
            tools=ineligible,
        )
    return None


def _spec_of(name: str) -> ToolSpec:
    """Registry spec for a name already known to exist (checked by _check_selection)."""
    spec = get_tool(name)
    if spec is None:  # unreachable after the unknown check; kept for narrowing
        raise KeyError(f"unknown tool: {name}")
    return spec


def _check_value(tool: str, param: ParamSpec, value: object) -> Diagnostic | None:
    """Type, choice, and bounds checks for one supplied/defaulted value."""
    name = param.name
    kind = param.type
    if not _type_ok(kind, value):
        return Diagnostic.error(
            "PROFILER_BAD_PARAM", f"{tool}.{name} must be {kind}, got {value!r}", tool=tool, param=name
        )
    if param.choices and value not in param.choices:
        return Diagnostic.error(
            "PROFILER_BAD_PARAM",
            f"{tool}.{name} must be one of {list(param.choices)}, got {value!r}",
            tool=tool,
            param=name,
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return Diagnostic.error("PROFILER_BAD_PARAM", f"{tool}.{name} must be finite", tool=tool, param=name)
        minimum = param.minimum
        maximum = param.maximum
        if minimum is not None and value < minimum:
            return Diagnostic.error(
                "PROFILER_BAD_PARAM", f"{tool}.{name} must be >= {minimum}, got {value!r}", tool=tool, param=name
            )
        if maximum is not None and value > maximum:
            return Diagnostic.error(
                "PROFILER_BAD_PARAM", f"{tool}.{name} must be <= {maximum}, got {value!r}", tool=tool, param=name
            )
    return None


def _canonical_model(param: ParamSpec, value: object) -> object:
    """A model named any older way ("google-bert/bert-base-uncased") becomes its id.

    Saved runs and recipes carry the Hugging Face names the tools used to
    take; the model choices are now registry ids, and a re-run must not fail
    over a spelling.
    """
    if param.name != "model" or not isinstance(value, str) or not param.choices:
        return value
    spec = get_model(value)
    return spec.id if spec is not None and spec.id in param.choices else value


def _fill_params(tool: str, supplied: dict[str, object]) -> tuple[dict[str, object], Diagnostic | None]:
    return _fill_spec_params(_spec_of(tool), supplied)


def validate_parameters(spec: ToolSpec, supplied: dict[str, object]) -> Result[dict[str, object]]:
    """Shared parameter validation for batch and explicitly non-batch UIs."""
    filled, bad = _fill_spec_params(spec, supplied)
    return Result.failure(bad) if bad is not None else Result.success(filled)


def _fill_spec_params(spec: ToolSpec, supplied: dict[str, object]) -> tuple[dict[str, object], Diagnostic | None]:
    """Merge supplied params over spec defaults; first violation wins."""
    tool = spec.name
    for key in supplied:
        if all(param.name != key for param in spec.params):
            return {}, Diagnostic.error(
                "PROFILER_UNKNOWN_PARAM", f"{tool} has no parameter {key!r}", tool=tool, param=key
            )
    filled: dict[str, object] = {}
    for param in spec.params:
        if param.name in supplied:
            value = supplied[param.name]
        elif param.required:
            return {}, Diagnostic.error(
                "PROFILER_MISSING_PARAM", f"{tool} requires parameter {param.name!r}", tool=tool, param=param.name
            )
        else:
            value = param.default
        if value is None and not param.required:
            filled[param.name] = None
            continue
        value = _canonical_model(param, value)
        bad = _check_value(tool, param, value)
        if bad is not None:
            return {}, bad
        filled[param.name] = value
    return filled, None


def build_plan(
    selected: Sequence[str],
    params: Mapping[str, Mapping[str, object]] | None = None,
) -> Result[Plan]:
    """Validate *selected* tool names + params into an ordered Plan."""
    wanted: dict[str, None] = {}
    for name in selected:
        wanted.setdefault(name)
    order = list(wanted)
    given = dict(params) if params else {}
    bad_shape = _check_selection(order, given)
    if bad_shape is not None:
        return Result.failure(bad_shape)

    index = {spec.name: pos for pos, spec in enumerate(TOOL_REGISTRY)}
    planned: list[PlannedTool] = []
    merged: dict[str, dict[str, object]] = {}
    for name in sorted(order, key=lambda n: (_spec_of(n).execution_phase, index[n])):
        spec = _spec_of(name)
        filled, bad_params = _fill_params(name, dict(given.get(name, {})))
        if bad_params is not None:
            return Result.failure(bad_params)
        merged[name] = filled
        planned.append(
            PlannedTool(
                name=name,
                params=filled,
                requires_parse=spec.requires_parse or (name == "search" and filled.get("mode") == "conll"),
                phase=spec.execution_phase,
            )
        )
    return Result.success(Plan(tools=tuple(planned), params=merged))
