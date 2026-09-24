"""Declarative form builders (FR-8.3) — registry specs to widget descriptors.

Pure data mapping with no Streamlit import: the app page renders the
descriptors, and tests assert them offline. Widget selection mirrors the
registry param types (``selectbox`` wins when choices exist).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.profiler.registry import ParamSpec, get_tool

__all__ = ["WidgetDescriptor", "human_label", "widgets_for"]


@dataclass(frozen=True, slots=True)
class WidgetDescriptor:
    """One input widget for one registry parameter."""

    key: str
    widget: str  # text | number | checkbox | selectbox | path
    label: str
    help: str
    default: Any = None
    required: bool = False
    choices: tuple[object, ...] = ()
    minimum: float | None = None
    maximum: float | None = None


def human_label(name: str) -> str:
    """plate-name to Title Words (``top-n`` -> ``Top N``)."""
    return name.replace("-", " ").replace("_", " ").title()


def _widget_for(param: ParamSpec) -> WidgetDescriptor:
    if param.choices:
        widget = "selectbox"
    elif param.type == "bool":
        widget = "checkbox"
    elif param.type in ("int", "float"):
        widget = "number"
    elif param.type == "path":
        widget = "path"
    else:
        widget = "text"
    return WidgetDescriptor(
        key=param.name,
        widget=widget,
        label=human_label(param.name),
        help=param.help,
        default=param.default,
        required=param.required,
        choices=tuple(param.choices),
        minimum=param.minimum,
        maximum=param.maximum,
    )


def widgets_for(tool_name: str) -> tuple[WidgetDescriptor, ...]:
    """Widget descriptors for every parameter of one registered tool."""
    spec = get_tool(tool_name)
    if spec is None:
        raise KeyError(f"unknown tool: {tool_name}")
    return tuple(_widget_for(param) for param in spec.params)
