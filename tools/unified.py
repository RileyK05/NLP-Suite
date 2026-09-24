"""nlp-suite — one unified command for every tool (FR-9.1).

``nlp-suite --list`` names every registry tool plus the utility CLIs;
``nlp-suite --list --json`` prints one JSON array of objects (name,
description, source, optional_package) for machine-readable discovery;
``nlp-suite <tool> [args ...]`` routes to that tool's ``main`` with the
same flags as its standalone entry point. Unknown tools fail with exit 2
and the tool list on stderr — never a traceback. A tool module that exists
but cannot be imported (a missing optional dependency raised inside the
module) is distinguished from an unknown tool: it fails with exit 3 and an
explicit ``IMPORT_FAILED`` diagnostic naming the module and the error.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from core.profiler.registry import get_tool, tool_names

__all__ = ["main"]

_UTILITY_EXCLUDE = frozenset({"__init__", "_cli", "unified"})


def _utility_clis() -> list[str]:
    """Non-registry CLI modules (cheap filename scan, no imports)."""
    tools_dir = Path(__file__).resolve().parent
    names = sorted(p.stem for p in tools_dir.glob("*.py") if p.stem not in _UTILITY_EXCLUDE)
    return [name for name in names if get_tool(name) is None]


def _list_tools() -> int:
    for name in tool_names():
        spec = get_tool(name)
        description = spec.description if spec is not None else ""
        print(f"{name:<24} {description}")
    print("\nutility CLIs (same routing, no registry spec):")
    for name in _utility_clis():
        print(f"{name:<24} standalone CLI (python -m tools.{name})")
    return 0


def _list_tools_json() -> int:
    """One JSON object on stdout: registry tools + utility CLIs, typed."""
    registry = []
    for name in tool_names():
        spec = get_tool(name)
        registry.append(
            {
                "name": name,
                "description": spec.description if spec is not None else "",
                "source": "registry",
                "optional_package": spec.optional_package if spec is not None else "",
            }
        )
    utilities = [
        {
            "name": name,
            "description": "standalone CLI (python -m tools." + name + ")",
            "source": "utility",
            "optional_package": "",
        }
        for name in _utility_clis()
    ]
    print(
        json.dumps(
            {"tools": registry, "utility_clis": utilities},
            ensure_ascii=False,
        )
    )
    return 0


def _route(name: str, rest: list[str]) -> int:
    """Import tools.<name> and call its main; distinguish unknown vs broken.

    A ``ModuleNotFoundError`` whose module name is exactly ``tools.<name>``
    (or its parent package) means the requested tool does not exist —
    unknown tool. A ``ModuleNotFoundError`` naming any OTHER module (a
    dependency imported inside the tool) or any other ``ImportError`` means
    the tool exists but cannot load (missing optional dependency) — a
    different failure with a different exit code.
    """
    try:
        module = importlib.import_module(f"tools.{name}")
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "") or ""
        if missing == f"tools.{name}" or missing.startswith(f"tools.{name}."):
            print(f"nlp-suite: unknown tool {name!r}", file=sys.stderr)
            print("run 'nlp-suite --list' to see every tool", file=sys.stderr)
            return 2
        print(
            f"nlp-suite: tool {name!r} failed to import (missing dependency {missing!r}): {exc}",
            file=sys.stderr,
        )
        print("the tool exists but a package it needs is not installed (nlp-doctor names the extra)", file=sys.stderr)
        return 3
    except ImportError as exc:
        # The module exists but its imports fail for another reason.
        print(f"nlp-suite: tool {name!r} failed to import (missing dependency?): {exc}", file=sys.stderr)
        print(
            f"the tool exists ('python -m tools.{name}' or nlp-doctor names its extra) but could not load",
            file=sys.stderr,
        )
        return 3
    entry = getattr(module, "main", None)
    if not callable(entry):
        print(f"nlp-suite: tool {name!r} has no main()", file=sys.stderr)
        return 2
    # Tool parsers set no ``prog``, so argparse derives it from sys.argv[0] —
    # which is this router, not the tool. Without this the first thing a user
    # sees from ``nlp-suite kwic --help`` is ``usage: unified.py``, and every
    # argparse error names a command they did not run. Restored in ``finally``
    # so a tool that inspects sys.argv itself is unaffected afterwards.
    original_argv0 = sys.argv[0]
    sys.argv[0] = f"nlp-suite {name}"
    try:
        return int(entry(rest))
    finally:
        sys.argv[0] = original_argv0


def main(argv: list[str] | None = None) -> int:
    from tools._cli import make_console_encoding_tolerant

    make_console_encoding_tolerant()
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        stream = sys.stdout if args else sys.stderr
        print("usage: nlp-suite [--list [--json]] <tool> [tool args ...]", file=stream)
        print("run 'nlp-suite --list' to see every tool", file=stream)
        return 0 if args else 2
    if args[0] == "--list":
        if "--json" in args[1:]:
            return _list_tools_json()
        return _list_tools()
    name, rest = args[0], args[1:]
    return _route(name, rest)


if __name__ == "__main__":
    raise SystemExit(main())
