"""FR-9.5 — migration guide drift guards.

`docs/MIGRATION.md` is a contract: every command it names must resolve,
and every registry tool must be named in it. Either direction of drift
fails here.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import re

from core.profiler.registry import tool_names

GUIDE = Path(__file__).resolve().parent.parent / "docs" / "MIGRATION.md"


def _guide_text() -> str:
    return GUIDE.read_text(encoding="utf-8")


def _resolves(name: str) -> bool:
    try:
        module = importlib.import_module(f"tools.{name}")
    except ImportError:
        return False
    return callable(getattr(module, "main", None))


class TestMigrationGuide:
    def test_unified_commands_resolve(self) -> None:
        # Tool names may carry digits (word2vec_bert), so the pattern includes them.
        names = set(re.findall(r"nlp-suite ([a-z_0-9]+)", _guide_text()))
        assert names, "guide names no unified commands"
        for name in sorted(names):
            assert _resolves(name), f"guide names unresolvable tool: nlp-suite {name}"

    def test_module_commands_resolve(self) -> None:
        names = set(re.findall(r"python -m tools\.([a-z_0-9]+)", _guide_text()))
        for name in sorted(names):
            assert _resolves(name), f"guide names unresolvable module: tools.{name}"

    def test_every_registry_tool_is_mapped(self) -> None:
        text = _guide_text()
        missing = [name for name in tool_names() if name not in text]
        assert not missing, f"registry tools missing from the migration guide: {missing}"
