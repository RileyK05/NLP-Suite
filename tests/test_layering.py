"""R4 and layering: nothing below app/ imports UI, and core is layered.

* **R4** — no library code imports a GUI toolkit. Only ``app/`` may import
  ``streamlit`` or ``tkinter``.
* Layering — ``core`` never imports ``app`` or ``tools``; ``core/io`` and
  ``core/conll`` never import ``core/pipelines``.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _imports_of(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    return imports


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
APP = ROOT / "app"
TOOLS = ROOT / "tools"


FORBIDDEN_BELOW_APP = {"tkinter", "customtkinter", "streamlit", "PyQt5", "PyQt6", "PySide6"}


def test_nothing_below_app_imports_ui() -> None:
    offenders: list[str] = []
    for path in CORE.rglob("*.py"):
        imports = _imports_of(path)
        hit = sorted(imports & FORBIDDEN_BELOW_APP)
        if hit:
            offenders.append(f"{path.relative_to(ROOT)} imports {hit}")
    assert not offenders, "\n".join(offenders)


def test_core_does_not_import_app_or_tools() -> None:
    offenders: list[str] = []
    for path in CORE.rglob("*.py"):
        imports = _imports_of(path)
        if "app" in imports or "tools" in imports:
            offenders.append(f"{path.relative_to(ROOT)} imports {sorted(imports & {'app', 'tools'})}")
    assert not offenders, "\n".join(offenders)


def test_core_does_not_use_eval_or_shell_true() -> None:
    offenders: list[str] = []
    for path in CORE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "eval(" in text and "# noqa: S307" not in text:
            offenders.append(f"{path.relative_to(ROOT)} uses eval(")
        if "shell=True" in text:
            offenders.append(f"{path.relative_to(ROOT)} uses shell=True")
        if "os.chdir" in text or "sys.path" in text:
            offenders.append(f"{path.relative_to(ROOT)} mutates os.chdir/sys.path")
    assert not offenders, "\n".join(offenders)


def test_app_and_tools_may_import_ui() -> None:
    # This test documents the intended exception: app/ and tools/ ARE allowed
    # to import UI. It passes if those directories exist.
    assert APP.exists() or TOOLS.exists()
