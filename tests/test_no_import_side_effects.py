"""R1: no import-time side effects.

Every module must be importable with the network disabled and in an
arbitrary working directory. The legacy suite violated this everywhere:
``GUI_util`` created a Tk root at import, ``IO_libraries_util`` shelled out
to pip, and stanza/spaCy models were downloaded at import.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _core_modules() -> list[str]:
    root = Path(__file__).resolve().parents[1]
    modules: list[str] = []
    for path in (root / "core").rglob("*.py"):
        if path.name == "__init__.py":
            # package init — importing the package itself is sufficient
            pkg = path.parent.relative_to(root).as_posix().replace("/", ".")
            modules.append(pkg)
            continue
        mod = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
        modules.append(mod)
    return sorted(set(modules))


CORE_MODULES = _core_modules()


def test_core_imports_without_network_and_with_changed_cwd(tmp_path: Path) -> None:
    """Import every core module in a subprocess with a different cwd."""
    root = Path(__file__).resolve().parents[1]
    imports = "; ".join(f"import {mod}" for mod in CORE_MODULES)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", imports],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )
    assert result.returncode == 0, f"import failed:\n{result.stderr}"
    assert not list((tmp_path).rglob("*.csv"))
    assert not list((tmp_path).rglob("*.json"))


def test_import_does_not_change_cwd() -> None:
    before = Path.cwd().resolve()
    for mod in CORE_MODULES:
        __import__(mod)
    assert Path.cwd().resolve() == before
