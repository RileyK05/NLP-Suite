"""FR-9.4 — install contract tests (entry points + docs)."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest


def _dist() -> importlib.metadata.Distribution | None:
    try:
        return importlib.metadata.distribution("nlp-suite-ng")
    except importlib.metadata.PackageNotFoundError:
        return None


class TestInstall:
    def test_console_scripts_registered(self) -> None:
        dist = _dist()
        if dist is None:
            pytest.skip("nlp-suite-ng not installed as a distribution")
        names = {ep.name for ep in dist.entry_points}
        assert {"nlp-suite", "nlp-doctor"} <= names

    def test_entry_targets_importable(self) -> None:
        import importlib

        for dotted in ("tools.unified:main", "tools.doctor:main"):
            module_name, attr = dotted.split(":")
            assert callable(getattr(importlib.import_module(module_name), attr)), dotted

    def test_install_doc_lists_extras(self) -> None:
        import tomllib

        text = (Path(__file__).resolve().parent.parent / "docs" / "INSTALL.md").read_text(encoding="utf-8")
        pyproject = tomllib.loads(
            (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
        )
        for extra in pyproject["project"]["optional-dependencies"]:
            if extra in ("all", "dev"):
                continue
            assert extra in text, f"INSTALL.md never mentions [{extra}]"
