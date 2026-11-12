"""Tests for optional-dependency import resolution."""

from __future__ import annotations

import json
from typing import Any

import pytest

from core.deps import OptionalDependencyError, import_optional

pytestmark = pytest.mark.filterwarnings("error")


def test_missing_root_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """ModuleNotFoundError for requested root package reports missing=True."""

    def fake_import(_name: str) -> Any:
        raise ModuleNotFoundError("No module named 'gensim'", name="gensim")

    monkeypatch.setattr("core.deps.importlib.import_module", fake_import)
    with pytest.raises(OptionalDependencyError) as excinfo:
        import_optional("gensim")

    assert excinfo.value.missing is True
    assert excinfo.value.name == "gensim"


def test_missing_parent_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """ModuleNotFoundError for parent package of requested submodule reports missing=True."""

    def fake_import(_name: str) -> Any:
        raise ModuleNotFoundError("No module named 'gensim'", name="gensim")

    monkeypatch.setattr("core.deps.importlib.import_module", fake_import)
    with pytest.raises(OptionalDependencyError) as excinfo:
        import_optional("gensim.models.ldamodel")

    assert excinfo.value.missing is True
    assert excinfo.value.name == "gensim.models.ldamodel"


def test_missing_downstream_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """ModuleNotFoundError for another dependency reports missing=False with reason."""

    def fake_import(_name: str) -> Any:
        raise ModuleNotFoundError("No module named 'smart_open'", name="smart_open")

    monkeypatch.setattr("core.deps.importlib.import_module", fake_import)
    with pytest.raises(OptionalDependencyError) as excinfo:
        import_optional("gensim")

    assert excinfo.value.missing is False
    assert "smart_open" in excinfo.value.reason


def test_broken_import_dll_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """ImportError such as DLL load failure reports missing=False with reason."""

    def fake_import(_name: str) -> Any:
        raise ImportError("DLL load failed while importing _x: The specified module could not be found.")

    monkeypatch.setattr("core.deps.importlib.import_module", fake_import)
    with pytest.raises(OptionalDependencyError) as excinfo:
        import_optional("gensim")

    assert excinfo.value.missing is False
    assert "DLL load failed while importing _x" in excinfo.value.reason


def test_arbitrary_exception_reported_broken(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arbitrary exception during import reports missing=False with reason."""

    def fake_import(_name: str) -> Any:
        raise ValueError("bad pandas")

    monkeypatch.setattr("core.deps.importlib.import_module", fake_import)
    with pytest.raises(OptionalDependencyError) as excinfo:
        import_optional("gensim")

    assert excinfo.value.missing is False
    assert "bad pandas" in excinfo.value.reason


def test_real_import_with_attr() -> None:
    """Real import of 'json' with attr 'dumps' returns json.dumps."""
    dumps = import_optional("json", attr="dumps")
    assert dumps is json.dumps
