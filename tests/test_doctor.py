"""The doctor: environment checks that name the fix, and exit codes that mean something."""

from __future__ import annotations

import pytest

from tools import doctor
from tools.doctor import Check, run_doctor


class TestDoctor:
    def test_doctor_reports_broken_model_with_fix_and_exits_1(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing parsers must fail and print the download command everywhere."""
        monkeypatch.setattr(doctor, "_check_core", lambda: [])
        monkeypatch.setattr(doctor, "_check_extras", lambda: [])
        monkeypatch.setattr(doctor, "_spacy_model_ok", lambda language: (False, "missing test model", False))
        monkeypatch.setattr(doctor, "_stanza_model_ok", lambda language: (False, "missing test model", False))
        code, checks = run_doctor(["en"])
        assert code == 1
        names = {check.name: check for check in checks}
        spacy_model_check = names["spacy model (en)"]
        assert not spacy_model_check.ok
        assert "python -m spacy download" in spacy_model_check.fix
        captured = capsys.readouterr()
        assert "fix: python -m spacy download" in captured.out

    def test_doctor_rejects_unknown_language(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, checks = run_doctor(["xx"])
        assert code == 2
        assert checks == []
        captured = capsys.readouterr()
        assert "not a language" in captured.err

    def test_check_dataclass_holds_fix(self) -> None:
        check = Check(name="thing", ok=False, detail="missing", fix="pip install thing")
        assert check.fix == "pip install thing"
        ok_check = Check(name="thing", ok=True, detail="fine")
        assert ok_check.fix == ""


class TestC614Corrections:
    """C6-14: explicit required field, full core deps, injected versions."""

    def test_required_field_present(self) -> None:
        from tools.doctor import Check

        assert Check(name="x", ok=False, detail="d", required=True).required
        assert not Check(name="x", ok=False, detail="d").required

    def test_version_check_required_and_exit_affecting(self, capsys: object) -> None:
        from tools import doctor

        checks = doctor._check_core()
        python_check = next(c for c in checks if c.name.startswith("python"))
        assert python_check.required
        # scipy + sklearn are core deps now
        names = {c.name for c in checks}
        assert {"scipy", "scikit-learn"} <= names

    @pytest.mark.parametrize(("required", "exit_code"), [(False, 0), (True, 1)])
    def test_exit_code_uses_required_field(
        self, monkeypatch: pytest.MonkeyPatch, required: bool, exit_code: int
    ) -> None:
        checks = [
            Check(name="dependency", ok=False, detail="missing", required=required),
        ]
        monkeypatch.setattr(doctor, "_check_core", lambda: checks)
        monkeypatch.setattr(doctor, "_check_parsers", lambda languages: [])
        monkeypatch.setattr(doctor, "_check_extras", lambda: [])
        code, reported = run_doctor(["en"])
        assert code == exit_code
        assert reported == checks

    def test_doctor_reports_sklearn_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core import deps
        from tools import doctor

        def fake_import(name: str, *, attr: str | None = None, suppress_warnings: bool = False):  # type: ignore[no-untyped-def]
            if name == "sklearn":
                raise deps.OptionalDependencyError("sklearn", missing=True)
            return deps.importlib.import_module(name)

        monkeypatch.setattr(deps, "import_optional", fake_import)
        checks = doctor._check_core()
        sklearn_check = next(c for c in checks if c.name == "scikit-learn")
        assert not sklearn_check.ok and sklearn_check.required


class TestImportTimeFailures:
    """A package present on disk but refusing to import is broken, not OK.

    ``find_spec`` locates a module without importing it, so a version gate at
    import time ("this version of pandas is not supported by gensim") used to
    read as "OK" and then crash mid-run. The doctor must import to see it.
    """

    def test_present_but_unimportable_is_not_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core import deps

        def fake_import(name: str, *, attr: str | None = None, suppress_warnings: bool = False):  # type: ignore[no-untyped-def]
            raise deps.OptionalDependencyError(
                "gensim", missing=False, reason="this version of pandas is not supported by gensim"
            )

        monkeypatch.setattr(deps, "import_optional", fake_import)
        ok, detail, damaged = doctor._import_state("gensim")
        assert not ok
        assert damaged
        assert "will not import" in detail
        assert "not supported by gensim" in detail

    def test_missing_module_says_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core import deps

        def fake_import(name: str, *, attr: str | None = None, suppress_warnings: bool = False):  # type: ignore[no-untyped-def]
            raise deps.OptionalDependencyError("gensim", missing=True)

        monkeypatch.setattr(deps, "import_optional", fake_import)
        ok, detail, damaged = doctor._import_state("gensim")
        assert not ok and detail == "missing"
        assert not damaged

    def test_importable_module_is_ok(self) -> None:
        ok, detail, damaged = doctor._import_state("json")
        assert ok and detail == "ok"
        assert not damaged


class TestOptionalDependencyImport:
    """core.deps.import_optional separates missing from broken."""

    def test_returns_module_and_attr(self) -> None:
        from core.deps import import_optional

        assert import_optional("json").dumps([1]) == "[1]"
        assert import_optional("json", attr="dumps")([2]) == "[2]"

    def test_absent_module_is_missing(self) -> None:
        from core.deps import OptionalDependencyError, import_optional

        with pytest.raises(OptionalDependencyError) as caught:
            import_optional("definitely_not_a_real_module_xyz")
        assert caught.value.missing
        assert "is not installed" in str(caught.value)

    def test_module_that_raises_on_import_is_broken_not_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core import deps

        def boom(name: str):  # type: ignore[no-untyped-def]
            raise ValueError("This version of pandas is not supported by gensim")

        monkeypatch.setattr(deps.importlib, "import_module", boom)
        with pytest.raises(deps.OptionalDependencyError) as caught:
            deps.import_optional("gensim")
        assert not caught.value.missing
        assert "not supported by gensim" in caught.value.reason
        # Still an ImportError, so an existing `except ImportError` catches it.
        assert isinstance(caught.value, ImportError)

    def test_windows_dll_oserror_is_broken(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core import deps

        def boom(name: str):  # type: ignore[no-untyped-def]
            raise OSError("[WinError 126] The specified module could not be found")

        monkeypatch.setattr(deps.importlib, "import_module", boom)
        with pytest.raises(deps.OptionalDependencyError) as caught:
            deps.import_optional("something")
        assert not caught.value.missing
        assert "WinError 126" in caught.value.reason

    def test_suppress_warnings_swallows_a_deprecation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import types
        import warnings

        from core import deps

        def import_and_warn(name: str):  # type: ignore[no-untyped-def]
            warnings.warn("deprecated", DeprecationWarning, stacklevel=2)
            return types.ModuleType(name)

        monkeypatch.setattr(deps.importlib, "import_module", import_and_warn)
        assert deps.import_optional("some_deprecated_package", suppress_warnings=True) is not None


@pytest.mark.filterwarnings("error")
class TestMissingVersusBroken:
    def test_absent(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        def fake_import_state(module: str) -> tuple[bool, str, bool]:
            if module == "plotly":
                return False, "missing", False
            return True, "ok", False

        def fake_has_module(name: str) -> bool:
            return name != "stanza"

        def fake_stanza_model_ok(_language: str) -> tuple[bool, str, bool]:
            return False, "stanza package not installed", False

        def fake_spacy_model_ok(_language: str) -> tuple[bool, str, bool]:
            return True, "en_core_web_sm", False

        monkeypatch.setattr(doctor, "_import_state", fake_import_state)
        monkeypatch.setattr(doctor, "_has_module", fake_has_module)
        monkeypatch.setattr(doctor, "_stanza_model_ok", fake_stanza_model_ok)
        monkeypatch.setattr(doctor, "_spacy_model_ok", fake_spacy_model_ok)

        code, _ = run_doctor(["en"])
        assert code == 0

        captured = capsys.readouterr().out
        assert "[MISSING] plotly" in captured
        assert "[MISSING] stanza package" in captured
        assert "[MISSING] stanza model (en)" in captured
        assert "Not installed: stanza model (en)" in captured
        assert "BROKEN" not in captured
        assert "Damaged or incomplete" not in captured

    def test_damaged(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        def fake_import_state(module: str) -> tuple[bool, str, bool]:
            if module == "gensim":
                return False, "present but will not import (boom)", True
            return True, "ok", False

        def fake_has_module(_name: str) -> bool:
            return True

        def fake_stanza_model_ok(_language: str) -> tuple[bool, str, bool]:
            return False, "model files corrupt", True

        def fake_spacy_model_ok(_language: str) -> tuple[bool, str, bool]:
            return True, "en_core_web_sm", False

        monkeypatch.setattr(doctor, "_import_state", fake_import_state)
        monkeypatch.setattr(doctor, "_has_module", fake_has_module)
        monkeypatch.setattr(doctor, "_stanza_model_ok", fake_stanza_model_ok)
        monkeypatch.setattr(doctor, "_spacy_model_ok", fake_spacy_model_ok)

        code, _ = run_doctor(["en"])
        assert code == 0

        captured = capsys.readouterr().out
        assert "[BROKEN ] gensim" in captured
        assert "[BROKEN ] stanza model (en)" in captured
        assert "Damaged or incomplete: stanza model (en)" in captured
        assert "Not installed" not in captured

    def test_unit_stanza_model_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_has_module(_name: str) -> bool:
            return False

        monkeypatch.setattr(doctor, "_has_module", fake_has_module)
        result = doctor._stanza_model_ok("en")
        assert result[2] is False
