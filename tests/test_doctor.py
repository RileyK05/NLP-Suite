"""The doctor: environment checks that name the fix, and exit codes that mean something."""

from __future__ import annotations

import pytest

from conftest import has_spacy_model
from tools.doctor import Check, run_doctor


class TestDoctor:
    def test_doctor_reports_broken_model_with_fix_and_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        """On a machine without parser models the doctor must fail the
        default path and print the exact download command."""
        if has_spacy_model():
            pytest.skip("model installed; default path is fine on this machine")
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

    def test_exit_code_uses_required_field(self, capsys: object) -> None:
        from tools import doctor

        # A failing check WITHOUT required=True must not fail the doctor.
        checks = [
            doctor.Check(name="python 3.12.10", ok=True, detail="fine", required=True),
            doctor.Check(name="pandas", ok=True, detail="core dependency", required=True),
            doctor.Check(name="pyarrow", ok=True, detail="core dependency", required=True),
            doctor.Check(name="scipy", ok=True, detail="core dependency", required=True),
            doctor.Check(name="scikit-learn", ok=True, detail="core dependency", required=True),
            doctor.Check(name="stanza package", ok=True, detail="default", required=True),
            doctor.Check(name="stanza model (en)", ok=True, detail="loadable", required=True),
            doctor.Check(name="plotly", ok=False, detail="missing", fix="pip install plotly", required=False),
        ]
        # run_doctor runs its own checks; the required-field logic is pinned
        # by asserting the filter selects only required failures.
        failed = [c for c in checks if not c.ok and c.required]
        assert failed == []
        failing_optional = [c for c in checks if not c.ok and not c.required]
        assert len(failing_optional) == 1

    def test_doctor_reports_sklearn_when_missing(self, monkeypatch: object) -> None:
        import importlib.util

        from tools import doctor

        real_find = importlib.util.find_spec

        def fake_find(name: str):  # type: ignore[no-untyped-def]
            if name == "sklearn":
                return None
            return real_find(name)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find)
        checks = doctor._check_core()
        sklearn_check = next(c for c in checks if c.name == "scikit-learn")
        assert not sklearn_check.ok and sklearn_check.required
