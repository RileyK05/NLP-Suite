"""Result[T] and Diagnostic contracts (ARCHITECTURE.md R4, R7, section 2.5)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from core.result import Diagnostic, Result, Severity


class TestDiagnostic:
    def test_error_severity(self) -> None:
        d = Diagnostic.error("EMPTY_DOC", "04.txt had no tokens", doc_id=4)
        assert d.severity is Severity.ERROR
        assert d.code == "EMPTY_DOC"
        assert d.context == {"doc_id": 4}

    def test_warning_and_info_helpers(self) -> None:
        assert Diagnostic.warning("W", "m").severity is Severity.WARNING
        assert Diagnostic.info("I", "m").severity is Severity.INFO

    def test_empty_code_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Diagnostic.error("", "no code")

    def test_bad_severity_type_rejected(self) -> None:
        with pytest.raises(TypeError):
            Diagnostic("ERROR", "CODE", "message")  # type: ignore[arg-type]

    def test_context_defaults_to_empty_dict_not_shared(self) -> None:
        a = Diagnostic.error("A", "m")
        b = Diagnostic.error("B", "m")
        assert a.context == {}
        a.context["x"] = 1
        assert b.context == {}

    def test_to_dict_is_json_shaped(self) -> None:
        d = Diagnostic.warning("UNPARSEABLE_DATE", "bad date", doc_id=7)
        assert d.to_dict() == {
            "severity": "WARNING",
            "code": "UNPARSEABLE_DATE",
            "message": "bad date",
            "context": {"doc_id": 7},
        }

    def test_equality_is_structural(self) -> None:
        assert Diagnostic.error("A", "m", x=1) == Diagnostic.error("A", "m", x=1)
        assert Diagnostic.error("A", "m", x=1) != Diagnostic.error("A", "m", x=2)

    def test_is_frozen(self) -> None:
        d = Diagnostic.error("A", "m")
        with pytest.raises(FrozenInstanceError):
            d.code = "B"  # type: ignore[misc]


class TestResultBasics:
    def test_success_ok(self) -> None:
        r = Result.success(42)
        assert r.ok
        assert r.value == 42
        assert not r.has_errors

    def test_failure_not_ok(self) -> None:
        r = Result[int].failure(Diagnostic.error("BOOM", "it broke"))
        assert not r.ok
        assert r.value is None
        assert r.has_errors
        assert len(r.errors) == 1

    def test_failure_requires_a_diagnostic(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            Result[int].failure()

    def test_warning_does_not_fail_a_result(self) -> None:
        r = Result.success(1, Diagnostic.warning("W", "just so you know"))
        assert r.ok
        assert len(r.warnings) == 1
        assert not r.has_errors

    def test_value_plus_error_is_not_ok(self) -> None:
        """Partial success: a value was produced, but something genuinely failed."""
        r = Result.success([1, 2, 3], Diagnostic.error("EMPTY_DOC", "04.txt empty"))
        assert r.value == [1, 2, 3]
        assert not r.ok
        assert len(r.errors) == 1


class TestResultSeverityPartitioning:
    def test_partitioning(self) -> None:
        r = Result.success(
            1,
            Diagnostic.info("I", "i"),
            Diagnostic.warning("W", "w"),
            Diagnostic.error("E", "e"),
        )
        assert len(r.infos) == 1
        assert len(r.warnings) == 1
        assert len(r.errors) == 1

    def test_empty_result_has_no_errors(self) -> None:
        assert not Result[int]().has_errors
        assert Result[int]().errors == ()


class TestResultUnwrap:
    def test_unwrap_returns_value(self) -> None:
        assert Result.success("x").unwrap() == "x"

    def test_unwrap_raises_on_failure(self) -> None:
        r = Result[str].failure(Diagnostic.error("BOOM", "it broke"))
        with pytest.raises(ValueError, match="failed Result"):
            r.unwrap()

    def test_unwrap_or(self) -> None:
        assert Result.success("x").unwrap_or("d") == "x"
        assert Result[str].failure(Diagnostic.error("E", "no")).unwrap_or("d") == "d"


class TestResultCombinators:
    def test_map_transforms_value(self) -> None:
        assert Result.success(2).map(lambda n: n * 10).unwrap() == 20

    def test_map_is_identity_on_failure(self) -> None:
        r = Result[int].failure(Diagnostic.error("E", "no"))
        mapped = r.map(lambda n: n * 10)
        assert mapped.value is None
        assert mapped.errors == r.errors

    def test_with_diagnostics_preserves_value_and_order(self) -> None:
        first = Diagnostic.warning("W", "first")
        r = Result.success(1, first).with_diagnostics(Diagnostic.error("E", "second"))
        assert r.value == 1
        assert [d.code for d in r.diagnostics] == ["W", "E"]
        assert not r.ok

    def test_map_carries_diagnostics(self) -> None:
        r = Result.success(2, Diagnostic.warning("W", "w")).map(lambda n: n + 1)
        assert r.unwrap() == 3
        assert len(r.warnings) == 1

    def test_is_frozen(self) -> None:
        r = Result.success(1)
        with pytest.raises(FrozenInstanceError):
            r.value = 2  # type: ignore[misc]


class TestPartialSuccessShape:
    """The shape the legacy suite could not express: 97 good docs + 3 bad ones."""

    def test_corpus_with_bad_documents(self) -> None:
        good = list(range(97))
        diags = tuple(Diagnostic.error("EMPTY_DOC", f"{i}.txt had no tokens", doc_id=i) for i in (4, 18, 91))
        r = Result.success(good, *diags)

        assert len(r.unwrap()) == 97
        assert len(r.errors) == 3
        assert not r.ok
        assert {(d.severity, d.code) for d in r.diagnostics} == {(Severity.ERROR, "EMPTY_DOC")}
