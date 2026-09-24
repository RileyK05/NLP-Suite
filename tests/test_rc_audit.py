"""FR-9.8 — RC audit contract tests (pure check functions, tmp roots)."""

from __future__ import annotations

from pathlib import Path

from scripts.rc_audit import audit_capabilities, audit_docs, audit_ledger_files, audit_triple, main


class TestAudit:
    def test_clean_tree_passes(self) -> None:
        assert main([]) == 0

    def test_capabilities_covered_on_clean_tree(self) -> None:
        assert audit_capabilities() == []

    def test_capabilities_flags_id_with_no_ledger_row(self, tmp_path: Path) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        # A ledger that defines nothing: every registry claim is then unbacked.
        (docs / "REPLACEMENT_LEDGER.md").write_text("| ID | Capability |", encoding="utf-8")
        failures = audit_capabilities(tmp_path)
        assert failures, "an empty ledger must not look fully covered"
        assert all("has no ledger row" in f for f in failures)

    def test_docs_detects_missing(self, tmp_path: Path) -> None:
        failures = audit_docs(tmp_path)
        assert any("MIGRATION.md" in f for f in failures)

    def test_ledger_flags_missing_path(self, tmp_path: Path) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "REPLACEMENT_LEDGER.md").write_text(
            "| FR-9.9 | Nope | review | — | — | NEW `core/nope.py` done |\n", encoding="utf-8"
        )
        failures = audit_ledger_files(tmp_path)
        assert failures == ["ledger names missing path: core/nope.py"]

    def test_ledger_strips_symbol_refs(self, tmp_path: Path) -> None:
        (tmp_path / "core").mkdir()
        (tmp_path / "core" / "real.py").write_text("x = 1\n", encoding="utf-8")
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "REPLACEMENT_LEDGER.md").write_text(
            "| FR-9.9 | Yup | review | — | — | `core/real.py::thing` done |\n", encoding="utf-8"
        )
        assert audit_ledger_files(tmp_path) == []

    def test_ledger_ignores_non_review_rows(self, tmp_path: Path) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "REPLACEMENT_LEDGER.md").write_text(
            "| FR-9.9 | Nope | todo | — | — | NEW `core/nope.py` planned |\n", encoding="utf-8"
        )
        assert audit_ledger_files(tmp_path) == []

    def test_triple_passes_on_real_tree(self) -> None:
        assert audit_triple() == []
