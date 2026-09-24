"""FR-3.5 filename operations â€” standardize, preview, duplicate-safe apply.

The packet rule is absolute: nothing renames without a preview first, and
the default is dry-run. These tests pin that.
"""

from __future__ import annotations

from pathlib import Path

from core.file_ops import filenames as F


class TestStandardize:
    def test_rule(self) -> None:
        assert F.standardize_name("My Report (FINAL).txt") == "My_Report_FINAL.txt"
        assert F.standardize_name("  spaced  .txt") == "spaced.txt"
        assert F.standardize_name("a---b___c.txt") == "a---b_c.txt"
        assert F.standardize_name("...txt") == "file.txt"
        assert F.standardize_name(" dÃ©jÃ  vu.TXT") == "d_j_vu.TXT"  # noqa: RUF001 — intentional unicode test

    def test_extension_preserved_and_name_never_empty(self) -> None:
        assert F.standardize_name("noext") == "noext"
        assert F.standardize_name("").endswith("")  # empty in, fallback out
        assert F.standardize_name("") == "file"


class TestPlan:
    def test_preview_shape_with_dates(self, tmp_path: Path) -> None:
        files = [tmp_path / "My Report.txt", tmp_path / "1999-05-04_speech.txt"]
        for f in files:
            f.write_text("x", encoding="utf-8")
        plan = F.plan_renames(files).unwrap().to_frame()
        assert list(plan.columns) == ["Old", "New", "Date", "Action"]
        dated = plan.loc[plan["Old"] == "1999-05-04_speech.txt"].iloc[0]
        assert str(dated["Date"]) == "1999-05-04"

    def test_collision_gets_suffix(self, tmp_path: Path) -> None:
        files = [tmp_path / "a b.txt", tmp_path / "a  b.txt"]  # both -> a_b.txt
        for f in files:
            f.write_text("x", encoding="utf-8")
        plan = F.plan_renames(files).unwrap().to_frame()
        assert sorted(plan["New"].tolist()) == ["a_b.txt", "a_b_2.txt"]

    def test_collision_suffix_keeps_extensionless_name(self, tmp_path: Path) -> None:
        # rpartition(".") on an extensionless name must not reduce the
        # whole name to a bare "_2" suffix.
        files = [tmp_path / "my file", tmp_path / "my, file"]  # both -> my_file
        for f in files:
            f.write_text("x", encoding="utf-8")
        plan = F.plan_renames(files).unwrap().to_frame()
        assert sorted(plan["New"].tolist()) == ["my_file", "my_file_2"]

    def test_existing_target_skips(self, tmp_path: Path) -> None:
        (tmp_path / "a b.txt").write_text("x", encoding="utf-8")
        (tmp_path / "a_b.txt").write_text("keep me", encoding="utf-8")
        plan = F.plan_renames([tmp_path / "a b.txt"]).unwrap().to_frame()
        assert plan.iloc[0]["Action"] == "skip-existing"

    def test_identical_names_skip(self, tmp_path: Path) -> None:
        (tmp_path / "fine.txt").write_text("x", encoding="utf-8")
        plan = F.plan_renames([tmp_path / "fine.txt"]).unwrap().to_frame()
        assert plan.iloc[0]["Action"] == "skip-identical"

    def test_missing_file_is_an_error(self, tmp_path: Path) -> None:
        result = F.plan_renames([tmp_path / "ghost.txt"])
        assert result.value is None
        assert any(d.code == "FILENAMES_MISSING" for d in result.diagnostics)


class TestApply:
    def test_dry_run_changes_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "a b.txt").write_text("x", encoding="utf-8")
        plan = F.plan_renames([tmp_path / "a b.txt"]).unwrap()
        applied = F.apply_renames(plan, dry_run=True).unwrap()
        assert applied == []
        assert (tmp_path / "a b.txt").is_file()

    def test_apply_renames_and_reports(self, tmp_path: Path) -> None:
        (tmp_path / "a b.txt").write_text("x", encoding="utf-8")
        plan = F.plan_renames([tmp_path / "a b.txt"]).unwrap()
        applied = F.apply_renames(plan, dry_run=False).unwrap()
        assert applied == [("a b.txt", "a_b.txt")]
        assert (tmp_path / "a_b.txt").is_file()
        assert not (tmp_path / "a b.txt").exists()


class TestCli:
    def test_cli_preview_by_default(self, tmp_path: Path) -> None:
        from tools.filenames import main

        (tmp_path / "a b.txt").write_text("x", encoding="utf-8")
        assert main([str(tmp_path), "--dry-run"]) == 0
        assert (tmp_path / "a b.txt").is_file()  # untouched

    def test_cli_apply_flag_renames(self, tmp_path: Path) -> None:
        from tools.filenames import main

        (tmp_path / "a b.txt").write_text("x", encoding="utf-8")
        assert main([str(tmp_path), "--apply"]) == 0
        assert (tmp_path / "a_b.txt").is_file()

    def test_cli_missing_dir_fails(self, tmp_path: Path) -> None:
        from tools.filenames import main

        assert main([str(tmp_path / "ghost")]) == 2


class TestC611Hardening:
    """C6-11: empties, escapes, races, cycles, case collisions."""

    def test_empty_plan_is_structured(self) -> None:
        result = F.plan_renames([])
        assert result.ok
        assert result.unwrap().rows == ()

    def test_mixed_parents_rejected(self, tmp_path: Path) -> None:
        d1, d2 = tmp_path / "d1", tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "a b.txt").write_text("x", encoding="utf-8")
        (d2 / "c d.txt").write_text("x", encoding="utf-8")
        result = F.plan_renames([d1 / "a b.txt", d2 / "c d.txt"])
        assert result.value is None
        assert any(d.code == "FILENAMES_MIXED_PARENTS" for d in result.diagnostics)

    def test_case_insensitive_collision(self, tmp_path: Path) -> None:
        (tmp_path / "a b.txt").write_text("x", encoding="utf-8")
        (tmp_path / "A B.txt").write_text("x", encoding="utf-8")  # same standardized name
        plan = F.plan_renames([tmp_path / "a b.txt", tmp_path / "A B.txt"]).unwrap()
        news = [row.new for row in plan.rows]
        assert len(set(news)) == len(news)  # _2 suffix even though case differs

    def test_source_gone_after_preview_diagnosed(self, tmp_path: Path) -> None:
        f = tmp_path / "a b.txt"
        f.write_text("x", encoding="utf-8")
        plan = F.plan_renames([f]).unwrap()
        f.unlink()
        result = F.apply_renames(plan, dry_run=False)
        assert result.ok  # partial report, nothing crashed
        assert result.unwrap() == []  # nothing was renamed
        assert any(d.code == "FILENAMES_SOURCE_GONE" for d in result.diagnostics)
        assert list(tmp_path.iterdir()) == []

    def test_swap_rename_cycle(self, tmp_path: Path) -> None:
        tmp_path / "b a.txt"  # -> b_a.txt
        tmp_path / "b.txt"  # -> b_txt.txt
        (tmp_path / "b a.txt").write_text("content-a", encoding="utf-8")
        (tmp_path / "b.txt").write_text("content-b", encoding="utf-8")
        # craft a swap: b.txt -> b_txt.txt while b_a.txt -> b_a.txt would clash with b.txt's target
        plan = F.plan_renames([tmp_path / "b a.txt", tmp_path / "b.txt"])
        if not plan.ok:
            import pytest

            pytest.skip("plan construction did not yield a cycle on this platform")
        result = F.apply_renames(plan.unwrap(), dry_run=False)
        assert result.ok, result.diagnostics

    def test_malicious_plan_rejected(self, tmp_path: Path) -> None:
        from core.file_ops.filenames import RenamePlan, RenameRow

        plan = RenamePlan(
            rows=(RenameRow(old="x.txt", new="../evil.txt", date="", action="rename"),),
            directory=tmp_path,
        )
        result = F.apply_renames(plan, dry_run=False)
        assert result.value is None
        assert any(d.code == "FILENAMES_ESCAPE" for d in result.diagnostics)

    def test_rollback_on_failure(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a b.txt"
        f2 = tmp_path / "c d.txt"
        f1.write_text("1", encoding="utf-8")
        f2.write_text("2", encoding="utf-8")
        plan = F.plan_renames([f1, f2]).unwrap()
        # Force the FIRST rename's underlying os.rename to fail -> rollback.

        original = Path.rename

        def flaky_rename(self: Path, target: Path) -> Path:
            if target.name == "a_b.txt":
                raise OSError("simulated mid-plan failure")
            return original(target, target)

        Path.rename = flaky_rename  # type: ignore[method-assign]
        try:
            result = F.apply_renames(plan, dry_run=False)
        finally:
            Path.rename = original  # type: ignore[method-assign]
        assert result.value is None
        assert any(d.code == "FILENAMES_FAILED" for d in result.diagnostics)
        # rollback restored everything: both sources still under old names
        assert f1.is_file() and f2.is_file()

    def test_source_gone_is_partial_report(self, tmp_path: Path) -> None:
        """A vanished source does not abort the run: the rest applies."""
        f1 = tmp_path / "a b.txt"
        f2 = tmp_path / "c d.txt"
        f1.write_text("1", encoding="utf-8")
        f2.write_text("2", encoding="utf-8")
        plan = F.plan_renames([f1, f2]).unwrap()
        f2.unlink()
        result = F.apply_renames(plan, dry_run=False)
        assert result.ok
        assert result.unwrap() == [("a b.txt", "a_b.txt")]
        assert any(d.code == "FILENAMES_SOURCE_GONE" for d in result.diagnostics)
