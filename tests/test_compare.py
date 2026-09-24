"""FR-1.5 comparison harness — ordering, tolerance, schema, missing-value, artifact rules.

The comparer is the Gate B instrument: goldens must be compared by declared
rules, never by eyeballing diffs. Every rule below is pinned by a test.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd

from core.artifacts.envelope import Artifact, Envelope, InputRef
from core.compare import (
    CompareConfig,
    compare_csv_files,
    compare_envelopes,
    compare_frames,
    compare_runs,
)


def _frame(rows: list[list[object]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def _small() -> pd.DataFrame:
    return _frame(
        [[1, "cats", 0.5], [2, "dogs", 1.5], [3, "birds", 2.5]],
        ["ID", "Token", "Score"],
    )


def _make_run(
    path: Path,
    *,
    params: dict[str, object] | None = None,
    rows: list[list[object]] | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    df = _frame(
        rows if rows is not None else [[1, "cats", 0.5], [2, "dogs", 1.5]],
        ["ID", "Token", "Score"],
    )
    df.to_csv(path / "out.csv", index=False)
    env = Envelope(
        tool="demo",
        params=dict(params) if params else {},
        inputs=(InputRef(path="a.txt", sha256="abc123"),),
        artifacts=(Artifact(kind="table", path="out.csv"),),
        diagnostics=(),
        corpus_sha256="deadbeef",
    )
    assert env.write(path).ok
    return path


class TestOrdering:
    def test_identical_frames_pass(self) -> None:
        result = compare_frames(_small(), _small().copy())
        assert result.ok
        assert result.unwrap().passed
        assert any(d.code == "COMPARE_MATCH" for d in result.diagnostics)

    def test_shuffled_rows_pass_with_keys(self) -> None:
        shuffled = _small().iloc[[2, 0, 1]].reset_index(drop=True)
        result = compare_frames(_small(), shuffled, CompareConfig(key_columns=("ID",)))
        assert result.ok, [str(d) for d in result.diagnostics]

    def test_shuffled_rows_pass_without_keys_by_default(self) -> None:
        shuffled = _small().iloc[[2, 0, 1]].reset_index(drop=True)
        assert compare_frames(_small(), shuffled).ok

    def test_shuffled_rows_fail_when_order_matters(self) -> None:
        shuffled = _small().iloc[[2, 0, 1]].reset_index(drop=True)
        result = compare_frames(_small(), shuffled, CompareConfig(order_matters=True))
        assert not result.ok
        assert any(d.code == "COMPARE_VALUE_MISMATCH" for d in result.diagnostics)

    def test_duplicate_keys_fail_fast(self) -> None:
        dup = _frame([[1, "a", 0.1], [1, "b", 0.2]], ["ID", "Token", "Score"])
        result = compare_frames(dup, dup.copy(), CompareConfig(key_columns=("ID",)))
        assert not result.ok
        assert any(d.code == "COMPARE_DUPLICATE_KEYS" for d in result.diagnostics)

    def test_unknown_key_column_is_a_harness_error(self) -> None:
        result = compare_frames(_small(), _small().copy(), CompareConfig(key_columns=("Nope",)))
        assert not result.ok
        assert any(d.code == "COMPARE_BAD_KEY" for d in result.diagnostics)


class TestNumericTolerance:
    def test_floats_within_tolerance_pass(self) -> None:
        other = _small()
        other.loc[0, "Score"] = 0.5000001  # envelope rounds to 6dp; goldens wobble there
        assert compare_frames(_small(), other).ok

    def test_floats_outside_tolerance_fail(self) -> None:
        other = _small()
        other.loc[0, "Score"] = 0.6
        result = compare_frames(_small(), other)
        assert not result.ok
        diff = result.unwrap().diff
        assert (diff["Column"] == "Score").any()
        assert "0.5" in str(diff.iloc[0]["Expected"])

    def test_int_float_indifference(self) -> None:
        other = _small()
        other["ID"] = other["ID"].astype(float)  # CSV round-trips may change dtype
        assert compare_frames(_small(), other).ok

    def test_custom_tolerance(self) -> None:
        other = _small()
        other.loc[0, "Score"] = 0.54
        assert not compare_frames(_small(), other).ok
        assert compare_frames(_small(), other, CompareConfig(rtol=0.1)).ok

    def test_numeric_strings_compare_numerically(self) -> None:
        left = _frame([["1.50"]], ["Score"])
        right = _frame([[1.5]], ["Score"])
        assert compare_frames(left, right).ok


class TestSchema:
    def test_missing_column_fails_with_names(self) -> None:
        result = compare_frames(_small(), _small().drop(columns=["Score"]))
        assert not result.ok
        diag = next(d for d in result.diagnostics if d.code == "COMPARE_SCHEMA_MISMATCH")
        assert "Score" in str(diag.context)

    def test_extra_column_fails(self) -> None:
        wider = _small().copy()
        wider["New"] = 1
        result = compare_frames(_small(), wider)
        assert not result.ok
        assert any(d.code == "COMPARE_SCHEMA_MISMATCH" for d in result.diagnostics)

    def test_column_order_is_warning_only(self) -> None:
        reordered = _small()[["Score", "Token", "ID"]]
        result = compare_frames(_small(), reordered)
        assert result.ok
        assert any(d.code == "COMPARE_COLUMN_ORDER" for d in result.diagnostics)

    def test_ignore_columns(self) -> None:
        wider = _small().copy()
        wider["RunPath"] = "outputs/some_run"
        result = compare_frames(_small(), wider, CompareConfig(ignore_columns=("RunPath",)))
        assert result.ok


class TestMissingValues:
    def test_all_pandas_missing_forms_are_equal(self) -> None:
        left = _frame([[None]], ["A"])
        right = _frame([[float("nan")]], ["A"])
        assert compare_frames(left, right).ok

    def test_missing_vs_value_fails(self) -> None:
        left = _frame([[None]], ["A"])
        right = _frame([["x"]], ["A"])
        result = compare_frames(left, right)
        assert not result.ok
        assert "<missing>" in str(result.unwrap().diff.iloc[0]["Expected"])

    def test_empty_string_is_a_value_by_default(self) -> None:
        left = _frame([[""]], ["A"])
        right = _frame([[None]], ["A"])
        assert not compare_frames(left, right).ok

    def test_empty_string_counts_as_missing_when_configured(self) -> None:
        left = _frame([[""]], ["A"])
        right = _frame([[None]], ["A"])
        assert compare_frames(left, right, CompareConfig(empty_as_missing=True)).ok

    def test_na_string_is_a_value_never_silently_missing(self) -> None:
        left = _frame([["NA"]], ["A"])
        right = _frame([[None]], ["A"])
        assert not compare_frames(left, right).ok


class TestRowCounts:
    def test_extra_row_fails_with_row_count_code(self) -> None:
        longer = pd.concat([_small(), _frame([[4, "fish", 3.5]], ["ID", "Token", "Score"])])
        result = compare_frames(_small(), longer, CompareConfig(key_columns=("ID",)))
        assert not result.ok
        assert any(d.code == "COMPARE_ROW_COUNT" for d in result.diagnostics)
        assert "extra row" in " ".join(map(str, result.unwrap().diff["Issue"].tolist()))

    def test_mismatch_result_stays_inspectable(self) -> None:
        other = _small()
        other.loc[0, "Score"] = 99.0
        result = compare_frames(_small(), other)
        assert not result.ok  # ERROR diagnostic present...
        assert len(result.unwrap().diff) == 1  # ...but the diff is still there to read


class TestCsvFiles:
    def test_self_copy_passes(self, tmp_path: Path) -> None:
        csv = tmp_path / "golden.csv"
        _small().to_csv(csv, index=False)
        assert compare_csv_files(csv, csv).ok

    def test_missing_file_is_a_harness_error(self, tmp_path: Path) -> None:
        result = compare_csv_files(tmp_path / "nope.csv", tmp_path / "alsono.csv")
        assert result.value is None
        assert any(d.code == "COMPARE_FILE_NOT_FOUND" for d in result.diagnostics)

    def test_csv_default_na_strings_are_values(self, tmp_path: Path) -> None:
        # pd.read_csv would normally turn "NA" into NaN; the harness must not.
        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        _frame([["NA"]], ["A"]).to_csv(left, index=False)
        _frame([["NA"]], ["A"]).to_csv(right, index=False)
        assert compare_csv_files(left, right).ok


class TestEnvelopes:
    def _env(self, **overrides: object) -> Envelope:
        base: dict[str, object] = {
            "tool": "demo",
            "params": {"alpha": 0.5},
            "inputs": (InputRef(path="a.txt", sha256="abc"),),
            "artifacts": (Artifact(kind="table", path="out.csv"),),
            "diagnostics": (),
            "corpus_sha256": "deadbeef",
        }
        base.update(overrides)
        return Envelope(
            tool=str(base["tool"]),
            params=dict(base["params"]),  # type: ignore[arg-type]
            inputs=tuple(base["inputs"]),  # type: ignore[arg-type]
            artifacts=tuple(base["artifacts"]),  # type: ignore[arg-type]
            diagnostics=tuple(base["diagnostics"]),  # type: ignore[arg-type]
            corpus_sha256=str(base["corpus_sha256"]),
        )

    def test_envelope_self_passes(self) -> None:
        assert compare_envelopes(self._env(), self._env()).ok

    def test_created_timestamp_is_ignored(self) -> None:
        left = self._env()
        right = Envelope.from_dict({**left.to_dict(), "created": "1999-01-01T00:00:00+00:00"})
        assert compare_envelopes(left, right).ok

    def test_param_change_is_caught(self) -> None:
        result = compare_envelopes(self._env(), self._env(params={"alpha": 0.9}))
        assert not result.ok
        assert any(d.code == "COMPARE_ENVELOPE_MISMATCH" for d in result.diagnostics)

    def test_float_params_use_tolerance(self) -> None:
        assert compare_envelopes(self._env(), self._env(params={"alpha": 0.5000001})).ok


class TestRuns:
    def test_self_copy_passes_the_gate(self, tmp_path: Path) -> None:
        golden = _make_run(tmp_path / "golden")
        copy = _make_run(tmp_path / "copy")  # same logical run, fresh timestamps
        result = compare_runs(golden, copy)
        assert result.ok, [str(d) for d in result.diagnostics]

    def test_tampered_cell_points_at_artifact_and_column(self, tmp_path: Path) -> None:
        golden = _make_run(tmp_path / "golden")
        copy_dir = tmp_path / "copy"
        shutil.copytree(golden, copy_dir)
        tampered = pd.read_csv(copy_dir / "out.csv")
        tampered.loc[0, "Score"] = 99.0
        tampered.to_csv(copy_dir / "out.csv", index=False)
        result = compare_runs(golden, copy_dir)
        assert not result.ok
        cols = result.unwrap().diff["Column"].tolist()
        assert cols and all(str(c).startswith("out.csv::") for c in cols)

    def test_missing_artifact_fails(self, tmp_path: Path) -> None:
        golden = _make_run(tmp_path / "golden")
        copy_dir = tmp_path / "copy"
        shutil.copytree(golden, copy_dir)
        (copy_dir / "out.csv").unlink()
        result = compare_runs(golden, copy_dir)
        assert not result.ok
        assert "missing artifact" in " ".join(map(str, result.unwrap().diff["Issue"].tolist()))

    def test_missing_envelope_is_a_harness_error(self, tmp_path: Path) -> None:
        golden = _make_run(tmp_path / "golden")
        empty = tmp_path / "empty"
        empty.mkdir()
        result = compare_runs(golden, empty)
        assert result.value is None
        assert any(d.code == "ENVELOPE_MISSING" for d in result.diagnostics)

    def test_ignore_params_for_relocated_runs(self, tmp_path: Path) -> None:
        golden = _make_run(tmp_path / "golden", params={"input": str(tmp_path / "golden")})
        moved = _make_run(tmp_path / "moved", params={"input": str(tmp_path / "moved")})
        assert not compare_runs(golden, moved).ok
        assert compare_runs(golden, moved, ignore_params=("input",)).ok


class TestCli:
    def test_csv_mode_exit_codes(self, tmp_path: Path) -> None:
        from tools.compare import main

        good = tmp_path / "good.csv"
        bad = tmp_path / "bad.csv"
        _small().to_csv(good, index=False)
        tampered = _small()
        tampered.loc[0, "Score"] = 99.0
        tampered.to_csv(bad, index=False)
        assert main(["--csv", str(good), str(good)]) == 0
        assert main(["--csv", str(good), str(bad)]) == 1
        assert main(["--csv", str(good), str(tmp_path / "ghost.csv")]) == 2

    def test_run_mode_matches_and_mismatches(self, tmp_path: Path) -> None:
        from tools.compare import main

        golden = _make_run(tmp_path / "golden")
        copy = _make_run(tmp_path / "copy")
        assert main([str(golden), str(copy)]) == 0
        payload = json.loads((copy / "result.json").read_text(encoding="utf-8"))
        payload["params"] = {"tampered": True}
        (copy / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        assert main([str(golden), str(copy)]) == 1


class TestC617Hardening:
    """C6-17: path validation, duplicates, order, NaN."""

    def test_artifact_escape_rejected(self, tmp_path: Path) -> None:
        golden = _make_run(tmp_path / "golden")
        tampered = tmp_path / "tampered"
        tampered.mkdir()
        payload = json.loads((golden / "result.json").read_text(encoding="utf-8"))
        payload["artifacts"] = [{"kind": "table", "path": "../escape.csv"}]
        (tampered / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        result = compare_runs(golden, tampered)
        assert result.value is None
        assert any(d.code == "COMPARE_ARTIFACT_ESCAPE" for d in result.diagnostics)

    def test_duplicate_artifact_paths_rejected(self, tmp_path: Path) -> None:
        golden = _make_run(tmp_path / "golden")
        tampered = tmp_path / "dup"
        tampered.mkdir()
        payload = json.loads((golden / "result.json").read_text(encoding="utf-8"))
        payload["artifacts"] = payload["artifacts"] * 2
        (tampered / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        result = compare_runs(golden, tampered)
        assert result.value is None
        assert any(d.code == "COMPARE_DUPLICATE_ARTIFACT" for d in result.diagnostics)

    def test_artifact_order_not_parity_critical(self, tmp_path: Path) -> None:
        golden = _make_run(tmp_path / "golden")
        reordered = tmp_path / "reordered"
        shutil.copytree(golden, reordered)
        payload = json.loads((reordered / "result.json").read_text(encoding="utf-8"))
        if len(payload["artifacts"]) > 1:
            payload["artifacts"].reverse()
            (reordered / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        assert compare_runs(golden, reordered).ok  # same set, different order

    def test_nan_cells_compare_as_missing(self, tmp_path: Path) -> None:
        csv = tmp_path / "t.csv"
        pd.DataFrame({"A": [1.0, float("nan")]}).to_csv(csv, index=False)
        assert compare_csv_files(csv, csv).ok
        left = tmp_path / "l.csv"
        right = tmp_path / "r.csv"
        pd.DataFrame({"A": [1.0, float("nan")]}).to_csv(left, index=False)
        pd.DataFrame({"A": [1.0, float("inf")]}).to_csv(right, index=False)
        # NaN vs inf: one missing, one present-with-inf -> a real difference
        assert not compare_csv_files(left, right).ok

    def test_malicious_envelope_rejected(self, tmp_path: Path) -> None:
        golden = _make_run(tmp_path / "golden")
        evil = tmp_path / "evil"
        evil.mkdir()
        payload = json.loads((golden / "result.json").read_text(encoding="utf-8"))
        payload["artifacts"] = [{"kind": "table", "path": "C:/Windows/evil.csv"}]
        (evil / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        result = compare_runs(golden, evil)
        assert result.value is None
        assert any(d.code == "COMPARE_ARTIFACT_ESCAPE" for d in result.diagnostics)
