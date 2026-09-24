"""Golden comparison harness (FR-1.5) — the Gate B instrument.

Goldens are compared by declared rules, never by eyeballing diffs. The rules:

* **Ordering.** With ``key_columns`` both frames sort by key before comparing,
  so row order never matters. Without keys, rows are aligned as an
  order-insensitive multiset using the same numeric tolerance as cell
  comparison, unless ``order_matters`` is set. Duplicate keys are a harness
  error — ambiguous alignment must never silently pass.
* **Numeric tolerance.** Cells that both parse as numbers compare with
  ``math.isclose(rtol, atol)`` (defaults cover the envelope's 6dp rounding).
  Everything else compares as exact strings. ``1`` equals ``1.0``; ``"01"``
  equals ``1``. This is documented, not accidental.
* **Schema.** Column-name sets must match exactly. Column *order* differences
  are a WARNING only — values compare by name. Dtypes are ignored; a CSV
  round-trip that turns ``int`` into ``float`` is not a regression.
* **Missing values.** In-memory pandas NA (``None``, ``NaN``, ``NaT``,
  ``pd.NA``) all equal each other. Empty string and sentinels like ``"NA"``
  are real values — never silently missing — unless ``empty_as_missing``.
  CSV files are read with ``keep_default_na=False`` so the reader cannot
  turn ``"NA"`` into NaN behind our backs.
* **Artifacts.** ``compare_runs`` pairs two run directories via their
  envelopes: envelope provenance (minus ``created`` timestamps), artifact
  presence, CSV contents, and byte equality for anything else.

Contract: every comparison returns a ``Result[CompareReport]`` whose value
is *always* present when the harness itself worked — even on mismatch, so
the diff stays inspectable. ``result.ok`` mirrors ``report.passed``. A
``None`` value means the harness failed (unreadable file, bad config),
never "the goldens differ".
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from core.artifacts.envelope import Envelope
from core.result import Diagnostic, Result

__all__ = [
    "CompareConfig",
    "CompareReport",
    "compare_csv_files",
    "compare_envelopes",
    "compare_frames",
    "compare_runs",
]


_DIFF_COLUMNS = ["Row", "Column", "Expected", "Actual", "Issue"]
_MAX_DIFF_ROWS = 500
_PREVIEW_LEN = 120


@dataclass(frozen=True, slots=True)
class CompareConfig:
    """How to align and compare two tables."""

    key_columns: tuple[str, ...] = ()
    rtol: float = 1e-6
    atol: float = 1e-9
    order_matters: bool = False
    empty_as_missing: bool = False
    ignore_columns: tuple[str, ...] = ()
    artifact_configs: Mapping[str, CompareConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.rtol, bool)
            or isinstance(self.atol, bool)
            or not math.isfinite(float(self.rtol))
            or not math.isfinite(float(self.atol))
            or self.rtol < 0
            or self.atol < 0
        ):
            raise ValueError("rtol and atol must be non-negative")


_DEFAULT_CONFIG = CompareConfig()


@dataclass(frozen=True, slots=True)
class CompareReport:
    """The outcome of one comparison plus the row-level diff.

    ``diff`` has columns ``Row, Column, Expected, Actual, Issue``. For
    envelope comparison ``compared_rows`` counts field checks and
    ``compared_columns`` is 1 (one envelope pair = one record).
    """

    passed: bool
    compared_rows: int
    compared_columns: int
    diff: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        return self.diff.copy()


def _is_missing(value: object, *, empty_as_missing: bool) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass  # array-like cell: not missing, compared as a value
    return empty_as_missing and value == ""


def _values_equal(expected: object, actual: object, *, rtol: float, atol: float, empty_as_missing: bool) -> bool:  # noqa: PLR0911
    exp_missing = _is_missing(expected, empty_as_missing=empty_as_missing)
    act_missing = _is_missing(actual, empty_as_missing=empty_as_missing)
    if exp_missing and act_missing:
        return True
    if exp_missing or act_missing:
        return False
    try:
        exp_num = float(expected)  # type: ignore[arg-type]
        act_num = float(actual)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(expected) == str(actual)
    if math.isnan(exp_num) and math.isnan(act_num):
        return True  # literal "nan" strings on both sides
    if math.isnan(exp_num) or math.isnan(act_num):
        return False
    # Infinity is a real non-finite value, not a missing value. It is never a
    # valid golden value (including inf == inf), and must not pass tolerance.
    if not math.isfinite(exp_num) or not math.isfinite(act_num):
        return False
    return math.isclose(exp_num, act_num, rel_tol=rtol, abs_tol=atol)


def _fmt(value: object) -> str:
    if _is_missing(value, empty_as_missing=False):
        return "<missing>"
    text = repr(value) if isinstance(value, str) else str(value)
    return text if len(text) <= _PREVIEW_LEN else text[: _PREVIEW_LEN - 3] + "..."


def _empty_report() -> CompareReport:
    return CompareReport(
        passed=False,
        compared_rows=0,
        compared_columns=0,
        diff=pd.DataFrame(columns=_DIFF_COLUMNS),
    )


def _cap_diff(rows: list[dict[str, object]]) -> tuple[pd.DataFrame, bool]:
    truncated = len(rows) > _MAX_DIFF_ROWS
    kept = rows[:_MAX_DIFF_ROWS]
    return pd.DataFrame(kept, columns=_DIFF_COLUMNS), truncated


def _row_label(values: tuple[object, ...], keys: tuple[str, ...]) -> str:
    if keys:
        return ", ".join(f"{k}={_fmt(v)}" for k, v in zip(keys, values, strict=True))
    return ""


def _check_keys(exp: pd.DataFrame, act: pd.DataFrame, keys: tuple[str, ...]) -> Diagnostic | None:
    """Validate key columns exist and are unique on both sides."""
    if not keys:
        return None
    for key in keys:
        if key not in exp.columns:
            return Diagnostic.error(
                "COMPARE_BAD_KEY",
                f"key column {key!r} not in frames",
                key=key,
                columns=list(exp.columns),
            )
    exp_dup = bool(exp.duplicated(subset=list(keys)).any())
    act_dup = bool(act.duplicated(subset=list(keys)).any())
    if exp_dup or act_dup:
        dup_side = "expected" if exp_dup else "actual"
        return Diagnostic.error(
            "COMPARE_DUPLICATE_KEYS",
            f"duplicate key values in {dup_side}; alignment would be ambiguous",
            side=dup_side,
            key_columns=list(keys),
        )
    return None


def _order_index(frame: pd.DataFrame, config: CompareConfig) -> list[int]:
    """Deterministic row order: by key, stored order, or full-row string form."""
    if config.key_columns:
        locs = [frame.columns.get_loc(k) for k in config.key_columns]
        return sorted(range(len(frame)), key=lambda i: tuple(str(frame.iat[i, loc]) for loc in locs))
    if config.order_matters:
        return list(range(len(frame)))
    return sorted(range(len(frame)), key=lambda i: tuple(str(v) for v in frame.iloc[i].tolist()))


def _unordered_alignment(
    expected: pd.DataFrame, actual: pd.DataFrame, columns: list[str], config: CompareConfig
) -> tuple[list[int], list[int]]:
    """Align unordered rows using the declared numeric tolerance.

    String sorting makes tolerance-equivalent rows appear different (for
    example 10.0 and 9.9999999). Match equal rows first, deterministically,
    then pair remaining rows for an inspectable value/extra-row diff.
    """
    edges_exp: list[int] = []
    edges_act: list[int] = []
    exp_locs = {col: expected.columns.get_loc(col) for col in columns}
    act_locs = {col: actual.columns.get_loc(col) for col in columns}
    for exp_idx in range(len(expected)):
        matches = [
            act_idx
            for act_idx in range(len(actual))
            if all(
                _values_equal(
                    expected.iat[exp_idx, exp_locs[col]],
                    actual.iat[act_idx, act_locs[col]],
                    rtol=config.rtol,
                    atol=config.atol,
                    empty_as_missing=config.empty_as_missing,
                )
                for col in columns
            )
        ]
        edges_exp.extend([exp_idx] * len(matches))
        edges_act.extend(matches)
    # Tolerance equality is not transitive. Greedily consuming the first
    # match can strand a later row even when a complete matching exists.
    graph = csr_matrix(([1] * len(edges_exp), (edges_exp, edges_act)), shape=(len(expected), len(actual)))
    matched = maximum_bipartite_matching(graph, perm_type="column")
    exp_order = [idx for idx, match in enumerate(matched) if match >= 0]
    act_order = [int(matched[idx]) for idx in exp_order]
    used_actual = set(act_order)
    unmatched_exp = [idx for idx, match in enumerate(matched) if match < 0]
    unmatched_act = [idx for idx in range(len(actual)) if idx not in used_actual]
    exp_order.extend(unmatched_exp)
    act_order.extend(unmatched_act)
    return exp_order, act_order


def _frame_labels(frame: pd.DataFrame, keys: tuple[str, ...]) -> list[str]:
    if not keys:
        return [f"row {pos}" for pos in range(len(frame))]
    locs = [frame.columns.get_loc(k) for k in keys]
    return [_row_label(tuple(frame.iat[pos, loc] for loc in locs), keys) for pos in range(len(frame))]


def _cell_diffs(
    exp_s: pd.DataFrame,
    act_s: pd.DataFrame,
    columns: list[str],
    config: CompareConfig,
) -> list[dict[str, object]]:
    """Cell-by-cell diff over two aligned frames, plus extra/missing rows."""
    rows: list[dict[str, object]] = []
    common = min(len(exp_s), len(act_s))
    exp_labels = _frame_labels(exp_s, config.key_columns)
    act_labels = _frame_labels(act_s, config.key_columns)
    for pos in range(common):
        for col in columns:
            loc = exp_s.columns.get_loc(col)
            exp_val = exp_s.iat[pos, loc]
            act_val = act_s.iat[pos, loc]
            if _values_equal(
                exp_val,
                act_val,
                rtol=config.rtol,
                atol=config.atol,
                empty_as_missing=config.empty_as_missing,
            ):
                continue
            issue = "value differs"
            if _is_missing(exp_val, empty_as_missing=config.empty_as_missing) or _is_missing(
                act_val, empty_as_missing=config.empty_as_missing
            ):
                issue = "missing value"
            rows.append(
                {
                    "Row": exp_labels[pos],
                    "Column": col,
                    "Expected": _fmt(exp_val),
                    "Actual": _fmt(act_val),
                    "Issue": issue,
                }
            )
    for pos in range(common, len(exp_s)):
        rows.append(
            {
                "Row": exp_labels[pos],
                "Column": "<row>",
                "Expected": "present",
                "Actual": "<missing>",
                "Issue": "missing row in actual",
            }
        )
    for pos in range(common, len(act_s)):
        rows.append(
            {
                "Row": act_labels[pos],
                "Column": "<row>",
                "Expected": "<missing>",
                "Actual": "present",
                "Issue": "extra row in actual",
            }
        )
    return rows


def compare_frames(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    config: CompareConfig | None = None,
) -> Result[CompareReport]:
    """Compare two tables by the declared ordering/tolerance/schema rules."""
    cfg = _DEFAULT_CONFIG if config is None else config
    if cfg.order_matters and cfg.key_columns:
        return Result.success(
            _empty_report(),
            Diagnostic.error(
                "COMPARE_BAD_CONFIG",
                "order_matters and key_columns are mutually exclusive",
                order_matters=True,
                key_columns=list(cfg.key_columns),
            ),
        )

    exp = expected.drop(columns=[c for c in cfg.ignore_columns if c in expected.columns])
    act = actual.drop(columns=[c for c in cfg.ignore_columns if c in actual.columns])
    exp_cols = list(exp.columns)

    if set(exp_cols) != set(act.columns):
        missing = sorted(set(exp_cols) - set(act.columns))
        extra = sorted(set(act.columns) - set(exp_cols))
        return Result.success(
            _empty_report(),
            Diagnostic.error(
                "COMPARE_SCHEMA_MISMATCH",
                f"column sets differ (missing {missing}, extra {extra})",
                missing=missing,
                extra=extra,
            ),
        )

    diags: list[Diagnostic] = []
    if exp_cols != list(act.columns):
        diags.append(
            Diagnostic.warning(
                "COMPARE_COLUMN_ORDER",
                "column order differs; values compared by name",
                expected_order=list(exp_cols),
                actual_order=list(act.columns),
            )
        )
    act = act[exp_cols]

    key_problem = _check_keys(exp, act, cfg.key_columns)
    if key_problem is not None:
        return Result.success(_empty_report(), key_problem)

    if not cfg.key_columns and not cfg.order_matters:
        exp_order, act_order = _unordered_alignment(exp, act, exp_cols, cfg)
    else:
        exp_order, act_order = _order_index(exp, cfg), _order_index(act, cfg)
    exp_s = exp.iloc[exp_order].reset_index(drop=True)
    act_s = act.iloc[act_order].reset_index(drop=True)

    if len(exp_s) != len(act_s):
        diags.append(
            Diagnostic.error(
                "COMPARE_ROW_COUNT",
                f"row counts differ (expected {len(exp_s)}, actual {len(act_s)})",
                expected_n=len(exp_s),
                actual_n=len(act_s),
            )
        )
    diff_rows = _cell_diffs(exp_s, act_s, exp_cols, cfg)

    diff, truncated = _cap_diff(diff_rows)
    if truncated:
        diags.append(
            Diagnostic.info(
                "COMPARE_DIFF_TRUNCATED",
                f"diff capped at {_MAX_DIFF_ROWS} of {len(diff_rows)} rows",
                shown=_MAX_DIFF_ROWS,
                total=len(diff_rows),
            )
        )
    common = min(len(exp_s), len(act_s))
    report = CompareReport(
        passed=not diff_rows and len(exp_s) == len(act_s),
        compared_rows=common,
        compared_columns=len(exp_cols),
        diff=diff,
    )
    if report.passed:
        diags.append(
            Diagnostic.info(
                "COMPARE_MATCH",
                f"tables match ({common} row(s), {len(exp_cols)} column(s))",
                rows=common,
                columns=len(exp_cols),
            )
        )
    else:
        bad_cols = sorted({str(r["Column"]) for r in diff_rows[:_MAX_DIFF_ROWS] if r["Column"] != "<row>"})[:5]
        diags.append(
            Diagnostic.error(
                "COMPARE_VALUE_MISMATCH",
                f"{len(diff_rows)} differing cell(s) in columns {bad_cols}",
                diff_rows=len(diff_rows),
                columns=bad_cols,
            )
        )
    return Result.success(report, *diags)


def compare_csv_files(
    expected_path: Path,
    actual_path: Path,
    config: CompareConfig | None = None,
) -> Result[CompareReport]:
    """Compare two CSV files. Missing/unreadable files are harness errors."""
    for label, path in (("expected", expected_path), ("actual", actual_path)):
        if not Path(path).is_file():
            return Result.failure(
                Diagnostic.error("COMPARE_FILE_NOT_FOUND", f"{label} CSV not found: {path}", path=str(path), side=label)
            )
    try:
        exp_df = pd.read_csv(expected_path, encoding="utf-8", keep_default_na=False)
        act_df = pd.read_csv(actual_path, encoding="utf-8", keep_default_na=False)
    except Exception as exc:
        return Result.failure(
            Diagnostic.error(
                "COMPARE_CSV_UNREADABLE",
                f"could not read CSV pair: {exc}",
                expected=str(expected_path),
                actual=str(actual_path),
            )
        )
    return compare_frames(exp_df, act_df, config)


def _params_equal(expected: Any, actual: Any, *, rtol: float, atol: float) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual or expected == actual
    if isinstance(expected, float) and isinstance(actual, (float, int)):
        return (
            math.isfinite(expected)
            and math.isfinite(float(actual))
            and math.isclose(expected, float(actual), rel_tol=rtol, abs_tol=atol)
        )
    if isinstance(expected, int) and isinstance(actual, float):
        return math.isfinite(actual) and math.isclose(float(expected), actual, rel_tol=rtol, abs_tol=atol)
    if isinstance(expected, dict) and isinstance(actual, dict):
        return set(expected) == set(actual) and all(
            _params_equal(expected[k], actual[k], rtol=rtol, atol=atol) for k in expected
        )
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return len(expected) == len(actual) and all(
            _params_equal(e, a, rtol=rtol, atol=atol) for e, a in zip(expected, actual, strict=True)
        )
    return bool(expected == actual)


def _check_equal(
    field: str, exp_val: object, act_val: object, issue: str, note: Callable[[str, object, object, str], None]
) -> int:
    if exp_val != act_val:
        note(field, exp_val, act_val, issue)
    return 1


def _diff_dict_lists(  # noqa: PLR0913
    label: str,
    singular: str,
    expected: list[dict[str, str]],
    actual: list[dict[str, str]],
    note: Callable[[str, object, object, str], None],
    *,
    unordered: bool = False,
) -> int:
    """Diff two dict lists (inputs are ordered; artifacts are a multiset)."""
    if len(expected) != len(actual):
        note(label, len(expected), len(actual), f"{label} count differs")
        return 1
    if unordered:
        # Artifact declaration order is an implementation detail. Preserve
        # duplicate detection elsewhere, but compare the complete records as
        # a multiset so reordering cannot fail a run comparison.
        exp_counts = Counter(_canonical_dict(item) for item in expected)
        act_counts = Counter(_canonical_dict(item) for item in actual)
        if exp_counts != act_counts:
            note(label, expected, actual, f"{label} differ")
        return max(len(expected), 1)
    for i, (exp_item, act_item) in enumerate(zip(expected, actual, strict=True)):
        if exp_item != act_item:
            note(f"{label}[{i}]", exp_item, act_item, f"{singular} differs")
    return max(len(expected), 1)


def _canonical_dict(item: Mapping[str, object]) -> str:
    return json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def compare_envelopes(
    expected: Envelope,
    actual: Envelope,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> Result[CompareReport]:
    """Compare two envelopes. ``created`` timestamps are always ignored."""
    diff_rows: list[dict[str, object]] = []

    def note(field: str, exp_val: object, act_val: object, issue: str) -> None:
        diff_rows.append(
            {"Row": "envelope", "Column": field, "Expected": _fmt(exp_val), "Actual": _fmt(act_val), "Issue": issue}
        )

    checks = 0
    checks += _check_equal("tool", expected.tool, actual.tool, "tool differs", note)
    checks += _check_equal(
        "schema_version", expected.schema_version, actual.schema_version, "envelope version differs", note
    )
    checks += 1
    if not _params_equal(expected.params, actual.params, rtol=rtol, atol=atol):
        note("params", expected.params, actual.params, "params differ")
    checks += _diff_dict_lists(
        "inputs", "input", [i.to_dict() for i in expected.inputs], [i.to_dict() for i in actual.inputs], note
    )
    checks += _diff_dict_lists(
        "artifacts",
        "artifact",
        [a.to_dict() for a in expected.artifacts],
        [a.to_dict() for a in actual.artifacts],
        note,
        unordered=True,
    )
    checks += 1
    exp_diag = Counter((d.severity.value, d.code) for d in expected.diagnostics)
    act_diag = Counter((d.severity.value, d.code) for d in actual.diagnostics)
    if exp_diag != act_diag:
        note("diagnostics", dict(exp_diag), dict(act_diag), "diagnostic (severity, code) counts differ")
    checks += 1
    if expected.corpus_sha256 != actual.corpus_sha256:
        note("corpus_sha256", expected.corpus_sha256, actual.corpus_sha256, "corpus fingerprint differs")

    diff, truncated = _cap_diff(diff_rows)
    report = CompareReport(passed=not diff_rows, compared_rows=checks, compared_columns=1, diff=diff)
    diags: list[Diagnostic] = []
    if truncated:
        diags.append(Diagnostic.info("COMPARE_DIFF_TRUNCATED", "envelope diff truncated", total=len(diff_rows)))
    if report.passed:
        diags.append(Diagnostic.info("COMPARE_MATCH", f"envelopes match ({checks} field check(s))", checks=checks))
    else:
        diags.append(
            Diagnostic.error(
                "COMPARE_ENVELOPE_MISMATCH",
                f"{len(diff_rows)} envelope field(s) differ",
                diff_rows=len(diff_rows),
            )
        )
    return Result.success(report, *diags)


def _without_params(env: Envelope, ignore: tuple[str, ...]) -> Envelope:
    if not ignore:
        return env
    return Envelope(
        tool=env.tool,
        created=env.created,
        params={k: v for k, v in env.params.items() if k not in ignore},
        inputs=env.inputs,
        artifacts=env.artifacts,
        diagnostics=env.diagnostics,
        schema_version=env.schema_version,
        corpus_sha256=env.corpus_sha256,
    )


def _compare_artifact_pair(
    path: str, exp_file: Path, act_file: Path, config: CompareConfig
) -> Result[tuple[list[dict[str, object]], int]]:
    """Compare one paired artifact. Value is (diff_rows, checks performed)."""
    if not exp_file.is_file() or not act_file.is_file():
        missing_side = "expected" if not exp_file.is_file() else "actual"
        row: dict[str, object] = {
            "Row": path,
            "Column": "artifact",
            "Expected": "on disk" if exp_file.is_file() else "<missing>",
            "Actual": "on disk" if act_file.is_file() else "<missing>",
            "Issue": f"missing artifact file on disk ({missing_side})",
        }
        return Result.success(([row], 1))
    if exp_file.suffix.lower() != ".csv":
        if exp_file.read_bytes() == act_file.read_bytes():
            return Result.success(([], 1))
        row = {
            "Row": path,
            "Column": "artifact",
            "Expected": "bytes match",
            "Actual": "bytes differ",
            "Issue": "non-CSV artifact bytes differ",
        }
        return Result.success(
            ([row], 1),
            Diagnostic.error("COMPARE_ARTIFACT_MISMATCH", f"non-CSV artifact {path} differs", artifact=path),
        )
    sub = compare_csv_files(exp_file, act_file, config)
    if sub.value is None:
        return Result.failure(*sub.diagnostics)
    sub_report = sub.unwrap()
    if not sub_report.passed and sub_report.diff.empty:
        return Result.failure(*sub.diagnostics)
    rows: list[dict[str, object]] = []
    for record_raw in sub_report.diff.to_dict(orient="records"):
        record = {str(k): v for k, v in dict(record_raw).items()}
        record["Column"] = f"{path}::{record['Column']}"
        rows.append(record)
    diags = [
        Diagnostic(
            severity=diag.severity,
            code=diag.code,
            message=diag.message,
            context={**dict(diag.context), "artifact": path},
        )
        for diag in sub.diagnostics
        if diag.code != "COMPARE_MATCH"
    ]
    return Result.success((rows, sub_report.compared_rows), *diags)


def _validate_artifact_paths(env: Envelope, run_dir: Path) -> Diagnostic | None:
    """C6-17: artifact paths must be relative, inside the run dir, unique."""
    seen: set[str] = set()
    for artifact in env.artifacts:
        raw_path = str(artifact.path)
        win_path = PureWindowsPath(raw_path)
        posix_path = PurePosixPath(raw_path)
        if (
            win_path.is_absolute()
            or posix_path.is_absolute()
            or win_path.drive
            or ".." in win_path.parts
            or ".." in posix_path.parts
        ):
            return Diagnostic.error(
                "COMPARE_ARTIFACT_ESCAPE",
                f"artifact path {artifact.path!r} is not a safe relative path",
                artifact=artifact.path,
            )
        if artifact.path in seen:
            return Diagnostic.error(
                "COMPARE_DUPLICATE_ARTIFACT",
                f"envelope declares artifact {artifact.path!r} twice; refusing to collapse",
                artifact=artifact.path,
            )
        seen.add(artifact.path)
        candidate = (run_dir / artifact.path).resolve()
        try:
            candidate.relative_to(run_dir.resolve())
        except ValueError:
            return Diagnostic.error(
                "COMPARE_ARTIFACT_ESCAPE",
                f"artifact path {artifact.path!r} escapes run directory",
                artifact=artifact.path,
            )
    return None


def compare_runs(
    expected_dir: Path,
    actual_dir: Path,
    config: CompareConfig | None = None,
    *,
    ignore_params: tuple[str, ...] = (),
) -> Result[CompareReport]:
    """Compare two run directories artifact-by-artifact via their envelopes.

    C6-17: artifact paths are validated (relative, inside the run dir, no
    duplicates); artifact collections compare as SETS of (kind, path)
    pairs — envelope insertion order is not parity-critical. NaN cells
    compare as missing; infinity is a non-finite mismatch, never a valid
    approximate value.
    """
    cfg = _DEFAULT_CONFIG if config is None else config
    exp_env_result = Envelope.read(Path(expected_dir))
    if exp_env_result.value is None:
        return Result.failure(*exp_env_result.diagnostics)
    act_env_result = Envelope.read(Path(actual_dir))
    if act_env_result.value is None:
        return Result.failure(*act_env_result.diagnostics)
    exp_env = _without_params(exp_env_result.unwrap(), ignore_params)
    act_env = _without_params(act_env_result.unwrap(), ignore_params)
    for env, directory in ((exp_env, Path(expected_dir)), (act_env, Path(actual_dir))):
        problem = _validate_artifact_paths(env, directory)
        if problem is not None:
            return Result.failure(problem)

    diags: list[Diagnostic] = []
    all_rows: list[dict[str, object]] = []
    total_rows = 0

    env_result = compare_envelopes(exp_env, act_env, rtol=cfg.rtol, atol=cfg.atol)
    env_report = env_result.unwrap()
    total_rows += env_report.compared_rows
    for row in env_report.diff.to_dict(orient="records"):
        all_rows.append({str(k): v for k, v in dict(row).items()})
    for diag in env_result.diagnostics:
        if diag.code != "COMPARE_MATCH":
            diags.append(diag)

    # Set identity: (path, kind) pairs; envelope order is not parity-critical.
    exp_artifacts = {a.path: a for a in exp_env.artifacts}
    act_artifacts = {a.path: a for a in act_env.artifacts}
    for path in sorted(set(exp_artifacts) - set(act_artifacts)):
        all_rows.append(
            {
                "Row": path,
                "Column": "artifact",
                "Expected": exp_artifacts[path].kind,
                "Actual": "<missing>",
                "Issue": "missing artifact in actual",
            }
        )
    for path in sorted(set(act_artifacts) - set(exp_artifacts)):
        all_rows.append(
            {
                "Row": path,
                "Column": "artifact",
                "Expected": "<missing>",
                "Actual": act_artifacts[path].kind,
                "Issue": "extra artifact in actual",
            }
        )
    for path in sorted(set(exp_artifacts) & set(act_artifacts)):
        pair_cfg = cfg.artifact_configs.get(path, cfg)
        pair = _compare_artifact_pair(path, Path(expected_dir) / path, Path(actual_dir) / path, pair_cfg)
        if pair.value is None:
            return Result.failure(*pair.diagnostics)
        pair_rows, pair_checks = pair.unwrap()
        all_rows.extend(pair_rows)
        total_rows += pair_checks
        diags.extend(pair.diagnostics)

    diff, truncated = _cap_diff(all_rows)
    if truncated:
        diags.append(
            Diagnostic.info(
                "COMPARE_DIFF_TRUNCATED",
                f"run diff capped at {_MAX_DIFF_ROWS} of {len(all_rows)} rows",
                shown=_MAX_DIFF_ROWS,
                total=len(all_rows),
            )
        )
    report = CompareReport(passed=not all_rows, compared_rows=total_rows, compared_columns=5, diff=diff)
    if report.passed:
        diags.append(
            Diagnostic.info(
                "COMPARE_MATCH",
                f"run directories match ({total_rows} check(s))",
                expected=str(expected_dir),
                actual=str(actual_dir),
            )
        )
    else:
        diags.append(
            Diagnostic.error(
                "COMPARE_RUN_MISMATCH",
                f"runs differ in {len(all_rows)} place(s)",
                diff_rows=len(all_rows),
                expected=str(expected_dir),
                actual=str(actual_dir),
            )
        )
    return Result.success(report, *diags)
