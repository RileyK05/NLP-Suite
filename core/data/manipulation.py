"""Data manipulation — merge/concat/reshape."""

from __future__ import annotations

import pandas as pd

from core.result import Diagnostic, Result

__all__ = ["concat", "melt", "merge", "pivot"]


def concat(frames: list[pd.DataFrame], axis: int = 0) -> Result[pd.DataFrame]:
    """Concatenate frames vertically (0) or horizontally (1)."""
    if not frames:
        return Result.failure(Diagnostic.error("DATA_EMPTY_INPUT", "no frames to concat"))
    if axis not in (0, 1):
        return Result.failure(Diagnostic.error("DATA_BAD_AXIS", f"axis must be 0 or 1, got {axis}"))
    if any(not isinstance(f, pd.DataFrame) for f in frames):
        return Result.failure(Diagnostic.error("DATA_BAD_TYPE", "all inputs must be DataFrames"))
    try:
        out = pd.concat(frames, axis=axis, ignore_index=(axis == 0))
    except Exception as exc:
        return Result.failure(Diagnostic.error("DATA_CONCAT_FAILED", f"concat failed: {exc}"))
    return Result.success(out)


def merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str],
    how: str = "inner",
) -> Result[pd.DataFrame]:
    """Merge two frames on key(s)."""
    if how not in ("inner", "left", "right", "outer", "cross"):
        return Result.failure(
            Diagnostic.error("DATA_BAD_HOW", f"how must be inner/left/right/outer/cross, got {how!r}")
        )
    keys = [on] if isinstance(on, str) else list(on)
    if how != "cross":
        for k in keys:
            if k not in left.columns or k not in right.columns:
                return Result.failure(Diagnostic.error("DATA_MISSING_KEY", f"key {k!r} must be in both frames"))
    try:
        if how == "cross":
            out = left.merge(right, how="cross")
        else:
            out = left.merge(right, on=keys, how=how)
    except Exception as exc:
        return Result.failure(Diagnostic.error("DATA_MERGE_FAILED", f"merge failed: {exc}"))
    return Result.success(out)


def pivot(
    frame: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    aggfunc: str = "mean",
) -> Result[pd.DataFrame]:
    """Pivot table."""
    for col in [index, columns, values]:
        if col not in frame.columns:
            return Result.failure(Diagnostic.error("DATA_MISSING_COLUMN", f"column {col!r} not in frame"))
    if aggfunc not in ("mean", "sum", "count", "min", "max", "median"):
        return Result.failure(Diagnostic.error("DATA_BAD_AGGFUNC", f"aggfunc {aggfunc!r} not supported"))
    try:
        out = frame.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
        out = out.reset_index()
        out.columns.name = None
    except Exception as exc:
        return Result.failure(Diagnostic.error("DATA_PIVOT_FAILED", f"pivot failed: {exc}"))
    return Result.success(out)


def melt(
    frame: pd.DataFrame,
    id_vars: list[str],
    value_vars: list[str] | None = None,
) -> Result[pd.DataFrame]:
    """Wide to long."""
    for col in id_vars:
        if col not in frame.columns:
            return Result.failure(Diagnostic.error("DATA_MISSING_COLUMN", f"id_var {col!r} not in frame"))
    if value_vars is not None:
        for col in value_vars:
            if col not in frame.columns:
                return Result.failure(Diagnostic.error("DATA_MISSING_COLUMN", f"value_var {col!r} not in frame"))
    try:
        out = frame.melt(id_vars=id_vars, value_vars=value_vars)
    except Exception as exc:
        return Result.failure(Diagnostic.error("DATA_MELT_FAILED", f"melt failed: {exc}"))
    return Result.success(out)
