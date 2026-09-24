"""PC-ACE analysis — stats and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.pcace.core import parse_code, read_db
from core.result import Diagnostic, Result

__all__ = ["analyze", "validate_codes"]


def validate_codes(codes: list[str]) -> Result[pd.DataFrame]:
    """Per-code validation."""
    if not codes:
        return Result.failure(Diagnostic.error("PCACE_VALIDATE_EMPTY", "no codes to validate"))
    rows: list[dict[str, object]] = []
    diags: list[Diagnostic] = []
    for code in codes:
        res = parse_code(code)
        if res.ok:
            info = res.unwrap()
            rows.append(
                {"Code": info["code"], "Category": info["category"], "Depth": int(info["depth"]), "Status": "OK"}
            )
        else:
            rows.append({"Code": code, "Category": "", "Depth": 0, "Status": "INVALID"})
            diags.append(Diagnostic.warning("PCACE_INVALID_CODE", f"code {code!r} invalid", value=code))
    df = pd.DataFrame(rows, columns=["Code", "Category", "Depth", "Status"])
    if diags:
        return Result.success(df, *diags)
    return Result.success(df)


def analyze(db_path: Path) -> Result[pd.DataFrame]:
    """Per-category frequencies from the pcace DB."""
    res = read_db(db_path)
    if not res.ok:
        return Result[pd.DataFrame](None, res.diagnostics)
    df = res.unwrap()
    if df.empty:
        return Result.success(pd.DataFrame(columns=["Category", "Count", "Codes"]))

    # Group by category
    grouped = df.groupby("category", sort=False).agg(
        Count=("code", "size"), Codes=("code", lambda s: ", ".join(sorted(s)))
    )
    out = grouped.reset_index().rename(columns={"category": "Category"})
    out = out[["Category", "Count", "Codes"]].sort_values("Count", ascending=False).reset_index(drop=True)
    return Result.success(out)
