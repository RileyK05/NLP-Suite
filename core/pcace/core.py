"""PC-ACE core — database and grammar stub."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
import re
import sqlite3

import pandas as pd

from core.result import Diagnostic, Result

__all__ = ["create_db", "grammar_info", "parse_code", "read_db"]

# Simplified PC-ACE grammar: codes like "1.1.2" or "A.12"
_CODE_RE = re.compile(r"^[A-Z0-9]+(\.[A-Z0-9]+)*$", re.IGNORECASE)

_GRAMMAR: dict[str, str] = {
    "1": "Actor",
    "1.1": "Government",
    "1.2": "Opposition",
    "2": "Action",
    "2.1": "Cooperation",
    "2.2": "Conflict",
    "3": "Target",
}


def parse_code(code: str) -> Result[dict[str, str]]:
    """Parse a PC-ACE code into its parts."""
    code = code.strip()
    if not code:
        return Result.failure(Diagnostic.error("PCACE_EMPTY_CODE", "code must be non-empty"))
    if not _CODE_RE.match(code):
        return Result.failure(Diagnostic.error("PCACE_BAD_CODE", f"code {code!r} does not match grammar", value=code))
    parts = code.split(".")
    category = _GRAMMAR.get(code, _GRAMMAR.get(parts[0], "Unknown"))
    return Result.success({"code": code, "parts": ".".join(parts), "category": category, "depth": str(len(parts))})


def grammar_info() -> pd.DataFrame:
    """Return the baked grammar as a DataFrame."""
    rows = [{"Code": k, "Label": v} for k, v in sorted(_GRAMMAR.items())]
    return pd.DataFrame(rows, columns=["Code", "Label"])


def create_db(
    db_path: Path,
    codes: list[str],
) -> Result[Path]:
    """Create a sqlite DB with a pcace table and insert codes."""
    db_path = Path(db_path)
    if not codes:
        return Result.failure(Diagnostic.error("PCACE_NO_CODES", "no codes to insert"))
    # Validate all codes first
    for c in codes:
        res = parse_code(c)
        if not res.ok:
            return Result[Path](None, res.diagnostics)

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(db_path))) as conn:
            conn.execute("DROP TABLE IF EXISTS pcace")
            conn.execute("CREATE TABLE pcace (id INTEGER PRIMARY KEY, code TEXT UNIQUE, category TEXT, depth INTEGER)")
            for c in codes:
                info = parse_code(c).unwrap()
                conn.execute(
                    "INSERT INTO pcace (code, category, depth) VALUES (?, ?, ?)",
                    (info["code"], info["category"], int(info["depth"])),
                )
            conn.commit()
    except Exception as exc:
        return Result.failure(Diagnostic.error("PCACE_DB_FAILED", f"failed to create DB: {exc}", path=str(db_path)))
    return Result.success(db_path)


def read_db(db_path: Path) -> Result[pd.DataFrame]:
    """Read the pcace table."""
    db_path = Path(db_path)
    if not db_path.is_file():
        return Result.failure(Diagnostic.error("PCACE_NO_DB", f"database {db_path} not found"))
    try:
        with closing(sqlite3.connect(str(db_path))) as conn:
            df = pd.read_sql_query("SELECT code, category, depth FROM pcace ORDER BY code", conn)
    except Exception as exc:
        return Result.failure(Diagnostic.error("PCACE_READ_FAILED", f"read failed: {exc}"))
    return Result.success(df)
