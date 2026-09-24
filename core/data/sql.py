"""SQL — safe parameterized query layer over sqlite."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3

import pandas as pd

from core.result import Diagnostic, Result

__all__ = ["execute", "query"]

_ALLOWED_OPS = frozenset(["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP"])


def _check_sql(sql: str) -> Diagnostic | None:
    stripped = sql.lstrip().upper()
    first = stripped.split(None, 1)[0] if stripped else ""
    # Strip leading '(' for subqueries
    first = first.lstrip("(")
    if first not in _ALLOWED_OPS:
        return Diagnostic.error("SQL_BAD_OP", f"SQL must start with one of {sorted(_ALLOWED_OPS)}, got {first!r}")
    # Basic guard: no multiple statements. A trailing ';' on a single
    # statement is fine; anything that leaves more than one non-empty
    # statement is not (previously "SELECT 1; DELETE FROM t" slipped through
    # because a single ';' was permitted anywhere).
    statements = [s for s in sql.strip().split(";") if s.strip()]
    if len(statements) > 1:
        return Diagnostic.error("SQL_MULTI_STATEMENT", "multiple statements not allowed")
    return None


def query(
    db_path: Path,
    sql: str,
    params: tuple[object, ...] | None = None,
) -> Result[pd.DataFrame]:
    """Execute a SELECT and return a DataFrame."""
    db_path = Path(db_path)
    if not db_path.is_file():
        return Result.failure(Diagnostic.error("SQL_NO_DB", f"database {db_path} not found"))
    diag = _check_sql(sql)
    if diag is not None:
        return Result.failure(diag)
    if not sql.lstrip().upper().startswith("SELECT"):
        return Result.failure(
            Diagnostic.error("SQL_NOT_SELECT", "query() is for SELECT only; use execute() for writes")
        )

    try:
        with closing(sqlite3.connect(str(db_path))) as conn:
            df = pd.read_sql_query(sql, conn, params=params or ())
    except Exception as exc:
        return Result.failure(Diagnostic.error("SQL_FAILED", f"query failed: {exc}", sql=sql))
    return Result.success(df)


def execute(
    db_path: Path,
    sql: str,
    params: tuple[object, ...] | None = None,
) -> Result[int]:
    """Execute a write statement and return affected row count."""
    db_path = Path(db_path)
    # Allow creating a new DB for CREATE
    diag = _check_sql(sql)
    if diag is not None:
        return Result.failure(diag)
    try:
        with closing(sqlite3.connect(str(db_path))) as conn:
            cur = conn.execute(sql, params or ())
            conn.commit()
            return Result.success(cur.rowcount)
    except Exception as exc:
        return Result.failure(Diagnostic.error("SQL_FAILED", f"execute failed: {exc}", sql=sql))
