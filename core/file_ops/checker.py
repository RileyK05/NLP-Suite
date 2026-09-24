"""File checker — validates that a file can be read as text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.io.reader import read_text
from core.result import Diagnostic, Result

__all__ = ["CheckReport", "check_file"]


@dataclass(frozen=True, slots=True)
class CheckReport:
    path: Path
    ok: bool
    encoding: str | None
    is_empty: bool
    diagnostics: tuple[Diagnostic, ...]


def check_file(path: Path, encoding: str | None = None) -> Result[CheckReport]:
    """Check that *path* is a readable text file."""
    source = Path(path)
    if not source.exists():
        return Result.failure(Diagnostic.error("FILE_NOT_FOUND", f"{source} does not exist", path=str(source)))
    if source.is_dir():
        return Result.failure(Diagnostic.error("FILE_IS_DIR", f"{source} is a directory", path=str(source)))
    result = read_text(source, encoding=encoding)
    if result.value is None:
        return Result.failure(*result.diagnostics)
    text = result.unwrap()
    is_empty = not text.strip()
    diagnostics = tuple(result.diagnostics)
    if is_empty:
        diagnostics = diagnostics + (Diagnostic.warning("EMPTY_FILE", f"{source.name} is empty", path=str(source)),)
    encoding_used = encoding or "utf-8-sig"
    if result.diagnostics and any(d.code == "ENCODING_FALLBACK" for d in result.diagnostics):
        encoding_used = str(result.diagnostics[0].context.get("encoding", encoding_used))
    return Result.success(
        CheckReport(path=source, ok=not is_empty, encoding=encoding_used, is_empty=is_empty, diagnostics=diagnostics),
        *diagnostics,
    )
