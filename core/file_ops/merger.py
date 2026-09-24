"""Merger and matcher — combining and finding files."""

from __future__ import annotations

from pathlib import Path

from core.result import Diagnostic, Result

__all__ = ["match_files", "merge_texts"]


def match_files(directory: Path, pattern: str = "*.txt", recursive: bool = True) -> list[Path]:
    """Return files matching *pattern* under *directory*, sorted."""
    root = Path(directory)
    if not root.is_dir():
        return []
    globber = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(p for p in globber if p.is_file())


def merge_texts(texts: list[str], separator: str = "\n") -> Result[str]:
    """Merge *texts* into one string."""
    if not texts:
        return Result.failure(Diagnostic.error("MERGE_EMPTY", "no texts to merge"))
    return Result.success(separator.join(texts))


def merge_files(paths: list[Path], output_path: Path, separator: str = "\n") -> Result[Path]:
    """Read *paths* and write the merged content to *output_path*."""
    if not paths:
        return Result.failure(Diagnostic.error("MERGE_NO_INPUTS", "no input paths provided"))
    texts: list[str] = []
    for path in paths:
        try:
            texts.append(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            return Result.failure(
                Diagnostic.error("MERGE_READ_FAILED", f"could not read {path}: {exc}", path=str(path))
            )
    merged = merge_texts(texts, separator=separator)
    if merged.value is None:
        return Result[Path](None, merged.diagnostics)
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(merged.unwrap(), encoding="utf-8")
    except OSError as exc:
        return Result.failure(
            Diagnostic.error("MERGE_WRITE_FAILED", f"could not write {output_path}: {exc}", path=str(output_path))
        )
    return Result.success(Path(output_path))
