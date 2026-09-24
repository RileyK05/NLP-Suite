"""Corpus validation — PC-ACE style checker."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.io.reader import read_text
from core.result import Diagnostic, Result

__all__ = ["validate"]


def validate(
    corpus_dir: Path,
    *,
    min_tokens: int = 1,
    required_ext: str = ".txt",
) -> Result[pd.DataFrame]:
    """Per-file validation: existence, extension, token count, emptiness.

    Files are decoded through the intake encoding chain (``read_text``), not
    with ``errors="ignore"`` — a validation report built on silently dropped
    bytes would report the wrong token counts and never say why.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        return Result.failure(
            Diagnostic.error("VALIDATE_NO_DIR", f"{corpus_dir} is not a directory", path=str(corpus_dir))
        )

    files = sorted(corpus_dir.rglob(f"*{required_ext}")) if required_ext else sorted(corpus_dir.iterdir())
    # Filter to files
    files = [p for p in files if p.is_file()]

    rows: list[dict[str, object]] = []
    read_diags: list[Diagnostic] = []
    for p in files:
        rel = p.relative_to(corpus_dir).as_posix()
        text_result = read_text(p)
        if text_result.value is None:
            for diag in text_result.diagnostics:
                rows.append(
                    {
                        "File": rel,
                        "Status": "ERROR",
                        "Tokens": 0,
                        "Issue": str(diag),
                    }
                )
            continue
        # A fallback-decoded file is still counted, but the fallback must be
        # visible in the report — otherwise the validator would bless silently
        # reinterpreted bytes (fail-big: fallbacks are knowable or they go).
        fallback_note = ""
        for diag in text_result.diagnostics:
            read_diags.append(diag)
            if diag.code == "ENCODING_FALLBACK":
                fallback_note = f"decoded as {diag.context.get('encoding', '?')} (not utf-8)"
        text = text_result.unwrap()
        toks = [t for t in text.split() if t.strip()]
        n = len(toks)
        if n == 0:
            rows.append({"File": rel, "Status": "EMPTY", "Tokens": 0, "Issue": "empty file"})
        elif n < min_tokens:
            rows.append(
                {
                    "File": rel,
                    "Status": "SHORT",
                    "Tokens": n,
                    "Issue": f"{n} < min_tokens {min_tokens}",
                }
            )
        else:
            rows.append({"File": rel, "Status": "OK", "Tokens": n, "Issue": fallback_note})

    # If no files, return diagnostic frame
    if not rows:
        df = pd.DataFrame(columns=["File", "Status", "Tokens", "Issue"])
        return Result.success(df, Diagnostic.info("VALIDATE_EMPTY", f"no *{required_ext} files under {corpus_dir}"))

    df = pd.DataFrame(rows, columns=["File", "Status", "Tokens", "Issue"])
    diags: list[Diagnostic] = list(read_diags)
    bad = df[df["Status"] != "OK"]
    if not bad.empty:
        diags.append(Diagnostic.warning("VALIDATE_ISSUES", f"{len(bad)} file(s) have issues", count=len(bad)))
    return Result.success(df, *diags)
