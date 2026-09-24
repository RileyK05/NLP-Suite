"""Key Words In Context (KWIC) concordance over a parsed corpus.

One row per match: the hit word plus the left and right context windows,
with document and sentence provenance. Works over the canonical CoNLL
table (``field`` selects Form or Lemma), so every viewer/search convention
of the suite applies — regex with a clear ``KWIC_BAD_REGEX`` diagnostic,
case-sensitive opt-in, and a max-hits bound so a common stopword on a
large corpus stops after ``max_hits`` matches with a warning instead of a
gigabyte frame.

The legacy suite had a KWIC entry in its menu (TIPS_NLP_KWIC PDF) but no
surviving module; this is a new-from-convention capability (CAP-INTAKE-19),
matching the modern ``search`` semantics where they exist.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["concordance"]

_COLUMNS = ["Document ID", "Document", "Sentence ID", "Hit", "Left Context", "Right Context"]
_MAX_WINDOW = 50
_MAX_HITS_CAP = 100_000


def _tokens(frame: pd.DataFrame, field_col: str) -> Iterator[tuple[object, object, list[str]]]:
    """Yield (document_id, sentence_id, token strings) per sentence, in row order."""
    for (did, sid), group in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        yield did, sid, [str(x) for x in group[field_col].tolist()]


def concordance(
    frame: pd.DataFrame,
    query: str,
    *,
    field: Col = Col.FORM,
    window: int = 5,
    max_hits: int = 1000,
    case_sensitive: bool = False,
    regex: bool = False,
) -> Result[pd.DataFrame]:
    """KWIC rows for every token matching *query*.

    The hit word is the center; ``Left Context``/``Right Context`` hold up to
    *window* neighbor tokens joined with spaces. Regex matches test each token
    with ``re.search``; literal matches are substring tests (case-insensitive
    unless ``case_sensitive``), matching the ``search`` tool's semantics.
    """
    if not query:
        return Result.failure(Diagnostic.error("KWIC_EMPTY_QUERY", "query must be non-empty"))
    if window < 1 or window > _MAX_WINDOW:
        return Result.failure(Diagnostic.error("KWIC_BAD_WINDOW", f"window must be 1..{_MAX_WINDOW}, got {window}"))
    if max_hits < 1 or max_hits > _MAX_HITS_CAP:
        return Result.failure(
            Diagnostic.error("KWIC_BAD_MAX_HITS", f"max_hits must be 1..{_MAX_HITS_CAP}, got {max_hits}")
        )
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("KWIC_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_COLUMNS))

    pattern: re.Pattern[str] | None = None
    if regex:
        try:
            pattern = re.compile(query, flags=0 if case_sensitive else re.IGNORECASE)
        except re.error as exc:
            return Result.failure(Diagnostic.error("KWIC_BAD_REGEX", f"invalid regex: {exc}"))

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    names: dict[object, str] = {}
    if doc_col is not None:
        names = frame.groupby(Col.DOCUMENT_ID.value)[doc_col].first().astype(str).to_dict()

    rows: list[dict[str, object]] = []
    truncated = False
    for did, sid, toks in _tokens(frame, field.value):
        if truncated:
            break
        for i, token in enumerate(toks):
            if not token.strip():
                continue
            if pattern is not None:
                if pattern.search(token) is None:
                    continue
            else:
                haystack = token if case_sensitive else token.lower()
                needle = query if case_sensitive else query.lower()
                if needle not in haystack:
                    continue
            left = " ".join(toks[max(0, i - window) : i])
            right = " ".join(toks[i + 1 : i + 1 + window])
            rows.append(
                {
                    "Document ID": str(did),
                    "Document": names.get(did, ""),
                    "Sentence ID": sid,
                    "Hit": token,
                    "Left Context": left,
                    "Right Context": right,
                }
            )
            if len(rows) >= max_hits:
                truncated = True
                break
    out = pd.DataFrame(rows, columns=_COLUMNS)
    diags: tuple[Diagnostic, ...] = ()
    if truncated:
        diags = (
            Diagnostic.warning(
                "KWIC_TRUNCATED",
                f"stopped at max_hits={max_hits}; raise --max-hits to see more matches",
            ),
        )
    return Result.success(out, *diags)
