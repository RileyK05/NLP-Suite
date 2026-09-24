"""HTML annotator — dictionary, extractor, gender."""

from __future__ import annotations

import html
import re

import pandas as pd

from core.result import Diagnostic, Result

__all__ = ["annotate", "extract_text", "gender_spans"]

_TAG_RE = re.compile(r"<[^>]+>")
_PRONOUN_GENDER: dict[str, str] = {
    "he": "M",
    "him": "M",
    "his": "M",
    "himself": "M",
    "she": "F",
    "her": "F",
    "hers": "F",
    "herself": "F",
    "they": "N",
    "them": "N",
    "their": "N",
    "themself": "N",
}


def extract_text(html_text: str) -> Result[str]:
    """Strip tags and unescape entities. No network, no file IO."""
    if not isinstance(html_text, str):
        return Result.failure(Diagnostic.error("HTML_BAD_TYPE", "html_text must be str"))  # type: ignore[unreachable]
    text = _TAG_RE.sub(" ", html_text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return Result.success(text)


def annotate(
    text: str,
    dictionary: dict[str, str],
    *,
    case_sensitive: bool = False,
) -> Result[str]:
    """Wrap dictionary terms in <mark data-tag=\"...\"> tags.

    All terms are matched in a single pass (longest first) so that later
    substitutions can never match inside markup injected by earlier ones —
    sequential ``re.sub`` over the annotated text corrupted the output when
    one dictionary term was a substring of another term's tag or of the
    inserted markup itself.
    """
    if not isinstance(text, str):
        return Result.failure(Diagnostic.error("HTML_BAD_TEXT", "text must be str"))  # type: ignore[unreachable]
    if not dictionary:
        return Result.failure(Diagnostic.error("HTML_EMPTY_DICT", "dictionary must be non-empty"))
    terms = [t for t in sorted(dictionary.keys(), key=len, reverse=True) if t.strip()]
    if not terms:
        return Result.failure(Diagnostic.error("HTML_EMPTY_DICT", "dictionary must contain a non-blank term"))

    flags = 0 if case_sensitive else re.IGNORECASE
    alternatives: list[str] = []
    tags: list[str] = []
    for term in terms:
        # Word boundaries for plain words; escaped literal otherwise.
        if re.match(r"^\w+$", term):
            alternatives.append(rf"\b{re.escape(term)}\b")
        else:
            alternatives.append(re.escape(term))
        tags.append(html.escape(dictionary[term]))
    combined = re.compile("|".join(f"({alt})" for alt in alternatives), flags)
    tag_by_index = {i: tags[i] for i in range(len(tags))}

    def _wrap(match: re.Match[str]) -> str:
        # Which alternative fired: exactly one capture group participates.
        group = match.lastindex or 1
        return f'<mark data-tag="{tag_by_index[group - 1]}">{match.group(0)}</mark>'

    try:
        out = combined.sub(_wrap, text)
    except re.error as exc:
        return Result.failure(Diagnostic.error("HTML_REGEX_FAILED", f"dictionary terms: {exc}"))
    return Result.success(out)


def gender_spans(frame: pd.DataFrame, field: str = "Form") -> Result[pd.DataFrame]:
    """Find gendered pronouns in a token DataFrame."""
    # Accept a generic DataFrame with a Form-like column; used by tests and CLI.
    if field not in frame.columns:
        return Result.failure(Diagnostic.error("HTML_MISSING_COLUMN", f"missing {field!r}"))
    if frame.empty:
        return Result.success(pd.DataFrame(columns=["Token", "Gender", "Sentence ID", "Document ID"]))
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        tok = str(row[field])
        g = _PRONOUN_GENDER.get(tok.lower())
        if g is None:
            continue
        rows.append(
            {
                "Token": tok,
                "Gender": g,
                "Sentence ID": row.get("Sentence ID", ""),
                "Document ID": str(row.get("Document ID", "")),
            }
        )
    if not rows:
        return Result.success(pd.DataFrame(columns=["Token", "Gender", "Sentence ID", "Document ID"]))
    df = pd.DataFrame(rows, columns=["Token", "Gender", "Sentence ID", "Document ID"])
    return Result.success(df)
