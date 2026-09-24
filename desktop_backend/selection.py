"""Validated, reproducible document selections for interactive desktop runs.

Selections are kept separate from the runner on purpose.  The runner can
persist the validated request verbatim, while this module remains a small,
side-effect-free policy boundary for the API and its tests.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from core.io.reader import date_from_filename

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_MAX_DOCUMENTS = 2000


class CorpusSelection(BaseModel):
    """A user-owned subset of a project's imported documents.

    ``document_ids=None`` means all documents.  An empty list is explicit and
    is rejected by :func:`resolve_selection` so a typo cannot silently run a
    zero-document analysis.
    """

    model_config = ConfigDict(extra="forbid")

    document_ids: list[StrictStr] | None = Field(default=None, max_length=_MAX_DOCUMENTS)
    date_from: StrictStr | None = None
    date_to: StrictStr | None = None
    include_undated: bool = False

    @field_validator("document_ids")
    @classmethod
    def _valid_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if any(not item.strip() for item in value):
            raise ValueError("document_ids must contain non-empty IDs")
        if len(set(value)) != len(value):
            raise ValueError("document_ids must not contain duplicates")
        return value

    @field_validator("date_from", "date_to")
    @classmethod
    def _strict_iso_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if _ISO_DATE.fullmatch(value) is None:
            raise ValueError("dates must use YYYY-MM-DD")
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("dates must be real calendar dates") from exc
        return value

    def bounds(self) -> tuple[date | None, date | None]:
        """Return validated date bounds and reject an inverted interval."""

        lower = date.fromisoformat(self.date_from) if self.date_from else None
        upper = date.fromisoformat(self.date_to) if self.date_to else None
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("date_from must be on or before date_to")
        return lower, upper


def _document_date(doc: dict[str, Any]) -> date | None:
    return date_from_filename(Path(str(doc.get("name", ""))))


def document_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    """Return safe document metadata with a stable, filename-derived date.

    The returned mapping is a copy.  ``document_date`` is ISO text for JSON,
    and ``date_source`` makes the best-effort provenance visible to the UI.
    ``created`` is retained as the import timestamp; it is not confused with
    the document's date.
    """

    result = dict(doc)
    found = _document_date(doc)
    result["document_date"] = found.isoformat() if found else None
    result["date_source"] = "filename" if found else None
    return result


def resolve_selection(documents: list[dict[str, Any]], selection: CorpusSelection | None) -> list[dict[str, Any]]:
    """Resolve a validated selection, preserving project document order.

    Returned rows are copies augmented with date provenance.  Unknown IDs and
    a selection that excludes every document are errors at this boundary.
    """

    chosen = selection or CorpusSelection()
    lower, upper = chosen.bounds()
    by_id = {str(doc.get("id")): doc for doc in documents}
    if chosen.document_ids is not None:
        unknown = [identifier for identifier in chosen.document_ids if identifier not in by_id]
        if unknown:
            raise ValueError(f"Unknown document ID(s): {', '.join(unknown)}")
        allowed = set(chosen.document_ids)
    else:
        allowed = None

    resolved: list[dict[str, Any]] = []
    for doc in documents:
        if allowed is not None and str(doc.get("id")) not in allowed:
            continue
        found = _document_date(doc)
        if found is None:
            if (lower is not None or upper is not None) and not chosen.include_undated:
                continue
        else:
            if lower is not None and found < lower:
                continue
            if upper is not None and found > upper:
                continue
        resolved.append(document_metadata(doc))

    if not resolved:
        raise ValueError("The selected documents are empty. Choose at least one document.")
    return resolved


__all__ = ["CorpusSelection", "document_metadata", "resolve_selection"]
