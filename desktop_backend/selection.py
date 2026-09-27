"""Validated, reproducible document selections for interactive desktop runs.

Selections are kept separate from the runner on purpose.  The runner can
persist the validated request verbatim, while this module remains a small,
side-effect-free policy boundary for the API and its tests.
"""

from __future__ import annotations

from datetime import date
import math
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from core.io.document_fields import document_date, document_order, field_value

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
    #: The same window along an order axis: chapters or sessions by their
    #: Order value, with ``include_unordered`` for documents that have none.
    order_from: float | None = None
    order_to: float | None = None
    include_unordered: bool = False
    #: Keep documents whose detail is one of the listed values, ANDed across
    #: details: ``{"Kind": ["sotu"], "Speaker": ["Harry S Truman"]}``.
    #: ``"(empty)"`` selects documents without that detail.
    fields: dict[StrictStr, list[StrictStr]] | None = Field(default=None, max_length=40)

    @field_validator("fields")
    @classmethod
    def _valid_fields(cls, value: dict[str, list[str]] | None) -> dict[str, list[str]] | None:
        if value is None:
            return None
        for name, allowed in value.items():
            if not name.strip():
                raise ValueError("A detail filter needs the detail's name")
            if not allowed:
                raise ValueError(f"The filter on “{name}” lists no values; remove it or choose at least one")
        return value

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

    @field_validator("order_from", "order_to")
    @classmethod
    def _finite_order(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("order bounds must be real numbers")
        return value

    def bounds(self) -> tuple[date | None, date | None]:
        """Return validated date bounds and reject an inverted interval."""

        lower = date.fromisoformat(self.date_from) if self.date_from else None
        upper = date.fromisoformat(self.date_to) if self.date_to else None
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("date_from must be on or before date_to")
        return lower, upper

    def order_bounds(self) -> tuple[float | None, float | None]:
        """Return validated order bounds and reject an inverted interval."""
        lower, upper = self.order_from, self.order_to
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("order_from must be on or before order_to")
        return lower, upper


def _document_date(doc: dict[str, Any]) -> date | None:
    """The effective Date detail (typed, imported or from the name), else the name's date."""
    return document_date(doc.get("fields"), str(doc.get("name", "")))


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
    # Where the date came from: "filename" (read from the name), or the source
    # of a Date detail someone set ("user", "csv", "split").
    sources = {name.casefold(): source for name, source in (doc.get("field_sources") or {}).items()}
    result["date_source"] = (sources.get("date") or "filename") if found else None
    result["document_order"] = document_order(doc.get("fields"))
    return result


_EMPTY = "(empty)"


def _matches_fields(doc: dict[str, Any], wanted: dict[str, list[str]]) -> bool:
    fields = doc.get("fields") or {}
    for name, allowed in wanted.items():
        value = field_value(fields, name)
        choices = {choice.casefold() for choice in allowed}
        if (value.casefold() if value else _EMPTY.casefold()) not in choices:
            return False
    return True


def _check_field_names(documents: list[dict[str, Any]], wanted: dict[str, list[str]]) -> None:
    known = {name.casefold(): name for doc in documents for name in (doc.get("fields") or {})}
    unknown = [name for name in wanted if name.casefold() not in known]
    if unknown:
        listed = ", ".join(sorted(known.values(), key=str.casefold)) or "none yet"
        raise ValueError(f"No document has the detail {', '.join(map(repr, unknown))}. Details here: {listed}.")


def _within_dates(found: date | None, lower: date | None, upper: date | None, include_undated: bool) -> bool:
    """Inside the bounds; an undated document passes only when no bound is set or undated ones are asked for."""
    if found is None:
        return (lower is None and upper is None) or include_undated
    return not ((lower is not None and found < lower) or (upper is not None and found > upper))


def _within_orders(found: float | None, lower: float | None, upper: float | None, include_unordered: bool) -> bool:
    """The same window over Order values, for chapters and sessions."""
    if found is None:
        return (lower is None and upper is None) or include_unordered
    return not ((lower is not None and found < lower) or (upper is not None and found > upper))


def resolve_selection(documents: list[dict[str, Any]], selection: CorpusSelection | None) -> list[dict[str, Any]]:
    """Resolve a validated selection, preserving project document order.

    Returned rows are copies augmented with date provenance.  Unknown IDs and
    a selection that excludes every document are errors at this boundary.
    """

    chosen = selection or CorpusSelection()
    lower, upper = chosen.bounds()
    order_lower, order_upper = chosen.order_bounds()
    by_id = {str(doc.get("id")): doc for doc in documents}
    if chosen.document_ids is not None:
        unknown = [identifier for identifier in chosen.document_ids if identifier not in by_id]
        if unknown:
            raise ValueError(f"Unknown document ID(s): {', '.join(unknown)}")
        allowed = set(chosen.document_ids)
    else:
        allowed = None
    if chosen.fields:
        _check_field_names(documents, chosen.fields)

    resolved: list[dict[str, Any]] = []
    for doc in documents:
        if allowed is not None and str(doc.get("id")) not in allowed:
            continue
        if chosen.fields and not _matches_fields(doc, chosen.fields):
            continue
        if not _within_dates(_document_date(doc), lower, upper, chosen.include_undated):
            continue
        if not _within_orders(document_order(doc.get("fields")), order_lower, order_upper, chosen.include_unordered):
            continue
        resolved.append(document_metadata(doc))

    if not resolved:
        raise ValueError("The selected documents are empty. Choose at least one document.")
    return resolved


__all__ = ["CorpusSelection", "document_metadata", "resolve_selection"]
