"""Document details: Date, Speaker, Kind, Chapter ... (docs/internal/PLAN_0.5.0.md section 1).

A document's details come from four places, and when two disagree the more
deliberate one wins: ``user`` (typed on the Corpus page) over ``csv`` (a
spreadsheet) over ``split`` (the book splitter) over ``filename`` (read from
the name by :mod:`core.io.filename_fields`).

File-name details are **computed, not stored**. Detection is a few regexes
over at most 2,000 names, so doing it on every read costs nothing and means
an import, a trash or a restore can never leave them stale. Only the other
three sources have rows (``document_fields``). The project's choices about
details -- the axis, what one step along an order axis is called, the names
given to the file-name parts -- live in ``project_settings``.
"""

from __future__ import annotations

from datetime import date
import json
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from core.corpus_axis import axis_of
from core.io.document_fields import document_date
from core.io.filename_fields import Template, apply_template, detect_template
from desktop_backend.store import now

if TYPE_CHECKING:  # pragma: no cover - typing only
    from desktop_backend.store import Workspace

__all__ = [
    "MAX_EVENTS",
    "MAX_FIELDS_PER_PROJECT",
    "MAX_FIELD_VALUE",
    "RESERVED_FIELD_NAMES",
    "SOURCES",
    "FieldRow",
    "FieldSettings",
    "ProjectEvent",
    "TextCleaning",
    "attach_fields",
    "cleaning",
    "delete_field",
    "details",
    "event_position",
    "field_names",
    "filename_template",
    "import_fields",
    "name_problem",
    "project_axis",
    "project_events",
    "resolved_axis",
    "save_settings",
    "set_fields",
    "settings",
]

#: Lowest precedence first.
SOURCES = ("filename", "split", "csv", "user")
MAX_FIELDS_PER_PROJECT = 40
MAX_FIELD_VALUE = 500
#: The most dated events one project may list; enough for a century of milestones.
MAX_EVENTS = 50
#: A stage-direction term is a short word or two, not a sentence.
_MAX_TERM_LENGTH = 40
_FIELD_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,39}$")
#: Column names the suite's own tables use. A detail named like one would sit
#: beside a tool's column of the same name and be mistaken for it.
RESERVED_FIELD_NAMES = frozenset(
    name.casefold()
    for name in (
        "Document",
        "Document ID",
        "Year",
        "Position",
        "Position label",
        "Side",
        "Project",
        "Words",
        "Tokens",
        "Sentence ID",
        "Token ID",
        "Form",
        "Lemma",
        "POS",
        "Head",
        "DepRel",
    )
)


class TextCleaning(BaseModel):
    """What is left out of the text the tools read (plan 5.2).

    Stage directions -- "(Applause.)", "[inaudible]" -- are not the author's
    words, and every tool counts them. The imported file is never changed
    (R3); only what analyses see. On by default for projects made after
    0.5.0 (Riley, D6); older projects opt in from the Corpus page.
    """

    model_config = ConfigDict(extra="forbid")

    #: Remove bracketed stage directions at read time.
    stage_directions: bool = False
    #: The project's own words that make a bracketed span a stage direction.
    extra_terms: list[StrictStr] = Field(default_factory=list, max_length=50)

    @field_validator("extra_terms")
    @classmethod
    def _terms(cls, value: list[str]) -> list[str]:
        cleaned = [term.strip() for term in value if term.strip()]
        for term in cleaned:
            if len(term) > _MAX_TERM_LENGTH:
                raise ValueError(f"A stage-direction term is at most 40 characters: “{term[:40]}…” is too long.")
        return cleaned


class ProjectEvent(BaseModel):
    """One dated event to draw as a stopline: a name and a year (or a date).

    The plan's section 4: "the single largest lever on figure quality for
    historical corpora -- it turns every time figure into a check against the
    world." The date is a decimal year when possible (1941.5 for mid-1941) so
    it can sit between two addresses, or an ISO date parsed to its exact
    position.
    """

    model_config = ConfigDict(extra="forbid")

    name: StrictStr = Field(min_length=1, max_length=60)
    #: A year (1941), a decimal year (1941.5), or an ISO date (1941-12-07).
    date: StrictStr = Field(min_length=4, max_length=10)
    note: StrictStr = Field(default="", max_length=200)

    @property
    def position(self) -> float | None:
        """The decimal-year position of the event, or None when unparseable."""
        return event_position(self.date)


def event_position(value: str) -> float | None:
    """A year, a decimal year, or an ISO date as a decimal year for an x axis.

    "1941" -> 1941.0, "1941.5" -> 1941.5 (mid-year), "1941-12-07" -> 1941.93.
    Anything else is None: an event with no position is dropped rather than
    silently placed at the axis origin.
    """
    text = value.strip()
    iso = _ISO_DATE.match(text)
    if iso:
        try:
            day = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return None
        start = date(day.year, 1, 1)
        span = (date(day.year + 1, 1, 1) - start).days
        return day.year + (day - start).days / span
    try:
        number = float(text)
    except ValueError:
        return None
    return number if _MIN_EVENT_YEAR <= number <= _MAX_EVENT_YEAR else None


_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
#: The range a decimal year must fall in to be a plausible event date.
_MIN_EVENT_YEAR = 1000.0
_MAX_EVENT_YEAR = 2500.0


class FieldSettings(BaseModel):
    """A project's choices about its documents' details and its text."""

    model_config = ConfigDict(extra="forbid")

    #: How documents line up: by Date, by Order, not at all, or chosen from what exists.
    axis: Literal["auto", "time", "order", "none"] = "auto"
    #: What one step along an order axis is called ("Chapter", "Session").
    #: "Document" until named: figures then read "across the documents".
    order_label: StrictStr = Field(default="Document", min_length=1, max_length=40)
    #: Read details from file names at all.
    filename_detection: bool = True
    #: A reader's names for the detected parts: {"Kind": "Genre"}.
    part_names: dict[str, str] = Field(default_factory=dict)
    #: What is left out of the text the tools read.
    text_cleaning: TextCleaning = Field(default_factory=TextCleaning)
    #: Dated events to draw as stoplines on every time figure (SHOWCASE_FIGURES_PLAN
    #: section 4): a historical corpus's figures become a check against the world.
    events: list[ProjectEvent] = Field(default_factory=list, max_length=MAX_EVENTS)

    @field_validator("part_names")
    @classmethod
    def _names(cls, value: dict[str, str]) -> dict[str, str]:
        for new in value.values():
            _check_name(new)
        return value

    @field_validator("events")
    @classmethod
    def _events(cls, value: list[ProjectEvent]) -> list[ProjectEvent]:
        seen: set[str] = set()
        for event in value:
            key = event.name.casefold()
            if key in seen:
                raise ValueError(f"“{event.name}” is listed twice; event names must be unique.")
            seen.add(key)
        return value


class FieldRow(BaseModel):
    """One detail to set. An empty value from ``user`` removes that override."""

    model_config = ConfigDict(extra="forbid")

    document_id: StrictStr
    name: StrictStr
    value: StrictStr = Field(max_length=MAX_FIELD_VALUE)
    source: Literal["user", "csv", "split"] = "user"


def _check_name(name: str) -> str:
    cleaned = name.strip()
    if not _FIELD_NAME.match(cleaned):
        raise ValueError(
            f"“{name}” cannot name a detail: use 1 to 40 letters, digits, spaces, - or _, starting with a letter or digit."
        )
    if cleaned.casefold() in RESERVED_FIELD_NAMES:
        raise ValueError(f"“{cleaned}” is a column name the suite's tables already use. Choose another name.")
    return cleaned


# --------------------------------------------------------------- settings --


def settings(workspace: Workspace, project_id: str) -> dict[str, Any]:
    """``{"settings": FieldSettings as a dict, "revision": n}``; defaults until first saved."""
    with workspace.connect() as db:
        row = db.execute("SELECT settings, revision FROM project_settings WHERE project_id=?", (project_id,)).fetchone()
    stored = json.loads(row["settings"]) if row else {}
    known = {key: value for key, value in stored.items() if key in FieldSettings.model_fields}
    return {"settings": FieldSettings(**known).model_dump(), "revision": row["revision"] if row else 0}


def _bump(db: Any, project_id: str, new_settings: dict[str, Any] | None = None) -> int:
    """Advance the project's details revision (and optionally store new settings)."""
    row = db.execute("SELECT settings, revision FROM project_settings WHERE project_id=?", (project_id,)).fetchone()
    if row is None:
        db.execute(
            "INSERT INTO project_settings (project_id, settings, revision, updated) VALUES (?, ?, 1, ?)",
            (project_id, json.dumps(new_settings or {}), now()),
        )
        return 1
    revision = int(row["revision"]) + 1
    db.execute(
        "UPDATE project_settings SET settings=?, revision=?, updated=? WHERE project_id=?",
        (json.dumps(new_settings) if new_settings is not None else row["settings"], revision, now(), project_id),
    )
    return revision


def save_settings(
    workspace: Workspace, project_id: str, body: FieldSettings, expected_revision: int | None = None
) -> dict[str, Any]:
    """Store a project's detail settings; refuse when someone else saved in between."""
    _writable(workspace, project_id)
    with workspace.connect() as db:
        row = db.execute("SELECT revision FROM project_settings WHERE project_id=?", (project_id,)).fetchone()
        current = int(row["revision"]) if row else 0
        if expected_revision is not None and expected_revision != current:
            raise ValueError(f"These settings moved on to revision {current}. Reload them before saving.")
        _bump(db, project_id, body.model_dump())
    return settings(workspace, project_id)


def cleaning(workspace: Workspace, project_id: str) -> TextCleaning:
    """What the project leaves out of the text the tools read (stage directions)."""
    return FieldSettings(**settings(workspace, project_id)["settings"]).text_cleaning


def project_axis(workspace: Workspace, project_id: str) -> dict[str, str]:
    """The project's axis choice as a run records it: ``{"kind": "auto"|"time"|"order"|"none", "noun": ...}``.

    Frozen into every run's request at submission, like its documents, so a
    result can be reproduced after the Corpus page's choice changes.
    """
    chosen = FieldSettings(**settings(workspace, project_id)["settings"])
    return {"kind": chosen.axis, "noun": chosen.order_label}


def project_events(workspace: Workspace, project_id: str) -> list[dict[str, Any]]:
    """The project's dated events with resolved decimal-year positions.

    ``[{"name", "date", "note", "position"}]``, positioned ones only, so a
    figure can put a stopline where the event belongs. An event whose date
    cannot be read is dropped here rather than drawn at the origin; the Corpus
    page is where it is reported (the settings store keeps it for editing).
    """
    chosen = FieldSettings(**settings(workspace, project_id)["settings"])
    found: list[dict[str, Any]] = []
    for event in chosen.events:
        position = event.position
        if position is not None:
            found.append({"name": event.name, "date": event.date, "note": event.note, "position": position})
    return found


def resolved_axis(
    workspace: Workspace, project_id: str, documents: list[dict[str, Any]], found: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """The axis the project's runs will use, for the Corpus page's shape line.

    ``{kind, noun, placed, first, last}``: "auto" resolved to what the
    documents carry, how many documents it places, and the ends of the range
    (ISO dates for time, order values for order). *found* is :func:`details`.
    """
    chosen = project_axis(workspace, project_id)
    placed = [
        SimpleNamespace(
            doc_id=str(doc["id"]),
            date=document_date({n: item["value"] for n, item in found.get(str(doc["id"]), {}).items()}, doc["name"]),
            details={n: item["value"] for n, item in found.get(str(doc["id"]), {}).items()},
        )
        for doc in documents
    ]
    axis = axis_of(placed, chosen["kind"], chosen["noun"])
    ends: list[str] = []
    if axis.positions:
        ordered = sorted(axis.positions, key=lambda doc: axis.positions[doc])
        if axis.kind == "time":
            ends = [axis.labels[ordered[0]], axis.labels[ordered[-1]]]
        else:
            ends = [f"{axis.positions[doc]:g}" for doc in (ordered[0], ordered[-1])]
    return {
        "kind": axis.kind,
        "noun": axis.noun,
        "placed": len(axis.positions),
        "first": ends[0] if ends else None,
        "last": ends[-1] if ends else None,
    }


def _writable(workspace: Workspace, project_id: str) -> None:
    project = workspace.project(project_id)
    if project["trashed"]:
        raise ValueError("Restore this project from Trash before changing its documents' details.")
    if project["archived"]:
        raise ValueError("Restore this project from the archive before changing its documents' details.")


# ------------------------------------------------------------------ values --


def _detection(workspace: Workspace, project_id: str) -> tuple[Template | None, dict[str, str]]:
    """The detector's template over the active names, and the project's names for its parts.

    Parsing always uses the detector's own part names -- a Date renamed "When"
    must still be read as a date -- and the renaming is applied to the output.
    """
    chosen = FieldSettings(**settings(workspace, project_id)["settings"])
    if not chosen.filename_detection:
        return None, {}
    with workspace.connect() as db:
        names = [
            row["name"]
            for row in db.execute(
                "SELECT name FROM documents WHERE project_id=? AND trashed=0 ORDER BY name", (project_id,)
            )
        ]
    return detect_template(names), dict(chosen.part_names)


def filename_template(workspace: Workspace, project_id: str) -> Template | None:
    """The pattern the project's document names share, as the Corpus page describes it (renamed parts)."""
    template, renames = _detection(workspace, project_id)
    if template is None or not renames:
        return template
    renamed = tuple(renames.get(part, part) for part in template.parts)
    return Template(renamed, template.separator, template.fits, template.total, template.misses)


def _detected(template: Template | None, renames: dict[str, str], name: str) -> dict[str, str]:
    """File-name details for *name*, under the project's names for the parts."""
    if template is None:
        return {}
    return {renames.get(part, part): value for part, value in apply_template(name, template).items()}


def details(workspace: Workspace, project_id: str, documents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each document's details with their sources: ``{doc_id: {name: {value, source, overridden}}}``.

    ``overridden`` lists the lower-precedence values a higher one hides, so the
    Corpus page can show "file name says 1953-02-02" under a typed date.
    """
    template, renames = _detection(workspace, project_id)
    with workspace.connect() as db:
        stored = db.execute(
            "SELECT document_id, name, key, value, source FROM document_fields WHERE project_id=?", (project_id,)
        ).fetchall()
    by_document: dict[str, list[tuple[str, str, str, str]]] = {}
    for row in stored:
        by_document.setdefault(row["document_id"], []).append((row["key"], row["name"], row["value"], row["source"]))
    found: dict[str, dict[str, Any]] = {}
    for doc in documents:
        doc_id = str(doc["id"])
        candidates: dict[str, list[tuple[int, str, str, str]]] = {}
        for name, value in _detected(template, renames, str(doc["name"])).items():
            candidates.setdefault(name.casefold(), []).append((0, name, value, "filename"))
        for key, name, value, source in by_document.get(doc_id, []):
            candidates.setdefault(key, []).append((SOURCES.index(source), name, value, source))
        effective: dict[str, Any] = {}
        for options in candidates.values():
            options.sort(reverse=True)
            _rank, name, value, source = options[0]
            if not value:
                continue
            effective[name] = {
                "value": value,
                "source": source,
                "overridden": [{"value": v, "source": s} for _r, _n, v, s in options[1:] if v],
            }
        found[doc_id] = effective
    return found


def attach_fields(workspace: Workspace, project_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*rows* (document dicts) each with ``fields`` (``{name: value}``) and ``field_sources`` (``{name: source}``)."""
    found = details(workspace, project_id, rows)
    for row in rows:
        mine = found.get(str(row["id"]), {})
        row["fields"] = {name: item["value"] for name, item in mine.items()}
        row["field_sources"] = {name: item["source"] for name, item in mine.items()}
    return rows


def field_names(workspace: Workspace, project_id: str) -> list[dict[str, Any]]:
    """Every detail the project's documents have, with counts and sample values, for pickers."""
    documents = workspace.documents(project_id)
    found = details(workspace, project_id, documents)
    summary: dict[str, dict[str, Any]] = {}
    for per_document in found.values():
        for name, item in per_document.items():
            entry = summary.setdefault(name.casefold(), {"name": name, "count": 0, "sources": set(), "values": set()})
            entry["count"] += 1
            entry["sources"].add(item["source"])
            entry["values"].add(item["value"])
    order = {"date": 0, "order": 1}
    return [
        {
            "name": entry["name"],
            "count": entry["count"],
            "sources": sorted(entry["sources"], key=SOURCES.index),
            "distinct": len(entry["values"]),
            "sample_values": sorted(entry["values"])[:10],
        }
        for key, entry in sorted(summary.items(), key=lambda pair: (order.get(pair[0], 2), pair[0]))
    ]


def set_fields(
    workspace: Workspace, project_id: str, rows: list[FieldRow], expected_revision: int | None = None
) -> dict[str, Any]:
    """Set details in one transaction. An empty ``user`` value removes that override."""
    _writable(workspace, project_id)
    known = {str(doc["id"]) for doc in workspace.documents(project_id, include_trashed=True)}
    existing = {item["name"].casefold() for item in field_names(workspace, project_id)}
    with workspace.connect() as db:
        current_row = db.execute("SELECT revision FROM project_settings WHERE project_id=?", (project_id,)).fetchone()
        current = int(current_row["revision"]) if current_row else 0
        if expected_revision is not None and expected_revision != current:
            raise ValueError(f"The document details moved on to revision {current}. Reload them before saving.")
        for row in rows:
            if row.document_id not in known:
                raise KeyError(f"Unknown document: {row.document_id}")
            name = _check_name(row.name)
            existing.add(name.casefold())
            if len(existing) > MAX_FIELDS_PER_PROJECT:
                raise ValueError(f"A project may have at most {MAX_FIELDS_PER_PROJECT} kinds of detail.")
            if not row.value.strip():
                db.execute(
                    "DELETE FROM document_fields WHERE document_id=? AND key=? AND source=?",
                    (row.document_id, name.casefold(), row.source),
                )
                continue
            db.execute(
                "INSERT INTO document_fields (project_id, document_id, name, key, value, source, updated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(document_id, key, source) "
                "DO UPDATE SET name=excluded.name, value=excluded.value, updated=excluded.updated",
                (project_id, row.document_id, name, name.casefold(), row.value.strip(), row.source, now()),
            )
        revision = _bump(db, project_id)
    return {"revision": revision}


def name_problem(name: str) -> str:
    """Why *name* cannot name a detail, or "" when it can (for the import preview)."""
    try:
        _check_name(name)
    except ValueError as exc:
        return str(exc)
    return ""


def import_fields(
    workspace: Workspace,
    project_id: str,
    values: dict[str, dict[str, str]],
    targets: list[str],
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """Store a spreadsheet's details (source ``csv``), replacing the last import of the same details.

    *values* is document name -> {detail: value} from
    :func:`core.io.metadata_table.match_metadata`; *targets* are the details
    the sheet supplies. Every active document gets a row for each of them, an
    empty one where the sheet had no value, so a corrected sheet imported
    again replaces the old one instead of leaving its stale cells behind.
    Typed values (``user``) still win over anything imported.
    """
    for target in targets:
        _check_name(target)
    rows = [
        FieldRow(
            document_id=str(doc["id"]),
            name=target,
            value=values.get(str(doc["name"]), {}).get(target, ""),
            source="csv",
        )
        for doc in workspace.documents(project_id)
        for target in targets
    ]
    return set_fields(workspace, project_id, rows, expected_revision)


def stored_rows(workspace: Workspace, project_id: str) -> list[dict[str, Any]]:
    """The details someone set (not the file-name ones, which are recomputed), for backups."""
    with workspace.connect() as db:
        return [
            dict(row)
            for row in db.execute(
                "SELECT document_id, name, value, source, updated FROM document_fields "
                "WHERE project_id=? ORDER BY document_id, key, source",
                (project_id,),
            )
        ]


def delete_field(workspace: Workspace, project_id: str, name: str) -> None:
    """Remove every stored value of a detail (file-name values return if the name still has them)."""
    _writable(workspace, project_id)
    with workspace.connect() as db:
        db.execute("DELETE FROM document_fields WHERE project_id=? AND key=?", (project_id, name.strip().casefold()))
        _bump(db, project_id)
