"""Saved corpus comparisons (docs/internal/PLAN_0.5.0.md section 3).

A comparison is two to six *sides* -- each a name, a project and a selection
within it -- plus how to line them up, what to focus on, and which methods to
run. It is stored as a definition only. Running it is an ordinary job of the
``contrast`` tool in the comparison's home project (its first side's), so Past
runs, panels, views, exports, trash and backups all work on it unchanged.

The storage follows the questions and notebooks tables: names unique per
project, a revision that refuses a stale save, duplicate and delete.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from core.profiler.registry import ParamSpec, ToolSpec
from desktop_backend.selection import CorpusSelection, resolve_selection
from desktop_backend.store import free_name, now

if TYPE_CHECKING:  # pragma: no cover - typing only
    from desktop_backend.store import Workspace

__all__ = [
    "CONTRAST_TOOL",
    "MAX_COMPARISONS_PER_PROJECT",
    "METHODS",
    "ComparisonBody",
    "ComparisonDefinition",
    "Side",
    "comparison",
    "comparisons",
    "delete_comparison",
    "duplicate_comparison",
    "save_comparison",
    "update_comparison",
]

CONTRAST_TOOL = "contrast"
MAX_COMPARISONS_PER_PROJECT = 100
MAX_SIDES = 6
#: The methods a comparison can run, in the order the page lists them (plan 3.5).
METHODS = ("measures", "keyness", "focus", "topics", "tone", "meaning", "carryover")
#: Methods that compare two sides only: pairs of passages, one centroid each.
TWO_SIDED = frozenset({"carryover"})
Method = Literal["measures", "keyness", "focus", "topics", "tone", "meaning", "carryover"]


def _default_methods() -> list[Method]:
    """The quick methods: what a first comparison should show without waiting on a model."""
    return ["measures", "keyness", "tone"]


class Side(BaseModel):
    """One document set: a name for it, the project it is in, and which of that project's documents."""

    model_config = ConfigDict(extra="forbid")

    name: StrictStr = Field(min_length=1, max_length=60)
    project_id: StrictStr = Field(min_length=1, max_length=64)
    selection: CorpusSelection | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Give each side a name")
        return value


class ComparisonDefinition(BaseModel):
    """What to compare, and how."""

    model_config = ConfigDict(extra="forbid")

    sides: list[Side] = Field(min_length=2, max_length=MAX_SIDES)
    #: "none", "period", or "field:<Name>" -- compare within each shared value.
    alignment: StrictStr = "none"
    methods: list[Method] = Field(default_factory=_default_methods, min_length=1)
    #: Word groups the comparison focuses on: {"immigration": ["immigration", "border", ...]}.
    focus: dict[StrictStr, list[StrictStr]] | None = None
    #: How focus words are counted (the same rule as nlp.term_rates).
    match: Literal["lemma", "form", "exact-lowercase"] = "lemma"
    topics: int = Field(default=8, ge=2, le=40)

    @field_validator("alignment")
    @classmethod
    def _alignment(cls, value: str) -> str:
        if value in ("none", "period") or (value.startswith("field:") and value[6:].strip()):
            return value
        raise ValueError('Line the sides up by "none", "period" or "field:<detail name>"')

    @model_validator(mode="after")
    def _consistent(self) -> ComparisonDefinition:
        names = [side.name.casefold() for side in self.sides]
        if len(set(names)) != len(names):
            raise ValueError("Two sides have the same name; name each side for what it holds")
        if len(self.sides) > 2 and TWO_SIDED & set(self.methods):  # noqa: PLR2004 - pairs need two sides
            raise ValueError("Carried-over passages compares two sides; remove it or keep two sides")
        if "focus" in self.methods and not self.focus:
            raise ValueError("Focus rates need focus words")
        if "carryover" in self.methods and not self.focus:
            raise ValueError("Carried-over passages need focus words")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("A method is listed twice")
        return self


class ComparisonBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: StrictStr = Field(min_length=1, max_length=120)
    definition: ComparisonDefinition
    expected_revision: int | None = None


#: The ``contrast`` tool as a plan knows it. Kept out of the general catalog
#: and the CLI (like ``phrase_distribution``): it has its own page, and its
#: request is a set of sides, not a parameter form.
CONTRAST_SPEC = ToolSpec(
    name=CONTRAST_TOOL,
    capability_ids=("CAP-CONTRAST-01",),
    packet="corpus-comparison",
    description="Two to six document sets compared on the same measures, words, focus and topics",
    requires_parse=True,
    parser_backend="config",
    input_kind="corpus",
    profiler_eligible=False,
    params=(
        ParamSpec("sides", "str", None, True, "side names, in order (JSON list)"),
        ParamSpec("alignment", "str", "none", False, "none, period, or field:<detail>"),
        ParamSpec("methods", "str", '["measures", "keyness", "tone"]', False, "methods to run (JSON list)"),
        ParamSpec("focus", "str", "{}", False, "focus word groups (JSON object)"),
        ParamSpec(
            "match", "str", "lemma", False, "how focus words are counted", choices=("lemma", "form", "exact-lowercase")
        ),
        ParamSpec("topics", "int", 8, False, "topics in the shared model", minimum=2, maximum=40),
        ParamSpec("comparison-name", "str", "", False, "the saved comparison's name"),
        ParamSpec("comparison-id", "str", "", False, "the saved comparison's ID"),
    ),
    outputs=(
        "contrast_sides.csv",
        "contrast_notes.csv",
        "contrast_measures.csv",
        "contrast_measure_tests.csv",
        "contrast_keyness.csv",
        "contrast_focus_rates.csv",
        "contrast_focus_by_group.csv",
        "contrast_topics.csv",
        "contrast_topic_prevalence.csv",
        "contrast_tone.csv",
        "contrast_tone_tests.csv",
        "contrast_carryover.csv",
    ),
    version="2",
)


def contrast_params(definition: ComparisonDefinition, name: str = "", comparison_id: str = "") -> dict[str, Any]:
    """The plan parameters of a comparison (the sides' documents travel in the request)."""
    return {
        "sides": json.dumps([side.name for side in definition.sides]),
        "alignment": definition.alignment,
        "methods": json.dumps(list(definition.methods)),
        "focus": json.dumps(definition.focus or {}),
        "match": definition.match,
        "topics": definition.topics,
        "comparison-name": name,
        "comparison-id": comparison_id,
    }


def resolve_side(workspace: Workspace, side: Side) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One side's project and selected documents, or a refusal that names the side."""
    try:
        project = workspace.project(side.project_id)
    except KeyError as exc:
        raise ValueError(f"The project for side “{side.name}” is not in this workspace.") from exc
    if project["trashed"]:
        raise ValueError(f"The project for side “{side.name}” is in Trash. Restore it first.")
    try:
        return project, resolve_selection(workspace.documents(side.project_id), side.selection)
    except ValueError as exc:
        raise ValueError(f"Side “{side.name}”: {exc}") from exc


def side_documents(workspace: Workspace, definition: ComparisonDefinition) -> list[dict[str, Any]]:
    """Every side's documents, each read from its own project and marked with its side.

    ``project_id`` on each document tells the runner which project's corpus
    folder to read it from; ``Side`` and ``Project`` go into its details,
    which is how the contrast engine knows who is who. A document on two
    sides is refused: every word it has would count for both.
    """
    documents: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], str] = {}
    for side in definition.sides:
        project, chosen = resolve_side(workspace, side)
        for doc in chosen:
            key = (side.project_id, str(doc["id"]))
            if key in seen:
                raise ValueError(
                    f"“{doc['name']}” is on both “{seen[key]}” and “{side.name}”. A document can be on one side only."
                )
            seen[key] = side.name
            details = {**(doc.get("fields") or {}), "Side": side.name, "Project": str(project["name"])}
            documents.append({**doc, "project_id": side.project_id, "fields": details})
    return documents


def _row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "definition": json.loads(row["definition"]),
        "revision": row["revision"],
        "created": row["created"],
        "updated": row["updated"],
    }


def comparisons(workspace: Workspace, project_id: str) -> list[dict[str, Any]]:
    workspace.project(project_id)
    with workspace.connect() as db:
        rows = db.execute(
            "SELECT * FROM comparisons WHERE project_id=? ORDER BY updated DESC, name", (project_id,)
        ).fetchall()
    return [_row(row) for row in rows]


def comparison(workspace: Workspace, project_id: str, comparison_id: str) -> dict[str, Any]:
    with workspace.connect() as db:
        row = db.execute(
            "SELECT * FROM comparisons WHERE project_id=? AND id=?", (project_id, comparison_id)
        ).fetchone()
    if row is None:
        raise KeyError("Comparison not found")
    return _row(row)


def _writable(workspace: Workspace, project_id: str) -> None:
    project = workspace.project(project_id)
    if project["trashed"]:
        raise ValueError("Restore this project from Trash before changing its comparisons.")


def save_comparison(workspace: Workspace, project_id: str, body: ComparisonBody) -> dict[str, Any]:
    _writable(workspace, project_id)
    name = body.name.strip()
    with workspace.connect() as db:
        count = db.execute("SELECT COUNT(*) FROM comparisons WHERE project_id=?", (project_id,)).fetchone()[0]
        if count >= MAX_COMPARISONS_PER_PROJECT:
            raise ValueError(f"A project may keep at most {MAX_COMPARISONS_PER_PROJECT} comparisons.")
        if db.execute("SELECT 1 FROM comparisons WHERE project_id=? AND name=?", (project_id, name)).fetchone():
            raise ValueError(f"This project already has a comparison called “{name}”.")
        identifier = uuid.uuid4().hex
        stamp = now()
        db.execute(
            "INSERT INTO comparisons (id, project_id, name, definition, revision, created, updated) "
            "VALUES (?, ?, ?, ?, 1, ?, ?)",
            (identifier, project_id, name, body.definition.model_dump_json(), stamp, stamp),
        )
    return comparison(workspace, project_id, identifier)


def update_comparison(
    workspace: Workspace, project_id: str, comparison_id: str, body: ComparisonBody
) -> dict[str, Any]:
    _writable(workspace, project_id)
    current = comparison(workspace, project_id, comparison_id)
    if body.expected_revision is not None and body.expected_revision != current["revision"]:
        raise ValueError(f"This comparison moved on to revision {current['revision']}. Reload it before saving.")
    name = body.name.strip()
    with workspace.connect() as db:
        clash = db.execute(
            "SELECT 1 FROM comparisons WHERE project_id=? AND name=? AND id<>?", (project_id, name, comparison_id)
        ).fetchone()
        if clash:
            raise ValueError(f"This project already has a comparison called “{name}”.")
        db.execute(
            "UPDATE comparisons SET name=?, definition=?, revision=revision+1, updated=? WHERE project_id=? AND id=?",
            (name, body.definition.model_dump_json(), now(), project_id, comparison_id),
        )
    return comparison(workspace, project_id, comparison_id)


def duplicate_comparison(workspace: Workspace, project_id: str, comparison_id: str) -> dict[str, Any]:
    source = comparison(workspace, project_id, comparison_id)
    taken = {item["name"] for item in comparisons(workspace, project_id)}
    body = ComparisonBody(name=free_name(source["name"], taken), definition=source["definition"])
    return save_comparison(workspace, project_id, body)


def delete_comparison(workspace: Workspace, project_id: str, comparison_id: str) -> None:
    _writable(workspace, project_id)
    comparison(workspace, project_id, comparison_id)
    with workspace.connect() as db:
        db.execute("DELETE FROM comparisons WHERE project_id=? AND id=?", (project_id, comparison_id))
