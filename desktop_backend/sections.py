"""Cutting a book into its chapters, as documents (docs/PLAN_0.5.0.md 2.4).

:mod:`core.file_ops.sections` finds the cuts; this stores them. Each section
becomes a document of its own -- a new file holding exactly its slice of the
book, and a row that names the book it came from (``derived_from``) and how it
was cut (``derivation``: the rule, the offsets, and the book's sha256). Its
details come with it, from source ``split``: Work, Order, Chapter, Title, and
one per level above it (Volume, Book, Part).

The book itself is never changed (R3). By default it goes to Trash: a corpus
holding both the book and its chapters counts every word twice.

Purging the book later is allowed. Refusing it would leave a person unable to
tidy their project for the sake of a link they can see in each chapter's
derivation anyway: the book's sha256 stays there, so which text a chapter was
cut from is still on record after the book's file is gone.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any
import uuid

from pydantic import BaseModel, ConfigDict, Field

from core.file_ops.sections import RULES, SectionPlan, TurnPlan, detect_sections, detect_turns
from core.io.reader import read_text
from core.result import Result
from desktop_backend.fields import FieldSettings, _bump, _check_name, details, resolved_axis, settings
from desktop_backend.paths import portable_key, storage_filename
from desktop_backend.store import MAX_DOCUMENTS, now

if TYPE_CHECKING:  # pragma: no cover - typing only
    from desktop_backend.store import Workspace

__all__ = ["SplitBody", "plan_for", "public_plan", "public_turns", "split_document", "split_transcript"]

DASH = chr(0x2013)
#: A section's text shown in the preview list.
PREVIEW_CHARS = 120
#: Characters a document name may not hold on any of the systems a workspace moves between.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_TITLE = 80


class SplitBody(BaseModel):
    """How to cut one document: the rule and its options, for the preview and the apply alike."""

    model_config = ConfigDict(extra="forbid")

    rule: str = Field(default="auto", pattern="^(" + "|".join(RULES) + ")$")
    pattern: str = Field(default="", max_length=200)
    regex: bool = False
    block_words: int = Field(default=2000, ge=100, le=100_000)
    #: Keep a Project Gutenberg header and footer as part of the text.
    keep_front_matter: bool = False
    #: Keep the whole book as a document beside its chapters (words then count twice).
    keep_whole: bool = False
    #: The book's sha256 the preview was made from; apply refuses a changed book.
    expected_sha256: str | None = None
    #: A transcript: cut by speaker (plan 2.5) instead of by chapter.
    transcript: bool = False
    #: One document per turn rather than one per speaker.
    per_turn: bool = False
    #: The speakers to keep; None keeps everyone (deselect the moderator here).
    speakers: list[str] | None = Field(default=None, max_length=200)


def _book(workspace: Workspace, project_id: str, document_id: str) -> tuple[dict[str, Any], str]:
    document = next((doc for doc in workspace.documents(project_id) if doc["id"] == document_id), None)
    if document is None:
        raise KeyError("Document not found (it may be in Trash)")
    if document.get("derived_from"):
        raise ValueError("This document is already a section of a book; split the book itself instead.")
    text = read_text(workspace.project_dir(project_id) / "corpus" / document["stored_name"]).unwrap()
    return document, text


def plan_for(workspace: Workspace, project_id: str, document_id: str, body: SplitBody) -> Result[SectionPlan]:
    """Where *body*'s rule would cut the document (nothing is written)."""
    _document, text = _book(workspace, project_id, document_id)
    return detect_sections(
        text,
        rule=body.rule,
        pattern=body.pattern,
        regex=body.regex,
        block_words=body.block_words,
        keep_front_matter=body.keep_front_matter,
    )


def public_plan(workspace: Workspace, project_id: str, document_id: str, body: SplitBody) -> dict[str, Any]:
    """The preview: every section with its words and opening, what was left out, and why to look twice."""
    document, text = _book(workspace, project_id, document_id)
    planned = plan_for(workspace, project_id, document_id, body)
    if planned.value is None:
        raise ValueError(" ".join(d.message for d in planned.diagnostics))
    plan = planned.value
    return {
        "document": {"id": document["id"], "name": document["name"], "sha256": document["sha256"]},
        "rule": plan.rule,
        "work": _work(document["name"]),
        "level": _level(plan),
        "sections": [
            {
                "order": section.order,
                "title": section.title,
                "level": section.level,
                "parents": [list(pair) for pair in section.parents],
                "words": section.words,
                "opening": " ".join(text[section.start : section.end].split())[:PREVIEW_CHARS],
            }
            for section in plan.sections
        ],
        "left_out": [{"what": what, "size": size} for what, size in plan.left_out],
        "diagnostics": [
            {"severity": d.severity.name, "code": d.code, "message": d.message, **dict(d.context)}
            for d in planned.diagnostics
        ],
    }


def _work(name: str) -> str:
    return Path(name).stem.strip() or "Book"


def _level(plan: SectionPlan) -> str:
    """What one section is called on the axis: "Chapter", or "Section" for blocks and gaps."""
    levels = {section.level for section in plan.sections}
    return levels.pop().capitalize() if len(levels) == 1 else "Section"


def _document_name(work: str, order: int, title: str) -> str:
    """The work, a dash, the zero-padded order and the title, as a name every file system accepts."""
    safe = _UNSAFE.sub(" ", title).strip()
    safe = re.sub(r"\s+", " ", safe)[:_MAX_TITLE].strip(" .")
    return f"{_UNSAFE.sub(' ', work).strip()} {DASH} {order:03d} {safe}.txt"


def _details(plan: SectionPlan, work: str) -> list[list[tuple[str, str]]]:
    """Each section's split details: Work, Order, Chapter (or its level), Title, and its parents."""
    rows = []
    for section in plan.sections:
        level = section.level.capitalize() if section.level != "section" else "Section"
        pairs = [
            ("Work", work),
            ("Order", str(section.order)),
            (level, str(section.number if section.number is not None else section.order)),
            ("Title", section.name or section.title),
        ]
        pairs += [(name, value) for name, value in section.parents]
        rows.append(pairs)
    return rows


@dataclass(frozen=True, slots=True)
class _Piece:
    """One new document: its name, its text, how it was cut, and its split details."""

    name: str
    text: str
    order: int
    derivation: dict[str, Any]
    details: list[tuple[str, str]]


def _store(  # noqa: PLR0913 - the one transaction every kind of split shares
    workspace: Workspace,
    project_id: str,
    document: dict[str, Any],
    pieces: list[_Piece],
    *,
    keep_whole: bool,
    order_label: str,
) -> list[dict[str, Any]]:
    """Write *pieces* as documents in one transaction, and return them.

    Everything happens or nothing does: the files are written first, then one
    transaction adds the rows, their details and the axis, and a failure
    removes the files again. The source goes to Trash unless *keep_whole*.
    A project with nothing to line its documents up by is lined up by
    *order_label* afterwards, unless someone already chose an axis.
    """
    project = workspace.project(project_id)
    if project["trashed"] or project["archived"]:
        raise ValueError("Restore this project before splitting its documents.")
    for piece in pieces:
        for name, _value in piece.details:
            _check_name(name)
    documents_before = workspace.documents(project_id)
    axis_before = resolved_axis(
        workspace, project_id, documents_before, details(workspace, project_id, documents_before)
    )
    chosen = FieldSettings(**settings(workspace, project_id)["settings"])
    root = workspace.project_dir(project_id) / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    created: list[str] = []
    stamp = now()
    try:
        with workspace.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM documents WHERE project_id=?", (project_id,)).fetchone()[0]
            if count + len(pieces) > MAX_DOCUMENTS:
                raise ValueError(f"Splitting into {len(pieces)} would pass the {MAX_DOCUMENTS:,}-document limit.")
            used = {
                portable_key(row[0])
                for row in db.execute("SELECT stored_name FROM documents WHERE project_id=?", (project_id,))
            }
            stem = Path(document["stored_name"]).stem
            for piece in pieces:
                data = piece.text.encode("utf-8")
                stored = storage_filename(f"{stem}__{piece.order:03d}.txt", root)
                if portable_key(stored) in used or (root / stored).exists():
                    stored = f"{uuid.uuid4().hex[:8]}__{stored}"
                used.add(portable_key(stored))
                target = root / stored
                with target.open("xb") as handle:
                    handle.write(data)
                written.append(target)
                identifier = uuid.uuid4().hex
                digest = hashlib.sha256(data).hexdigest()
                derivation = {**piece.derivation, "source_name": document["name"], "source_sha256": document["sha256"]}
                db.execute(
                    "INSERT INTO documents(id,project_id,name,stored_name,sha256,bytes,words,created,"
                    "source_name,source_sha256,import_diagnostics,derived_from,derivation) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        identifier,
                        project_id,
                        piece.name,
                        stored,
                        digest,
                        len(data),
                        len(piece.text.split()),
                        stamp,
                        None,
                        digest,
                        "[]",
                        document["id"],
                        json.dumps(derivation),
                    ),
                )
                for name, value in piece.details:
                    db.execute(
                        "INSERT INTO document_fields (project_id, document_id, name, key, value, source, updated) "
                        "VALUES (?, ?, ?, ?, ?, 'split', ?)",
                        (project_id, identifier, name, name.casefold(), value, stamp),
                    )
                created.append(identifier)
            if not keep_whole:
                db.execute("UPDATE documents SET trashed=1 WHERE project_id=? AND id=?", (project_id, document["id"]))
            if chosen.axis == "auto" and axis_before["kind"] == "none" and order_label:
                new_settings = chosen.model_copy(update={"axis": "order", "order_label": order_label})
                _bump(db, project_id, new_settings.model_dump())
            else:
                _bump(db, project_id)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    by_id = {doc["id"]: doc for doc in workspace.documents(project_id)}
    return [by_id[identifier] for identifier in created]


def _fresh(document: dict[str, Any], body: SplitBody) -> None:
    if body.expected_sha256 is not None and body.expected_sha256 != document["sha256"]:
        raise ValueError("This document changed since the preview. Preview the split again.")


def split_document(workspace: Workspace, project_id: str, document_id: str, body: SplitBody) -> list[dict[str, Any]]:
    """Store the document's sections -- or, for a transcript, its speakers -- as documents."""
    if body.transcript:
        return split_transcript(workspace, project_id, document_id, body)
    document, text = _book(workspace, project_id, document_id)
    _fresh(document, body)
    planned = plan_for(workspace, project_id, document_id, body)
    if planned.value is None:
        raise ValueError(" ".join(d.message for d in planned.diagnostics))
    plan = planned.value
    work = _work(document["name"])
    pieces = [
        _Piece(
            name=_document_name(work, section.order, section.title),
            text=text[section.start : section.end],
            order=section.order,
            derivation={"rule": plan.rule, "start": section.start, "end": section.end, "order": section.order},
            details=pairs,
        )
        for section, pairs in zip(plan.sections, _details(plan, work), strict=True)
    ]
    return _store(workspace, project_id, document, pieces, keep_whole=body.keep_whole, order_label=_level(plan))


# ------------------------------------------------------------ transcripts --


def _turns(workspace: Workspace, project_id: str, document_id: str) -> tuple[dict[str, Any], TurnPlan]:
    document, text = _book(workspace, project_id, document_id)
    found = detect_turns(text)
    if found.value is None:
        raise ValueError(" ".join(d.message for d in found.diagnostics))
    return document, found.value


def public_turns(workspace: Workspace, project_id: str, document_id: str, body: SplitBody) -> dict[str, Any]:
    """The transcript preview: each speaker's turns and words, and the stage directions left out."""
    document, plan = _turns(workspace, project_id, document_id)
    kept = _kept_speakers(plan, body)
    return {
        "document": {"id": document["id"], "name": document["name"], "sha256": document["sha256"]},
        "event": _work(document["name"]),
        "speakers": [
            {"name": name, "turns": turns, "words": words, "kept": name in kept} for name, turns, words in plan.speakers
        ],
        "directions": [{"text": text, "count": count} for text, count in plan.directions],
        "preamble_words": plan.preamble_words,
        "documents": sum(turns for name, turns, _w in plan.speakers if name in kept) if body.per_turn else len(kept),
    }


def _kept_speakers(plan: TurnPlan, body: SplitBody) -> set[str]:
    everyone = {name for name, _t, _w in plan.speakers}
    if body.speakers is None:
        return everyone
    wanted = {name.casefold() for name in body.speakers}
    return {name for name in everyone if name.casefold() in wanted}


def split_transcript(workspace: Workspace, project_id: str, document_id: str, body: SplitBody) -> list[dict[str, Any]]:
    """One document per speaker (all their turns, a blank line between) or, with ``per_turn``, per turn.

    Per speaker is the default: a turn is often a sentence, and per-document
    measures on one sentence are noise.
    """
    document, plan = _turns(workspace, project_id, document_id)
    _fresh(document, body)
    kept = _kept_speakers(plan, body)
    if not kept:
        raise ValueError("Keep at least one speaker.")
    event = _work(document["name"])
    date = str((document.get("fields") or {}).get("Date", ""))
    dated = [("Date", date)] if date else []
    pieces: list[_Piece] = []
    if body.per_turn:
        chosen = [turn for turn in plan.turns if turn.speaker in kept]
        for order, turn in enumerate(chosen, start=1):
            pieces.append(
                _Piece(
                    name=_document_name(event, order, turn.speaker),
                    text=turn.text + "\n",
                    order=order,
                    derivation={"rule": "turns", "speaker": turn.speaker, "turns": [[turn.start, turn.end]]},
                    details=[("Speaker", turn.speaker), ("Event", event), ("Order", str(order)), *dated],
                )
            )
    else:
        for order, (name, turns, _words) in enumerate(
            [speaker for speaker in plan.speakers if speaker[0] in kept], start=1
        ):
            own = [turn for turn in plan.turns if turn.speaker == name]
            pieces.append(
                _Piece(
                    name=f"{_UNSAFE.sub(' ', event).strip()} {DASH} {_UNSAFE.sub(' ', name).strip()}.txt",
                    text="\n\n".join(turn.text for turn in own) + "\n",
                    order=order,
                    derivation={"rule": "turns", "speaker": name, "turns": [[t.start, t.end] for t in own]},
                    details=[("Speaker", name), ("Event", event), ("Turns", str(turns)), *dated],
                )
            )
    return _store(
        workspace, project_id, document, pieces, keep_whole=body.keep_whole, order_label="Turn" if body.per_turn else ""
    )
