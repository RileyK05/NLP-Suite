"""Bounded, checksummed project backups; restoration never overwrites a project."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import stat
import tempfile
from typing import Any
import uuid
import zipfile

from pydantic import ValidationError

from desktop_backend.comparisons import MAX_COMPARISONS_PER_PROJECT, ComparisonBody, comparisons
from desktop_backend.fields import MAX_FIELDS_PER_PROJECT, FieldRow, FieldSettings, settings, stored_rows
from desktop_backend.notebooks import MAX_NOTEBOOKS_PER_PROJECT, NotebookBody
from desktop_backend.paths import portable_key, safe_relative
from desktop_backend.questions import MAX_QUESTIONS_PER_PROJECT, QuestionBody
from desktop_backend.store import MAX_DOCUMENTS, MAX_PROJECT_NAME, TERMINAL_JOB_STATES, Workspace, now
from desktop_backend.views import MAX_VIEWS_PER_PROJECT, ViewBody

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
MAX_MEMBERS = 20000
MAX_MANIFEST_BYTES = 16 * 1024 * 1024


def safe_member(name: str) -> str:
    return safe_relative(name)


def validate_members(names: list[str]) -> None:
    spellings: dict[str, str] = {}
    leaves = {portable_key(name) for name in names}
    if len(leaves) != len(names):
        raise ValueError("Duplicate portable archive paths")
    for name in names:
        safe_member(name)
        parts = name.split("/")
        for length in range(1, len(parts) + 1):
            prefix = "/".join(parts[:length])
            key = portable_key(prefix)
            if key in spellings and spellings[key] != prefix:
                raise ValueError("Archive contains case/Unicode path aliases")
            if length < len(parts) and key in leaves:
                raise ValueError("Archive path is both a file and a directory")
            spellings[key] = prefix


def backup(workspace: Workspace, project_id: str, destination: Path) -> None:
    """Caller serializes this operation with project mutation and job submission."""
    project = workspace.project(project_id)
    if project.get("trashed"):
        raise ValueError("Restore this project from Trash before backing it up.")
    jobs = workspace.jobs(project_id)
    if any(j["state"] in ("RUNNING", "QUEUED") for j in jobs):
        raise ValueError("Wait for or cancel active runs before backing up this project.")
    root = workspace.project_dir(project_id)
    manifest: dict[str, Any] = {
        "format": "nlp-suite-project",
        # Version 2 adds saved views; version 3 adds questions; version 4
        # records trashed documents and runs; version 5 adds notebooks;
        # version 6 adds document details someone set, the project's detail
        # settings, and saved comparisons. Trashed projects cannot be backed up.
        # An older build refuses this archive outright rather than restoring it
        # with every saved view silently missing. Refusing is the safe
        # direction: a backup that quietly comes back smaller than it went in
        # is only discovered by the person who needed the missing part.
        "version": 6,
        "project": project,
        "documents": workspace.documents(project_id, include_trashed=True),
        "jobs": workspace.jobs(project_id, include_trashed=True),
        # ``source`` is recomputed against whatever the restored workspace
        # actually holds, so storing this run's answer would only be a chance
        # to disagree with it.
        "views": [{k: v for k, v in view.items() if k != "source"} for view in workspace.views(project_id)],
        "questions": workspace.questions(project_id),
        "notebooks": workspace.notebooks(project_id),
        # Details read from file names are not stored anywhere, so they come
        # back by themselves; these are the ones a person or spreadsheet set.
        "details": stored_rows(workspace, project_id),
        "settings": settings(workspace, project_id)["settings"],
        "comparisons": comparisons(workspace, project_id),
        "files": {},
    }
    total = 0
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("Project contains symbolic links; backup refused.")
            if not path.is_file():
                continue
            relative = safe_member(path.relative_to(root).as_posix())
            total += path.stat().st_size
            if total > MAX_EXPANDED_BYTES or len(manifest["files"]) >= MAX_MEMBERS:
                raise ValueError("Project exceeds backup size limits.")
            digest = hashlib.sha256()
            with path.open("rb") as source, archive.open("files/" + relative, "w") as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    target.write(block)
            manifest["files"][relative] = digest.hexdigest()
        payload = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
        validate_members(list(manifest["files"]))
        if len(payload) > MAX_MANIFEST_BYTES:
            raise ValueError("Project manifest exceeds the restore limit.")
        archive.writestr("project.json", payload)
    if destination.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Compressed project exceeds the 512 MB restore limit.")


def restore(workspace: Workspace, source: Path) -> dict[str, Any]:
    if source.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Project archive exceeds 512 MB.")
    try:
        with (
            zipfile.ZipFile(source) as archive,
            tempfile.TemporaryDirectory(prefix="restore-", dir=workspace.root) as temporary,
        ):
            members = archive.infolist()
            names = [item.filename for item in members]
            validate_members(names)
            if len(names) != len({portable_key(name) for name in names}) or len(names) > MAX_MEMBERS + 1:
                raise ValueError("Duplicate members or too many archive entries.")
            if sum(item.file_size for item in members) > MAX_EXPANDED_BYTES:
                raise ValueError("Expanded project exceeds 2 GB.")
            for item in members:
                safe_member(item.filename)
                mode = stat.S_IFMT(item.external_attr >> 16)
                if item.is_dir() or mode not in (0, stat.S_IFREG):
                    raise ValueError("Only regular file entries are accepted.")
            if archive.getinfo("project.json").file_size > MAX_MANIFEST_BYTES:
                raise ValueError("Project manifest is too large.")
            manifest = json.loads(archive.read("project.json"))
            if manifest.get("format") != "nlp-suite-project" or manifest.get("version") not in (1, 2, 3, 4, 5, 6):
                raise ValueError("Unsupported project archive version.")
            files = manifest["files"]
            if not isinstance(files, dict) or set(names) != {
                "project.json",
                *("files/" + safe_member(p) for p in files),
            }:
                raise ValueError("Archive members do not match the manifest.")
            stage = Path(temporary) / "project"
            stage.mkdir()
            for relative, expected in files.items():
                target = stage / safe_member(relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with archive.open("files/" + relative) as incoming, target.open("xb") as outgoing:
                    for block in iter(lambda: incoming.read(1024 * 1024), b""):
                        digest.update(block)
                        outgoing.write(block)
                if digest.hexdigest() != expected:
                    raise ValueError(f"Backup checksum mismatch: {relative}")
            return _publish(workspace, stage, manifest)
    except (
        KeyError,
        TypeError,
        AttributeError,
        sqlite3.IntegrityError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(f"Invalid project archive: {exc}") from exc


def _validate_inventory(stage: Path, manifest: dict[str, Any]) -> None:
    documents, jobs = manifest["documents"], manifest["jobs"]
    if not isinstance(documents, list) or len(documents) > MAX_DOCUMENTS or not isinstance(jobs, list):
        raise ValueError("Invalid document or job inventory.")
    for document in documents:
        if not isinstance(document.get("name"), str) or not isinstance(document.get("created"), str):
            raise ValueError("Invalid document metadata")
        name = safe_member(document["stored_name"])
        if "/" in name or manifest["files"].get("corpus/" + name) != document["sha256"]:
            raise ValueError("Document inventory does not match its copied text.")
        if (
            document.get("bytes") != (stage / "corpus" / name).stat().st_size
            or not isinstance(document.get("words"), int)
            or document["words"] < 0
        ):
            raise ValueError("Invalid document size or word count")
        if not isinstance(json.loads(document.get("import_diagnostics", "[]")), list):
            raise ValueError("Invalid import diagnostics")
        derivation = document.get("derivation")
        if derivation is not None and not isinstance(json.loads(derivation), dict):
            raise ValueError("Invalid record of how a chapter was cut from its book")
        original = document.get("source_name")
        if original:
            normalized = safe_member(original)
            if "/" in normalized or manifest["files"].get("originals/" + normalized) != document["source_sha256"]:
                raise ValueError("Original-source inventory does not match its checksum.")
    for job in jobs:
        if job["state"] not in TERMINAL_JOB_STATES:
            raise ValueError("A backup must contain only terminal jobs.")
        if not isinstance(job.get("diagnostics"), list) or not isinstance(json.loads(job["request"]), dict):
            raise ValueError("Invalid job metadata")
        if job["run_dir"]:
            directory = safe_member(job["run_dir"])
            if not directory.startswith("runs/") or directory + "/result.json" not in manifest["files"]:
                raise ValueError("Run inventory does not match its result envelope.")
    _validate_views(manifest)
    _validate_questions(manifest)
    _validate_notebooks(manifest)
    _validate_details(manifest)


def _validate_views(manifest: dict[str, Any]) -> None:
    """Reject a saved view this build could not reopen.

    Settings are checked against the same model that accepts them from the
    browser, so a view written by a newer build -- or by hand -- fails here,
    during a restore somebody is watching, rather than as a broken chart weeks
    later. Version 1 archives predate views and carry none.
    """
    views = manifest.get("views", [])
    if not isinstance(views, list) or len(views) > MAX_VIEWS_PER_PROJECT:
        raise ValueError("Invalid saved-view inventory.")
    names: set[str] = set()
    for view in views:
        if not isinstance(view, dict):
            raise ValueError("Invalid saved view.")
        try:
            ViewBody(
                name=view["name"],
                job=view["job"],
                index=view["index"],
                settings=view["settings"],
                filters=view.get("filters", []),
            )
        except (ValidationError, TypeError) as exc:
            raise ValueError(f"Invalid saved view “{view.get('name', '?')}”: {exc}") from exc
        if not isinstance(view.get("revision"), int) or view["revision"] < 1:
            raise ValueError("Invalid saved-view revision")
        for stamp in ("created", "updated"):
            if not isinstance(view.get(stamp), str):
                raise ValueError("Invalid saved-view timestamps")
        if view["name"] in names:
            raise ValueError(f"Two saved views are both called “{view['name']}”.")
        names.add(view["name"])


def _validate_questions(manifest: dict[str, Any]) -> None:
    questions = manifest.get("questions", [])
    if not isinstance(questions, list) or len(questions) > MAX_QUESTIONS_PER_PROJECT:
        raise ValueError("Invalid saved-question inventory.")
    document_ids = {str(document["id"]) for document in manifest["documents"]}
    names: set[str] = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Invalid saved question.")
        try:
            body = QuestionBody(
                name=question["name"],
                snapshot_id=question["snapshot_id"],
                parser=question["parser"],
                document_ids=question["document_ids"],
                specification=question["specification"],
            )
        except (ValidationError, TypeError) as exc:
            raise ValueError(f"Invalid saved question “{question.get('name', '?')}”: {exc}") from exc
        if not set(body.document_ids) <= document_ids:
            raise ValueError("A saved question names a document outside this backup.")
        if not isinstance(question.get("revision"), int) or question["revision"] < 1:
            raise ValueError("Invalid saved-question revision")
        if question["name"] in names:
            raise ValueError(f"Two saved questions are both called “{question['name']}”.")
        names.add(question["name"])


def _validate_notebooks(manifest: dict[str, Any]) -> None:
    """Reject a notebook this build could not open. Versions 1-4 carry none."""
    notebooks = manifest.get("notebooks", [])
    if not isinstance(notebooks, list) or len(notebooks) > MAX_NOTEBOOKS_PER_PROJECT:
        raise ValueError("Invalid notebook inventory.")
    names: set[str] = set()
    for notebook in notebooks:
        if not isinstance(notebook, dict):
            raise ValueError("Invalid notebook.")
        try:
            NotebookBody(name=notebook["name"], content=notebook["content"])
        except (ValidationError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid notebook “{notebook.get('name', '?')}”: {exc}") from exc
        if not isinstance(notebook.get("revision"), int) or notebook["revision"] < 1:
            raise ValueError("Invalid notebook revision")
        if notebook["name"] in names:
            raise ValueError(f"Two notebooks are both called “{notebook['name']}”.")
        names.add(notebook["name"])


def _validate_details(manifest: dict[str, Any]) -> None:
    """Reject details, settings or comparisons this build could not use. Versions 1-5 carry none."""
    documents = {str(doc["id"]) for doc in manifest.get("documents", [])}
    rows = manifest.get("details", [])
    if not isinstance(rows, list) or len(rows) > MAX_DOCUMENTS * MAX_FIELDS_PER_PROJECT:
        raise ValueError("Invalid document details.")
    for row in rows:
        if not isinstance(row, dict) or str(row.get("document_id")) not in documents:
            raise ValueError("A document detail names a document the archive does not hold.")
        try:
            FieldRow(document_id=str(row["document_id"]), name=row["name"], value=row["value"], source=row["source"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid document detail: {exc}") from exc
    try:
        FieldSettings(**(manifest.get("settings") or {}))
    except ValueError as exc:
        raise ValueError(f"Invalid project settings: {exc}") from exc
    saved = manifest.get("comparisons", [])
    if not isinstance(saved, list) or len(saved) > MAX_COMPARISONS_PER_PROJECT:
        raise ValueError("Invalid comparison inventory.")
    names: set[str] = set()
    for item in saved:
        try:
            ComparisonBody(name=item["name"], definition=item["definition"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid comparison: {exc}") from exc
        if item["name"] in names:
            raise ValueError(f"Two comparisons are both called “{item['name']}”.")
        names.add(item["name"])


def _publish(workspace: Workspace, stage: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_inventory(stage, manifest)
    documents, jobs = manifest["documents"], manifest["jobs"]
    name = str(manifest["project"]["name"]).strip()
    if not name or len(name) > MAX_PROJECT_NAME:
        raise ValueError("Invalid project name.")
    identifier = uuid.uuid4().hex
    destination = workspace.root / "projects" / identifier
    destination.parent.mkdir(exist_ok=True)
    published = False
    try:
        with workspace.connect() as db:
            db.execute(
                "INSERT INTO projects(id,name,created) VALUES(?,?,?)",
                (identifier, name, manifest["project"]["created"]),
            )
            document_ids: dict[str, str] = {}
            for doc in documents:
                document_ids[str(doc["id"])] = uuid.uuid4().hex
                db.execute(
                    "INSERT INTO documents(id,project_id,name,stored_name,sha256,bytes,words,created,source_name,source_sha256,import_diagnostics,trashed,derivation) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        document_ids[str(doc["id"])],
                        identifier,
                        doc["name"],
                        doc["stored_name"],
                        doc["sha256"],
                        doc["bytes"],
                        doc["words"],
                        doc["created"],
                        doc.get("source_name"),
                        doc.get("source_sha256"),
                        doc.get("import_diagnostics", "[]"),
                        int(bool(doc.get("trashed", False))),
                        doc.get("derivation"),
                    ),
                )
            # A chapter names the book it was cut from by ID, and IDs are new
            # here. A book purged before the backup maps to nothing; the
            # chapter's derivation still carries the book's sha256.
            for doc in documents:
                parent = document_ids.get(str(doc.get("derived_from") or ""))
                if parent is not None:
                    db.execute("UPDATE documents SET derived_from=? WHERE id=?", (parent, document_ids[str(doc["id"])]))
            # Restored jobs are given fresh IDs, so anything that referred to a
            # job by ID has to be carried across with them. Saved views do.
            job_ids: dict[str, str] = {}
            for job in jobs:
                job_ids[str(job["id"])] = uuid.uuid4().hex
                db.execute(
                    "INSERT INTO jobs (id, project_id, tool, state, stage, created, finished, run_dir, diagnostics, request, trashed) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        job_ids[str(job["id"])],
                        identifier,
                        job["tool"],
                        job["state"],
                        job["stage"],
                        job["created"],
                        job["finished"],
                        job["run_dir"],
                        json.dumps(job["diagnostics"]),
                        job["request"],
                        int(bool(job.get("trashed", False))),
                    ),
                )
            for view in manifest.get("views", []):
                # A view naming a run this archive does not contain keeps its
                # settings and loses its source: the empty string matches no
                # job, so the view reopens saying its table is gone rather than
                # attaching itself to whichever run inherited that ID.
                db.execute(
                    "INSERT INTO views(id,project_id,name,job_id,artifact_index,artifact_path,source_sha256,"
                    "settings,filters,revision,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        uuid.uuid4().hex,
                        identifier,
                        view["name"],
                        job_ids.get(str(view["job"]), ""),
                        view["index"],
                        view["artifact_path"],
                        view["source_sha256"],
                        json.dumps(view["settings"]),
                        json.dumps(view.get("filters", [])),
                        view["revision"],
                        view["created"],
                        view["updated"],
                    ),
                )
            for question in manifest.get("questions", []):
                specification = dict(question["specification"])
                filtered = specification.get("evidence_document_id")
                if filtered is not None:
                    specification["evidence_document_id"] = document_ids[str(filtered)]
                db.execute(
                    "INSERT INTO questions(id,project_id,name,snapshot_id,parser,document_ids,specification,"
                    "revision,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        uuid.uuid4().hex,
                        identifier,
                        question["name"],
                        question["snapshot_id"],
                        question["parser"],
                        json.dumps([document_ids[str(item)] for item in question["document_ids"]]),
                        json.dumps(specification),
                        question["revision"],
                        question["created"],
                        question["updated"],
                    ),
                )
            for notebook in manifest.get("notebooks", []):
                # Normalized again on the way in, so a notebook restored from a
                # backup is exactly as clean as one saved in the app.
                content = NotebookBody(name=notebook["name"], content=notebook["content"]).content
                db.execute(
                    "INSERT INTO notebooks(id,project_id,name,content,revision,created,updated) VALUES(?,?,?,?,?,?,?)",
                    (
                        uuid.uuid4().hex,
                        identifier,
                        notebook["name"],
                        json.dumps(content),
                        notebook["revision"],
                        notebook["created"],
                        notebook["updated"],
                    ),
                )
            for row in manifest.get("details", []):
                db.execute(
                    "INSERT INTO document_fields(project_id,document_id,name,key,value,source,updated) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (
                        identifier,
                        document_ids[str(row["document_id"])],
                        row["name"],
                        str(row["name"]).strip().casefold(),
                        row["value"],
                        row["source"],
                        row.get("updated") or row.get("created") or manifest["project"]["created"],
                    ),
                )
            if manifest.get("settings"):
                db.execute(
                    "INSERT INTO project_settings(project_id,settings,revision,updated) VALUES(?,?,1,?)",
                    (identifier, json.dumps(FieldSettings(**manifest["settings"]).model_dump()), now()),
                )
            old_project = str(manifest["project"].get("id", ""))
            for item in manifest.get("comparisons", []):
                # A side on this project follows it to its new id; a side on
                # another project keeps that id and shows as missing if this
                # workspace does not have it.
                definition = ComparisonBody(name=item["name"], definition=item["definition"]).definition
                for side in definition.sides:
                    if side.project_id == old_project:
                        side.project_id = identifier
                        if side.selection is not None and side.selection.document_ids is not None:
                            side.selection.document_ids = [
                                document_ids.get(doc_id, doc_id) for doc_id in side.selection.document_ids
                            ]
                db.execute(
                    "INSERT INTO comparisons(id,project_id,name,definition,revision,created,updated) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (
                        uuid.uuid4().hex,
                        identifier,
                        item["name"],
                        definition.model_dump_json(),
                        item.get("revision", 1),
                        item.get("created") or now(),
                        item.get("updated") or now(),
                    ),
                )
            # Move only after all SQL statements have validated. On a commit error,
            # roll the files back into the temporary directory for safe cleanup.
            stage.rename(destination)
            published = True
    except BaseException:
        if published:
            destination.rename(stage)
        raise
    return workspace.project(identifier)
