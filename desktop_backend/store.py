"""Persistent, copy-on-import desktop projects and manifest-backed run access."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any
import uuid

from core.artifacts.envelope import Envelope
from core.file_ops.converter import convert_document_to_text, supported_suffixes
from core.io.reader import read_text
from desktop_backend.paths import portable_key, storage_filename
from desktop_backend.questions import MAX_QUESTIONS_PER_PROJECT, QuestionBody
from desktop_backend.views import MAX_VIEW_NAME, MAX_VIEWS_PER_PROJECT, ViewBody, describe_source

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_DOCUMENTS = 2000
MAX_PROJECT_NAME = 120
MAX_FILENAME = 240
PREVIEW_CHARACTERS = 30000

# The single definition of "no more work will happen on this job". Backup
# validation, restore validation and job handling all consume this tuple so
# a new terminal state (e.g. PARTIAL) is accepted everywhere at once.
# PARTIAL is terminal: usable artifacts plus diagnostics the user must read.
TERMINAL_JOB_STATES = frozenset({"DONE", "PARTIAL", "FAILED", "INTERRUPTED", "CANCELLED"})


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def free_name(base: str, taken: set[str]) -> str:
    """``"Trend"`` -> ``"Trend (copy)"`` -> ``"Trend (copy 2)"``.

    Names are unique per project, so duplicating has to choose one rather than
    failing and asking the reader to. The base is trimmed first so that
    duplicating a name already at the length limit cannot produce one over it,
    and the search is bounded: at the limit the caller gets a plain refusal
    instead of a loop.
    """
    room = MAX_VIEW_NAME - len(" (copy 999)")
    stem = base[:room].rstrip() or "View"
    for attempt in range(1, 1000):
        candidate = f"{stem} (copy)" if attempt == 1 else f"{stem} (copy {attempt})"
        if candidate not in taken:
            return candidate
    raise ValueError(f"Too many copies of “{base}”. Rename or delete some first.")


def extract_text(root: Path, suffix: str, data: bytes) -> tuple[bytes, list[dict[str, Any]]]:
    if suffix == ".txt":
        return data, []
    with tempfile.TemporaryDirectory(prefix="conversion-", dir=root) as temporary:
        source = Path(temporary) / ("input" + suffix)
        source.write_bytes(data)
        converted = convert_document_to_text(source)
    if not converted.ok or converted.value is None:
        raise ValueError("; ".join(d.message for d in converted.diagnostics))
    text = converted.unwrap().encode("utf-8")
    if len(text) > MAX_FILE_BYTES:
        raise ValueError("Extracted text exceeds 20 MB.")
    return text, [d.to_dict() for d in converted.diagnostics]


class Workspace:
    """SQLite metadata with immutable original document copies on disk."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
                    name TEXT NOT NULL, stored_name TEXT NOT NULL, sha256 TEXT NOT NULL,
                    bytes INTEGER NOT NULL, words INTEGER NOT NULL, created TEXT NOT NULL,
                    UNIQUE(project_id, name, sha256));
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
                    tool TEXT NOT NULL, state TEXT NOT NULL, stage TEXT NOT NULL,
                    created TEXT NOT NULL, finished TEXT, run_dir TEXT,
                    diagnostics TEXT NOT NULL DEFAULT '[]', request TEXT NOT NULL);
                -- Saved chart views. job_id is deliberately NOT a foreign key:
                -- a view whose run has gone is reported as dangling and kept,
                -- because the settings someone worked out are worth more than
                -- the tidiness of deleting them with their source.
                CREATE TABLE IF NOT EXISTS views (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
                    name TEXT NOT NULL, job_id TEXT NOT NULL, artifact_index INTEGER NOT NULL,
                    artifact_path TEXT NOT NULL, source_sha256 TEXT NOT NULL,
                    settings TEXT NOT NULL, filters TEXT NOT NULL DEFAULT '[]',
                    revision INTEGER NOT NULL DEFAULT 1,
                    created TEXT NOT NULL, updated TEXT NOT NULL,
                    UNIQUE(project_id, name));
                CREATE TABLE IF NOT EXISTS questions (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
                    name TEXT NOT NULL, snapshot_id TEXT NOT NULL, parser TEXT NOT NULL,
                    document_ids TEXT NOT NULL, specification TEXT NOT NULL,
                    matching TEXT NOT NULL DEFAULT '{}',
                    revision INTEGER NOT NULL DEFAULT 1,
                    created TEXT NOT NULL, updated TEXT NOT NULL,
                    UNIQUE(project_id, name));
            """)
            if "archived" not in {row[1] for row in db.execute("PRAGMA table_info(projects)")}:
                db.execute("ALTER TABLE projects ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
            if "trashed" not in {row[1] for row in db.execute("PRAGMA table_info(documents)")}:
                db.execute("ALTER TABLE documents ADD COLUMN trashed INTEGER NOT NULL DEFAULT 0")
            if "trashed" not in {row[1] for row in db.execute("PRAGMA table_info(projects)")}:
                db.execute("ALTER TABLE projects ADD COLUMN trashed INTEGER NOT NULL DEFAULT 0")
            if "trashed" not in {row[1] for row in db.execute("PRAGMA table_info(jobs)")}:
                db.execute("ALTER TABLE jobs ADD COLUMN trashed INTEGER NOT NULL DEFAULT 0")
            if "matching" not in {row[1] for row in db.execute("PRAGMA table_info(questions)")}:
                # Questions saved before the matching rules were recorded keep
                # an empty profile, which reads as "unrecorded" rather than as
                # "matched by today's rules".
                db.execute("ALTER TABLE questions ADD COLUMN matching TEXT NOT NULL DEFAULT '{}'")
            columns = {row[1] for row in db.execute("PRAGMA table_info(documents)")}
            for name, declaration in (
                ("source_name", "TEXT"),
                ("source_sha256", "TEXT"),
                ("import_diagnostics", "TEXT NOT NULL DEFAULT '[]'"),
            ):
                if name not in columns:
                    db.execute(f"ALTER TABLE documents ADD COLUMN {name} {declaration}")
            definition = db.execute("SELECT sql FROM sqlite_master WHERE name='documents'").fetchone()[0]
            if "UNIQUE(project_id, name, sha256)" in definition:
                # Transactional table replacement retains every column and row;
                # converted source identity must not collapse equal extracted text.
                db.execute(
                    definition.replace("documents", "documents_v2", 1).replace(
                        "UNIQUE(project_id, name, sha256)", "UNIQUE(project_id, name, source_sha256)"
                    )
                )
                db.execute("INSERT INTO documents_v2 SELECT * FROM documents")
                db.execute("DROP TABLE documents")
                db.execute("ALTER TABLE documents_v2 RENAME TO documents")
                db.execute(
                    "CREATE UNIQUE INDEX documents_identity ON documents(project_id,name,COALESCE(source_sha256,sha256))"
                )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.root / "workspace.sqlite3", timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def projects(self, archived: bool = False, trashed: bool = False) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    """
                SELECT p.*, COUNT(CASE WHEN d.trashed=0 THEN d.id END) AS documents, COALESCE(SUM(CASE WHEN d.trashed=0 THEN d.words ELSE 0 END),0) AS words
                FROM projects p LEFT JOIN documents d ON p.id=d.project_id
                WHERE p.trashed=? AND (? OR p.archived=?)
                GROUP BY p.id ORDER BY p.created DESC, p.rowid DESC
            """,
                    (int(trashed), int(trashed), int(archived)),
                )
            ]

    def project(self, project_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
            if row is None:
                raise KeyError("Project not found")
            counts = db.execute(
                "SELECT COUNT(*) AS documents,COALESCE(SUM(words),0) AS words FROM documents WHERE project_id=? AND trashed=0",
                (project_id,),
            ).fetchone()
            return {**dict(row), **dict(counts)}

    def project_dir(self, project_id: str) -> Path:
        self.project(project_id)
        return self.root / "projects" / project_id

    def create(self, name: str) -> dict[str, Any]:
        name = name.strip()
        if not name or len(name) > MAX_PROJECT_NAME:
            raise ValueError("Give your project a name between 1 and 120 characters.")
        project_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO projects(id,name,created) VALUES (?, ?, ?)", (project_id, name, now()))
        (self.project_dir(project_id) / "corpus").mkdir(parents=True)
        return self.project(project_id)

    def rename(self, project_id: str, name: str) -> dict[str, Any]:
        self.project(project_id)
        name = name.strip()
        if not name or len(name) > MAX_PROJECT_NAME:
            raise ValueError("Give your project a name between 1 and 120 characters.")
        with self.connect() as db:
            db.execute("UPDATE projects SET name=? WHERE id=?", (name, project_id))
        return self.project(project_id)

    def archive(self, project_id: str, archived: bool) -> dict[str, Any]:
        if any(job["state"] in ("RUNNING", "QUEUED") for job in self.jobs(project_id)):
            raise ValueError("Finish or cancel active runs before archiving this project.")
        with self.connect() as db:
            db.execute("UPDATE projects SET archived=? WHERE id=?", (int(archived), project_id))
        return self.project(project_id)

    def trash_project(self, project_id: str, trashed: bool) -> dict[str, Any]:
        project = self.project(project_id)
        if any(job["state"] in ("RUNNING", "QUEUED") for job in self.jobs(project_id, include_trashed=True)):
            raise ValueError("Finish or cancel active runs before deleting this project.")
        trash_root = self.root / "trash"
        if trash_root.is_symlink() or not trash_root.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Project Trash storage is unsafe.")
        trash_root.mkdir(exist_ok=True)
        active = self.root / "projects" / project_id
        held = trash_root / ("project-" + project_id)
        if (
            active.is_symlink()
            or held.is_symlink()
            or not active.resolve(strict=False).is_relative_to(self.root.resolve())
            or not held.resolve(strict=False).is_relative_to(self.root.resolve())
        ):
            raise ValueError("Project storage contains a symbolic link; deletion refused.")
        source, destination = (active, held) if trashed else (held, active)
        if not source.is_dir() or destination.exists():
            raise ValueError("Project storage is missing or already exists at the destination.")
        if any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("Project storage contains a symbolic link; deletion refused.")
        source.rename(destination)
        try:
            with self.connect() as db:
                db.execute("UPDATE projects SET trashed=? WHERE id=?", (int(trashed), project_id))
        except BaseException:
            destination.rename(source)
            raise
        return {**project, "trashed": int(trashed)}

    def purge_project(self, project_id: str) -> None:  # noqa: PLR0912, PLR0915
        project = self.project(project_id)
        if not project["trashed"]:
            raise ValueError("Move this project to Trash before permanently deleting it.")
        if any(job["state"] in ("RUNNING", "QUEUED") for job in self.jobs(project_id, include_trashed=True)):
            raise ValueError("Finish or cancel active runs before permanently deleting this project.")
        trash_root = self.root / "trash"
        if trash_root.is_symlink() or not trash_root.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Project Trash storage is unsafe.")
        held = trash_root / ("project-" + project_id)
        if (
            held.is_symlink()
            or not held.resolve(strict=False).is_relative_to(trash_root.resolve())
            or not held.is_dir()
        ):
            raise ValueError("Project Trash storage is unsafe or missing.")
        if any(path.is_symlink() for path in held.rglob("*")):
            raise ValueError("Project Trash contains a symbolic link; purge refused.")
        staged = trash_root / ("purge-project-" + project_id + "-" + uuid.uuid4().hex)
        if not staged.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Project purge staging path is unsafe.")
        held.rename(staged)
        staged_logs = trash_root / ("purge-project-logs-" + project_id + "-" + uuid.uuid4().hex)
        moved_logs: list[tuple[Path, Path]] = []
        try:
            logs_root = self.root / "logs"
            if logs_root.is_symlink():
                raise ValueError("Workspace log storage is unsafe; project deletion was refused.")
            for job in self.jobs(project_id, include_trashed=True):
                log = logs_root / (job["id"] + ".log")
                if log.is_symlink() or not log.resolve(strict=False).is_relative_to(self.root):
                    raise ValueError("A run log path is unsafe; project deletion was refused.")
                if log.exists():
                    target = staged_logs / log.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    log.rename(target)
                    moved_logs.append((target, log))
            with self.connect() as db:
                # These tables refer to a project but are intentionally not
                # cascaded, so purge their rows in dependency order.
                db.execute("DELETE FROM views WHERE project_id=?", (project_id,))
                db.execute("DELETE FROM questions WHERE project_id=?", (project_id,))
                db.execute("DELETE FROM jobs WHERE project_id=?", (project_id,))
                db.execute("DELETE FROM documents WHERE project_id=?", (project_id,))
                db.execute("DELETE FROM projects WHERE id=?", (project_id,))
        except BaseException:
            for source, target in reversed(moved_logs):
                source.rename(target)
            staged.rename(held)
            raise
        try:
            shutil.rmtree(staged)
        except OSError as exc:
            raise OSError(f"Project metadata was purged; remaining files are safely staged at {staged}: {exc}") from exc
        if staged_logs.exists():
            if not staged_logs.resolve().is_relative_to(self.root.resolve()):
                raise ValueError("Project log purge staging path is unsafe.")
            try:
                shutil.rmtree(staged_logs)
            except OSError as exc:
                raise OSError(
                    f"Project metadata was purged; remaining run logs are safely staged at {staged_logs}: {exc}"
                ) from exc

    def documents(
        self, project_id: str, *, trashed: bool = False, include_trashed: bool = False
    ) -> list[dict[str, Any]]:
        self.project(project_id)
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM documents WHERE project_id=? AND (? OR trashed=?) ORDER BY name, id",
                    (project_id, int(include_trashed), int(trashed)),
                )
            ]

    def trash_document(self, project_id: str, document_id: str, trashed: bool) -> dict[str, Any]:
        """Hide or restore one imported document without moving its identity or files."""
        if self.project(project_id)["trashed"]:
            raise ValueError("Restore this project from Trash before changing its documents.")
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM documents WHERE project_id=? AND id=?", (project_id, document_id)
            ).fetchone()
            if row is None:
                raise KeyError("Document not found")
            db.execute(
                "UPDATE documents SET trashed=? WHERE project_id=? AND id=?", (int(trashed), project_id, document_id)
            )
            updated = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            return dict(updated)

    def purge_document(self, project_id: str, document_id: str) -> None:  # noqa: PLR0912
        """Permanently remove a trashed document and both owned copies."""
        if self.project(project_id)["trashed"]:
            raise ValueError("Restore this project from Trash before permanently deleting documents.")
        project_root = self.project_dir(project_id).resolve()
        if not project_root.is_relative_to(self.root.resolve()):
            raise ValueError("Project storage path is unsafe.")
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM documents WHERE project_id=? AND id=?", (project_id, document_id)
            ).fetchone()
            if row is None:
                raise KeyError("Document not found")
            if not row["trashed"]:
                raise ValueError("Move this document to Trash before permanently deleting it.")
            references = db.execute(
                "SELECT name,document_ids FROM questions WHERE project_id=?", (project_id,)
            ).fetchall()
            for question in references:
                if document_id in json.loads(question["document_ids"]):
                    raise ValueError(
                        f"Saved question '{question['name']}' still refers to this document. Update or delete that question first."
                    )
            paths = [project_root / "corpus" / row["stored_name"]]
            if row["source_name"]:
                paths.append(project_root / "originals" / row["source_name"])
            for path in paths:
                resolved = path.resolve(strict=False)
                if not resolved.is_relative_to(project_root) or path.is_symlink():
                    raise ValueError("Document storage path is unsafe; it was not deleted.")
            # Stage both copies by rename first. If either move or the metadata
            # transaction fails, put every file back before returning.
            stage = project_root / ".trash-purge" / document_id
            if not stage.resolve().is_relative_to(project_root) or stage.parent.is_symlink():
                raise ValueError("Document purge staging path is unsafe.")
            stage.parent.mkdir(exist_ok=True)
            moved: list[tuple[Path, Path]] = []
            try:
                for index, path in enumerate(paths):
                    if path.exists():
                        target = stage / str(index)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        path.rename(target)
                        moved.append((target, path))
                db.execute("DELETE FROM documents WHERE project_id=? AND id=?", (project_id, document_id))
            except BaseException:
                for source, target in reversed(moved):
                    source.rename(target)
                raise
        try:
            shutil.rmtree(stage)
        except OSError as exc:
            raise OSError(
                f"Document metadata was purged, but staged files remain recoverable at {stage}: {exc}"
            ) from exc
        if stage.parent.exists() and not any(stage.parent.iterdir()):
            stage.parent.rmdir()

    def import_document(self, project_id: str, name: str, data: bytes) -> dict[str, Any]:  # noqa: PLR0912, PLR0915
        if self.project(project_id)["trashed"]:
            raise ValueError("Restore this project from Trash before importing documents.")
        if self.project(project_id)["archived"]:
            raise ValueError("Restore this project from the archive before importing documents.")
        root = self.project_dir(project_id) / "corpus"
        name = name.replace("\\", "/").split("/")[-1].strip()
        suffix = Path(name).suffix.lower()
        if suffix not in supported_suffixes() or len(name) > MAX_FILENAME:
            raise ValueError(
                "Supported formats: TXT, CSV, TSV, HTML, PDF, DOCX and RTF. Legacy DOC and scanned PDFs need external conversion/OCR."
            )
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("A document may be at most 20 MB.")
        if not data or (suffix == ".txt" and b"\0" in data):
            raise ValueError("Document is empty or contains binary/NUL data. Export it as plain text first.")
        source_data = data
        source_digest = hashlib.sha256(data).hexdigest()
        data, import_diagnostics = extract_text(self.root, suffix, data)
        digest = hashlib.sha256(data).hexdigest()
        identifier = uuid.uuid4().hex
        # Keep recognizable document labels while reserving unique storage paths.
        safe_name = storage_filename(name, root)
        if suffix != ".txt":
            safe_name += ".txt"
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM documents WHERE project_id=? AND name=? AND COALESCE(source_sha256,sha256)=?",
                (project_id, name, source_digest),
            ).fetchone()
            if existing is not None:
                if existing["trashed"]:
                    raise ValueError("An identical document is in Trash. Restore it before importing another copy.")
                return {**dict(existing), "duplicate": True}
            count = db.execute("SELECT COUNT(*) FROM documents WHERE project_id=?", (project_id,)).fetchone()[0]
            if count >= MAX_DOCUMENTS:
                raise ValueError("This project has reached the 2,000-document limit.")
            used = {
                portable_key(row[0])
                for row in db.execute("SELECT stored_name FROM documents WHERE project_id=?", (project_id,))
            }
            stored = (
                safe_name
                if portable_key(safe_name) not in used and not (root / safe_name).exists()
                else f"{identifier[:8]}__{safe_name}"
            )
            target = root / stored
            original = self.project_dir(project_id) / "originals" / (identifier + suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            created = False
            try:
                with target.open("xb") as handle:
                    created = True
                    handle.write(data)
                read = read_text(target)
                if read.value is None or not read.unwrap().strip():
                    raise ValueError("Document contains no readable text.")
                words = len(read.unwrap().split())
                if suffix != ".txt":
                    original.parent.mkdir(exist_ok=True)
                    with original.open("xb") as handle:
                        handle.write(source_data)
                db.execute(
                    "INSERT INTO documents(id,project_id,name,stored_name,sha256,bytes,words,created,source_name,source_sha256,import_diagnostics) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        identifier,
                        project_id,
                        name,
                        stored,
                        digest,
                        len(data),
                        words,
                        now(),
                        original.name if suffix != ".txt" else None,
                        source_digest,
                        json.dumps(import_diagnostics),
                    ),
                )
            except Exception:
                if created:
                    target.unlink(missing_ok=True)
                    original.unlink(missing_ok=True)
                raise
        return next(doc for doc in self.documents(project_id) if doc["id"] == identifier)

    def preview(self, project_id: str, document_id: str) -> dict[str, Any]:
        doc = next((d for d in self.documents(project_id) if d["id"] == document_id), None)
        if doc is None:
            raise KeyError("Document not found")
        text = read_text(self.project_dir(project_id) / "corpus" / doc["stored_name"]).unwrap()
        return {**doc, "text": text[:PREVIEW_CHARACTERS], "truncated": len(text) > PREVIEW_CHARACTERS}

    def jobs(self, project_id: str, *, trashed: bool = False, include_trashed: bool = False) -> list[dict[str, Any]]:
        self.project(project_id)
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM jobs WHERE project_id=? AND (? OR trashed=?) ORDER BY created DESC, rowid DESC",
                (project_id, int(include_trashed), int(trashed)),
            )
            return [{**dict(row), "diagnostics": json.loads(row["diagnostics"])} for row in rows]

    def trash_run(self, project_id: str, job_id: str, trashed: bool) -> dict[str, Any]:
        if self.project(project_id)["trashed"]:
            raise ValueError("Restore this project from Trash before changing its runs.")
        job = next((item for item in self.jobs(project_id, include_trashed=True) if item["id"] == job_id), None)
        if job is None:
            raise KeyError("Run not found")
        if job["state"] in ("RUNNING", "QUEUED"):
            raise ValueError("Wait for or cancel this run before deleting it.")
        with self.connect() as db:
            db.execute("UPDATE jobs SET trashed=? WHERE project_id=? AND id=?", (int(trashed), project_id, job_id))
        return {**job, "trashed": int(trashed)}

    def purge_run(self, project_id: str, job_id: str) -> None:  # noqa: PLR0912
        if self.project(project_id)["trashed"]:
            raise ValueError("Restore this project from Trash before permanently deleting runs.")
        job = next((item for item in self.jobs(project_id, include_trashed=True) if item["id"] == job_id), None)
        if job is None:
            raise KeyError("Run not found")
        if not job["trashed"]:
            raise ValueError("Move this run to Trash before permanently deleting it.")
        if job["state"] in ("RUNNING", "QUEUED"):
            raise ValueError("Wait for or cancel this run before permanently deleting it.")
        project_root = self.project_dir(project_id).resolve()
        if not project_root.is_relative_to(self.root.resolve()):
            raise ValueError("Project storage path is unsafe.")
        paths: list[tuple[Path, Path]] = []
        if job["run_dir"]:
            rel = Path(job["run_dir"])
            if rel.is_absolute() or rel.parent != Path("runs") or rel.name in {".", ".."}:
                raise ValueError("Run artifact path is unsafe.")
            if any(
                other["id"] != job_id and other["run_dir"] == job["run_dir"]
                for other in self.jobs(project_id, include_trashed=True)
            ):
                raise ValueError("Another run still refers to these artifacts; purge refused.")
            paths.append((project_root / rel, project_root))
        paths.extend(
            [
                (project_root / "job-inputs" / job_id, project_root),
                (self.root / "logs" / (job_id + ".log"), self.root.resolve()),
            ]
        )
        for path, root in paths:
            if (
                path.is_symlink()
                or any(parent.is_symlink() for parent in path.parents if parent != root.parent)
                or not path.resolve(strict=False).is_relative_to(root)
            ):
                raise ValueError("Run storage path is unsafe.")
        trash_root = self.root / "trash"
        if trash_root.is_symlink() or not trash_root.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Run purge staging path is unsafe.")
        trash_root.mkdir(exist_ok=True)
        staged = trash_root / ("purge-run-" + job_id + "-" + uuid.uuid4().hex)
        if not staged.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Run purge staging path is unsafe.")
        staged.mkdir()
        moved: list[tuple[Path, Path]] = []
        try:
            for i, (path, _) in enumerate(paths):
                if path.exists():
                    target = staged / str(i)
                    path.rename(target)
                    moved.append((target, path))
            with self.connect() as db:
                db.execute("DELETE FROM jobs WHERE project_id=? AND id=?", (project_id, job_id))
        except BaseException:
            for source, target in reversed(moved):
                source.rename(target)
            staged.rmdir()
            raise
        try:
            shutil.rmtree(staged)
        except OSError as exc:
            recovery = trash_root / ("purge-run-failed-" + job_id + "-" + uuid.uuid4().hex)
            staged.rename(recovery)
            raise OSError(f"Run metadata was purged; remaining files are safely staged at {recovery}: {exc}") from exc

    def update_job(
        self,
        job_id: str,
        *,
        state: str,
        stage: str,
        run_dir: str | None = None,
        diagnostics: list[dict[str, Any]] | None = None,
    ) -> None:
        finished = None if state in ("QUEUED", "RUNNING") else now()
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET state=?, stage=?, finished=?, run_dir=?, diagnostics=? WHERE id=? AND state != 'CANCELLED'",
                (state, stage, finished, run_dir, json.dumps(diagnostics or []), job_id),
            )

    def artifacts(self, project_id: str, job_id: str) -> tuple[Path, Envelope]:
        job = next((j for j in self.jobs(project_id) if j["id"] == job_id), None)
        if job is None or not job["run_dir"]:
            raise KeyError("This run has no published results yet")
        root = self.project_dir(project_id).resolve()
        directory = (root / job["run_dir"]).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("Run path is outside this project")
        return directory, Envelope.read(directory).unwrap()

    def tables(self, project_id: str) -> list[dict[str, Any]]:
        """Every CSV result table in this project, newest run first.

        The interactive workbench needs a table to open before it can show
        anything, and making the reader first find the run that produced one
        is how an exploration tool turns back into a filing cabinet. This is
        the front door: one list of everything there is to look at.

        A run whose directory has been moved, deleted or half-written is
        skipped rather than raising. One unreadable envelope should cost its
        own row, not the whole list.

        Each file is confirmed on disk rather than taken from the envelope's
        word. An envelope records what a run published, which is not the same
        claim as "this file is here now" -- and a picker whose entries fail
        when chosen is worse than a shorter picker.
        """
        found: list[dict[str, Any]] = []
        for job in self.jobs(project_id):
            if job["state"] not in ("DONE", "PARTIAL") or not job["run_dir"]:
                continue
            try:
                directory, envelope = self.artifacts(project_id, job["id"])
            except (KeyError, ValueError, OSError):
                continue
            found.extend(
                {
                    "job": job["id"],
                    "index": index,
                    "tool": job["tool"],
                    "path": artifact.path,
                    "description": artifact.description,
                    "created": job["created"],
                    "state": job["state"],
                }
                for index, artifact in enumerate(envelope.artifacts)
                if artifact.path.lower().endswith(".csv") and (directory / artifact.path).is_file()
            )
        return found

    def artifact(self, project_id: str, job_id: str, index: int) -> Path:
        directory, envelope = self.artifacts(project_id, job_id)
        if index < 0 or index >= len(envelope.artifacts):
            raise KeyError("Artifact not found")
        path = (directory / envelope.artifacts[index].path).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("Artifact is missing or outside its run directory")
        return path

    # ----------------------------------------------------------- saved views --

    def source_digest(self, project_id: str, job_id: str, index: int) -> str | None:
        """The SHA-256 of a view's source table, or None if it is not there.

        A view records this at save time so reopening can tell "the same
        numbers" from "the same filename". Returning None rather than raising
        keeps one dangling view from emptying the whole list.
        """
        try:
            path = self.artifact(project_id, job_id, index)
        except (KeyError, ValueError, OSError):
            return None
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError:
            return None
        return digest.hexdigest()

    def _view(self, row: sqlite3.Row, digests: dict[tuple[str, int], str | None] | None = None) -> dict[str, Any]:
        """One stored row as the API shape.

        ``job``/``index`` rather than the column names, so a view addresses its
        source with exactly the words :meth:`tables` uses. Two spellings for one
        address is how a lookup ends up silently pointed at nothing.
        """
        view = {
            "id": row["id"],
            "name": row["name"],
            "job": row["job_id"],
            "index": row["artifact_index"],
            "artifact_path": row["artifact_path"],
            "source_sha256": row["source_sha256"],
            "settings": json.loads(row["settings"]),
            "filters": json.loads(row["filters"]),
            "revision": row["revision"],
            "created": row["created"],
            "updated": row["updated"],
        }
        # Several views commonly share one source table, and each answer costs
        # a full read of the file. Listing a project's views hashes each
        # distinct source once rather than once per view.
        key = (str(row["job_id"]), int(row["artifact_index"]))
        if digests is None:
            digest = self.source_digest(row["project_id"], *key)
        else:
            if key not in digests:
                digests[key] = self.source_digest(row["project_id"], *key)
            digest = digests[key]
        view["source"] = describe_source(view, digest=digest)
        return view

    def views(self, project_id: str) -> list[dict[str, Any]]:
        """Every saved view in this project, most recently touched first."""
        self.project(project_id)
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM views WHERE project_id=? ORDER BY updated DESC, rowid DESC",
                (project_id,),
            ).fetchall()
        digests: dict[tuple[str, int], str | None] = {}
        return [self._view(row, digests) for row in rows]

    def view(self, project_id: str, view_id: str) -> dict[str, Any]:
        self.project(project_id)
        with self.connect() as db:
            row = db.execute("SELECT * FROM views WHERE project_id=? AND id=?", (project_id, view_id)).fetchone()
        if row is None:
            raise KeyError("View not found")
        return self._view(row)

    def save_view(self, project_id: str, body: ViewBody) -> dict[str, Any]:
        """Store a new view, refusing a source that is not a readable CSV table.

        Checked now rather than on reopening: a view saved against something
        that was never openable is a bad row in a picker, and the person who
        finds it is not the person who made it.
        """
        self.project(project_id)
        path = self.artifact(project_id, body.job, body.index)
        if path.suffix.lower() != ".csv":
            raise ValueError("A view is saved from a CSV result table.")
        digest = self.source_digest(project_id, body.job, body.index)
        if digest is None:
            raise ValueError("That result table could not be read.")
        _, envelope = self.artifacts(project_id, body.job)
        stamp = now()
        identifier = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM views WHERE project_id=?", (project_id,)).fetchone()[0]
            if count >= MAX_VIEWS_PER_PROJECT:
                raise ValueError(f"This project has reached the {MAX_VIEWS_PER_PROJECT}-view limit.")
            if db.execute("SELECT 1 FROM views WHERE project_id=? AND name=?", (project_id, body.name)).fetchone():
                raise ValueError(f"This project already has a view called “{body.name}”.")
            db.execute(
                "INSERT INTO views(id,project_id,name,job_id,artifact_index,artifact_path,source_sha256,"
                "settings,filters,revision,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    project_id,
                    body.name,
                    body.job,
                    body.index,
                    envelope.artifacts[body.index].path,
                    digest,
                    json.dumps(body.settings.model_dump()),
                    json.dumps([f.model_dump() for f in body.filters]),
                    1,
                    stamp,
                    stamp,
                ),
            )
        return self.view(project_id, identifier)

    def update_view(self, project_id: str, view_id: str, body: ViewBody) -> dict[str, Any]:
        """Replace a view's contents and count the revision.

        The revision is what a published chart's provenance names, so it has to
        move whenever anything a reader could see moves -- including the source,
        since a view may be re-pointed at a rerun of the same analysis.
        """
        existing = self.view(project_id, view_id)
        path = self.artifact(project_id, body.job, body.index)
        if path.suffix.lower() != ".csv":
            raise ValueError("A view is saved from a CSV result table.")
        digest = self.source_digest(project_id, body.job, body.index)
        if digest is None:
            raise ValueError("That result table could not be read.")
        _, envelope = self.artifacts(project_id, body.job)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            clash = db.execute(
                "SELECT 1 FROM views WHERE project_id=? AND name=? AND id != ?",
                (project_id, body.name, view_id),
            ).fetchone()
            if clash:
                raise ValueError(f"This project already has a view called “{body.name}”.")
            db.execute(
                "UPDATE views SET name=?, job_id=?, artifact_index=?, artifact_path=?, source_sha256=?, "
                "settings=?, filters=?, revision=?, updated=? WHERE project_id=? AND id=?",
                (
                    body.name,
                    body.job,
                    body.index,
                    envelope.artifacts[body.index].path,
                    digest,
                    json.dumps(body.settings.model_dump()),
                    json.dumps([f.model_dump() for f in body.filters]),
                    existing["revision"] + 1,
                    now(),
                    project_id,
                    view_id,
                ),
            )
        return self.view(project_id, view_id)

    def duplicate_view(self, project_id: str, view_id: str) -> dict[str, Any]:
        """Copy a view under a free name, so the original survives the next edit.

        Duplicating is the cheap way to try a variant without losing the thing
        that already works, which is the whole reason experimenting must not
        overwrite. The copy starts its own revision count: it is a new view,
        not another revision of the one it came from.
        """
        original = self.view(project_id, view_id)
        stamp = now()
        identifier = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM views WHERE project_id=?", (project_id,)).fetchone()[0]
            if count >= MAX_VIEWS_PER_PROJECT:
                raise ValueError(f"This project has reached the {MAX_VIEWS_PER_PROJECT}-view limit.")
            taken = {row[0] for row in db.execute("SELECT name FROM views WHERE project_id=?", (project_id,))}
            name = free_name(original["name"], taken)
            db.execute(
                "INSERT INTO views(id,project_id,name,job_id,artifact_index,artifact_path,source_sha256,"
                "settings,filters,revision,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    project_id,
                    name,
                    original["job"],
                    original["index"],
                    original["artifact_path"],
                    original["source_sha256"],
                    json.dumps(original["settings"]),
                    json.dumps(original["filters"]),
                    1,
                    stamp,
                    stamp,
                ),
            )
        return self.view(project_id, identifier)

    def delete_view(self, project_id: str, view_id: str) -> None:
        self.view(project_id, view_id)
        with self.connect() as db:
            db.execute("DELETE FROM views WHERE project_id=? AND id=?", (project_id, view_id))

    @staticmethod
    def _question(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "snapshot_id": row["snapshot_id"],
            "parser": row["parser"],
            "document_ids": json.loads(row["document_ids"]),
            "specification": json.loads(row["specification"]),
            "matching": json.loads(row["matching"] or "{}"),
            "revision": row["revision"],
            "created": row["created"],
            "updated": row["updated"],
        }

    def questions(self, project_id: str) -> list[dict[str, Any]]:
        """Every named research question, most recently touched first."""
        self.project(project_id)
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM questions WHERE project_id=? ORDER BY updated DESC, rowid DESC",
                (project_id,),
            ).fetchall()
        return [self._question(row) for row in rows]

    def question(self, project_id: str, question_id: str) -> dict[str, Any]:
        self.project(project_id)
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM questions WHERE project_id=? AND id=?", (project_id, question_id)
            ).fetchone()
        if row is None:
            raise KeyError("Saved question not found")
        return self._question(row)

    def _validate_question_documents(self, project_id: str, body: QuestionBody) -> None:
        available = {str(doc["id"]) for doc in self.documents(project_id)}
        missing = [identifier for identifier in body.document_ids if identifier not in available]
        if missing:
            raise ValueError("A document used by this question is no longer in the project.")
        selected = set(body.document_ids)
        filtered = body.specification.evidence_document_id
        if filtered is not None and filtered not in selected:
            raise ValueError("The evidence filter names a document outside the saved corpus.")

    def save_question(self, project_id: str, body: QuestionBody) -> dict[str, Any]:
        self._validate_question_documents(project_id, body)
        stamp = now()
        identifier = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM questions WHERE project_id=?", (project_id,)).fetchone()[0]
            if count >= MAX_QUESTIONS_PER_PROJECT:
                raise ValueError(f"This project has reached the {MAX_QUESTIONS_PER_PROJECT}-question limit.")
            if db.execute("SELECT 1 FROM questions WHERE project_id=? AND name=?", (project_id, body.name)).fetchone():
                raise ValueError(f"This project already has a question called “{body.name}”.")
            db.execute(
                "INSERT INTO questions(id,project_id,name,snapshot_id,parser,document_ids,specification,"
                "matching,revision,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    project_id,
                    body.name,
                    body.snapshot_id,
                    body.parser,
                    json.dumps(body.document_ids),
                    json.dumps(body.specification.model_dump()),
                    json.dumps(body.matching.model_dump()),
                    1,
                    stamp,
                    stamp,
                ),
            )
        return self.question(project_id, identifier)

    def update_question(self, project_id: str, question_id: str, body: QuestionBody) -> dict[str, Any]:
        existing = self.question(project_id, question_id)
        self._validate_question_documents(project_id, body)
        if body.expected_revision is not None and body.expected_revision != existing["revision"]:
            # Somebody else -- or this reader in another window -- has already
            # saved over what this edit started from. Replacing it would keep
            # one of the two answers and lose the other without saying which.
            raise ValueError(
                f"“{existing['name']}” has moved on to revision {existing['revision']} since this was opened. "
                "Reopen it to see the current version, or save this as a new question."
            )
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            clash = db.execute(
                "SELECT 1 FROM questions WHERE project_id=? AND name=? AND id != ?",
                (project_id, body.name, question_id),
            ).fetchone()
            if clash:
                raise ValueError(f"This project already has a question called “{body.name}”.")
            db.execute(
                "UPDATE questions SET name=?,snapshot_id=?,parser=?,document_ids=?,specification=?,"
                "matching=?,revision=?,updated=? WHERE project_id=? AND id=?",
                (
                    body.name,
                    body.snapshot_id,
                    body.parser,
                    json.dumps(body.document_ids),
                    json.dumps(body.specification.model_dump()),
                    json.dumps(body.matching.model_dump()),
                    existing["revision"] + 1,
                    now(),
                    project_id,
                    question_id,
                ),
            )
        return self.question(project_id, question_id)

    def duplicate_question(self, project_id: str, question_id: str) -> dict[str, Any]:
        original = self.question(project_id, question_id)
        stamp = now()
        identifier = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM questions WHERE project_id=?", (project_id,)).fetchone()[0]
            if count >= MAX_QUESTIONS_PER_PROJECT:
                raise ValueError(f"This project has reached the {MAX_QUESTIONS_PER_PROJECT}-question limit.")
            taken = {row[0] for row in db.execute("SELECT name FROM questions WHERE project_id=?", (project_id,))}
            name = free_name(str(original["name"]), taken)
            db.execute(
                "INSERT INTO questions(id,project_id,name,snapshot_id,parser,document_ids,specification,"
                "matching,revision,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    project_id,
                    name,
                    original["snapshot_id"],
                    original["parser"],
                    json.dumps(original["document_ids"]),
                    json.dumps(original["specification"]),
                    # A copy of a question is the same question: it has to be
                    # matched by the same rules, or the copy answers something
                    # else under a name that says "copy".
                    json.dumps(original["matching"]),
                    1,
                    stamp,
                    stamp,
                ),
            )
        return self.question(project_id, identifier)

    def delete_question(self, project_id: str, question_id: str) -> None:
        self.question(project_id, question_id)
        with self.connect() as db:
            db.execute("DELETE FROM questions WHERE project_id=? AND id=?", (project_id, question_id))
