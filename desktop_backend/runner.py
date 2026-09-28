"""Bounded desktop worker processes, with a frozen document selection per run."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import replace
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any
import uuid

from core.corpus_axis import axis_of
from core.profiler.plan import Plan
from core.profiler.registry import ParamSpec, ToolSpec, get_tool
from core.research.phrase import Tokenization
from core.result import Diagnostic, Result
from desktop_backend.catalog import CORPUS_TOOLS, desktop_spec
from desktop_backend.comparisons import CONTRAST_TOOL
from desktop_backend.fields import cleaning as project_cleaning, project_axis, project_events
from desktop_backend.live import as_parser
from desktop_backend.project_corpus import load_corpus, parse_cached, resolve_parser
from desktop_backend.selection import CorpusSelection, resolve_selection
from desktop_backend.store import Workspace, now
from desktop_backend.tables import TABLE_TOOLS, run_table

# Deliberately scoped to workflows with an exercised desktop contract.
DESKTOP_TOOLS = (
    *CORPUS_TOOLS,
    *TABLE_TOOLS,
)
# Research-workspace publishers are submitted by dedicated endpoints. Keeping
# them out of DESKTOP_TOOLS prevents the general analysis catalog from
# exposing an implementation detail as a second, confusing way to ask the
# same question.
PHRASE_DISTRIBUTION_SPEC = ToolSpec(
    name="phrase_distribution",
    capability_ids=("CAP-RESEARCH-01",),
    packet="research-workspace",
    description="A saved literal phrase question across time, document position, documents, and source evidence",
    requires_parse=True,
    parser_backend="config",
    input_kind="corpus",
    params=(
        ParamSpec("phrase", "str", None, True, "literal phrase to track"),
        ParamSpec("comparison", "str", "", False, "optional literal phrase to compare"),
        ParamSpec("case-sensitive", "bool", False, False, "match capitalization exactly"),
        ParamSpec("normalize", "bool", False, False, "fold dashes, quotes and accents before comparing"),
        ParamSpec("match-lemma", "bool", False, False, "count every inflection of the word"),
        ParamSpec("match-nominalization", "bool", False, False, "count the noun a verb turns into"),
        ParamSpec("position-bins", "int", 10, False, "sections per document", minimum=5, maximum=50),
        ParamSpec("evidence-year", "int", 0, False, "saved evidence year (0 for all)", minimum=0, maximum=9999),
        ParamSpec("evidence-document", "str", "", False, "saved evidence document ID (empty for all)"),
        ParamSpec("question-name", "str", None, True, "saved question name"),
        ParamSpec("question-id", "str", None, True, "saved question ID"),
        ParamSpec("question-revision", "int", 1, False, "saved question revision", minimum=1),
        ParamSpec("snapshot-id", "str", None, True, "interactive corpus snapshot ID"),
        # JSON arrays, because a plan parameter is a scalar. Empty means the
        # question recorded no tokens and the parser resolves its text again.
        ParamSpec("resolved-tokens", "str", "", False, "tokens the saved phrase was resolved into (JSON)"),
        ParamSpec("comparison-tokens", "str", "", False, "tokens the saved comparison was resolved into (JSON)"),
    ),
    outputs=(
        "phrase_question.csv",
        "phrase_summary.csv",
        "phrase_time.csv",
        "phrase_positions.csv",
        "phrase_documents.csv",
        "phrase_occurrences.csv",
    ),
    version="1",
)
QUESTION_PUBLISHER_SPECS = {PHRASE_DISTRIBUTION_SPEC.name: PHRASE_DISTRIBUTION_SPEC}
QUESTION_PUBLISHERS = frozenset(QUESTION_PUBLISHER_SPECS)


def _question_plan(tool: str, params: dict[str, Any]) -> Result[Plan]:
    """Validate one private research publisher without adding a public CLI."""
    return _private_plan(QUESTION_PUBLISHER_SPECS[tool], params)


def _private_plan(spec: ToolSpec, params: dict[str, Any]) -> Result[Plan]:
    """A one-tool plan for a tool the general catalog does not list (questions, comparisons)."""
    from core.profiler.plan import PlannedTool, validate_parameters

    tool = spec.name
    validated = validate_parameters(spec, params)
    if not validated.ok:
        return Result.failure(*validated.diagnostics)
    filled = validated.unwrap()
    return Result.success(Plan((PlannedTool(tool, filled, spec.requires_parse, spec.execution_phase),), {tool: filled}))


# Tools whose primary output is a picture, not a table of numbers. They get
# their own desktop page so the Analysis studio stays a question-driven list.
# Corpus viz (charts/wordcloud_gephi/shapes/narrative) + CSV-input viz siblings.
VISUALIZATION_TOOLS = frozenset(
    {"charts", "wordcloud_gephi", "shapes", "narrative", "table_charts", "table_wordcloud_gephi"}
)

# Every state a job row can hold. Declared rather than left implicit in the
# SQL below, because the desktop has to have a word for each one: until this
# existed only DONE had been given a word, and the rest were shown to the
# reader as the identifier in lower case.
JOB_STATES: tuple[str, ...] = (
    "QUEUED",
    "RUNNING",
    "DONE",
    "PARTIAL",
    "FAILED",
    "CANCELLED",
    "INTERRUPTED",
)

# The runner appends this frame to every run: it records which documents were
# read, not what the tool found. Anything choosing "the table this run
# produced" has to skip it, so it is named once here rather than spelled out
# as a literal at each of those places.
INPUT_MANIFEST = "desktop_inputs.csv"


def input_manifest(documents: list[dict[str, Any]]) -> Any:
    """The documents a run read, one row each, with a column per detail.

    Details are flattened into columns (Date, Speaker, Kind, Side ...) so an
    exported run says what each document was, not only which file it was. A
    detail named like one of the manifest's own columns is prefixed.
    """
    import pandas as pd

    rows = []
    for document in documents:
        row = {key: value for key, value in document.items() if key not in ("fields", "field_sources")}
        for name, value in (document.get("fields") or {}).items():
            row[f"Detail: {name}" if name in row else name] = value
        rows.append(row)
    return pd.DataFrame(rows)


def overlapping_documents(documents: list[dict[str, Any]]) -> list[Diagnostic]:
    """A warning when a run reads a book and chapters cut from it: every word would count twice."""
    ids = {str(document.get("id")) for document in documents}
    books = {str(document["derived_from"]) for document in documents if document.get("derived_from")} & ids
    if not books:
        return []
    names = [str(document["name"]) for document in documents if str(document.get("id")) in books]
    return [
        Diagnostic.warning(
            "CORPUS_OVERLAPPING_DOCUMENTS",
            f"This run read {', '.join(names[:3])} and the chapters cut from it, so their words count twice. "
            "Leave the whole book out of the selection, or move it to Trash.",
        )
    ]


# A warm worker that has had no job for this long is shut down to free memory.
WORKER_IDLE_SECONDS = 300.0
# How often the idle reaper checks.
WORKER_REAP_POLL_SECONDS = 30.0
WORKER_START_SECONDS = 90.0


def worker_command(root: Path) -> list[str]:
    args = ["--data-dir", str(root), "--worker"]
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    return [sys.executable, "-m", "desktop_backend.server", *args]


def worker_loop(root: Path) -> int:
    """Serve jobs from stdin, one id per line, until the parent disconnects.

    Keeps pandas/parsers loaded across jobs instead of paying the import and
    model load on every analysis. Startup errors surface on stderr; the
    protocol itself uses single-line JSON on stdout.
    """
    import contextlib

    print(json.dumps({"event": "ready"}), flush=True)
    for line in sys.stdin:
        job_id = line.strip()
        if not job_id:
            continue
        with contextlib.suppress(OSError, ValueError):
            log_path = Workspace(root).root / "logs" / f"{job_id}.log"
            if log_path.is_file():
                handle = log_path.open("ab")
                os.dup2(handle.fileno(), 2)
        code = run_job(root, job_id)
        print(json.dumps({"event": "done", "job": job_id, "returncode": code}), flush=True)
    return 0


def validate_desktop_mode(tool: str, params: dict[str, Any]) -> None:
    if tool == "sentiment_vader_anew" and params.get("analysis") == "both" and not params.get("anew-lexicon"):
        raise ValueError("Choose an ANEW lexicon file, or select VADER to use the included sentiment analysis.")
    if tool == "search" and params.get("mode") == "csv":
        raise ValueError("CSV search is a CLI workflow. Choose text or conll search for a desktop corpus.")
    if tool == "spellcheck" and params.get("correct"):
        raise ValueError("Desktop spellcheck reports suggestions only. Corrected copies require the CLI workflow.")


class Runner:
    """One analysis at a time, off the HTTP/UI thread; durable terminal states."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="desktop-job")
        self.lock = threading.RLock()
        self._worker: subprocess.Popen[str] | None = None
        self._worker_busy = False
        self._worker_job_id: str | None = None
        self._worker_last_used = time.monotonic()
        self._stopping = False
        self._reaper = threading.Thread(target=self._reap_idle_worker, daemon=True, name="desktop-worker-reaper")
        self._reaper.start()
        # The server is single-instance per workspace. A prior process that
        # was interrupted cannot truthfully leave its jobs marked running.
        with workspace.connect() as db:
            db.execute(
                "UPDATE jobs SET state='INTERRUPTED', stage='App closed before completion', finished=? "
                "WHERE state IN ('RUNNING','QUEUED')",
                (now(),),
            )

    # -- warm worker management ------------------------------------------------

    def _spawn_worker(self) -> subprocess.Popen[str] | None:
        """Start a warm worker; returns None if it fails to come up.

        Must be called WITHOUT the state lock: the ready-line wait blocks
        for the full cold import of the worker subprocess, and holding the
        lock here would stall cancel() and every other API operation.
        """
        log_dir = self.workspace.root / "logs"
        log_dir.mkdir(exist_ok=True)
        try:
            process = subprocess.Popen(  # noqa: S603
                worker_command(self.workspace.root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # Windows-only flag
            )
        except OSError:
            return None
        ready_lines: queue.Queue[str] = queue.Queue()
        threading.Thread(
            target=lambda: ready_lines.put(process.stdout.readline() if process.stdout else ""), daemon=True
        ).start()
        try:
            ready = ready_lines.get(timeout=WORKER_START_SECONDS)
        except queue.Empty:
            ready = ""
        if not ready or '"ready"' not in ready:
            process.kill()
            process.wait()
            _close_pipes(process)
            return None
        with self.lock:
            # A cancel may have run while we waited for the ready line.
            if self._stopping or self._worker is not None:
                process.kill()
                process.wait()
                _close_pipes(process)
                return self._worker
            self._worker = process
            self._worker_last_used = time.monotonic()
        return process

    def _dispatch(self, job_id: str) -> None:
        """Hand the job to the warm worker and wait for its completion line.

        The blocking pipe read happens OUTSIDE the shared state lock so
        cancel() can take the lock while an analysis runs. Interruption is
        cooperative: cancel() closes the pipes; a read on a closed pipe
        raises and the dispatch unwinds.
        """
        log_dir = self.workspace.root / "logs"
        log_dir.mkdir(exist_ok=True)
        try:
            self.workspace.update_job(job_id, state="RUNNING", stage="Starting the analysis engine")
            with self.lock, self.workspace.connect() as db:
                if db.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()[0] == "CANCELLED":
                    return
            # Spawn OUTSIDE the lock: the ready-line wait blocks for the
            # worker's cold import, and cancel() must stay responsive.
            with self.lock:
                worker = self._worker
            if worker is None:
                worker = self._spawn_worker()
            if worker is None:
                raise RuntimeError("The analysis engine could not be started.")
            with self.lock:
                if self._worker is not None and self._worker is not worker:
                    # Another dispatch spawned a worker while we spawned.
                    # Kill and close to satisfy resource finalization.
                    worker.kill()
                    worker.wait()
                    _close_pipes(worker)
                    worker = self._worker
                with self.workspace.connect() as db:
                    if db.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()[0] == "CANCELLED":
                        return
                if self._stopping:
                    raise RuntimeError("The analysis engine is shutting down.")
                assert worker.stdin is not None  # noqa: S101
                (log_dir / f"{job_id}.log").write_bytes(b"")
                self._worker_job_id = job_id
                worker.stdin.write(job_id + "\n")
                worker.stdin.flush()
                self._worker_busy = True
            # Blocking wait with no lock held; cancel() may close this pipe.
            assert worker.stdout is not None  # noqa: S101
            try:
                line = worker.stdout.readline()
            except (OSError, ValueError):
                line = ""
            with self.lock:
                self._worker_busy = False
                self._worker_job_id = None
                self._worker_last_used = time.monotonic()
            done: dict[str, Any] = {}
            with contextlib.suppress(ValueError):
                payload = json.loads(line)
                if isinstance(payload, dict):
                    done = payload
            if done.get("event") != "done" or done.get("job") != job_id:
                # Worker died (or was severed by cancel): reap and respawn
                # for the next submission. _stop_worker closes, waits and
                # kills the handle so no second live worker is left behind.
                with self.lock:
                    self._stop_worker()
                raise RuntimeError("The analysis engine stopped unexpectedly. See the job log and retry.")
            returncode = int(done.get("returncode", 1))
            with self.workspace.connect() as db:
                state = db.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
            if state in ("QUEUED", "RUNNING"):
                raise RuntimeError(f"Analysis process exited unexpectedly ({returncode}). See log {job_id}.log")
        except Exception as exc:
            with self.workspace.connect() as db:
                state = db.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
            if state not in ("QUEUED", "RUNNING"):
                return  # A terminal state (e.g. CANCELLED) outranks the dispatch failure.
            self.workspace.update_job(
                job_id,
                state="FAILED",
                stage="Analysis could not finish",
                diagnostics=[{"severity": "ERROR", "code": "DESKTOP_WORKER_FAILED", "message": str(exc)}],
            )

    def _reap_idle_worker(self) -> None:
        while not self._stopping:
            time.sleep(WORKER_REAP_POLL_SECONDS)
            with self.lock:
                if (
                    self._worker is not None
                    and not self._worker_busy
                    and time.monotonic() - self._worker_last_used > WORKER_IDLE_SECONDS
                ):
                    self._stop_worker()

    def _stop_worker(self) -> None:
        """Sever the worker.

        On Windows the kill MUST come before the pipe closes: a thread
        blocked in stdout.readline() is not woken by close() alone (it
        stays blocked until the writer process exits), but it returns
        promptly once the process dies. Kill first, then close.
        """
        worker, self._worker = self._worker, None
        if worker is None:
            return
        if worker.poll() is None:
            worker.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                worker.wait(timeout=10)
        _close_pipes(worker)

    def submit(  # noqa: PLR0912 -- validation plus immutable submission
        self,
        project_id: str,
        tool: str,
        params: dict[str, Any],
        parser: str,
        *,
        selection: CorpusSelection | None = None,
    ) -> dict[str, Any]:
        from core.profiler.plan import Plan, PlannedTool, build_plan, validate_parameters
        from core.result import Result
        from desktop_backend.glance import GLANCE_TOOL, current_key, glance_plan

        glance = tool == GLANCE_TOOL
        if tool not in DESKTOP_TOOLS and tool not in QUESTION_PUBLISHERS and not glance:
            raise ValueError("This analysis is not available in the desktop app.")
        if self._stopping:
            raise ValueError("The app is closing. Reopen it before starting an analysis.")
        if self.workspace.project(project_id)["archived"]:
            raise ValueError("Restore this project from the archive before starting an analysis.")
        # The same list live analysis turns into the engine's Literal, so a
        # third parser cannot become submittable without becoming warmable.
        as_parser(parser)
        if tool == "sentiment_vader_anew":
            params = {"analysis": "vader", **params}
        validate_desktop_mode(tool, params)
        if tool in TABLE_TOOLS:
            if selection is not None:
                raise ValueError("Document selections apply to corpus analyses. This workflow reads a CSV table.")
            validated = validate_parameters(desktop_spec(TABLE_TOOLS[tool]), params)
            if not validated.ok:
                raise ValueError("; ".join(d.message for d in validated.diagnostics))
            plan = Result.success(Plan((PlannedTool(tool, validated.unwrap(), False, 2),), {tool: validated.unwrap()}))
        elif tool in QUESTION_PUBLISHERS:
            plan = _question_plan(tool, params)
        elif glance:
            if selection is not None:
                raise ValueError("Corpus at a glance reads the whole corpus.")
            plan = glance_plan()
        else:
            plan = build_plan([tool], {tool: params})
        if not plan.ok:
            raise ValueError("; ".join(d.message for d in plan.diagnostics))
        documents = self.workspace.documents(project_id)
        if not documents and tool not in TABLE_TOOLS:
            raise ValueError("Import at least one text document before running an analysis.")
        if tool not in TABLE_TOOLS:
            documents = resolve_selection(documents, selection)
            scope = selection or CorpusSelection()
            documents = [
                {
                    **document,
                    "selection_mode": "explicit" if scope.document_ids is not None else "all",
                    "date_from": scope.date_from,
                    "date_to": scope.date_to,
                    "include_undated": scope.include_undated,
                    "order_from": scope.order_from,
                    "order_to": scope.order_to,
                    "include_unordered": scope.include_unordered,
                    # As text, so the input manifest says in one cell what
                    # narrowed the run ("Speaker = Barack Obama").
                    "detail_filter": "; ".join(
                        f"{name} = {' or '.join(values)}" for name, values in (scope.fields or {}).items()
                    ),
                }
                for document in documents
            ]
        if plan.unwrap().needs_parse:
            import importlib.util

            if importlib.util.find_spec(parser) is None:
                raise ValueError(f"The {parser} package is not installed. Open Setup for installation instructions.")
            if parser == "spacy" and importlib.util.find_spec("en_core_web_sm") is None:
                raise ValueError("The English spaCy model is missing. Open Setup for installation instructions.")
        job_id = uuid.uuid4().hex
        frozen_params = plan.unwrap().params
        spec = TABLE_TOOLS.get(tool) or QUESTION_PUBLISHER_SPECS.get(tool) or get_tool(tool)
        resources = []
        if spec is not None and not glance:
            for parameter in spec.params:
                value = frozen_params[tool].get(parameter.name)
                if parameter.type != "path" or value is None:
                    continue
                from core.io.reader import hash_file
                from desktop_backend.store import MAX_FILE_BYTES

                source = Path(str(value))
                with source.open("rb") as handle:
                    data = handle.read(MAX_FILE_BYTES + 1)
                if len(data) > MAX_FILE_BYTES:
                    raise ValueError(f"{parameter.name} exceeds 20 MB.")
                directory = self.workspace.project_dir(project_id) / "job-inputs" / job_id
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / (parameter.name + source.suffix)
                target.write_bytes(data)
                frozen_params[tool][parameter.name] = str(target)
                resources.append({"path": str(target), "source": str(source), "sha256": hash_file(target)})
        request = {
            "documents": documents,
            "params": frozen_params,
            "parser": parser,
            "resources": resources,
            "selection": selection.model_dump() if selection is not None else None,
            "axis": project_axis(self.workspace, project_id),
            # The project's dated events, frozen like the documents: a run's
            # figures draw the stoplines the reader set at submission time
            # (SHOWCASE_FIGURES_PLAN section 4).
            "events": project_events(self.workspace, project_id),
            # What each project leaves out of the text the tools read (stage
            # directions), frozen like the documents: a run records what it
            # read (plan 5.2).
            "text_cleaning": {
                home: project_cleaning(self.workspace, home).model_dump()
                for home in sorted({str(document.get("project_id") or project_id) for document in documents})
            },
        }
        if glance:
            # The cache key: the documents' contents and the recipe version.
            request["glance_key"] = current_key(self.workspace, project_id)
        with self.workspace.connect() as db:
            db.execute(
                "INSERT INTO jobs (id, project_id, tool, state, stage, created, request) "
                "VALUES (?, ?, ?, 'QUEUED', 'Waiting for an analysis slot', ?, ?)",
                (job_id, project_id, tool, now(), json.dumps(request)),
            )
        self.pool.submit(self._dispatch, job_id)
        return next(j for j in self.workspace.jobs(project_id) if j["id"] == job_id)

    def submit_contrast(
        self, project_id: str, definition: Any, parser: str, *, name: str = "", comparison_id: str = ""
    ) -> dict[str, Any]:
        """Queue a comparison: every side's documents, each read from its own project.

        The job belongs to *project_id* (the comparison's home), so its results
        are an ordinary run there. The documents are frozen now, as for every
        run; a side's project that is missing or trashed refuses here, with the
        side's name, rather than failing later in the worker.
        """
        import importlib.util

        from desktop_backend.comparisons import CONTRAST_SPEC, contrast_params, side_documents

        if self._stopping:
            raise ValueError("The app is closing. Reopen it before starting a comparison.")
        if self.workspace.project(project_id)["archived"]:
            raise ValueError("Restore this project from the archive before starting a comparison.")
        as_parser(parser)
        if importlib.util.find_spec(parser) is None:
            raise ValueError(f"The {parser} package is not installed. Open Setup for installation instructions.")
        documents = side_documents(self.workspace, definition)
        params = contrast_params(definition, name, comparison_id)
        plan = _private_plan(CONTRAST_SPEC, params)
        if not plan.ok:
            raise ValueError("; ".join(d.message for d in plan.diagnostics))
        request = {
            "documents": documents,
            "params": plan.unwrap().params,
            "parser": parser,
            "resources": [],
            "selection": None,
            "comparison": definition.model_dump(),
            "axis": project_axis(self.workspace, project_id),
        }
        job_id = uuid.uuid4().hex
        with self.workspace.connect() as db:
            db.execute(
                "INSERT INTO jobs (id, project_id, tool, state, stage, created, request) "
                "VALUES (?, ?, ?, 'QUEUED', 'Waiting for an analysis slot', ?, ?)",
                (job_id, project_id, CONTRAST_SPEC.name, now(), json.dumps(request)),
            )
        self.pool.submit(self._dispatch, job_id)
        return next(j for j in self.workspace.jobs(project_id) if j["id"] == job_id)

    def submit_notebook(self, project_id: str, notebook_id: str, parser: str) -> dict[str, Any]:
        """Queue "Run and save": every cell of the notebook's saved revision, in a fresh namespace.

        Queued like any analysis, so it waits its turn rather than competing
        with a run already using the parser. The request freezes the notebook's
        code and the project's documents as they are now, as every run does.
        """
        from desktop_backend.notebooks import NOTEBOOK_TOOL

        if self._stopping:
            raise ValueError("The app is closing. Reopen it before running a notebook.")
        if self.workspace.project(project_id)["archived"]:
            raise ValueError("Restore this project from the archive before running a notebook.")
        as_parser(parser)
        notebook = self.workspace.notebook(project_id, notebook_id)
        # Pasted code runs when its reader chooses to run it. A saved run is a
        # record that says what produced a figure, so it waits until every
        # pasted cell has been marked as read.
        for position, cell in enumerate(notebook["content"]["cells"], 1):
            if cell["cell_type"] == "code" and cell.get("metadata", {}).get("nlpsuite", {}).get("origin") == "pasted":
                raise ValueError(
                    f"Cell {position} was pasted in from outside the app and is not marked as read. "
                    "Read it, then press “I have read it”, before saving a run."
                )
        documents = resolve_selection(self.workspace.documents(project_id), None)
        request = {
            "notebook": notebook["content"],
            "notebook_id": notebook_id,
            "name": notebook["name"],
            "revision": notebook["revision"],
            "documents": documents,
            "parser": parser,
            "params": {},
            "resources": [],
        }
        job_id = uuid.uuid4().hex
        with self.workspace.connect() as db:
            db.execute(
                "INSERT INTO jobs (id, project_id, tool, state, stage, created, request) "
                "VALUES (?, ?, ?, 'QUEUED', 'Waiting for an analysis slot', ?, ?)",
                (job_id, project_id, NOTEBOOK_TOOL, now(), json.dumps(request)),
            )
        self.pool.submit(self._dispatch, job_id)
        return next(j for j in self.workspace.jobs(project_id) if j["id"] == job_id)

    def cancel(self, project_id: str, job_id: str) -> dict[str, Any]:
        """Cancel only a job belonging to this project; terminal results are retained.

        Safe to call while an analysis is running: the dispatch thread no
        longer holds the state lock across its blocking pipe read. Severing
        the worker's pipes interrupts that read promptly.
        """
        with self.lock:
            job = next((j for j in self.workspace.jobs(project_id) if j["id"] == job_id), None)
            if job is None:
                raise KeyError("Run not found")
            with self.workspace.connect() as db:
                changed = db.execute(
                    "UPDATE jobs SET state='CANCELLED', stage='Cancelled by user', finished=? "
                    "WHERE id=? AND state IN ('QUEUED','RUNNING')",
                    (now(), job_id),
                ).rowcount
            if changed and self._worker_busy and self._worker_job_id == job_id:
                # The worker is mid-job; sever it so the analysis stops promptly.
                # The next submission spawns a fresh warm worker.
                self._stop_worker()
                self._worker_busy = False
                self._worker_job_id = None
            return next(j for j in self.workspace.jobs(project_id) if j["id"] == job_id)

    def close(self) -> None:
        # Pending/running work finishes before a graceful server shutdown.
        # _stopping goes up only AFTER the pool drains: dispatch and the
        # idle reaper consult it, and queued jobs must still be able to
        # run to completion during the graceful wait.
        self.pool.shutdown(wait=True)
        self._stopping = True
        with self.lock:
            self._stop_worker()


def _close_pipes(process: subprocess.Popen[str]) -> None:
    """Close a worker's stdin/stdout, tolerating already-closed handles."""
    if process.stdin is not None:
        with contextlib.suppress(OSError, ValueError):
            process.stdin.close()
    if process.stdout is not None:
        with contextlib.suppress(OSError, ValueError):
            process.stdout.close()


def run_mallet(
    workspace: Workspace,
    job: dict[str, Any],
    params: dict[str, Any],
    documents: list[Any],
    project_dir: Path,
) -> int:
    """The MALLET LDA workflow (lda_mallet), staged outside core/ (R3).

    MALLET is a Java binary that reads a directory of .txt files and writes
    its own scratch outputs. Write custody says analysis modules write
    nothing, so the staging copies and MALLET's own files live in temporary
    directories and the run directory receives only the parsed tables,
    through the OutputWriter like every other result.

    Not a live tool (see desktop/src/live.ts): a train-topics call is a job.
    """
    import tempfile

    from core.analysis.mallet import parse_doc_topics, train_topics
    from core.io.writer import OutputWriter

    tool = "lda_mallet"
    with tempfile.TemporaryDirectory(prefix="mallet-in-") as staged:
        stage = Path(staged)
        for position, doc in enumerate(documents, 1):
            name = getattr(doc, "label", "") or Path(doc.path).name
            (stage / f"{position:04d}_{name}.txt").write_text(doc.text, encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="mallet-out-") as out:
            keys = train_topics(stage, out, n_topics=int(params.get("topics", 10)), seed=int(params.get("seed", 42)))
            if not keys.ok:
                workspace.update_job(
                    job["id"],
                    state="FAILED",
                    stage="MALLET could not run",
                    diagnostics=[d.to_dict() for d in keys.diagnostics],
                )
                return 1
            dominant = parse_doc_topics(Path(out) / "mallet_doc_topics.txt")
            inputs = tuple(doc.path for doc in documents)
            writer = OutputWriter(project_dir / "runs", tool=tool, params=params, inputs=inputs)
            try:
                written = writer.write_table(keys.unwrap(), "topics.csv", kind="table", description="MALLET topic keys")
                if not written.ok:
                    raise ValueError("; ".join(d.message for d in written.diagnostics))
                if dominant.ok:
                    written = writer.write_table(
                        dominant.unwrap(),
                        "topics_dominant.csv",
                        kind="table",
                        description="MALLET document-topic composition",
                    )
                    if not written.ok:
                        raise ValueError("; ".join(d.message for d in written.diagnostics))
                writer.add_diagnostics(*keys.diagnostics, *dominant.diagnostics)
                envelope = writer.finalize()
                if not envelope.ok:
                    raise ValueError("; ".join(d.message for d in envelope.diagnostics))
            except Exception:
                writer.abandon()
                raise
            workspace.update_job(
                job["id"],
                state="DONE",
                stage="Results ready",
                run_dir=writer.run_dir.relative_to(project_dir).as_posix(),
                diagnostics=[d.to_dict() for d in (*keys.diagnostics, *dominant.diagnostics)],
            )
            return 0


def run_job(root: Path, job_id: str) -> int:  # noqa: PLR0912 -- staged worker lifecycle
    """Entrypoint used by both Python development and the frozen executable."""
    from core.io.reader import hash_file
    from core.models.vector_cache import VectorCache
    from core.profiler.batch import BatchRequest, write_batch
    from core.profiler.executor import BatchResult, execute
    from core.profiler.plan import build_plan

    workspace = Workspace(root)
    try:
        with workspace.connect() as db:
            job = dict(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
        request = json.loads(job["request"])
        for resource in request.get("resources", []):
            if hash_file(Path(resource["path"])) != resource["sha256"]:
                raise ValueError("An analysis resource changed after submission. Submit a new run.")
        if job["tool"] in TABLE_TOOLS:
            return run_table(workspace, job, request["params"][job["tool"]])
        from desktop_backend.notebooks import NOTEBOOK_TOOL, run_notebook

        if job["tool"] == NOTEBOOK_TOOL:
            return run_notebook(workspace, job, request)
        project_dir = workspace.project_dir(job["project_id"])
        workspace.update_job(job_id, state="RUNNING", stage="Reading document snapshot")
        corpus, diagnostics = load_corpus(
            workspace, job["project_id"], request["documents"], text_cleaning=request.get("text_cleaning")
        )
        diagnostics.extend(overlapping_documents(request["documents"]))
        if job["tool"] == "lda_mallet":
            return run_mallet(workspace, job, request["params"][job["tool"]], list(corpus.docs), project_dir)
        from desktop_backend.glance import GLANCE_TOOL

        glance = job["tool"] == GLANCE_TOOL
        if job["tool"] in QUESTION_PUBLISHERS:
            plan = _question_plan(job["tool"], request["params"][job["tool"]]).unwrap()
        elif job["tool"] == CONTRAST_TOOL:
            from desktop_backend.comparisons import CONTRAST_SPEC

            plan = _private_plan(CONTRAST_SPEC, request["params"][job["tool"]]).unwrap()
        elif glance:
            plan = build_plan(list(request["params"]), request["params"]).unwrap()
        else:
            plan = build_plan([job["tool"]], request["params"]).unwrap()
        table = None
        parse_diags: tuple[Diagnostic, ...] = ()
        # Set with the table, because it belongs to it: a question about words
        # has to be split by the parser that produced the words it searches.
        tokenizer: Tokenization | None = None
        if plan.needs_parse:
            workspace.update_job(job_id, state="RUNNING", stage="Parsing English documents")
            from core.io.reader import ordered_fingerprint
            from desktop_backend.live import annotation_key, parser_identity

            # Same policy as the CLI: a backend whose model is missing or
            # damaged does not fail the job while another installed backend
            # can parse. The substitution rides along in the job diagnostics.
            pipeline = resolve_parser(request["parser"])
            if pipeline.value is None:
                workspace.update_job(
                    job_id,
                    state="FAILED",
                    stage="Parser setup required",
                    diagnostics=[d.to_dict() for d in pipeline.diagnostics],
                )
                return 1
            resolved_pipeline = pipeline.unwrap()
            if job["tool"] in QUESTION_PUBLISHERS:
                expected = str(request["params"][job["tool"]].get("snapshot-id", ""))
                # Order-sensitive, to match the parse key a question cites.
                current = annotation_key(ordered_fingerprint(corpus.docs), parser_identity(resolved_pipeline))
                if expected != current:
                    raise ValueError(
                        "This saved question belongs to an older document or parser snapshot. "
                        "Open it in Explore, review the refreshed answer, and save it again before publishing."
                    )
            # Through the same cache the Interactive page keeps, so the second
            # run over these documents reads the parse instead of redoing it.
            parsed = parse_cached(
                workspace.root,
                corpus,
                resolved_pipeline,
                stage=lambda text: workspace.update_job(job_id, state="RUNNING", stage=text),
            )
            table = parsed.table
            parse_diags = (*pipeline.diagnostics, *parsed.diagnostics)
            tokenizer = parsed.tokenizer
        workspace.update_job(job_id, state="RUNNING", stage="Computing analysis")
        # The axis the project chose when the run was submitted; runs queued
        # before axes existed choose from what the documents carry, as then.
        chosen = request.get("axis") or {}
        batch = execute(
            plan,
            corpus=corpus,
            table=table,
            parse_diagnostics=parse_diags,
            tokenizer=tokenizer,
            axis=axis_of(corpus, chosen.get("kind"), chosen.get("noun", "")),
            vectors=VectorCache(workspace.root),
        )
        batch = BatchResult(
            tuple(
                replace(
                    o,
                    frames={**o.frames, INPUT_MANIFEST: input_manifest(request["documents"])} if o.frames else {},
                    ok=o.ok and not any(d.severity.value == "ERROR" for d in diagnostics),
                    diagnostics=(*o.diagnostics, *diagnostics),
                )
                for o in batch.outcomes
            )
        )
        workspace.update_job(job_id, state="RUNNING", stage="Publishing results and provenance")
        # A finished run carries its publication figures (figures/*.png,
        # *.svg). NLP_SUITE_RUN_FIGURES=0 turns that off.
        figures = os.environ.get("NLP_SUITE_RUN_FIGURES", "1") != "0"
        # The dated events frozen into the request, as (position, name), for
        # the run's dated figures to draw as stoplines.
        events = tuple(
            (float(event["position"]), str(event["name"]))
            for event in request.get("events", [])
            if isinstance(event, dict) and "position" in event
        )
        selection_names = tuple(tool.name for tool in plan.tools) if glance else (job["tool"],)
        report = write_batch(
            project_dir / "runs",
            BatchRequest(selection_names, plan.params, batch, corpus, figures=figures, events=events),
        )
        outcome = batch.outcomes[0]
        if report.value is None:
            raise ValueError("; ".join(d.message for d in report.diagnostics))
        if glance:
            # One job, one parse, seven child runs: the parent batch is the result.
            every = all(item.ok for item in batch.outcomes)
            workspace.update_job(
                job_id,
                state="DONE" if every else "PARTIAL",
                stage="Results ready" if every else "Some analyses need review",
                run_dir=report.unwrap().run_dir.relative_to(project_dir).as_posix(),
                diagnostics=[d.to_dict() for item in batch.outcomes for d in item.diagnostics],
            )
            return 0 if every else 1
        child = report.unwrap().child_dirs.get(job["tool"])
        state = "DONE" if outcome.ok else "PARTIAL" if child is not None else "FAILED"
        workspace.update_job(
            job_id,
            state=state,
            stage="Results ready" if outcome.ok else "Review diagnostics",
            run_dir=child.relative_to(project_dir).as_posix() if child else None,
            diagnostics=[d.to_dict() for d in outcome.diagnostics],
        )
        return 0 if outcome.ok else 1
    except Exception as exc:
        workspace.update_job(
            job_id,
            state="FAILED",
            stage="Analysis failed",
            diagnostics=[{"severity": "ERROR", "code": "DESKTOP_ANALYSIS_FAILED", "message": str(exc)}],
        )
        return 1
