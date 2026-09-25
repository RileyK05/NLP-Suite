"""Loopback-only authenticated desktop API; also serves the production web UI."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager, suppress
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import secrets
import socket
import sys
import tempfile
import threading
from typing import Any, Literal
import uuid
import zipfile

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from core.file_ops.converter import supported_suffixes
from core.profiler.labels import param_label, tool_label
from core.result import Diagnostic, Result
from desktop_backend.archives import MAX_ARCHIVE_BYTES, backup, restore
from desktop_backend.catalog import desktop_spec
from desktop_backend.environment import availability, components, inventory
from desktop_backend.live import Bench, Session, StaleSnapshot, as_parser, as_table
from desktop_backend.live_panels import draw_live_panel, live_panel, live_panels_offered, prepare_live_panel
from desktop_backend.models import ModelDownloads
from desktop_backend.panels import BundleBody, PanelBody, StaticPanelBody, panel_failure
from desktop_backend.previews import MAX_PREVIEW_BYTES, PREVIEW_CSP, PreviewTickets
from desktop_backend.questions import QuestionBody
from desktop_backend.runner import (
    DESKTOP_TOOLS,
    INPUT_MANIFEST,
    VISUALIZATION_TOOLS,
    Runner,
    run_job,
    worker_loop,
)
from desktop_backend.schemas import AnalyseRequest, AnalyseResponse
from desktop_backend.selection import CorpusSelection, document_metadata, resolve_selection
from desktop_backend.store import MAX_DOCUMENTS, MAX_FILE_BYTES, Workspace
from desktop_backend.tables import TABLE_TOOLS
from desktop_backend.views import ViewBody, ViewSettings, chart_contract, publication_gaps

PREVIEW_ROWS = 500
# Artifacts the desktop app can render inline (HTML charts, wordclouds,
# static exports, raster images). Anything else stays download-only.
VIEWABLE_ARTIFACTS = {
    ".html": "text/html",
    ".htm": "text/html",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/plain",
    ".kml": "application/vnd.google-earth.kml+xml",
}


class ProjectBody(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class ArchiveBody(BaseModel):
    archived: bool


class RunBody(BaseModel):
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    parser: str = "spacy"
    selection: CorpusSelection | None = None


class GlanceBody(BaseModel):
    parser: str = "spacy"


class CompareBody(BaseModel):
    """Two finished runs to put side by side (the graded backend comparisons)."""

    expected: str
    actual: str


class WarmBody(BaseModel):
    """Which documents to bring up for live questioning, and with what parser."""

    selection: CorpusSelection | None = None
    parser: str = "spacy"


# The live-analysis envelope is defined once, in desktop_backend/schemas.py,
# and rendered into desktop/src/contract.ts (R-C6). This alias keeps the route
# signature readable while making the contract the validator: a request
# without a request_id (R-C2) or with an unknown field is refused here.
AnalyseBody = AnalyseRequest


class PhraseBody(BaseModel):
    """A phrase question plus paging/filtering of its occurrence evidence."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=300)
    #: Which loaded corpus the caller believes it is asking about. Empty means
    #: "whatever is loaded", which is only safe for a first question; paging,
    #: drilling in and comparing all cite the snapshot their answer came from
    #: so a reload underneath them is refused rather than silently answered.
    snapshot_id: str = Field(default="", max_length=128)
    case_sensitive: bool = False
    #: Fold typography (the several dashes, curly quotes, accents) on both
    #: sides before comparing.
    normalize: bool = False
    #: Count every inflection the corpus lemmatises to the same word.
    match_lemma: bool = False
    #: Count the noun a verb turns into, and the verb a noun came from.
    match_nominalization: bool = False
    position_bins: int = Field(default=10, ge=5, le=50)
    evidence_offset: int = Field(default=0, ge=0)
    evidence_limit: int = Field(default=100, ge=1, le=250)
    evidence_year: int | None = Field(default=None, ge=1, le=9999)
    evidence_document_id: str | None = Field(default=None, max_length=128)
    #: The position range a brush selected, in relative document coordinates.
    evidence_position_start: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_position_end: float | None = Field(default=None, ge=0.0, le=1.0)


class LivePanelBody(BaseModel):
    """One figure of a remembered live answer, with the reader's settings."""

    model_config = ConfigDict(extra="forbid")

    answer_id: str = Field(min_length=1, max_length=64)
    panel: str = Field(min_length=1, max_length=120)
    params: dict[str, Any] = Field(default_factory=dict)


class LiveStaticBody(LivePanelBody):
    """A live figure's publication version: the same figure, as an image."""

    format: Literal["png", "svg", "pdf"] = "png"
    dpi: int = Field(default=200, ge=72, le=400)
    notes: bool = False


def _figure_response(image: Result[bytes], fmt: str, name: str) -> Response:
    """Image bytes, or the refusal in the shape every panel refusal takes.

    A failure is a 200 with ``ok: False`` and diagnostics, never an HTTP
    error, so the app has one way to show why a figure did not draw.
    """
    from desktop_backend.panels import MEDIA_TYPES

    if image.value is None:
        return JSONResponse(panel_failure(list(image.diagnostics)))
    warnings = [d.message for d in image.diagnostics if d.severity.value == "WARNING"]
    return Response(
        content=image.unwrap(),
        media_type=MEDIA_TYPES[fmt],
        headers={
            "Content-Disposition": f'inline; filename="{name}.{fmt}"',
            # Drawn problems (overprinted text) travel with the picture, so
            # the app can say so without a second request.
            "X-Figure-Warnings": str(len(warnings)),
        },
    )


class VisualizeBody(BaseModel):
    tool: str
    index: int
    params: dict[str, Any] = Field(default_factory=dict)


class PathsBody(BaseModel):
    paths: list[str] = Field(max_length=2000)


#: What a response carries when there is nothing to read: no rows, or a run
#: that failed. Present with the same keys either way, so the interface has
#: one shape to render rather than a reading that is sometimes simply absent.
_NO_READING: dict[str, Any] = {
    "headline": "",
    "observations": [],
    "cautions": [],
    "recommended_charts": [],
    "figures": [],
    "table_first": "",
}
#: Past this a shortlist stops being short.
READING_CHARTS = 4


def reading_of(frame: Any, tool: str) -> dict[str, Any]:
    """A table read back in words, plus the charts worth drawing from it.

    Shared by the finished-run endpoint and the live bench because they are
    answering the same question about the same rows, and the pair of them
    drifting is the defect this codebase keeps finding in other shapes. The
    live bench had no reading at all: it drew whichever column happened to be
    first against whichever measure happened to be first, which over 87 dated
    speeches is a ranking of how long each speech is.

    Never raises. A reading is a convenience on top of a result; a result that
    cannot be described is still a result, and still charts.
    """
    from core.insight.readout import readout
    from core.insight.recommend import recommend_charts
    from core.viz.recipes import recipe_for

    recipe = recipe_for(tool)
    try:
        reading = readout(frame, tool=tool)
        # A tool with its own figures, or whose result is read as a table,
        # is offered those -- never generic column pairings. Those pairings
        # see only column types, and on a table of word pairs per speech
        # they proposed the mean of 9,407 unrelated pairs over time.
        charts = (
            [] if recipe.covered and not recipe.generic else recommend_charts(frame, tool=tool, limit=READING_CHARTS)
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return dict(_NO_READING)
    columns = {str(column) for column in frame.columns}
    return {
        "headline": reading.headline,
        "observations": list(reading.observations),
        "cautions": list(reading.cautions),
        # The figures this table can feed, by the question each answers. A
        # panel whose columns are not in this table belongs to another of the
        # run's tables and is not offered here.
        "figures": [
            {"panel": panel.name, "title": panel.title, "question": panel.question, "shape": panel.shape}
            for panel in recipe.panels
            if set(panel.requires) <= columns
        ],
        "table_first": recipe.table_first,
        "recommended_charts": [
            {
                "kind": r.spec.kind,
                "x": r.spec.x,
                "y": r.spec.y,
                # Carried so that pressing the button draws the chart that was
                # recommended. Without them a timeline opens summed and cut to
                # its twenty largest years, which is a different chart under
                # the recommendation's name.
                "agg": r.spec.agg or "",
                "top_n": r.spec.top_n,
                "question": r.question,
                "why": r.why,
            }
            for r in charts
        ],
    }


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    """Expose the frozen scope without leaking resource paths or the full request."""
    request = json.loads(job["request"])
    selected = request.get("selection") or {}
    scope = (
        None
        if job["tool"] in TABLE_TOOLS
        else {
            "document_count": len(request.get("documents", [])),
            "date_from": selected.get("date_from"),
            "date_to": selected.get("date_to"),
            "include_undated": selected.get("include_undated", False),
            "explicit_documents": selected.get("document_ids") is not None,
        }
    )
    return {**{key: value for key, value in job.items() if key != "request"}, "scope": scope}


def public_view(view: dict[str, Any]) -> dict[str, Any]:
    """A stored view plus what publishing it would not reproduce.

    Carried on the record itself so a list of saved views can be honest about
    each one without the interface deriving it a second time.
    """
    return {**view, "publication": publication_gaps(ViewSettings(**view["settings"]))}


def public_document(document: dict[str, Any]) -> dict[str, Any]:
    """Document metadata for the UI; storage filenames stay server-side."""
    return {
        key: value for key, value in document_metadata(document).items() if key not in {"stored_name", "source_name"}
    }


def create_app(workspace: Workspace, token: str, frontend: Path | None = None) -> FastAPI:
    runner = Runner(workspace)

    downloads = ModelDownloads()

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        yield
        runner.close()
        downloads.close()

    app = FastAPI(title="NLP Suite Desktop", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.runner = runner
    app.state.downloads = downloads
    previews = PreviewTickets()
    origins = ["http://tauri.localhost", "tauri://localhost", "http://127.0.0.1:1420", "http://localhost:1420"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        # A publication figure says how many drawn problems it has in a
        # header; the app reads it across the origin boundary.
        expose_headers=["X-Figure-Warnings", "Content-Disposition"],
    )

    async def authorize(request: Request) -> None:
        supplied = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if not secrets.compare_digest(supplied, token):
            raise HTTPException(401, "This workspace connection has expired. Reopen NLP Suite.")
        origin = request.headers.get("origin")
        if origin and origin not in [*origins, str(request.base_url).rstrip("/")]:
            raise HTTPException(403, "Untrusted request origin")

    @app.middleware("http")
    async def headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        # Defend against browser DNS-rebinding requests to the loopback server.
        hostname = request.url.hostname
        if hostname not in ("127.0.0.1", "localhost", "testserver"):
            return JSONResponse({"detail": "Invalid local host"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers.setdefault(
            "Content-Security-Policy",
            (
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: blob:; connect-src 'self'; frame-src 'self'; "
                "frame-ancestors 'none'; base-uri 'none'; object-src 'none'"
            ),
        )
        return response

    @app.exception_handler(StaleSnapshot)
    async def stale_snapshot(request: Request, exc: StaleSnapshot) -> JSONResponse:
        """409, not 400: the request was well formed and is simply out of date.

        Marked so the interface can offer to load the documents again and
        re-ask, instead of printing a sentence the reader can do nothing with.
        """
        return JSONResponse({"detail": str(exc), "stale_snapshot": True}, status_code=409)

    @app.exception_handler(ValueError)
    async def value_error(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(KeyError)
    async def missing(request: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(OSError)
    async def io_error(request: Request, exc: OSError) -> JSONResponse:
        return JSONResponse({"detail": f"File operation failed: {exc.strerror or str(exc)}"}, status_code=400)

    secured = [Depends(authorize)]

    @app.get("/api/health", dependencies=secured)
    def health() -> dict[str, Any]:
        return {"ok": True, "version": "0.4.0", "workspace": str(workspace.root)}

    @app.get("/api/setup", dependencies=secured)
    def setup() -> dict[str, Any]:
        return {
            "python": sys.version.split()[0],
            "workspace": str(workspace.root),
            "offline": True,
            # Named and explained, rather than as import names the reader would
            # have to know the packaging to decode.
            "components": components(),
            "sample_available": sample_dir().is_dir(),
        }

    @app.get("/api/notices", dependencies=secured)
    def notices() -> dict[str, str]:
        if getattr(sys, "frozen", False):
            path = Path(getattr(sys, "_MEIPASS", "")) / "desktop_resources/THIRD_PARTY_NOTICES.txt"
        else:
            path = Path(__file__).resolve().parents[1] / "out/desktop-package/resources/THIRD_PARTY_NOTICES.txt"
        return {
            "text": path.read_text(encoding="utf-8")
            if path.is_file()
            else "Dependency notices are generated during packaging. See docs/LICENSE_REVIEW.md for the current source and research-asset licensing policy."
        }

    @app.get("/api/models", dependencies=secured)
    def models() -> dict[str, Any]:
        """Every registered model: what it is, whether it is here, and progress."""
        return downloads.listing()

    @app.post("/api/models/{model_id}/download", dependencies=secured)
    def model_download(model_id: str) -> dict[str, Any]:
        """Start (or resume) a download; poll GET /api/models for progress."""
        return downloads.start(model_id)

    @app.post("/api/models/{model_id}/cancel", dependencies=secured)
    def model_cancel(model_id: str) -> dict[str, Any]:
        return downloads.cancel(model_id)

    @app.delete("/api/models/{model_id}", dependencies=secured)
    def model_delete(model_id: str) -> dict[str, Any]:
        return downloads.delete(model_id)

    def choice_labels(param: Any) -> dict[str, Any]:
        """Model choices by what they are, not their ids ("BERT base (uncased): word vectors").

        The ids (``granite-embedding-english-r2``) say nothing about which
        model does sentiment and which reads meaning; the form shows these.
        """
        if param.name != "model" or not param.choices:
            return {}
        from core.models.locate import status as model_status
        from core.models.registry import get_model
        from desktop_backend.models import KIND_LABELS

        labels: dict[str, str] = {}
        for choice in param.choices:
            spec = get_model(str(choice))
            if spec is None:
                continue
            text = f"{spec.display_name}: {KIND_LABELS[spec.kind].lower()}"
            if model_status(spec) != "ready":
                text += " (not installed; see Models)"
            labels[str(choice)] = text
        return {"choice_labels": labels}

    @app.get("/api/tools", dependencies=secured)
    def tools() -> list[dict[str, Any]]:
        from core.profiler.registry import ToolSpec, get_tool

        found = inventory()

        def payload(name: str, spec: ToolSpec) -> dict[str, Any]:
            """Ship the words the interface shows, so it never invents them.

            The desktop previously kept its own table of tool names and derived
            parameter labels from the flag. Both drifted out of step with the
            registry without anything failing. Labels are resolved here, once.
            """
            from core.profiler.registry import FAMILY_LABELS

            published = desktop_spec(spec)
            fields = {
                key: value
                for key, value in asdict(published).items()
                if key in {"name", "description", "requires_parse", "input_kind", "outputs", "family"}
            }
            return {
                **fields,
                "label": tool_label(name),
                "family_label": FAMILY_LABELS.get(published.family, published.family),
                "params": [
                    {**asdict(param), "label": param_label(name, param.name), **choice_labels(param)}
                    for param in published.params
                ],
                "category": "visualization" if name in VISUALIZATION_TOOLS else "analysis",
                "availability": availability(published, found),
            }

        return [
            payload(name, spec)
            for name in DESKTOP_TOOLS
            if (spec := TABLE_TOOLS.get(name) or get_tool(name)) is not None
        ]

    @app.post("/api/resources", dependencies=secured)
    async def resource_upload(name: str, request: Request) -> dict[str, str]:
        """User-selected auxiliary files are data, never executable commands."""
        suffix = Path(name.replace("\\", "/")).suffix.lower()
        if suffix not in (".txt", ".csv", ".tsv", ".json"):
            raise ValueError("Choose a TXT, CSV, TSV or JSON analysis resource.")
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > MAX_FILE_BYTES:
                raise HTTPException(413, "Resource exceeds 20 MB.")
            data.extend(chunk)
        directory = workspace.root / "resources"
        directory.mkdir(exist_ok=True)
        path = directory / (uuid.uuid4().hex + suffix)
        path.write_bytes(data)
        return {"path": str(path), "name": Path(name.replace("\\", "/")).name}

    @app.get("/api/projects", dependencies=secured)
    def projects(archived: bool = False, trashed: bool = False) -> list[dict[str, Any]]:
        return workspace.projects(archived=archived, trashed=trashed)

    @app.post("/api/projects", dependencies=secured)
    def create_project(body: ProjectBody) -> dict[str, Any]:
        return workspace.create(body.name)

    @app.post("/api/projects/{project_id}/rename", dependencies=secured)
    def rename_project(project_id: str, body: ProjectBody) -> dict[str, Any]:
        with runner.lock:
            return workspace.rename(project_id, body.name)

    @app.post("/api/projects/{project_id}/archive", dependencies=secured)
    def archive_project(project_id: str, body: ArchiveBody) -> dict[str, Any]:
        with runner.lock:
            return workspace.archive(project_id, body.archived)

    @app.post("/api/projects/{project_id}/trash", dependencies=secured)
    def trash_project(project_id: str) -> dict[str, Any]:
        with runner.lock:
            cool_project(project_id)
            return workspace.trash_project(project_id, True)

    @app.post("/api/projects/{project_id}/restore", dependencies=secured)
    def restore_project(project_id: str) -> dict[str, Any]:
        with runner.lock:
            result = workspace.trash_project(project_id, False)
            return result

    @app.post("/api/projects/{project_id}/purge", dependencies=secured)
    def purge_project(project_id: str) -> dict[str, bool]:
        with runner.lock:
            workspace.purge_project(project_id)
            return {"purged": True}

    @app.get("/api/projects/{project_id}/documents", dependencies=secured)
    def documents(project_id: str, trashed: bool = False) -> list[dict[str, Any]]:
        return [public_document(document) for document in workspace.documents(project_id, trashed=trashed)]

    def cool_project(project_id: str) -> None:
        held = benches.get(project_id)
        if held is not None:
            held.cool()

    @app.post("/api/projects/{project_id}/documents/{document_id}/trash", dependencies=secured)
    def trash_document(project_id: str, document_id: str) -> dict[str, Any]:
        with runner.lock:
            if any(job["state"] in ("RUNNING", "QUEUED") for job in workspace.jobs(project_id)):
                raise ValueError("Finish or cancel active runs before changing the corpus.")
            doc = workspace.trash_document(project_id, document_id, True)
            cool_project(project_id)
            return public_document(doc)

    @app.post("/api/projects/{project_id}/documents/{document_id}/restore", dependencies=secured)
    def restore_document(project_id: str, document_id: str) -> dict[str, Any]:
        with runner.lock:
            if any(job["state"] in ("RUNNING", "QUEUED") for job in workspace.jobs(project_id)):
                raise ValueError("Finish or cancel active runs before changing the corpus.")
            doc = workspace.trash_document(project_id, document_id, False)
            cool_project(project_id)
            return public_document(doc)

    @app.post("/api/projects/{project_id}/documents/{document_id}/purge", dependencies=secured)
    def purge_document(project_id: str, document_id: str) -> dict[str, bool]:
        with runner.lock:
            if any(job["state"] in ("RUNNING", "QUEUED") for job in workspace.jobs(project_id)):
                raise ValueError("Finish or cancel active runs before changing the corpus.")
            workspace.purge_document(project_id, document_id)
            cool_project(project_id)
            return {"purged": True}

    @app.get("/api/projects/{project_id}/documents/{document_id}", dependencies=secured)
    def preview(project_id: str, document_id: str) -> dict[str, Any]:
        return {
            key: value
            for key, value in workspace.preview(project_id, document_id).items()
            if key not in {"stored_name", "source_name"}
        }

    @app.post("/api/projects/{project_id}/documents", dependencies=secured)
    async def upload(project_id: str, name: str, request: Request) -> dict[str, Any]:
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > MAX_FILE_BYTES:
                raise HTTPException(413, "Each document must be under 20 MB.")
            data.extend(chunk)

        def commit() -> dict[str, Any]:
            with runner.lock:
                return public_document(workspace.import_document(project_id, name, bytes(data)))

        return await run_in_threadpool(commit)

    @app.post("/api/projects/{project_id}/import-paths", dependencies=secured)
    def paths(project_id: str, body: PathsBody) -> dict[str, Any]:
        with runner.lock:
            return import_paths(workspace, project_id, [Path(path) for path in body.paths])

    @app.post("/api/projects/{project_id}/import-sample", dependencies=secured)
    def sample(project_id: str) -> dict[str, Any]:
        if not sample_dir().is_dir():
            raise ValueError("The example documents are not included in this installation.")
        with runner.lock:
            return import_paths(workspace, project_id, [sample_dir()])

    @app.get("/api/projects/{project_id}/jobs", dependencies=secured)
    def jobs(project_id: str) -> list[dict[str, Any]]:
        return [public_job(job) for job in workspace.jobs(project_id)]

    @app.get("/api/projects/{project_id}/jobs/trash", dependencies=secured)
    def trashed_jobs(project_id: str) -> list[dict[str, Any]]:
        return [public_job(job) for job in workspace.jobs(project_id, trashed=True)]

    @app.post("/api/projects/{project_id}/jobs/{job_id}/trash", dependencies=secured)
    def trash_job(project_id: str, job_id: str) -> dict[str, Any]:
        with runner.lock:
            return public_job(workspace.trash_run(project_id, job_id, True))

    @app.post("/api/projects/{project_id}/jobs/{job_id}/restore", dependencies=secured)
    def restore_job(project_id: str, job_id: str) -> dict[str, Any]:
        with runner.lock:
            return public_job(workspace.trash_run(project_id, job_id, False))

    @app.post("/api/projects/{project_id}/jobs/{job_id}/purge", dependencies=secured)
    def purge_job(project_id: str, job_id: str) -> dict[str, bool]:
        with runner.lock:
            workspace.purge_run(project_id, job_id)
            return {"purged": True}

    @app.get("/api/projects/{project_id}/tables", dependencies=secured)
    def tables(project_id: str) -> list[dict[str, Any]]:
        """Everything in this project the interactive workbench can open.

        Input manifests are listed like any other table but flagged, so the
        app can open on a result instead. They are a record of what was read,
        which is worth being able to look at and a poor thing to land on.
        """
        return [
            {
                **table,
                "label": tool_label(table["tool"]),
                "manifest": Path(table["path"]).name == INPUT_MANIFEST,
            }
            for table in workspace.tables(project_id)
        ]

    # One warmed corpus per project. Parsing is what costs; everything after
    # it is cheap, so the parse is done once and the answers come from it.
    benches: dict[str, Session] = {}

    def session(project_id: str) -> Session:
        workspace.project(project_id)
        if project_id not in benches:
            benches[project_id] = Session(Bench(workspace.root))
        return benches[project_id]

    @app.get("/api/projects/{project_id}/live", dependencies=secured)
    def live_state(project_id: str) -> dict[str, Any]:
        return session(project_id).state()

    @app.post("/api/projects/{project_id}/live/warm", dependencies=secured)
    async def warm(project_id: str, body: WarmBody) -> dict[str, Any]:
        """Read and parse the chosen documents, so questions become cheap.

        This is the slow step and there is no making it fast: thirty seconds
        for a hundred thousand words, once. It runs off the event loop so the
        rest of the app stays alive, and reports its stage while it works.
        """
        from core.io.reader import (
            Corpus,
            Document,
            corpus_fingerprint,
            date_from_filename,
            hash_file,
            hash_text,
            read_text,
        )

        # Checked here, before a single document is read: a parser nobody has
        # is a mistake in the request, not a failure of the parse, and it
        # should come back as a refusal rather than as a warmed-then-failed
        # session the reader has to interpret.
        as_parser(body.parser)
        chosen = resolve_selection(
            [document_metadata(d) for d in workspace.documents(project_id)],
            body.selection,
        )
        project_dir = workspace.project_dir(project_id)
        documents = []
        for index, item in enumerate(chosen, 1):
            path = project_dir / "corpus" / item["stored_name"]
            if hash_file(path) != item["sha256"]:
                raise ValueError(f"{item['name']} has changed since it was imported. Reimport it.")
            text = read_text(path).unwrap()
            documents.append(
                Document(
                    doc_id=index,
                    path=path,
                    text=text,
                    sha256=hash_text(text),
                    date=date_from_filename(Path(item["name"])),
                    source_id=str(item["id"]),
                    label=str(item["name"]),
                )
            )
        if not documents:
            raise ValueError("No readable documents in that selection.")
        docs = tuple(documents)
        corpus = Corpus(docs, corpus_fingerprint(docs))
        summary = {
            "documents": len(docs),
            "words": sum(len(d.text.split()) for d in docs),
            "names": [d.name for d in docs[:12]],
            "ids": [str(item["id"]) for item in chosen],
        }
        held = session(project_id)
        return await run_in_threadpool(held.warm, corpus, body.parser, selection=summary)

    @app.delete("/api/projects/{project_id}/live/analyse/{request_id}", dependencies=secured)
    def cancel_analyse(project_id: str, request_id: str) -> dict[str, Any]:
        """Stop waiting for a live answer (R-C8): the reader walked away.

        Frees the abandoned question on the server side too, not just in the
        interface. The answer is discarded rather than delivered, and a
        question cancelled before it started is never computed at all. This is
        deliberately "discard the answer" and not "stop the CPU": model fits
        are not interruptible, and the honest contract says so.
        """
        return session(project_id).cancel_analysis(request_id)

    @app.post("/api/projects/{project_id}/live/analyse", dependencies=secured)
    async def analyse(project_id: str, body: AnalyseBody) -> dict[str, Any]:
        """Run an analysis over the warmed corpus and hand back the rows.

        No job, no run directory, no artifact. This is the same ``execute``
        a published run calls, so the rows here are the rows a publish would
        write -- which is what makes it safe to read them.

        Contract R-C2/R-C6: the caller's ``request_id`` comes back untouched so
        a superseded answer can be recognised, and the payload is validated
        against ``AnalyseResponse`` on the way out -- the engine does not get
        to invent fields the desktop never agreed to render.
        """
        held = session(project_id)

        def cancelled_answer(reason: str) -> dict[str, Any]:
            return AnalyseResponse.model_validate(
                {
                    "request_id": body.request_id,
                    "tool": body.tool,
                    "ok": False,
                    "snapshot_id": body.snapshot_id,
                    "diagnostics": [{"severity": "INFO", "code": "LIVE_CANCELLED", "message": reason}],
                    **_NO_READING,
                }
            ).model_dump()

        # R-C8: a question the reader abandoned before it started is never
        # computed -- the machine should not warm up for an answer nobody wants.
        if held.analysis_cancelled(body.request_id):
            held.forget_cancelled(body.request_id)
            return cancelled_answer("This question was cancelled before it ran; press Run to ask it again.")
        # R-C4: a question about documents that are no longer loaded is
        # refused (409), never silently answered against the wrong corpus.
        snapshot = held.resolved(body.snapshot_id) if body.snapshot_id else held.resolved()
        result = await run_in_threadpool(held.analyse, body.tool, body.params)
        # R-C8: abandoned mid-flight. The fit ran; its answer is discarded
        # rather than delivered to a question that is gone.
        if held.analysis_cancelled(body.request_id):
            held.forget_cancelled(body.request_id)
            return cancelled_answer("This question was cancelled while it ran; its answer was discarded.")
        named = result.table()
        reading = reading_of(named[1], result.tool) if named and result.ok else dict(_NO_READING)
        offered = live_panels_offered(result)
        if result.ok:
            # Every figure this answer can draw, from whichever of its tables
            # feeds it -- not only the figures of the one table on the page.
            reading["figures"] = [
                {"panel": info["name"], "title": info["title"], "question": info["question"], "shape": info["shape"]}
                for info in offered
            ]
        answered = AnalyseResponse.model_validate(
            {
                "request_id": body.request_id,
                "tool": result.tool,
                "ok": result.ok,
                "elapsed_ms": result.elapsed_ms,
                "snapshot_id": snapshot.id,
                "diagnostics": [
                    {
                        "severity": d.severity.value,
                        "code": d.code,
                        "message": d.message,
                    }
                    for d in result.diagnostics
                ],
                "path": named[0] if named else "",
                "table": as_table(named[1]) if named else None,
                "panel": live_panel(result, body.params),
                "panels": offered,
                # Kept server-side so a figure can be redrawn with other
                # settings without re-running the analysis.
                "answer_id": held.remember(result, body.params, snapshot.id) if result.ok else "",
                # Read from the whole frame, never from the page that travels.
                # `as_table` sends the first 500 rows; a reading of those is a
                # reading of an alphabetical prefix of the corpus presented as
                # a reading of the corpus.
                **reading,
            }
        )
        return answered.model_dump()

    @app.post("/api/projects/{project_id}/live/panel", dependencies=secured)
    def live_figure(project_id: str, body: LivePanelBody) -> dict[str, Any]:
        """Redraw one figure of a live answer with the reader's settings.

        The answer is the one the analyse call remembered, cited by id; one
        that has been dropped, or whose corpus was reloaded since, is refused
        in the same shape as any panel refusal rather than drawn from rows
        that no longer describe the loaded documents.
        """
        held = session(project_id)
        remembered = held.answer(body.answer_id)
        if remembered is None:
            return panel_failure(
                [
                    Diagnostic.error(
                        "LIVE_ANSWER_GONE",
                        "This answer is no longer held (it was replaced, or the documents were reloaded). Press Run to ask again.",
                    )
                ]
            )
        result, settings = remembered
        return draw_live_panel(result, settings, body.panel, body.params)

    @app.post("/api/projects/{project_id}/live/panel/static", dependencies=secured)
    def live_figure_static(project_id: str, body: LiveStaticBody) -> Response:
        """The publication figure (matplotlib + seaborn) of one live figure."""
        from desktop_backend.panels import render_prepared

        held = session(project_id)
        remembered = held.answer(body.answer_id)
        if remembered is None:
            return JSONResponse(
                panel_failure(
                    [Diagnostic.error("LIVE_ANSWER_GONE", "This answer is no longer held. Press Run to ask again.")]
                )
            )
        result, settings = remembered
        prepared = prepare_live_panel(result, settings, body.panel, body.params)
        return _figure_response(render_prepared(prepared, body.format, body.dpi, body.notes), body.format, body.panel)

    @app.post("/api/projects/{project_id}/live/phrase", dependencies=secured)
    async def phrase(project_id: str, body: PhraseBody) -> dict[str, Any]:
        """Track a requested phrase through time, documents and source context."""
        held = session(project_id)
        return await run_in_threadpool(held.track_phrase, body.model_dump())

    @app.get("/api/projects/{project_id}/live/source", dependencies=secured)
    def source_passage(
        project_id: str,
        document_id: str,
        character_start: int,
        character_end: int | None = None,
        radius: int = 100,
    ) -> dict[str, Any]:
        """A wider window of one loaded document around a verified offset.

        Bound to the live snapshot through ``session``'s resolved corpus, so a
        reload underneath the reader is refused exactly as a paged question
        is. Offsets are the imported text's code points -- the same unit the
        evidence rows declare -- never a page coordinate of a converted PDF.
        """
        held = session(project_id)
        corpus = held.resolved().warm.corpus
        from core.research.phrase import source_passage as read_passage

        return read_passage(corpus, document_id, character_start, character_end, radius)

    @app.post("/api/projects/{project_id}/live/cool", dependencies=secured)
    def cool(project_id: str) -> dict[str, Any]:
        held = session(project_id)
        held.cool()
        return held.state()

    @app.get("/api/projects/{project_id}/questions", dependencies=secured)
    def questions(project_id: str) -> list[dict[str, Any]]:
        return workspace.questions(project_id)

    def coherent_scope(project_id: str, body: QuestionBody) -> None:
        """Refuse a question whose snapshot, documents and parser disagree.

        Validating the document IDs alone establishes that those documents
        exist, not that the cited snapshot was built from them with that
        parser. The snapshot ID is a digest of the corpus text *and* the
        resolved parser identity, so nothing short of loading a parser can
        recompute it -- but a project that has one loaded already knows the
        answer, and that is the moment a question is saved.

        A project with nothing loaded cannot corroborate the snapshot here.
        The publish-time check in the runner, which does parse, stays the one
        that catches a mismatch in that case; it refuses the run rather than
        publishing an answer under a scope that never produced it.
        """
        state = session(project_id).state()
        if state.get("state") != "ready" or not state.get("snapshot_id"):
            return
        if state["snapshot_id"] != body.snapshot_id:
            raise ValueError(
                "These are not the documents that answer came from. "
                "Track the phrase again against the loaded documents before saving it."
            )
        if state.get("parser") != body.parser:
            raise ValueError(
                f"This question says it was read by {body.parser}, but the loaded documents "
                f"were read by {state.get('parser')}. Track the phrase again before saving it."
            )
        loaded = {str(item) for item in ((state.get("selection") or {}).get("ids") or [])}
        if loaded and loaded != set(body.document_ids):
            raise ValueError(
                "This question names a different set of documents from the ones that are loaded. "
                "Track the phrase again before saving it."
            )

    @app.post("/api/projects/{project_id}/questions", dependencies=secured)
    def create_question(project_id: str, body: QuestionBody) -> dict[str, Any]:
        coherent_scope(project_id, body)
        with runner.lock:
            return workspace.save_question(project_id, body)

    @app.post("/api/projects/{project_id}/questions/{question_id}", dependencies=secured)
    def save_question(project_id: str, question_id: str, body: QuestionBody) -> dict[str, Any]:
        coherent_scope(project_id, body)
        with runner.lock:
            return workspace.update_question(project_id, question_id, body)

    @app.post("/api/projects/{project_id}/questions/{question_id}/duplicate", dependencies=secured)
    def duplicate_question(project_id: str, question_id: str) -> dict[str, Any]:
        with runner.lock:
            return workspace.duplicate_question(project_id, question_id)

    @app.post("/api/projects/{project_id}/questions/{question_id}/publish", dependencies=secured)
    def publish_question(project_id: str, question_id: str) -> dict[str, Any]:
        """Turn a saved phrase question into a frozen, provenance-bearing run."""
        question = workspace.question(project_id, question_id)
        spec = question["specification"]
        # The tokens this question was saved as, so the published answer is the
        # saved question rather than today's reading of the same characters. A
        # question saved before these were recorded carries none, and the
        # published record says the parser resolved it instead.
        matching = question.get("matching") or {}
        params = {
            "resolved-tokens": json.dumps(matching.get("tokens") or [], ensure_ascii=False),
            "comparison-tokens": json.dumps(matching.get("comparison_tokens") or [], ensure_ascii=False),
            "phrase": spec["text"],
            "comparison": spec.get("comparison_text") or "",
            "case-sensitive": spec["case_sensitive"],
            # Defaulted rather than indexed: a question saved before these
            # existed has no key for them, and its literal reading is the
            # question it was saved as.
            "normalize": spec.get("normalize", False),
            "match-lemma": spec.get("match_lemma", False),
            "match-nominalization": spec.get("match_nominalization", False),
            "position-bins": spec["position_bins"],
            "evidence-year": spec.get("evidence_year") or 0,
            "evidence-document": spec.get("evidence_document_id") or "",
            "question-name": question["name"],
            "question-id": question["id"],
            "question-revision": question["revision"],
            "snapshot-id": question["snapshot_id"],
        }
        return public_job(
            runner.submit(
                project_id,
                "phrase_distribution",
                params,
                question["parser"],
                selection=CorpusSelection(document_ids=question["document_ids"]),
            )
        )

    @app.post("/api/projects/{project_id}/questions/{question_id}/delete", dependencies=secured)
    def delete_question(project_id: str, question_id: str) -> dict[str, bool]:
        with runner.lock:
            workspace.delete_question(project_id, question_id)
        return {"deleted": True}

    @app.get("/api/chart-contract", dependencies=secured)
    def contract() -> dict[str, dict[str, str]]:
        """Where a published chart differs from the live preview, in words.

        Shipped rather than restated in TypeScript for the same reason
        ``/api/tools`` ships its labels: a second copy on the far side of a
        language boundary drifts without anything failing.
        """
        return chart_contract()

    @app.get("/api/panels", dependencies=secured)
    def panel_catalogue() -> list[dict[str, Any]]:
        """Every panel and what it takes, so the app builds its own controls."""
        from desktop_backend.panels import panel_declarations

        return panel_declarations()

    @app.get("/api/projects/{project_id}/jobs/{job_id}/panels", dependencies=secured)
    def panels_for_run(project_id: str, job_id: str) -> list[dict[str, Any]]:
        """Which purpose-built figures this finished run can draw.

        Keyed on the tool that produced the run, so a reader is offered the
        figures that fit what they are looking at rather than a catalogue to
        pick through.
        """
        from desktop_backend.panels import panels_for_envelope

        directory, envelope = workspace.artifacts(project_id, job_id)
        return panels_for_envelope(directory, envelope)

    @app.post("/api/projects/{project_id}/jobs/{job_id}/panels", dependencies=secured)
    def draw_run_panel(project_id: str, job_id: str, body: PanelBody) -> dict[str, Any]:
        """Prepare one panel over this run: marks, evidence, caption, limits.

        Returns geometry rather than a picture -- the app draws it, so every
        mark stays clickable and can say what it stands for. Failures come
        back as ``ok: False`` plus diagnostics (R-C7), never as an HTTP error,
        so the interface has one shape to render.
        """
        from desktop_backend.panels import draw_panel

        directory, envelope = workspace.artifacts(project_id, job_id)
        return draw_panel(directory, envelope, body)

    @app.get("/api/projects/{project_id}/jobs/{job_id}/bundles", dependencies=secured)
    def bundles_for_run(project_id: str, job_id: str) -> list[dict[str, Any]]:
        """The publication-only figures this run can draw (correlation
        matrices, joint plots, topic-by-decade tables, similarity maps)."""
        from desktop_backend.panels import bundles_for_envelope

        directory, envelope = workspace.artifacts(project_id, job_id)
        return bundles_for_envelope(directory, envelope)

    @app.post("/api/projects/{project_id}/jobs/{job_id}/bundles", dependencies=secured)
    def draw_run_bundle(project_id: str, job_id: str, body: BundleBody) -> Response:
        """One publication-only figure over this run, as PNG, SVG or PDF."""
        from desktop_backend.panels import render_run_bundle

        directory, envelope = workspace.artifacts(project_id, job_id)
        return _figure_response(render_run_bundle(directory, envelope, body), body.format, body.bundle)

    @app.post("/api/projects/{project_id}/jobs/{job_id}/panels/static", dependencies=secured)
    def draw_run_panel_static(project_id: str, job_id: str, body: StaticPanelBody) -> Response:
        """One panel over this run as a publication figure: PNG, SVG or PDF.

        The same preparation as the interactive figure -- same table, same
        parameters, same provenance -- drawn by matplotlib and seaborn with
        what a static figure can add: spread bands, violins, a clustered
        matrix, collision-free labels.
        """
        from desktop_backend.panels import prepare_run_panel, render_prepared

        directory, envelope = workspace.artifacts(project_id, job_id)
        prepared = prepare_run_panel(directory, envelope, PanelBody(panel=body.panel, params=body.params))
        return _figure_response(render_prepared(prepared, body.format, body.dpi, body.notes), body.format, body.panel)

    @app.get("/api/projects/{project_id}/views", dependencies=secured)
    def views(project_id: str) -> list[dict[str, Any]]:
        return [public_view(view) for view in workspace.views(project_id)]

    @app.post("/api/projects/{project_id}/views", dependencies=secured)
    def create_view(project_id: str, body: ViewBody) -> dict[str, Any]:
        with runner.lock:
            return public_view(workspace.save_view(project_id, body))

    @app.get("/api/projects/{project_id}/views/{view_id}", dependencies=secured)
    def read_view(project_id: str, view_id: str) -> dict[str, Any]:
        return public_view(workspace.view(project_id, view_id))

    @app.post("/api/projects/{project_id}/views/{view_id}", dependencies=secured)
    def save_view(project_id: str, view_id: str, body: ViewBody) -> dict[str, Any]:
        with runner.lock:
            return public_view(workspace.update_view(project_id, view_id, body))

    @app.post("/api/projects/{project_id}/views/{view_id}/duplicate", dependencies=secured)
    def copy_view(project_id: str, view_id: str) -> dict[str, Any]:
        with runner.lock:
            return public_view(workspace.duplicate_view(project_id, view_id))

    @app.post("/api/projects/{project_id}/views/{view_id}/delete", dependencies=secured)
    def remove_view(project_id: str, view_id: str) -> dict[str, bool]:
        # A POST, not a DELETE: the CORS policy above allows GET and POST only,
        # and widening it for one route would widen it for every route.
        with runner.lock:
            workspace.delete_view(project_id, view_id)
        return {"ok": True}

    @app.get("/api/projects/{project_id}/glance", dependencies=secured)
    def glance_status(project_id: str) -> dict[str, Any]:
        """Corpus at a glance: the newest result, whether it is current, what it found."""
        from desktop_backend.glance import status

        answer = status(workspace, project_id)
        if "job" in answer:
            answer["job"] = public_job(answer["job"])
        return answer

    @app.post("/api/projects/{project_id}/glance", dependencies=secured)
    def glance_start(project_id: str, body: GlanceBody) -> dict[str, Any]:
        """Run the glance recipe over the whole corpus (one parse, one job)."""
        from desktop_backend.glance import GLANCE_TOOL

        with runner.lock:
            job = runner.submit(project_id, GLANCE_TOOL, {}, body.parser)
        return public_job(job)

    @app.get("/api/projects/{project_id}/glance/figure", dependencies=secured)
    def glance_figure(project_id: str, path: str = Query(...)) -> FileResponse:
        from desktop_backend.glance import figure_file

        return FileResponse(figure_file(workspace, project_id, path), media_type="image/png")

    @app.post("/api/projects/{project_id}/jobs", dependencies=secured)
    def submit(project_id: str, body: RunBody) -> dict[str, Any]:
        with runner.lock:
            job = runner.submit(project_id, body.tool, body.params, body.parser, selection=body.selection)
        return public_job(job)

    @app.post("/api/projects/{project_id}/jobs/{job_id}/visualize", dependencies=secured)
    def visualize_result(project_id: str, job_id: str, body: VisualizeBody) -> dict[str, Any]:
        """Re-run a published artifact's CSV through a visualization workflow.

        The artifact path never crosses the client boundary: the server
        resolves it by index and hands it to the runner as the frozen
        ``input`` parameter, so the visualization runs with the same
        provenance rules as any manually selected table.
        """
        artifact = workspace.artifact(project_id, job_id, body.index)
        if artifact.suffix.lower() != ".csv":
            raise ValueError("Visualize works on CSV result tables. Download non-CSV artifacts to inspect them.")
        with runner.lock:
            job = runner.submit(
                project_id,
                body.tool,
                {**body.params, "input": str(artifact)},
                "spacy",
            )
        return public_job(job)

    @app.get("/api/projects/{project_id}/jobs/{job_id}/results", dependencies=secured)
    def results(project_id: str, job_id: str) -> dict[str, Any]:
        _, envelope = workspace.artifacts(project_id, job_id)
        return envelope.to_dict()

    @app.get("/api/projects/{project_id}/jobs/{job_id}/insight", dependencies=secured)
    def insight(project_id: str, job_id: str) -> dict[str, Any]:
        """Plain-language reading of a finished run, plus charts worth drawing.

        Read from the published CSV rather than recomputed, so what the desktop
        shows is a reading of the artifact the user can also open and export.
        """
        import pandas as pd

        directory, envelope = workspace.artifacts(project_id, job_id)
        # The input manifest records what was run on (appended after the
        # tool's own frames by the runner); the reading describes the tool's
        # result, so it is skipped even if frame ordering ever changes.
        table = next(
            (a for a in envelope.artifacts if a.path.endswith(".csv") and Path(a.path).name != INPUT_MANIFEST),
            None,
        )
        if table is None:
            return {"tool": envelope.tool, "available": False, "reason": "This run produced no table to read."}
        try:
            frame = pd.read_csv(directory / table.path)
        except (OSError, ValueError) as exc:
            return {"tool": envelope.tool, "available": False, "reason": f"Could not read {table.path}: {exc}"}
        from desktop_backend.panels import panels_for_envelope

        reading = reading_of(frame, envelope.tool)
        # Every figure the run can draw, from whichever of its tables feeds
        # it, not only the figures of the one table this reading describes.
        reading["figures"] = [
            {"panel": info["name"], "title": info["title"], "question": info["question"], "shape": info["shape"]}
            for info in panels_for_envelope(directory, envelope)
        ]
        return {"tool": envelope.tool, "available": True, "table": table.path, **reading}

    @app.post("/api/projects/{project_id}/compare", dependencies=secured)
    def compare_two_runs(project_id: str, body: CompareBody) -> dict[str, Any]:
        """Put two finished runs side by side and name every disagreement.

        The graded questions are comparisons -- MALLET against Gensim, BERT
        against Gensim Word2Vec, four sentiment annotators -- and answering
        them used to mean two exports and a spreadsheet. ``core.compare`` has
        always known how to diff two run directories; this is that instrument
        behind one click, with the same diagnostic shape as everything else
        (R-C7).
        """
        from core.compare import compare_runs

        def failure(diagnostics: list[Any]) -> dict[str, Any]:
            return {
                "ok": False,
                "passed": False,
                "expected": body.expected,
                "actual": body.actual,
                "diff": None,
                "diagnostics": [
                    {"severity": d.severity.value, "code": d.code, "message": d.message} for d in diagnostics
                ],
            }

        try:
            expected_dir, _ = workspace.artifacts(project_id, body.expected)
            actual_dir, _ = workspace.artifacts(project_id, body.actual)
        except (KeyError, ValueError, OSError) as exc:
            return failure(
                [
                    Diagnostic.error(
                        "COMPARE_NO_RUN",
                        f"cannot compare those runs: {exc}",
                        fix="compare two finished runs (state DONE or PARTIAL) from this project",
                    )
                ]
            )
        result = compare_runs(expected_dir, actual_dir)
        if result.value is None:
            return failure(list(result.diagnostics))
        report = result.unwrap()
        return {
            "ok": True,
            "passed": report.passed,
            "expected": body.expected,
            "actual": body.actual,
            "compared_rows": report.compared_rows,
            "compared_columns": report.compared_columns,
            "diff": as_table(report.diff),
            "diagnostics": [
                {"severity": d.severity.value, "code": d.code, "message": d.message} for d in result.diagnostics
            ],
        }

    @app.get("/api/visualizations", dependencies=secured)
    def visualizations() -> list[dict[str, Any]]:
        """The visualization catalog, so the UI can badge legacy vs new."""
        from core.viz.catalog import VISUALIZATIONS

        return [
            {
                "name": v.name,
                "title": v.title,
                "origin": str(v.origin),
                "is_legacy": v.origin.is_legacy,
                "tool": v.tool,
                "produces": list(v.produces),
                "summary": v.summary,
                "legacy_module": v.legacy_module,
                "legacy_note": v.legacy_note,
                "advantage": v.advantage,
            }
            for v in VISUALIZATIONS
        ]

    @app.get("/api/projects/{project_id}/backup", dependencies=secured)
    def project_backup(project_id: str) -> FileResponse:
        temporary = tempfile.TemporaryDirectory(prefix="backup-", dir=workspace.root)
        path = Path(temporary.name) / "project.nlpsuite"
        try:
            with runner.lock:
                backup(workspace, project_id, path)
            return FileResponse(
                path,
                media_type="application/zip",
                filename=f"project-{project_id[:8]}.nlpsuite",
                background=BackgroundTask(temporary.cleanup),
            )
        except Exception:
            temporary.cleanup()
            raise

    @app.post("/api/projects/restore", dependencies=secured)
    async def project_restore(request: Request) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="upload-", dir=workspace.root) as temporary:
            path = Path(temporary) / "project.nlpsuite"
            size = 0
            with path.open("wb") as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_ARCHIVE_BYTES:
                        raise HTTPException(413, "Project archive exceeds 512 MB.")
                    handle.write(chunk)

            def commit() -> dict[str, Any]:
                with runner.lock:
                    return restore(workspace, path)

            return await run_in_threadpool(commit)

    @app.post("/api/projects/{project_id}/jobs/{job_id}/cancel", dependencies=secured)
    def cancel(project_id: str, job_id: str) -> dict[str, Any]:
        job = runner.cancel(project_id, job_id)
        return public_job(job)

    @app.post("/api/projects/{project_id}/jobs/{job_id}/artifacts/{index}/preview", dependencies=secured)
    def prepare_preview(project_id: str, job_id: str, index: int) -> dict[str, str]:
        path = workspace.artifact(project_id, job_id, index)
        if path.suffix.lower() not in VIEWABLE_ARTIFACTS or path.stat().st_size > MAX_PREVIEW_BYTES:
            raise ValueError("Artifact is not previewable or exceeds 32 MiB. Use Download.")
        return {"path": "/api/previews/" + previews.issue(project_id, job_id, index)}

    @app.get("/api/previews/{ticket}")
    def browser_preview(ticket: str) -> FileResponse:
        # A scoped expiring ticket replaces session auth for iframe navigation.
        # Re-resolve through Workspace to retain path/symlink confinement.
        path = workspace.artifact(*previews.resolve(ticket))
        if path.suffix.lower() not in VIEWABLE_ARTIFACTS or path.stat().st_size > MAX_PREVIEW_BYTES:
            raise ValueError("Preview unavailable. Use Download.")
        return FileResponse(
            path,
            media_type=VIEWABLE_ARTIFACTS[path.suffix.lower()],
            headers={
                "Content-Security-Policy": PREVIEW_CSP,
            },
        )

    @app.get("/api/projects/{project_id}/jobs/{job_id}/artifacts/{index}", dependencies=secured)
    def artifact(  # noqa: PLR0913 -- URL and bounded preview query parameters
        project_id: str,
        job_id: str,
        index: int,
        *,
        download: bool = False,
        q: str = Query(default="", max_length=500),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=PREVIEW_ROWS, ge=1, le=PREVIEW_ROWS),
    ) -> Any:
        path = workspace.artifact(project_id, job_id, index)
        if download:
            return FileResponse(path, media_type="application/octet-stream", filename=path.name)
        suffix = path.suffix.lower()
        if suffix != ".csv":
            if suffix not in VIEWABLE_ARTIFACTS:
                raise ValueError("Only CSV tables have an inline preview. Download other artifacts to inspect them.")
            return FileResponse(path, media_type=VIEWABLE_ARTIFACTS[suffix])
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows: list[dict[str | Any, Any]] = []
            total = 0
            matched = 0
            needle = q.casefold()
            for row in reader:
                total += 1
                if needle and not any(needle in str(value).casefold() for value in row.values()):
                    continue
                matched += 1
                if matched > offset and len(rows) < limit:
                    rows.append(row)
            return {
                "columns": reader.fieldnames or [],
                "rows": rows,
                "total": total,
                "filtered_total": matched,
                "offset": offset,
                "truncated": matched > offset + len(rows),
            }

    @app.get("/api/projects/{project_id}/jobs/{job_id}/export", dependencies=secured)
    def export(project_id: str, job_id: str) -> FileResponse:
        directory, envelope = workspace.artifacts(project_id, job_id)
        temporary = tempfile.TemporaryDirectory(prefix="export-", dir=workspace.root)
        path = Path(temporary.name) / "run.zip"
        try:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                for index, item in enumerate(envelope.artifacts):
                    archive.write(workspace.artifact(project_id, job_id, index), item.path)
                archive.write(directory / "result.json", "result.json")
            return FileResponse(
                path,
                media_type="application/zip",
                filename=f"nlp-suite-{job_id[:8]}.zip",
                background=BackgroundTask(temporary.cleanup),
            )
        except Exception:
            temporary.cleanup()
            raise

    if frontend is not None and frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


def sample_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", "")) / "desktop_resources" / "sample-corpus"
    return Path(__file__).resolve().parents[1] / "assets" / "sample-corpus"


def import_paths(workspace: Workspace, project_id: str, paths: list[Path]) -> dict[str, Any]:
    workspace.project(project_id)
    candidates: list[Path] = []
    for path in paths:
        if path.is_symlink():
            continue
        if path.is_dir():
            for directory, subdirs, files in os.walk(path, followlinks=False):
                subdirs[:] = sorted(d for d in subdirs if not (Path(directory) / d).is_symlink())
                candidates.extend(
                    Path(directory) / name
                    for name in sorted(files)
                    if Path(name).suffix.lower() in supported_suffixes()
                )
                if len(candidates) > MAX_DOCUMENTS:
                    raise ValueError("Import at most 2,000 documents at a time.")
        else:
            candidates.append(path)
    imported, duplicates = 0, 0
    errors = []
    for path in candidates:
        try:
            if path.is_symlink():
                raise ValueError("Symbolic links are not imported.")
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Document exceeds 20 MB.")
            result = workspace.import_document(project_id, path.name, path.read_bytes())
            if result.get("duplicate"):
                duplicates += 1
            else:
                imported += 1
        except (OSError, ValueError) as exc:
            errors.append({"name": path.name, "message": str(exc)})
    return {"imported": imported, "duplicates": duplicates, "errors": errors}


#: The analysis modules a tool adapter may reach first from a request thread.
_ANALYSIS_WRAPPERS = (
    "core.analysis.lda",
    # Paragraph segmentation for the topic-flow table, imported inside the LDA adapter.
    "core.analysis.topic_flow",
    "core.analysis.sentiment_vader_anew",
    "core.analysis.date_annotator",
    "core.analysis.gender_annotator",
    "core.analysis.gender_guess",
    "core.analysis.location_extract",
    "core.analysis.ngram_viewer",
    "core.analysis.quote_annotator",
    "core.analysis.sentiment_neural",
    "core.analysis.shape_reduction",
    "core.analysis.svo_map",
    "core.analysis.verb_analysis",
    "core.analysis.word2vec_bert",
    "core.analysis.doc_embeddings",
    # The model runtime every BERT-family tool reaches through default_backend.
    "core.models.onnx_backend",
)

#: The third-party backends those modules import *inside* their functions.
#:
#: Warming the wrapper is not enough and this is the bug that proved it:
#: ``core.analysis.lda`` imports gensim inside ``fit_lda``, so preloading the
#: wrapper left the DLL load exactly where the deadlock is -- in a request
#: worker -- while every check passed, because the check asked about
#: ``core.analysis`` names and those were all present. A topic model then hung
#: forever at 0% CPU, on a fit that takes ten seconds.
#:
#: torch and transformers are left out deliberately. They are imported lazily
#: by ``core.analysis.contextual``, which no desktop tool reaches, and warming
#: them would cost seconds of start-up and hundreds of megabytes for a
#: workflow this interface never offers. ``tests/test_live_analysis.py``
#: records that exclusion, so it stays a decision rather than an oversight.
LAZY_BACKENDS = (
    # Topic models and word embeddings. gensim pulls scipy and ~555 modules of
    # native extensions; this is the import that wedged the server.
    "gensim.corpora",
    "gensim.models",
    "gensim.models.ldamodel",
    # Sentiment.
    "vaderSentiment.vaderSentiment",
    # Statistics and stylometry, whose adapters import these inside functions.
    "scipy",
    "scipy.spatial.distance",
    "sklearn.decomposition",
    "sklearn.feature_extraction.text",
    "sklearn.manifold",
    # Word senses and document embeddings cluster inside the analysis.
    "sklearn.cluster",
    "sklearn.metrics",
    "scipy.cluster.hierarchy",
    # WordNet, FrameNet, VerbNet and the SentiWordNet adapters. Importing the
    # package loads no corpus data; a missing corpus is reported by the tool.
    "nltk",
    "nltk.corpus",
    "spacy",
    "stanza",
    # Figures and exports. Found cold by tests/test_no_lazy_backend_imports.py:
    # the word-embeddings adapter draws its t-SNE panel through
    # core.viz.embeddings (plotly) from inside the request path, and the table
    # workflows render wordclouds and Excel workbooks from inside worker jobs.
    "plotly.express",
    "plotly.graph_objects",
    "PIL",
    "wordcloud",
    "openpyxl",
    "openpyxl.chart",
    "kaleido",
    # The model runtime: native extensions both, imported inside
    # core.models.onnx_backend when the first model opens.
    "onnxruntime",
    "tokenizers",
)


def preload_engine() -> None:
    """Import the analysis engine before the server starts taking requests.

    Not an optimisation -- a deadlock fix, and a Windows one. The first live
    analysis imports :mod:`core.profiler.executor`, which pulls in scikit-learn
    and SciPy, which load native extension modules. Loading a DLL holds the
    Windows loader lock. That import runs in a request worker thread, and the
    moment a second request arrives the main thread grows the worker pool --
    and starting a Win32 thread also needs the loader lock. Neither side gives
    way, the whole server stops answering, and nothing times out or recovers.

    The interface polls while the first analysis runs, so the second request is
    not a coincidence: it is every time. Here there is one thread and no event
    loop, so the import is only an import.

    Kept to the module :meth:`Bench.analyse` actually imports, and checked by a
    test, so warming the wrong module cannot pass for warming the right one.
    """
    import core.profiler.executor  # noqa: F401

    # The adapters import their optional backends lazily, inside the request
    # worker: gensim for topic models, vaderSentiment for sentiment. The same
    # loader lock applies, and the wedge it produced was silent -- a topic
    # model over the real corpus left the server at 0% CPU with every thread
    # waiting, for as long as anyone cared to watch. Anything a tool adapter
    # may import for the first time belongs here, on the main thread before
    # the socket opens.
    for module in (*_ANALYSIS_WRAPPERS, *LAZY_BACKENDS):
        # A missing optional extra is not a problem here. The tool reports it
        # cleanly from its own guard, in a request thread, holding no lock: an
        # ImportError loads no native extension, so it cannot wedge anything.
        with suppress(ImportError):
            __import__(module)


def main() -> None:
    parser = argparse.ArgumentParser(description="NLP Suite desktop backend")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "NLP Suite" / "workspace",
    )
    parser.add_argument("--frontend", type=Path, default=Path(__file__).resolve().parents[1] / "desktop" / "dist")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--worker", nargs="?", const="", default=None)
    parser.add_argument("--desktop", action="store_true", help="shut down gracefully when the native shell disconnects")
    args = parser.parse_args()
    # Downloaded models live beside the workspace, in the app's own data
    # folder: they survive app updates, and every worker process this one
    # starts inherits where to find them. The app's data dir is always
    # <app data>/workspace; any other folder keeps its models inside itself
    # (a workspace at a drive root must not scatter C:\models).
    data_dir = args.data_dir.resolve()
    models_dir = data_dir.parent / "models" if data_dir.name == "workspace" else data_dir / "models"
    os.environ.setdefault("NLP_SUITE_MODELS_DIR", str(models_dir))
    # Both paths import the analysis engine before they take work (R-C5): a
    # job worker's first import is as fatal in its own way -- the job dies
    # silently at the point of the cold import -- as the request-thread wedge.
    preload_engine()
    if args.worker is not None:
        if args.worker:
            raise SystemExit(run_job(args.data_dir, args.worker))
        raise SystemExit(worker_loop(args.data_dir))
    import uvicorn

    workspace = Workspace(args.data_dir)
    # Keep an OS-held lock: crashes release it, unlike a stale PID file.
    lock = (workspace.root / ".server.lock").open("a+b")
    lock.seek(0)
    lock.write(b"0")
    lock.flush()
    lock.seek(0)
    if sys.platform == "win32":
        import msvcrt

        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise SystemExit(
                f"Workspace {workspace.root} is locked: another NLP Suite server is already running there.\n"
                "Close the other instance (its terminal window or the desktop app), then try again.\n"
                f"({exc})"
            ) from exc
    else:
        import fcntl

        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SystemExit(
                f"Workspace {workspace.root} is locked: another NLP Suite server is already running there.\n"
                "Close the other instance (its terminal window or the desktop app), then try again.\n"
                f"({exc})"
            ) from exc
    token = secrets.token_urlsafe(32)
    sock = socket.socket()
    sock.bind(("127.0.0.1", args.port))
    port = sock.getsockname()[1]
    print(json.dumps({"baseUrl": f"http://127.0.0.1:{port}", "token": token}), flush=True)
    # After the handshake, so the shell already has its URL and can start
    # drawing while this loads; before the server runs, so it happens on a
    # quiet main thread rather than under a request.
    preload_engine()
    app = create_app(workspace, token, args.frontend)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    if args.desktop:

        def disconnected() -> None:
            sys.stdin.read()
            server.should_exit = True

        threading.Thread(target=disconnected, daemon=True).start()
    try:
        server.run(sockets=[sock])
    finally:
        sock.close()
        lock.close()


if __name__ == "__main__":
    main()
