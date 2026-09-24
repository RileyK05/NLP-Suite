"""Batch envelopes (FR-7.2) — one run dir per tool, one parent manifest.

Each successful outcome gets its own child run directory with artifacts
and a ``result.json`` envelope (via :class:`OutputWriter`, exactly like a
standalone CLI run). The parent ``profiler`` run writes ``batch.json``: an
explicit list of children with statuses, timings, and artifact paths.
Child discovery reads ``batch.json`` — never a directory glob — and failed
tools get no child dir, only their recorded error (the legacy manifest
rule: every completed analysis survives whatever happens next). Manifest
paths are relative to the batch output root (not the parent run dir), so a
moved profile folder keeps working — the same portability lesson the
legacy manifest learned with relative paths.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.artifacts.envelope import Envelope
from core.io.reader import Corpus, hash_file
from core.io.writer import OutputWriter
from core.profiler.executor import BatchResult, ToolOutcome
from core.profiler.report import render_report
from core.profiler.resume import current_key
from core.result import Diagnostic, Result

__all__ = ["BatchReport", "BatchRequest", "write_batch"]


@dataclass(frozen=True, slots=True)
class BatchReport:
    """Published batch: parent run dir, child dirs by tool, parent envelope."""

    run_dir: Path
    child_dirs: dict[str, Path] = field(default_factory=dict)
    envelope: Envelope | None = None


@dataclass(frozen=True, slots=True)
class BatchRequest:
    """Everything one batch publication needs."""

    selection: tuple[str, ...]
    params: Mapping[str, Mapping[str, object]]
    batch: BatchResult
    corpus: Corpus | None = None
    reuse: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: Render each tool's default publication figures into ``figures/`` of
    #: its run (``core/viz/run_figures``). Off by default: a CLI batch writes
    #: exactly the tables it always has; the desktop turns it on.
    figures: bool = False


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_child_run(  # noqa: PLR0913 - figures is keyword-only
    output_root: Path,
    corpus: Corpus | None,
    name: str,
    params: Mapping[str, object],
    outcome: ToolOutcome,
    *,
    figures: bool = False,
) -> Result[tuple[dict[str, Any], Path]]:
    """Publish one successful outcome (callers route failures to the manifest)."""
    try:
        writer = OutputWriter(output_root, tool=name, params=dict(params), corpus=corpus)
    except (ValueError, OSError) as exc:
        return Result.failure(Diagnostic.error("BATCH_WRITE_FAILED", f"could not stage {name}: {exc}", tool=name))
    try:
        written_paths: dict[str, Path] = {}
        for filename, frame in outcome.frames.items():
            # An .html frame is a rendered document (e.g. the t-SNE scatter
            # from core.viz.embeddings), not tabular data — route by suffix.
            if filename.endswith(".html"):
                written = writer.write_html(
                    str(frame.iloc[0, 0]),
                    filename,
                    kind="chart",
                    description=f"{name} {filename}",
                )
            else:
                written = writer.write_table(frame, filename, kind="table", description=f"{name} {filename}")
            if written.value is None:
                writer.abandon()
                return Result.failure(*written.diagnostics)
            written_paths[filename] = written.unwrap()
        figure_notes = _write_figures(writer, name, params, outcome, written_paths) if figures else []
        writer.add_diagnostics(*outcome.diagnostics, *figure_notes)
        env = writer.finalize()
    except (ValueError, FileExistsError, OSError) as exc:
        writer.abandon()
        return Result.failure(Diagnostic.error("BATCH_WRITE_FAILED", f"could not publish {name}: {exc}", tool=name))
    if env.value is None:
        return Result.failure(*env.diagnostics)
    # Derive the cached artifact list from the published envelope rather than
    # from the frames this batch happened to pass in. resume._intact_record
    # compares the manifest against the envelope, so any artifact the writer
    # adds on its own -- readout.md, a schema sidecar -- must appear in both or
    # every child looks damaged and is silently re-run.
    published = [artifact.path for artifact in env.unwrap().artifacts]
    record: dict[str, Any] = {
        "tool": name,
        "ok": outcome.ok,
        "seconds": outcome.seconds,
        "run_dir": writer.run_dir.name,
        "artifacts": [f"{writer.run_dir.name}/{filename}" for filename in published],
        "diagnostics": [diag.to_dict() for diag in outcome.diagnostics],
        "reuse_key": current_key(name, params, corpus),
        "artifact_sha256": {
            f"{writer.run_dir.name}/{filename}": hash_file(writer.run_dir / filename)
            for filename in (*published, "result.json")
        },
    }
    return Result.success((record, writer.run_dir))


def _write_figures(
    writer: OutputWriter,
    name: str,
    params: Mapping[str, object],
    outcome: ToolOutcome,
    written: Mapping[str, Path],
) -> list[Diagnostic]:
    """The run's publication figures, through its own writer; never fatal.

    A figure is a view of the tables, not part of the result: whatever goes
    wrong drawing one is reported on the run and the run still publishes.
    """
    try:
        from core.viz.run_figures import run_figures  # noqa: PLC0415 - matplotlib loads only when figures are wanted

        hashes = {filename: hash_file(path) for filename, path in written.items() if path.suffix == ".csv"}
        drawn, notes = run_figures(name, outcome.frames, settings=dict(params), hashes=hashes)
    except Exception as exc:  # the tables stand without their pictures (R4)
        return [Diagnostic.warning("RUN_FIGURES_FAILED", f"figures were not drawn: {type(exc).__name__}: {exc}")]
    for figure in drawn:
        if figure.filename.endswith(".md"):
            result = writer.write_text(
                figure.data.decode("utf-8"), figure.filename, kind="report", description=figure.description
            )
        else:
            result = writer.write_bytes(figure.data, figure.filename, kind="image", description=figure.description)
        if result.value is None:
            notes.extend(result.diagnostics)
    return notes


def write_batch(output_root: str | Path, request: BatchRequest) -> Result[BatchReport]:
    """Publish one child run per successful outcome plus a parent manifest.

    ``request.reuse`` links prior child records (from
    :func:`resume.find_reusable`) for tools the batch did not re-run; a
    fresh outcome always wins over a stale link. Every batch also renders
    ``report.md`` beside ``batch.json``.
    """
    root = Path(output_root)
    by_outcome = {outcome.name: outcome for outcome in request.batch.outcomes}
    child_dirs: dict[str, Path] = {}
    children: list[dict[str, Any]] = []
    for name in request.selection:
        outcome = by_outcome.get(name)
        prior = request.reuse.get(name)
        if outcome is not None:
            if not outcome.ok and not outcome.frames:
                children.append(
                    {
                        "tool": outcome.name,
                        "ok": False,
                        "seconds": outcome.seconds,
                        "run_dir": None,
                        "artifacts": [],
                        "diagnostics": [diag.to_dict() for diag in outcome.diagnostics],
                    }
                )
            else:
                published = _write_child_run(
                    root, request.corpus, name, request.params.get(name, {}), outcome, figures=request.figures
                )
                if published.value is None:
                    return Result.failure(*published.diagnostics)
                record, child_dir = published.unwrap()
                child_dirs[name] = child_dir
                children.append(record)
        elif prior is not None:
            record = dict(prior)
            record["reused"] = True
            record["diagnostics"] = [
                *record.get("diagnostics", []),
                Diagnostic.info(
                    "PROFILER_REUSED", f"{name} reused from {record.get('run_dir', '?')}", tool=name
                ).to_dict(),
            ]
            children.append(record)
        else:
            children.append(
                {
                    "tool": name,
                    "ok": False,
                    "seconds": 0.0,
                    "run_dir": None,
                    "artifacts": [],
                    "diagnostics": [
                        Diagnostic.error("BATCH_OUTCOME_MISSING", f"no outcome and no reuse link for {name}").to_dict()
                    ],
                }
            )

    return _write_parent(root, request, children, child_dirs)


def _write_parent(
    root: Path,
    request: BatchRequest,
    children: list[dict[str, Any]],
    child_dirs: dict[str, Path],
) -> Result[BatchReport]:
    """Publish the parent manifest after child outcomes have been collected."""
    manifest: dict[str, Any] = {
        "tool": "profiler",
        "selection": list(request.selection),
        "children": children,
    }
    try:
        parent = OutputWriter(
            root,
            tool="profiler",
            params={"analyses": list(request.selection), "params": _jsonable(dict(request.params))},
            corpus=request.corpus,
        )
    except (ValueError, OSError) as exc:
        return Result.failure(Diagnostic.error("BATCH_WRITE_FAILED", f"could not stage batch: {exc}"))
    try:
        parent.add_diagnostics(*(diag for outcome in request.batch.outcomes for diag in outcome.diagnostics))
        written = parent.write_json(manifest, "batch.json", kind="manifest", description="batch manifest")
        if written.value is None:
            parent.abandon()
            return Result.failure(*written.diagnostics)
        text = parent.write_text(render_report(manifest), "report.md", kind="report", description="batch report")
        if text.value is None:
            parent.abandon()
            return Result.failure(*text.diagnostics)
        env = parent.finalize()
    except (ValueError, FileExistsError, OSError) as exc:
        parent.abandon()
        return Result.failure(Diagnostic.error("BATCH_WRITE_FAILED", f"could not publish batch manifest: {exc}"))
    if env.value is None:
        return Result.failure(*env.diagnostics)
    return Result.success(BatchReport(run_dir=parent.run_dir, child_dirs=child_dirs, envelope=env.unwrap()))
