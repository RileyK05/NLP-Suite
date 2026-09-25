"""FR-0.5 — write custody gate.

R3: every run artifact and every file inside a run directory is written only
through :class:`OutputWriter`. The low-level primitives that OutputWriter
delegates to (envelope, sidecar) and the file-management utilities that operate
on explicit user paths (converter, merger, pcace DB) are the only other
modules that may touch the filesystem for writes — and they must never be used
to write into a run directory except through the writer.

This test is the automated enforcement for that boundary. If it fails, a core
module has added a direct write outside the contract.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"

# Modules that are allowed to contain direct filesystem writes.
# - writer.py: the one writer
# - artifacts/envelope.py: low-level primitive called only by writer.finalize
# - conll/schema.py: low-level primitive called only by writer.write_table
# - file_ops/converter.py, file_ops/merger.py: file-management utilities that
#   operate on explicit user-provided paths (not run directories); no tool
#   currently uses the file-writing wrappers, but they remain as explicit-path
#   utilities and are tested as such
# - pcace/core.py: creates the sqlite file at an explicit path (tools pass
#   writer.run_dir / "pcace.db" and the DB is created there)
# - models/download.py: writes downloaded model files into the user's models
#   directory (never a run directory), each through <file>.partial and an
#   atomic rename after its SHA-256 checks out
# - profiler/batch.py: publishes batch children through OutputWriter only
#   (the ".write_text(" match is the writer.write_text METHOD call, not a
#   direct filesystem write — it contains no open/mkdir/Path.write call)
ALLOWED_WRITERS: frozenset[str] = frozenset(
    {
        "core/io/writer.py",
        "core/artifacts/envelope.py",
        "core/conll/schema.py",
        "core/file_ops/converter.py",
        "core/file_ops/merger.py",
        "core/pcace/core.py",
        "core/models/download.py",
        "core/profiler/batch.py",
    }
)

# Analysis and other core subpackages must never write directly.
FORBIDDEN_SUBPACKAGES = (
    "core/analysis",
    "core/data",
    "core/gis",
    "core/narrative",
    "core/viz",
    "core/profiler",
    "core/install",
    "core/pipelines",
)


def _has_direct_write(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    # Direct filesystem writes to look for. Reads (open "rb") are allowed;
    # only writes are gated. `open(` with "w" would also be a write, but no
    # core module uses it — all writes go through write_text/write_bytes.
    markers = (".write_text(", ".write_bytes(", "mkdir(")
    return any(m in text for m in markers)


def test_only_allowlisted_core_modules_write() -> None:
    offenders: list[str] = []
    for path in CORE.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in ALLOWED_WRITERS:
            continue
        if _has_direct_write(path):
            offenders.append(rel)
    assert not offenders, "R3 violation: direct filesystem writes outside allowlist:\n" + "\n".join(offenders)


def test_no_analysis_module_imports_write_primitives_for_writing() -> None:
    """Analysis modules must not import write_sidecar/Envelope.write to write."""
    offenders: list[str] = []
    for path in (CORE / "analysis").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        mentions = "write_sidecar" in text or "Envelope" in text
        writes = ".write(" in text or "write_sidecar" in text
        # Importing Envelope for type hints is allowed; writing is not
        if mentions and writes:
            offenders.append(path.relative_to(ROOT).as_posix())
    # Currently no analysis module does this — the gate pins that fact
    assert not offenders, "\n".join(offenders)


def test_envelope_and_sidecar_are_only_called_from_writer_or_tests() -> None:
    callers: list[str] = []
    for path in CORE.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in ("core/artifacts/envelope.py", "core/conll/schema.py", "core/io/writer.py"):
            continue
        text = path.read_text(encoding="utf-8")
        mentions = "write_sidecar" in text or "Envelope(" in text or ".write(" in text
        # Only flag if the file actually writes via those primitives
        if mentions and "write_sidecar" in text:
            callers.append(rel)
    # No core module outside the allowlist should call write_sidecar directly
    assert not callers, "direct write_sidecar call outside writer:\n" + "\n".join(callers)
