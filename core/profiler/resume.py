"""Batch resume (FR-7.5) — fingerprint-gated reuse of previous runs.

A tool result is reusable only when the tool version, parameters, corpus
fingerprint, and known backend/model/asset versions all match the stored
reuse key. Corpus identity reuses the reader's order-independent
fingerprint (``Corpus.sha256``). Automatic reuse fingerprints core source,
runtime package versions, and explicit file parameters, then verifies the
published artifacts and envelope hashes. Parsed/model/asset-dependent tools
are rerun until authoritative dependency fingerprints are wired through.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path, PureWindowsPath
from typing import Any

from core.artifacts.envelope import Envelope
from core.io.reader import Corpus, hash_file
from core.profiler.registry import get_tool

__all__ = ["find_reusable", "load_manifest", "reuse_key"]


def reuse_key(
    tool: str,
    spec_version: str,
    params: Mapping[str, object],
    corpus_fp: str,
    versions: Mapping[str, str] | None = None,
) -> str:
    """Stable reuse identity for one tool run."""
    payload = json.dumps(
        {
            "tool": tool,
            "spec": spec_version,
            "params": params,
            "corpus": corpus_fp,
            "versions": dict(versions or {}),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_key(tool: str, params: Mapping[str, object], corpus: Corpus | None) -> str | None:
    """Reuse key for *tool* as it would run right now."""
    spec = get_tool(tool)
    # Until the shared parse/model/asset identities are recorded, re-execute
    # those tools. An empty versions mapping must never authorize stale reuse.
    if (
        spec is None
        or corpus is None
        or spec.requires_parse
        or spec.assets
        or spec.optional_package
        or (tool == "search" and params.get("mode") != "text")
    ):
        return None
    versions: dict[str, str] = {}
    try:
        for package in ("numpy", "pandas", "scipy", "scikit-learn"):
            versions[package] = version(package)
        source_root = Path(__file__).resolve().parents[1]
        source_hash = hashlib.sha256()
        for source in sorted(source_root.rglob("*.py")):
            source_hash.update(source.relative_to(source_root).as_posix().encode("utf-8"))
            source_hash.update(hash_file(source).encode("ascii"))
        versions["core-source"] = source_hash.hexdigest()
        for param in spec.params:
            path = params.get(param.name)
            if param.type == "path" and path is not None:
                versions[f"file:{param.name}"] = hash_file(Path(str(path)))
    except (OSError, PackageNotFoundError):
        return None
    return reuse_key(tool, spec.version, params, corpus.sha256, versions)


def _safe_path(root: Path, relative: object) -> Path | None:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        return None
    path = Path(relative)
    if path.is_absolute() or PureWindowsPath(relative).drive or ".." in path.parts:
        return None
    resolved = (root / path).resolve()
    return resolved if resolved.is_relative_to(root) else None


def _intact_record(root: Path, name: str, record: dict[str, Any]) -> bool:
    """Verify every cached file against the hashes captured on publication."""
    directory = _safe_path(root, record.get("run_dir"))
    hashes = record.get("artifact_sha256")
    artifacts = record.get("artifacts")
    if directory is None or not isinstance(hashes, dict) or not isinstance(artifacts, list):
        return False
    envelope = Envelope.read(directory)
    if (
        not envelope.ok
        or envelope.unwrap().tool != name
        or any(diag.severity.value == "ERROR" for diag in envelope.unwrap().diagnostics)
    ):
        return False
    expected = [f"{record['run_dir']}/{item.path}" for item in envelope.unwrap().artifacts]
    if artifacts != expected or set(hashes) != {*expected, f"{record['run_dir']}/result.json"}:
        return False
    try:
        for relative, digest in hashes.items():
            target = _safe_path(root, relative)
            if target is None or not target.is_relative_to(directory) or hash_file(target) != digest:
                return False
    except OSError:
        return False
    return True


def load_manifest(path: str | Path) -> dict[str, Any] | None:
    """Read a prior batch.json (run dir or file); None when absent/unreadable."""
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "batch.json"
    if not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def find_reusable(
    prior: str | Path,
    selection: Sequence[str],
    params: Mapping[str, Mapping[str, object]],
    corpus: Corpus | None,
    *,
    output_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Prior child records reusable for this selection (ok + key match)."""
    manifest = load_manifest(prior)
    if not manifest:
        return {}
    children = manifest.get("children", [])
    if not isinstance(children, list):
        return {}
    prior_path = Path(prior).resolve()
    root = prior_path.parent if prior_path.is_dir() else prior_path.parent.parent
    if output_root is not None and Path(output_root).resolve() != root:
        return {}
    prior_records = {
        c["tool"]: c for c in children if isinstance(c, dict) and c.get("ok") is True and isinstance(c.get("tool"), str)
    }
    reuse: dict[str, dict[str, Any]] = {}
    for name in selection:
        record = prior_records.get(name)
        if record is None or record.get("reuse_key") is None:
            continue
        if record["reuse_key"] == current_key(name, params.get(name, {}), corpus) and _intact_record(
            root, name, record
        ):
            reuse[name] = record
    return reuse
