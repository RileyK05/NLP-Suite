"""The artifact envelope — the chain-of-custody log.

ARCHITECTURE.md invariants enforced here:

* **R8** — tools interoperate via envelopes, never by globbing filenames.
* **R3** — the envelope is written only by :class:`OutputWriter`.

Every run writes a ``result.json``. The file is the complete provenance:
what was run, with what parameters, on what inputs, producing which
artifacts, with which diagnostics. Downstream tools and the viewer consume
this file, not the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from core.result import Diagnostic, Result, Severity

__all__ = ["ENVELOPE_VERSION", "Artifact", "Envelope", "InputRef"]


ENVELOPE_VERSION: int = 1


def _round_floats(value: object) -> object:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {str(k): _round_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True, slots=True)
class InputRef:
    """One input file, as recorded in the envelope."""

    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: object) -> InputRef:
        if not isinstance(data, dict):
            raise TypeError(f"InputRef must be an object, got {type(data).__name__}")
        path = data.get("path")
        sha256 = data.get("sha256")
        if not isinstance(path, str) or not isinstance(sha256, str):
            raise TypeError("InputRef path and sha256 must be strings")
        return cls(path=path, sha256=sha256)


@dataclass(frozen=True, slots=True)
class Artifact:
    """One output artifact, as recorded in the envelope.

    ``kind`` is a free string (``table``, ``chart``, ``map``, ...) so new
    kinds need no schema change. The viewer degrades unknown kinds to
    download-only.
    """

    kind: str
    path: str
    description: str = ""

    def to_dict(self) -> dict[str, str]:
        data: dict[str, str] = {"kind": self.kind, "path": self.path}
        if self.description:
            data["description"] = self.description
        return data

    @classmethod
    def from_dict(cls, data: object) -> Artifact:
        if not isinstance(data, dict):
            raise TypeError(f"Artifact must be an object, got {type(data).__name__}")
        kind = data.get("kind")
        path = data.get("path")
        if not isinstance(kind, str) or not isinstance(path, str):
            raise TypeError("Artifact kind and path must be strings")
        description = data.get("description", "")
        if not isinstance(description, str):
            raise TypeError("Artifact description must be a string")
        if not kind:
            raise ValueError("Artifact kind must be a non-empty string")
        if not path:
            raise ValueError("Artifact path must be a non-empty string")
        return cls(kind=kind, path=path, description=description)


@dataclass(frozen=True, slots=True)
class Envelope:
    """The provenance envelope for a single tool run.

    Written once per run as ``result.json``. Every field is frozen; a
    finished envelope is never mutated.
    """

    tool: str
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    params: dict[str, Any] = field(default_factory=dict)
    inputs: tuple[InputRef, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    schema_version: int = ENVELOPE_VERSION
    corpus_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.tool:
            raise ValueError("Envelope tool must be a non-empty string")
        if self.schema_version != ENVELOPE_VERSION:
            raise ValueError(f"unsupported envelope version {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "created": self.created,
            "schema_version": self.schema_version,
            "params": _round_floats(self.params),
            "inputs": [item.to_dict() for item in self.inputs],
            "artifacts": [item.to_dict() for item in self.artifacts],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "corpus_sha256": self.corpus_sha256,
        }

    @classmethod
    def from_dict(cls, data: object) -> Envelope:
        if not isinstance(data, dict):
            raise TypeError(f"Envelope must be an object, got {type(data).__name__}")
        tool = data.get("tool")
        if not isinstance(tool, str) or not tool:
            raise ValueError("Envelope tool must be a non-empty string")
        created = data.get("created", "")
        if not isinstance(created, str):
            raise TypeError("Envelope created must be a string")
        schema_version = data.get("schema_version", ENVELOPE_VERSION)
        if not isinstance(schema_version, int):
            raise TypeError("Envelope schema_version must be an integer")
        params = data.get("params", {})
        if not isinstance(params, dict):
            raise TypeError("Envelope params must be an object")
        raw_inputs = data.get("inputs", [])
        if not isinstance(raw_inputs, list):
            raise TypeError("Envelope inputs must be a list")
        inputs = tuple(InputRef.from_dict(item) for item in raw_inputs)
        raw_artifacts = data.get("artifacts", [])
        if not isinstance(raw_artifacts, list):
            raise TypeError("Envelope artifacts must be a list")
        artifacts = tuple(Artifact.from_dict(item) for item in raw_artifacts)
        raw_diagnostics = data.get("diagnostics", [])
        if not isinstance(raw_diagnostics, list):
            raise TypeError("Envelope diagnostics must be a list")
        diagnostics: list[Diagnostic] = []
        for item in raw_diagnostics:
            if not isinstance(item, dict):
                raise TypeError("diagnostic must be an object")
            severity_raw = item.get("severity")
            code = item.get("code")
            message = item.get("message")
            context = item.get("context", {})
            if not isinstance(severity_raw, str) or not isinstance(code, str) or not isinstance(message, str):
                raise TypeError("diagnostic severity, code and message must be strings")
            if not isinstance(context, dict):
                raise TypeError("diagnostic context must be an object")
            try:
                severity = Severity(severity_raw)
            except ValueError as exc:
                raise ValueError(f"unknown diagnostic severity {severity_raw!r}") from exc
            diagnostics.append(Diagnostic(severity=severity, code=code, message=message, context=dict(context)))
        corpus_sha256 = data.get("corpus_sha256", "")
        if not isinstance(corpus_sha256, str):
            raise TypeError("Envelope corpus_sha256 must be a string")
        return cls(
            tool=tool,
            created=created,
            params=dict(params),
            inputs=inputs,
            artifacts=artifacts,
            diagnostics=tuple(diagnostics),
            schema_version=schema_version,
            corpus_sha256=corpus_sha256,
        )

    def write(self, run_dir: Path) -> Result[Path]:
        """Write this envelope as ``result.json`` inside *run_dir*."""
        target = Path(run_dir) / "result.json"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self.to_dict(), indent=2, sort_keys=False, ensure_ascii=False) + "\n"
            target.write_text(payload, encoding="utf-8")
        except OSError as exc:
            return Result.failure(
                Diagnostic.error("ENVELOPE_WRITE_FAILED", f"could not write {target}: {exc}", path=str(target))
            )
        return Result.success(target)

    @classmethod
    def read(cls, run_dir: Path) -> Result[Envelope]:
        """Read ``result.json`` from *run_dir*."""
        source = Path(run_dir) / "result.json"
        if not source.is_file():
            return Result.failure(Diagnostic.error("ENVELOPE_MISSING", f"no envelope at {source}", path=str(source)))
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return Result.failure(
                Diagnostic.error("ENVELOPE_UNREADABLE", f"could not read {source}: {exc}", path=str(source))
            )
        try:
            return Result.success(cls.from_dict(payload))
        except (ValueError, TypeError) as exc:
            return Result.failure(
                Diagnostic.error("ENVELOPE_INVALID", f"envelope {source} is not valid: {exc}", path=str(source))
            )
