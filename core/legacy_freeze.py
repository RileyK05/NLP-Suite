"""Legacy environment freeze (FR-1.3) — pin what "the oracle" means.

The freezer only reads plain files: no git binary, no legacy imports, no
network. It records the oracle commit (resolved from ``.git`` file reads),
the release version, dependency manifests (names + whether anything is
pinned), external-software rows, a top-level ``lib/`` inventory, ``src/``
size, a tree fingerprint, and capture metadata.

Honest limits, stated in the record itself (C6-16):

* Legacy requirements are unpinned (2/77 pinned), so exact legacy package
  versions are unrecoverable from manifests — the freeze says so instead of
  guessing. The capture ran under Python 3.12 (the available interpreter),
  NOT the legacy 3.10 runtime; a complete py3.10 freeze additionally needs
  the installed-package list, package hashes, model versions/checksums, and
  executable probes (java -version etc.). Until that exists, treat this
  record as a FILE inventory, not a reproducible environment freeze.
* The tree fingerprint covers relative paths + sizes (plus content hashes
  of the small manifests). It detects added/removed/resized files, not a
  same-size content edit.
* Files larger than ``hash_limit_bytes`` are sized, not hashed.
* ``working_tree`` remains an operator attestation; verify with git.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import hashlib
import os
from pathlib import Path
import platform
import re
import sys

from core.result import Diagnostic, Result

__all__ = [
    "ExternalSoftware",
    "FreezeRecord",
    "LibEntry",
    "RequirementFile",
    "freeze_oracle",
    "parse_requirement_names",
]

_DEFAULT_HASH_LIMIT = 64 * 1024 * 1024
_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_SOFTWARE_COLUMNS = 3
_PACKED_REF_COLUMNS = 2


@dataclass(frozen=True, slots=True)
class RequirementFile:
    path: str  # oracle-relative
    names: tuple[str, ...]
    pinned: tuple[bool, ...]  # parallel to names: True where a == pin was present


@dataclass(frozen=True, slots=True)
class ExternalSoftware:
    name: str
    install_path: str
    download_link: str


@dataclass(frozen=True, slots=True)
class LibEntry:
    name: str  # lib/-relative
    kind: str  # "dir" or "file"
    size: int
    sha256: str  # "" for dirs and for files over the hash limit
    hashed: bool


@dataclass(frozen=True, slots=True)
class FreezeRecord:
    oracle_dir: str
    captured: str  # ISO timestamp of the freeze, clearly capture metadata
    capture_platform: str
    capture_python: str
    is_git_checkout: bool
    head_raw: str
    commit: str  # "" when unresolvable
    release_version: str
    python_hints: tuple[str, ...]  # raw evidence lines, e.g. "python-version: '3.10'"
    requirements: tuple[RequirementFile, ...] = ()
    external_software: tuple[ExternalSoftware, ...] = ()
    lib_entries: tuple[LibEntry, ...] = ()
    src_file_count: int = 0
    src_total_bytes: int = 0
    tree_fingerprint: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)
    working_tree: str = ""  # operator-attested "clean"/"dirty"; "" means unattested

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_dir": self.oracle_dir,
            "captured": self.captured,
            "capture_platform": self.capture_platform,
            "capture_python": self.capture_python,
            "is_git_checkout": self.is_git_checkout,
            "head_raw": self.head_raw,
            "commit": self.commit,
            "release_version": self.release_version,
            "python_hints": list(self.python_hints),
            "requirements": [
                {"path": r.path, "names": list(r.names), "pinned": list(r.pinned)} for r in self.requirements
            ],
            "external_software": [
                {"name": s.name, "install_path": s.install_path, "download_link": s.download_link}
                for s in self.external_software
            ],
            "lib_entries": [
                {"name": e.name, "kind": e.kind, "size": e.size, "sha256": e.sha256, "hashed": e.hashed}
                for e in self.lib_entries
            ],
            "src_file_count": self.src_file_count,
            "src_total_bytes": self.src_total_bytes,
            "tree_fingerprint": self.tree_fingerprint,
            "warnings": list(self.warnings),
        }

    def to_markdown(self) -> str:
        lines = [
            "# Legacy environment freeze",
            "",
            "Frozen oracle record for the NLP Suite 1.6.x reference tree. "
            "Regenerate with `python -m tools.freeze_legacy --oracle <dir>` and diff.",
            "",
            "## Identity",
            "",
            f"- oracle dir: `{self.oracle_dir}`",
            f"- git checkout: {'yes' if self.is_git_checkout else 'no'}",
            f"- HEAD: `{self.head_raw or 'n/a'}`",
            f"- commit: `{self.commit or 'UNRESOLVED'}`",
            *([f"- working tree (operator-attested): {self.working_tree}"] if self.working_tree else []),
            f"- release version: `{self.release_version or 'unknown'}`",
            f"- captured: {self.captured} on {self.capture_platform} / Python {self.capture_python}",
            "",
            "## Python runtime",
            "",
        ]
        if self.python_hints:
            lines.extend(f"- `{hint}`" for hint in self.python_hints)
        else:
            lines.append("- no Python version evidence found in manifests")
        lines.extend(["", "## Requirements (names only; pins are the exception)", ""])
        if self.requirements:
            for req in self.requirements:
                pinned_n = sum(1 for p in req.pinned if p)
                count = len(req.names)
                lines.append(f"- `{req.path}`: {count} package{'s' if count != 1 else ''}, {pinned_n} pinned")
        else:
            lines.append("- no requirements files found")
        lines.extend(["", "## External software (legacy config, machine-specific paths)", ""])
        if self.external_software:
            lines.append("| Software | Install path | Download |")
            lines.append("|---|---|---|")
            for soft in self.external_software:
                lines.append(f"| {soft.name} | `{soft.install_path}` | {soft.download_link} |")
        else:
            lines.append("- none recorded")
        lines.extend(
            [
                "",
                "## lib/ inventory (top level)",
                "",
                f"- {len(self.lib_entries)} entries",
                f"- src/: {self.src_file_count} .py files, {self.src_total_bytes} bytes",
                f"- tree fingerprint: `{self.tree_fingerprint or 'n/a'}`",
                "",
                "## Warnings",
                "",
            ]
        )
        if self.warnings:
            lines.extend(f"- {warning}" for warning in self.warnings)
        else:
            lines.append("- none")
        lines.append("")
        return "\n".join(lines)


def parse_requirement_names(text: str) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    """Split requirements text into (package names, ==-pinned flags).

    Names keep extras (``fuzzywuzzy[speedup]``) but drop version specifiers
    and environment markers; comments and blanks are skipped.
    """
    names: list[str] = []
    pinned: list[bool] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line and "://" not in line:
            line = line.split("#", 1)[0].strip()
        clause = line.split(";", 1)[0].strip()
        if not clause:
            continue
        pinned.append("==" in clause)
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            if sep in clause:
                clause = clause.split(sep, 1)[0].strip()
                break
        names.append(clause)
    return tuple(names), tuple(pinned)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError):
        return None


def _resolve_commit(git_dir: Path) -> tuple[bool, str, str]:
    """Resolve the oracle commit from ``.git`` file reads (no git binary)."""
    head_file = git_dir / "HEAD"
    head_text = _read_text(head_file)
    if head_text is None:
        return False, "", ""
    head_raw = head_text.strip()
    if not head_raw.startswith("ref:"):
        return True, head_raw, head_raw  # detached HEAD: the ref IS the commit
    ref = head_raw[4:].strip()
    loose = _read_text(git_dir / ref)
    if loose is not None and loose.strip():
        return True, head_raw, loose.strip()
    packed_text = _read_text(git_dir / "packed-refs")
    if packed_text is not None:
        for packed_line in packed_text.splitlines():
            parts = packed_line.split()
            if len(parts) == _PACKED_REF_COLUMNS and parts[1] == ref:
                return True, head_raw, parts[0]
    return True, head_raw, ""


def _python_hints(root: Path) -> tuple[str, ...]:
    """Raw Python-version evidence lines from manifests (uninterpreted)."""
    hints: list[str] = []
    pyproject = _read_text(root / "pyproject.toml")
    if pyproject is not None:
        for line in pyproject.splitlines():
            stripped = line.strip()
            if stripped.startswith(("requires-python", "target-version")):
                hints.append(f"pyproject.toml: {stripped}")
    workflows = root / ".github" / "workflows"
    if workflows.is_dir():
        for workflow in sorted(workflows.glob("*.yml")):
            text = _read_text(workflow)
            if text is None:
                continue
            for line in text.splitlines():
                if "python-version" in line and re.search(r"\d+\.\d+", line):
                    hints.append(f"{workflow.name}: {line.strip()}")
    return tuple(hints[:10])


def _requirement_files(root: Path, warnings: list[str]) -> tuple[RequirementFile, ...]:
    found: list[RequirementFile] = []
    for name in ("requirements.txt", "requirements-windows.txt", "requirements-mac.txt"):
        text = _read_text(root / name)
        if text is None:
            continue
        names, pinned = parse_requirement_names(text)
        found.append(RequirementFile(path=name, names=names, pinned=pinned))
        if names and not any(pinned):
            count = len(names)
            warnings.append(
                f"{name} is fully unpinned ({count} package{'s' if count != 1 else ''}, "
                "no == pins): exact versions unrecoverable"
            )
    if not found:
        warnings.append("no requirements files found at the oracle root")
    return tuple(found)


def _external_software(root: Path, warnings: list[str]) -> tuple[ExternalSoftware, ...]:
    config = root / "config" / "NLP_setup_external_software_config.csv"
    text = _read_text(config)
    if text is None:
        return ()
    try:
        rows = list(csv.reader(text.splitlines()))
    except csv.Error:
        warnings.append("external software config is not parseable CSV")
        return ()
    expected_header = ["software", "installation_path", "download_link"]
    if not rows or [h.strip().lower() for h in rows[0][:_SOFTWARE_COLUMNS]] != expected_header:
        warnings.append("external software config has an unexpected header; rows ignored")
        return ()
    entries = []
    for row in rows[1:]:
        if len(row) < _SOFTWARE_COLUMNS or not row[0].strip():
            continue
        entries.append(ExternalSoftware(name=row[0].strip(), install_path=row[1].strip(), download_link=row[2].strip()))
    return tuple(entries)


def _sha_file(path: Path, limit: int) -> tuple[str, bool]:
    try:
        if path.stat().st_size > limit:
            return "", False
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
    except OSError:
        return "", False
    return digest.hexdigest(), True


def _lib_entries(root: Path, limit: int) -> tuple[LibEntry, ...]:
    lib = root / "lib"
    if not lib.is_dir():
        return ()
    entries: list[LibEntry] = []
    try:
        names = sorted(p.name for p in lib.iterdir())
    except OSError:
        return ()
    for name in names:
        path = lib / name
        if path.is_dir():
            entries.append(LibEntry(name=name, kind="dir", size=0, sha256="", hashed=False))
        elif path.is_file():
            try:
                size = path.stat().st_size
            except OSError:
                continue
            sha, hashed = _sha_file(path, limit)
            entries.append(LibEntry(name=name, kind="file", size=size, sha256=sha, hashed=hashed))
    return tuple(entries)


def _src_stats(root: Path) -> tuple[int, int]:
    src = root / "src"
    if not src.is_dir():
        return 0, 0
    count = 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            count += 1
            try:
                total += (Path(dirpath) / filename).stat().st_size
            except OSError:
                continue
    return count, total


def _tree_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    manifest: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for filename in filenames:
            full = Path(dirpath) / filename
            try:
                size = full.stat().st_size
            except OSError:
                continue
            manifest.append(f"{full.relative_to(root).as_posix()}\t{size}")
    for line in sorted(manifest):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def freeze_oracle(
    oracle_dir: Path,
    hash_limit_bytes: int = _DEFAULT_HASH_LIMIT,
    working_tree: str = "",
) -> Result[FreezeRecord]:
    """Freeze the legacy oracle at *oracle_dir* into a reproducible record.

    *working_tree* is an operator attestation (``"clean"``/``"dirty"``) from
    running ``git status`` by hand — the freezer never shells git.
    """
    if working_tree not in ("", "clean", "dirty"):
        raise ValueError("working_tree must be '', 'clean', or 'dirty'")
    root = Path(oracle_dir)
    if not root.is_dir():
        return Result.failure(
            Diagnostic.error("FREEZE_NO_ORACLE", f"oracle directory not found: {root}", path=str(root))
        )
    warnings: list[str] = []
    git_dir = root / ".git"
    if git_dir.is_dir():
        is_git, head_raw, commit = _resolve_commit(git_dir)
    else:
        is_git, head_raw, commit = False, "", ""
        warnings.append("oracle is not a git checkout: commit attribution is unavailable")
    if is_git and not commit:
        warnings.append("oracle HEAD commit is unresolvable from .git file reads")
    release_text = _read_text(root / "lib" / "release_version.txt")
    release_version = (release_text or "").strip()
    if not release_version:
        warnings.append("lib/release_version.txt missing or empty")
    src_count, src_bytes = _src_stats(root)
    record = FreezeRecord(
        oracle_dir=str(root),
        captured=datetime.now(UTC).isoformat(timespec="seconds"),
        capture_platform=platform.platform(),
        capture_python=sys.version.split()[0],
        is_git_checkout=is_git,
        head_raw=head_raw,
        commit=commit,
        working_tree=working_tree,
        release_version=release_version,
        python_hints=_python_hints(root),
        requirements=_requirement_files(root, warnings),
        external_software=_external_software(root, warnings),
        lib_entries=_lib_entries(root, hash_limit_bytes),
        src_file_count=src_count,
        src_total_bytes=src_bytes,
        tree_fingerprint=_tree_fingerprint(root),
        warnings=tuple(warnings),
    )
    if not record.python_hints:
        record = replace(
            record,
            warnings=(*record.warnings, "no Python version evidence found in manifests"),
        )
    return Result.success(record)
