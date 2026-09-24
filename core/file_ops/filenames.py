"""Filename operations (FR-3.5) — standardize with preview-first renames.

Standardize rule (explicit, documented): strip, replace every run of
characters outside ``[A-Za-z0-9._-]`` with one underscore, collapse repeats,
keep the extension, fall back to ``file`` (+ extension) when nothing remains.
Date extraction reuses ``reader.date_from_filename`` — never duplicated.

R3 note: renaming mutates inputs, which reads never do. This module is the
one sanctioned exception and it earns it structurally: ``plan_renames``
previews everything (including collisions and existing targets),
``apply_renames`` defaults to ``dry_run=True``, and existing files are never
overwritten (``skip-existing``).

C6-11 hardening: empty plans are a structured result (no paths[0] access);
all files must live in one directory (mixed parents rejected); the plan is
an immutable value (tuple of frozen row records, not a DataFrame callers
can mutate); every source and target is validated to resolve inside the
approved directory (no .., absolute, drive, junction, or symlink escapes);
case-insensitive collision detection (Windows/macOS); source identity is
re-checked between preview and apply (deleted/renamed sources are
diagnosed); rename cycles (a->b, b->a) are ordered via a temporary name;
on failure the applied renames are rolled back.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

import pandas as pd

from core.io.reader import date_from_filename, hash_file
from core.result import Diagnostic, Result

__all__ = ["RenamePlan", "RenameRow", "apply_renames", "plan_renames", "standardize_name"]

_NON_STANDARD_RE = re.compile(r"[^A-Za-z0-9._-]+")
_COLLAPSE_RE = re.compile(r"_+")


def standardize_name(name: str) -> str:
    """Apply the standardize rule to one filename (no filesystem access)."""
    stem, dot, suffix = str(name).rpartition(".")
    if not dot:
        stem, suffix = str(name), ""
    else:
        suffix = "." + suffix
    cleaned = _COLLAPSE_RE.sub("_", _NON_STANDARD_RE.sub("_", stem.strip())).strip("._")
    if not cleaned:
        cleaned = "file"
    return cleaned + suffix


@dataclass(frozen=True, slots=True)
class RenameRow:
    """One previewed rename. Immutable — callers cannot mutate Old/New."""

    old: str
    new: str
    date: str
    action: str  # rename | skip-identical | skip-existing
    source_sha256: str = ""
    source_size: int | None = None


@dataclass(frozen=True, slots=True)
class RenamePlan:
    """Immutable plan value: tuple of frozen rows plus the approved directory."""

    rows: tuple[RenameRow, ...]
    directory: Path

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"Old": row.old, "New": row.new, "Date": row.date, "Action": row.action} for row in self.rows],
            columns=["Old", "New", "Date", "Action"],
        )


def _resolve_inside(directory: Path, name: str) -> Path | None:
    """Resolve *name* inside *directory*, or None on any escape attempt."""
    candidate = (directory / name).resolve()
    try:
        candidate.relative_to(directory.resolve())
    except ValueError:
        return None
    return candidate


def plan_renames(files: list[Path]) -> Result[RenamePlan]:
    """Preview standardized renames: collisions gain _2/_3, clashes skip.

    C6-11: empty input is a structured empty plan (never an index error);
    mixed-parent inputs are rejected; case-insensitive collisions count on
    every platform (Windows/macOS case-insensitive, and case-only renames
    must not silently collide).
    """
    paths = [Path(f) for f in files]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        return Result.failure(Diagnostic.error("FILENAMES_MISSING", f"not files: {missing}", missing=missing))
    if not paths:
        return Result.success(RenamePlan(rows=(), directory=Path.cwd()))
    directory = paths[0].parent
    mixed = sorted({str(p.parent) for p in paths})
    if len(mixed) > 1:
        return Result.failure(
            Diagnostic.error(
                "FILENAMES_MIXED_PARENTS",
                "all files must be in one directory for a rename plan",
                parents=mixed,
            )
        )
    used: set[str] = set()
    used_casefold: set[str] = set()
    source_casefold = {path.name.casefold() for path in paths}
    existing_casefold = {entry.name.casefold() for entry in directory.iterdir()}
    rows: list[RenameRow] = []
    for path in paths:
        old = path.name
        new = standardize_name(old)
        if new == old:
            rows.append(
                RenameRow(
                    old=old,
                    new=new,
                    date=_date_cell(path),
                    action="skip-identical",
                    source_sha256=hash_file(path),
                    source_size=path.stat().st_size,
                )
            )
            used.add(new)
            used_casefold.add(new.casefold())
            continue
        candidate = new
        counter = 2
        while candidate.casefold() in used_casefold:
            stem, dot, suffix = new.rpartition(".")
            if dot:
                candidate = f"{stem}_{counter}.{suffix}"
            else:
                candidate = f"{new}_{counter}"
            counter += 1
        used.add(candidate)
        used_casefold.add(candidate.casefold())
        resolved = _resolve_inside(directory, candidate)
        if resolved is None:
            return Result.failure(
                Diagnostic.error("FILENAMES_ESCAPE", f"planned target escapes directory: {candidate}", target=candidate)
            )
        if candidate.casefold() in existing_casefold and candidate.casefold() not in source_casefold:
            action = "skip-existing"
        else:
            action = "rename"
        rows.append(
            RenameRow(
                old=old,
                new=candidate,
                date=_date_cell(path),
                action=action,
                source_sha256=hash_file(path),
                source_size=path.stat().st_size,
            )
        )
    return Result.success(RenamePlan(rows=tuple(rows), directory=directory))


def _date_cell(path: Path) -> str:
    found = date_from_filename(path)
    return found.isoformat() if found is not None else ""


def _unblocked_candidates(
    candidates: list[tuple[Path, Path, str, str]], diags: list[Diagnostic]
) -> list[tuple[Path, Path, str, str]]:
    """Remove blocked moves and their transitive predecessors before mutating."""
    pending = candidates
    while True:
        active_source_names = {source.name.casefold() for source, _, _, _ in pending}
        retained: list[tuple[Path, Path, str, str]] = []
        for candidate in pending:
            _source, target, _old, _new = candidate
            if target.exists() and target.name.casefold() not in active_source_names:
                diags.append(
                    Diagnostic.warning(
                        "FILENAMES_SKIPPED", f"target appeared since preview: {target.name}", target=target.name
                    )
                )
                continue
            retained.append(candidate)
        if len(retained) == len(pending):
            return retained
        pending = retained


def apply_renames(plan: RenamePlan, *, dry_run: bool = True) -> Result[list[tuple[str, str]]]:
    """Apply a plan. Dry-run (the default) changes nothing and reports [].

    C6-11: source identity is re-checked at apply time (missing sources are
    diagnosed, not crashed on); targets are re-resolved inside the approved
    directory; on any failure all renames applied so far are rolled back.
    a->b / b->a swaps are handled via temporary names.
    """
    if dry_run:
        return Result.success([])
    applied: list[tuple[str, str]] = []
    moves: list[tuple[Path, Path]] = []
    diags: list[Diagnostic] = []
    candidates: list[tuple[Path, Path, str, str]] = []
    for row in plan.rows:
        if row.action != "rename":
            continue
        source = _resolve_inside(plan.directory, row.old)
        target = _resolve_inside(plan.directory, row.new)
        if source is None or target is None:
            return Result.failure(
                Diagnostic.error(
                    "FILENAMES_ESCAPE",
                    f"plan path escapes directory: {row.old!r} -> {row.new!r}",
                    old=row.old,
                    new=row.new,
                )
            )
        if not source.is_file():
            # WARNING: a vanished source is a partial event, not a run
            # failure — the remaining renames still apply (documented
            # continue-with-partial-report transactional behavior).
            diags.append(
                Diagnostic.warning(
                    "FILENAMES_SOURCE_GONE",
                    f"source changed since preview: {row.old}; skipped",
                    source=row.old,
                )
            )
            continue
        if row.source_sha256 and hash_file(source) != row.source_sha256:
            diags.append(
                Diagnostic.warning(
                    "FILENAMES_SOURCE_CHANGED",
                    f"source changed since preview: {row.old}; skipped",
                    source=row.old,
                )
            )
            continue
        if row.source_size is not None and source.stat().st_size != row.source_size:
            diags.append(
                Diagnostic.warning(
                    "FILENAMES_SOURCE_CHANGED",
                    f"source size changed since preview: {row.old}; skipped",
                    source=row.old,
                )
            )
            continue
        candidates.append((source, target, row.old, row.new))
    # A target may be occupied by another source only when that source also
    # survived the identity checks above. If it changed or disappeared since
    # preview, treating its old name as available would overwrite a live file.
    source_names = [source.name.casefold() for source, _, _, _ in candidates]
    target_names = [target.name.casefold() for _, target, _, _ in candidates]
    if len(set(source_names)) != len(source_names) or len(set(target_names)) != len(target_names):
        return Result.failure(Diagnostic.error("FILENAMES_COLLISION", "plan repeats a source or destination"))
    # Removing b->c (c is occupied) makes a->b unsafe too. Recompute until
    # no further blocked predecessors remain; one pass can overwrite b.
    pending = _unblocked_candidates(candidates, diags)
    # Two-phase apply: move every source that is also a target to a private
    # temporary name first, then complete all final moves. This handles swaps
    # and longer cycles without overwriting a source.
    targets = {target for _, target, _, _ in pending}
    cycle_sources = {source for source, _, _, _ in pending if source in targets}
    temps: dict[Path, Path] = {}
    try:
        for source in sorted(cycle_sources, key=lambda p: p.name.casefold()):
            temp = source.with_name(f".renaming-{os.urandom(4).hex()}-{source.name}")
            source.rename(temp)
            moves.append((source, temp))
            temps[source] = temp
        for source, target, old, new in pending:
            current = temps.get(source, source)
            if target.exists():
                raise FileExistsError(f"destination is occupied: {target}")
            current.rename(target)
            moves.append((current, target))
            applied.append((old, new))
    except OSError as exc:
        return _rollback_moves(moves, plan.directory, diags, str(exc))
    return Result.success(applied, *diags)


def _rollback(
    applied: list[tuple[str, str]], directory: Path, diags: list[Diagnostic], failed: str, exc: OSError
) -> Result[list[tuple[str, str]]]:
    """Undo everything applied so far, then fail (C6-11 transactional)."""
    problems = list(diags)
    for old, new in reversed(applied):
        current = directory / new
        original = directory / old
        if current.is_file():
            try:
                current.rename(original)
            except OSError as rollback_exc:
                problems.append(
                    Diagnostic.error("FILENAMES_ROLLBACK_FAILED", f"{new} -> {old}: {rollback_exc}", source=new)
                )
    problems.append(
        Diagnostic.error("FILENAMES_FAILED", f"could not rename {failed}: {exc} — rolled back", source=failed)
    )
    return Result.failure(*problems)


def _rollback_moves(
    moves: list[tuple[Path, Path]], directory: Path, diags: list[Diagnostic], failure: str
) -> Result[list[tuple[str, str]]]:
    """Rollback exact path moves in reverse order, including cycle temps."""
    problems = list(diags)
    for source, target in reversed(moves):
        if not target.exists():
            continue
        try:
            target.rename(source)
        except OSError as exc:
            problems.append(
                Diagnostic.error(
                    "FILENAMES_ROLLBACK_FAILED", f"{target.name} -> {source.name}: {exc}", source=target.name
                )
            )
    problems.append(
        Diagnostic.error("FILENAMES_FAILED", f"rename failed: {failure} — rolled back", directory=str(directory))
    )
    return Result.failure(*problems)
