"""OutputWriter — the only thing in the system that writes files.

ARCHITECTURE.md invariants enforced here:

* **R3** — reads never write. This module is the ONLY place that writes. It
  refuses any output path that lies inside an input directory, so a tool
  cannot overwrite its own corpus even if asked.
* **R8** — every run carries a manifest (the envelope) with input hashes,
  parameters, outputs and diagnostics.

A run directory is ``{output_root}/{tool}__{timestamp}/``. The writer creates
it once and every artifact for that run lives inside it.

The writer never consults global state; everything it needs is passed to its
constructor alongside a frozen :class:`NLPConfig`.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any
import uuid

import pandas as pd

from core.artifacts.envelope import Artifact, Envelope, InputRef, _round_floats
from core.conll.schema import CoNLLSchema, write_sidecar
from core.io.reader import Corpus, Document, hash_file
from core.result import Diagnostic, Result

__all__ = ["OutputWriter", "is_inside"]


def _timestamp() -> str:
    base = datetime.now(UTC).isoformat(timespec="milliseconds").replace(":", "-").replace(".", "-")
    return f"{base}-{uuid.uuid4().hex[:6]}"


def is_inside(child: Path, parent: Path) -> bool:
    """Whether *child* is inside *parent* (or equal to it)."""
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _validate_run_path(state: str, run_dir: Path, filename: str) -> Result[Path]:
    """Shared pre-write checks: staging state, sane name, reserved name, no escape.

    Used by both the normal writes (which then require the target NOT to
    exist) and :meth:`OutputWriter.register_artifact` (which requires the
    opposite), so the two paths can never drift on what "inside the run"
    means.
    """
    if state != "staging":
        return Result.failure(
            Diagnostic.error(
                "WRITER_TERMINAL",
                f"run is already {state}; no further artifacts may be written",
                state=state,
            )
        )
    if not filename or filename.strip() != filename:
        return Result.failure(
            Diagnostic.error("WRITER_BAD_FILENAME", f"artifact filename must be non-empty and trimmed: {filename!r}")
        )
    if "\\" in filename:
        # A backslash is a separator on Windows and a filename character on
        # POSIX: the same name would land in different places per platform.
        return Result.failure(
            Diagnostic.error("WRITER_BAD_FILENAME", f"artifact filename must use '/' separators: {filename!r}")
        )
    target = run_dir / filename
    if target.resolve() == (run_dir / "result.json").resolve():
        return Result.failure(Diagnostic.error("WRITER_RESERVED", "result.json is reserved for the envelope"))
    try:
        target.resolve().relative_to(run_dir.resolve())
    except ValueError:
        return Result.failure(
            Diagnostic.error(
                "WRITER_ESCAPE",
                f"artifact path {filename!r} escapes run directory {run_dir}",
                filename=filename,
            )
        )
    return Result.success(target)


def _corpus_refs(corpus: Corpus | None) -> tuple[InputRef, ...]:
    """Provenance rows for a corpus's documents."""
    if corpus is None:
        return ()
    return tuple(InputRef(path=doc.path.as_posix(), sha256=doc.sha256) for doc in corpus.docs)


def _input_refs(
    inputs: Sequence[Document | InputRef | Path],
    corpus: Corpus | None = None,
) -> tuple[InputRef, ...]:
    refs: list[InputRef] = []
    if corpus is not None:
        refs.extend(_corpus_refs(corpus))
    for item in inputs:
        if isinstance(item, Document):
            refs.append(InputRef(path=item.path.as_posix(), sha256=item.sha256))
        elif isinstance(item, InputRef):
            refs.append(item)
        elif isinstance(item, Path):
            if item.is_dir():
                # A directory input is hashed over member names and CONTENTS.
                # Names+sizes are not provenance: replacing ``AAAA`` with
                # ``BBBB`` at the same path and size must change the run
                # fingerprint.
                digest = hashlib.sha256()
                for member in sorted(p for p in item.rglob("*") if p.is_file()):
                    digest.update(member.relative_to(item).as_posix().encode("utf-8"))
                    digest.update(b"\0")
                    try:
                        digest.update(hash_file(member).encode("ascii"))
                    except OSError:
                        digest.update(b"unreadable")
                    digest.update(b"\n")
                refs.append(InputRef(path=item.as_posix(), sha256=f"dir:{digest.hexdigest()}"))
            else:
                # A raw file input gets a REAL hash: provenance with sha256=""
                # is an advertisement, not evidence (C6-4).
                refs.append(InputRef(path=item.as_posix(), sha256=hash_file(item)))
        else:
            raise TypeError(f"unsupported input reference type: {type(item).__name__}")
    return tuple(refs)


def _corpus_sha256(corpus: Corpus | None) -> str:
    if corpus is not None:
        return corpus.sha256
    return ""


_READOUT_FILENAME = "readout.md"
# One bounded retry for the staging rename, long enough for a Windows
# indexer or antivirus to release a directory it briefly holds open.
_PUBLISH_RETRY_SECONDS = 0.5


def _readout_for(frame: pd.DataFrame, tool: str, filename: str) -> str | None:
    """Plain-language reading of a result table, or None if it cannot be made.

    Imported lazily so ``core.io`` does not depend on ``core.insight`` at
    module load, and wrapped because a readout is a convenience: a rule that
    trips over an unexpected table shape must not fail the run that produced
    it.
    """
    try:
        from core.insight.readout import readout  # noqa: PLC0415 -- lazy to keep core.io free of core.insight

        reading = readout(frame, tool=tool)
        heading = f"# {tool} - {filename}"
        return heading + "\n\n" + reading.to_markdown()
    except Exception:  # never let a convenience summary break a real result
        return None


class OutputWriter:
    """Creates a single run directory and writes all of that run's artifacts.

    Atomicity (C6-4): every visible run directory contains a valid
    ``result.json``. A run is staged in ``.staging-<uuid>`` and published
    (renamed into its final ``{tool}__{timestamp}`` name) by
    :meth:`finalize`. If finalize fails — or never runs — the caller can
    call :meth:`abandon` to remove the staging directory entirely, so a
    half-written run never looks like a finished one. A ``failure`` envelope
    is still published deliberately when a tool wants the record (see
    :meth:`publish`); what never happens is a partial run *without* an
    envelope advertising success.
    """

    def __init__(
        self,
        output_root: Path,
        tool: str,
        params: dict[str, Any],
        inputs: Sequence[Document | InputRef | Path] = (),
        *,
        corpus: Corpus | None = None,
        corpus_sha256: str = "",
        created: str | None = None,
    ) -> None:
        if not tool:
            raise ValueError("OutputWriter tool must be a non-empty string")
        self.tool: str = tool
        self.params: dict[str, Any] = dict(params)
        self.created: str = created or datetime.now(UTC).isoformat(timespec="seconds")
        self.output_root: Path = Path(output_root)
        self._staging: Path = self.output_root / f".staging-{uuid.uuid4().hex[:12]}"
        self.run_dir: Path = self._staging  # artifacts are written staged
        self._inputs: tuple[InputRef, ...] = _input_refs(inputs, corpus)
        raw_corpus_sha = corpus_sha256 or _corpus_sha256(corpus)
        self.corpus_sha256: str = raw_corpus_sha
        self._artifacts: list[Artifact] = []
        # Markdown reading of the first table this run writes, produced while
        # the frame is still in hand and published by finalize(). Held as a
        # string, never as a DataFrame reference, so a large result is not
        # kept alive until the run ends.
        self._readout_markdown: str | None = None
        # Fallback used only when every table published was empty.
        self._empty_readout_markdown: str | None = None
        self._diagnostics: list[Diagnostic] = []
        self._state = "staging"
        self._published_envelope: Envelope | None = None
        corpus_paths = {Path(ref.path).resolve() for ref in _corpus_refs(corpus) if ref.path}
        input_paths = [Path(ref.path) for ref in self._inputs if ref.path]
        for input_path in input_paths:
            if is_inside(self._staging, input_path) or is_inside(input_path, self._staging):
                raise ValueError(f"run directory {self._staging} may not be inside input {input_path}")
            if Path(input_path).resolve() == self.output_root.resolve():
                raise ValueError(f"output_root {self.output_root} may not be an input path {input_path}")
            # R3: never write into a directory the run reads *as a whole* --
            # a corpus directory, or a directory named as an input. Output
            # landing there would be picked up as corpus content by the next
            # run over the same folder.
            if input_path.is_dir():
                if is_inside(input_path, self.output_root) or is_inside(input_path, self._staging):
                    raise ValueError(f"output_root {self.output_root} may not be inside input directory {input_path}")
                continue  # a dir input's parent is not a file holder
            # A corpus document stands for the directory it was discovered in,
            # so its parent is off-limits too. A file named explicitly on the
            # command line does not: it is one file, not a folder being
            # scanned, and forbidding its whole parent tree meant a chart could
            # not be written beside the CSV it came from, and a wordlist in
            # your home directory made your home directory unwritable. The
            # narrower protections above already stop a run from landing on the
            # file itself, and a run directory is freshly named, so it cannot
            # collide with anything already there.
            if input_path.resolve() not in corpus_paths:
                continue
            input_dir = input_path.parent.resolve()
            if is_inside(self.output_root, input_dir) or is_inside(self._staging, input_dir):
                raise ValueError(f"output_root {self.output_root} may not be inside input directory {input_dir}")
        try:
            self._staging.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ValueError(f"cannot create staging directory {self._staging}: {exc}") from exc
        except OSError as exc:
            # Locked / read-only / not-a-directory targets must fail fast with
            # a catchable error — never a retry loop, never a raw traceback
            # the CLI cannot translate (lockfile probe, FR-1.2).
            raise ValueError(f"cannot create staging directory {self._staging}: {exc}") from exc

    def __del__(self) -> None:
        """Best-effort cleanup for a CLI that returns before finalization.

        Callers should still use :meth:`abandon` explicitly; the destructor is
        the last line of defense against orphaned ``.staging-*`` directories
        on early-return paths. Published and failed runs are never removed.
        """
        with suppress(Exception):
            staging = getattr(self, "_staging", None)
            if getattr(self, "_state", None) == "staging" and isinstance(staging, Path):
                shutil.rmtree(staging, ignore_errors=True)

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        return tuple(self._artifacts)

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        return tuple(self._diagnostics)

    def add_diagnostics(self, *diagnostics: Diagnostic) -> None:
        if self._state == "staging":
            self._diagnostics.extend(diagnostics)

    def register_artifact(
        self,
        filename: str,
        *,
        kind: str = "table",
        description: str = "",
    ) -> Result[Path]:
        """Register a file already written inside the run directory.

        R3 says this module is the only writer — but some libraries insist
        on writing bytes themselves (raster wordclouds, t-SNE dumps). Those
        callers write into ``run_dir`` and then register here, so the
        envelope still lists every artifact on disk. The file must already
        exist, must not collide with a registered artifact, and is checked
        against the same inside-run/escape rules as every write.
        """
        if any(existing.path == filename for existing in self._artifacts):
            return Result.failure(
                Diagnostic.error(
                    "WRITER_REGISTERED_TWICE", f"artifact {filename!r} is already registered", filename=filename
                )
            )
        checked = _validate_run_path(self._state, self.run_dir, filename)
        if checked.value is None:
            return checked
        target = checked.unwrap()
        if not target.is_file():
            return Result.failure(
                Diagnostic.error(
                    "WRITER_REGISTER_MISSING",
                    f"cannot register {filename!r}: not a file in {self.run_dir}",
                    filename=filename,
                )
            )
        self._artifacts.append(Artifact(kind=kind, path=filename, description=description))
        return Result.success(target)

    def _check_inside_run(self, filename: str) -> Result[Path]:
        checked = _validate_run_path(self._state, self.run_dir, filename)
        if checked.value is None:
            return checked
        target = checked.unwrap()
        if target.exists():
            return Result.failure(
                Diagnostic.error("WRITER_EXISTS", f"artifact {target} already exists", path=str(target))
            )
        return Result.success(target)

    def write_table(
        self,
        df: pd.DataFrame,
        filename: str,
        *,
        kind: str = "table",
        description: str = "",
        schema: CoNLLSchema | None = None,
    ) -> Result[Path]:
        """Write a CSV table as an artifact."""
        checked = self._check_inside_run(filename)
        if checked.value is None:
            return checked
        target = checked.unwrap()
        if schema is not None:
            sidecar_checked = self._check_inside_run(filename + ".schema.json")
            if sidecar_checked.value is None:
                return sidecar_checked
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(target, index=False, encoding="utf-8")
        except OSError as exc:
            return Result.failure(
                Diagnostic.error("WRITER_FAILED", f"could not write table {target}: {exc}", path=str(target))
            )
        self._artifacts.append(Artifact(kind=kind, path=filename, description=description))
        if kind == "table" and self._readout_markdown is None:
            # The reading describes the first table that has something in it,
            # not simply the first table. A tool that publishes several — an
            # empty `clauses.csv` beside a populated `svo.csv`, say — was
            # getting "The result is empty: no rows met the settings used for
            # this run", which is false about the run and sends the analyst off
            # to loosen thresholds that were never the problem.
            #
            # Guarded at the call site, not only inside the helper: the rule is
            # "a convenience summary never fails a real result", and that has
            # to hold however the helper behaves.
            with suppress(Exception):
                markdown = _readout_for(df, self.tool, filename)
                if df.empty:
                    # Kept in case every table turns out to be empty, which is
                    # the one case where "the result is empty" is the answer.
                    self._empty_readout_markdown = self._empty_readout_markdown or markdown
                else:
                    self._readout_markdown = markdown
        if schema is not None:
            sidecar = write_sidecar(target, schema)
            if not sidecar.ok:
                self._artifacts.pop()
                with suppress(OSError):
                    target.unlink()
                return Result[Path](None, sidecar.diagnostics)
        return Result.success(target)

    def write_json(
        self,
        data: dict[str, Any],
        filename: str,
        *,
        kind: str = "table",
        description: str = "",
    ) -> Result[Path]:
        checked = self._check_inside_run(filename)
        if checked.value is None:
            return checked
        target = checked.unwrap()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            rounded = _round_floats(json.loads(json.dumps(data, ensure_ascii=False)))
            target.write_text(json.dumps(rounded, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except OSError as exc:
            return Result.failure(
                Diagnostic.error("WRITER_FAILED", f"could not write json {target}: {exc}", path=str(target))
            )
        self._artifacts.append(Artifact(kind=kind, path=filename, description=description))
        return Result.success(target)

    def write_text(
        self,
        text: str,
        filename: str,
        *,
        kind: str = "report",
        description: str = "",
    ) -> Result[Path]:
        checked = self._check_inside_run(filename)
        if checked.value is None:
            return checked
        target = checked.unwrap()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        except OSError as exc:
            return Result.failure(
                Diagnostic.error("WRITER_FAILED", f"could not write text {target}: {exc}", path=str(target))
            )
        self._artifacts.append(Artifact(kind=kind, path=filename, description=description))
        return Result.success(target)

    def write_html(
        self,
        html: str,
        filename: str,
        *,
        kind: str = "chart",
        description: str = "",
    ) -> Result[Path]:
        return self.write_text(html, filename, kind=kind, description=description)

    def write_bytes(
        self,
        data: bytes,
        filename: str,
        *,
        kind: str = "image",
        description: str = "",
    ) -> Result[Path]:
        """Write binary bytes (an exported image) as an artifact.

        Mirrors :meth:`write_text` exactly — staging/terminal-state check,
        reserved-name and containment validation, duplicate protection,
        error-as-``Result`` — for binary payloads rendered in memory
        (kaleido PNG/SVG/PDF). The caller hands bytes over; the writer stays
        the only place bytes touch the filesystem (R3).
        """
        checked = self._check_inside_run(filename)
        if checked.value is None:
            return checked
        target = checked.unwrap()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        except OSError as exc:
            return Result.failure(
                Diagnostic.error("WRITER_FAILED", f"could not write bytes {target}: {exc}", path=str(target))
            )
        self._artifacts.append(Artifact(kind=kind, path=filename, description=description))
        return Result.success(target)

    def finalize(self) -> Result[Envelope]:
        """Publish the staged run atomically and return the finished envelope.

        The envelope is written into the staging directory first; only after
        it exists is the directory renamed to its final ``{tool}__{ts}``
        name. Any artifact write failure before finalize leaves the run in
        staging — the caller decides: fix and re-verify, publish as an
        explicit failure (:meth:`publish_failure`), or discard
        (:meth:`abandon`). A visible run directory therefore always has a
        valid envelope.
        """
        if self._state != "staging":
            return Result.failure(
                Diagnostic.error("WRITER_TERMINAL", f"cannot finalize a {self._state} run", state=self._state)
            )
        readout_markdown = self._readout_markdown or self._empty_readout_markdown
        if readout_markdown is not None:
            # Best-effort: a run must never fail because its plain-language
            # summary could not be written. The numbers are the artifact that
            # matters; this is a convenience on top of them.
            with suppress(OSError):
                (self._staging / _READOUT_FILENAME).write_text(readout_markdown, encoding="utf-8")
                self._artifacts.append(
                    Artifact(
                        kind="text",
                        path=_READOUT_FILENAME,
                        description="Plain-language reading of the result",
                    )
                )
        envelope = Envelope(
            tool=self.tool,
            created=self.created,
            params=dict(self.params),
            inputs=self._inputs,
            artifacts=tuple(self._artifacts),
            diagnostics=tuple(self._diagnostics),
            corpus_sha256=self.corpus_sha256,
        )
        written = envelope.write(self._staging)
        if not written.ok:
            return Result[Envelope](None, written.diagnostics)
        final = self.output_root / f"{self.tool}__{_timestamp()}"
        try:
            self._staging.rename(final)
        except OSError as exc:
            # On Windows the rename can lose a short race against an
            # indexer or antivirus briefly holding the fresh staging
            # directory (WinError 5), then succeed immediately. One
            # short, bounded retry keeps a published result from
            # failing on the scanner's timing; a genuinely broken
            # rename still fails, on the second attempt.
            time.sleep(_PUBLISH_RETRY_SECONDS)
            try:
                self._staging.rename(final)
            except OSError:
                return Result[Envelope].failure(
                    Diagnostic.error("WRITER_PUBLISH_FAILED", f"could not publish run: {exc}", path=str(final)),
                )
        self.run_dir = final
        self._state = "finalized"
        self._published_envelope = envelope
        return Result.success(envelope)

    def publish_failure(
        self,
        *diagnostics: Diagnostic,
        keep_artifacts: bool = False,
    ) -> Result[Path]:
        """Publish a deliberately-failed run envelope.

        Default (``keep_artifacts=False``, unchanged contract): artifacts
        recorded so far are dropped from the envelope AND removed from the
        staging directory — a failure record names what failed, never a
        success that did not happen.

        ``keep_artifacts=True`` is for the case where an explicitly-requested
        export fails AFTER earlier artifacts were produced successfully
        (charts CLI: HTML + prepared CSV survive an image-export failure):
        the already-written files stay on disk, the failure envelope declares
        them under their ``chart``/``table`` kinds, and the diagnostics carry
        the export error. The directory keeps its ``-failed`` suffix and its
        ERROR diagnostics, so nothing masquerades as a fully successful run.

        Either way the returned ``Result`` MUST be checked: a failed
        publication (envelope write or rename) leaves the run staged, and the
        caller is responsible for :meth:`abandon` + reporting the error.
        """
        if self._state != "staging":
            return Result.failure(
                Diagnostic.error("WRITER_TERMINAL", f"cannot publish a {self._state} run", state=self._state)
            )
        self._diagnostics.extend(diagnostics)
        if keep_artifacts:
            # Retain the staged files AND declare them: the envelope lists the
            # artifacts that genuinely exist, while the -failed suffix and the
            # ERROR diagnostics mark the run incomplete.
            envelope = Envelope(
                tool=self.tool,
                created=self.created,
                params=dict(self.params),
                inputs=self._inputs,
                artifacts=tuple(self._artifacts),
                diagnostics=tuple(self._diagnostics),
                corpus_sha256=self.corpus_sha256,
            )
        else:
            # A failure envelope deliberately advertises no successful artifacts.
            # Remove staged files too, so an undeclared physical artifact cannot
            # be mistaken for a usable output by consumers that inspect the
            # directory.
            for child in tuple(self._staging.iterdir()):
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    with suppress(OSError):
                        child.unlink()
            envelope = Envelope(
                tool=self.tool,
                created=self.created,
                params=dict(self.params),
                inputs=self._inputs,
                artifacts=(),
                diagnostics=tuple(self._diagnostics),
                corpus_sha256=self.corpus_sha256,
            )
        written = envelope.write(self._staging)
        if not written.ok:
            return Result[Path](None, written.diagnostics)
        final = self.output_root / f"{self.tool}__{_timestamp()}-failed"
        try:
            self._staging.rename(final)
        except OSError as exc:
            # Same bounded retry as finalize(): a scanner holding the fresh
            # staging directory is timing, not breakage.
            time.sleep(_PUBLISH_RETRY_SECONDS)
            try:
                self._staging.rename(final)
            except OSError:
                return Result[Path].failure(Diagnostic.error("WRITER_PUBLISH_FAILED", f"{exc}", path=str(final)))
        self.run_dir = final
        self._state = "failed"
        return Result.success(final)

    def abandon(self) -> None:
        """Discard the staged run entirely (no partial directory left behind)."""
        if self._state == "staging":
            shutil.rmtree(self._staging, ignore_errors=True)
            self._state = "abandoned"
