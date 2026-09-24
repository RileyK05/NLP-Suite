"""Corpus intake: reading documents off disk.

ARCHITECTURE.md invariants enforced here:

* **R3** — reads never write. Nothing in this module writes, and a document it
  hands back is never modified in place.
* **R8** — inputs are hashed so a run can later prove what it read.
* **R6** — no global state; the config is passed in.

Three deliberate departures from the legacy suite:

1. **`doc_id` is assigned, never derived.** The legacy suite computed document
   identity by slicing ID strings (``str(id)[:-2]``), which corrupts any corpus
   past nine documents. Here ids are dense ``1..N`` handed out at intake.
2. **A bad document is a diagnostic, not an aborted run.** The legacy reader
   ``break``-ed out of the corpus on the first empty file, silently dropping
   every document after it. Here an unreadable file is recorded and the rest
   of the corpus is still returned.
3. **Encoding is detected by trying, not assumed.** The legacy file checker
   crashed on a ``None`` encoding from chardet and then opened files with
   ``errors='ignore'``, which defeats the point of checking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import os
from pathlib import Path
import re

from core.result import Diagnostic, Result

__all__ = [
    "ENCODING_CHAIN",
    "Corpus",
    "Document",
    "date_from_filename",
    "display_names",
    "hash_file",
    "hash_text",
    "read_corpus",
    "read_text",
    "strip_nul_bytes",
]

# Tried in order. latin-1 never raises, so it terminates the chain.
ENCODING_CHAIN: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

_CHUNK_SIZE = 1 << 16

_DATE_IN_NAME = re.compile(r"(?<!\d)(\d{4}[-_]\d{2}[-_]\d{2}|\d{8}|\d{2}[-_]\d{2}[-_]\d{4}|\d{4}[-_]\d{2})(?!\d)")
_DATE_FORMATS = ("%Y-%m-%d", "%Y_%m_%d", "%m-%d-%Y", "%m_%d_%Y", "%Y%m%d", "%Y-%m", "%Y_%m")


@dataclass(frozen=True, slots=True)
class Document:
    """One document in a corpus.

    ``doc_id`` is assigned at intake and is dense over the corpus. It is never
    parsed out of a filename.
    """

    doc_id: int
    path: Path
    text: str
    date: date | None = None
    sha256: str = ""
    # Stable identity and reader-facing label supplied by project-backed
    # callers.  Directory/CLI corpora leave both empty and retain the existing
    # path-based behaviour.
    source_id: str = ""
    label: str = ""

    @property
    def name(self) -> str:
        return self.label or self.path.name

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class Corpus:
    """An immutable set of documents plus a fingerprint over all of them."""

    docs: tuple[Document, ...]
    sha256: str

    def __len__(self) -> int:
        return len(self.docs)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.docs)

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(doc.path for doc in self.docs)

    @property
    def doc_ids(self) -> tuple[int, ...]:
        return tuple(doc.doc_id for doc in self.docs)

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(doc.text for doc in self.docs)


def _common_root(paths: Sequence[Path]) -> Path | None:
    """Deepest directory every path sits under, or None when there isn't one.

    ``os.path.commonpath`` raises when the paths mix absolute with relative or
    span two Windows drives, and returns ``""`` when they share nothing. All
    three mean the same thing here: there is no root to shorten against.
    """
    try:
        shared = os.path.commonpath([str(path) for path in paths])
    except ValueError:
        return None
    return Path(shared) if shared else None


def display_names(docs: Sequence[Document]) -> dict[int, str]:
    """The shortest label that still tells two documents apart, by ``doc_id``.

    Every result table that names a document should name it the way the person
    who assembled the corpus would: ``report.txt``, not the absolute path it
    happened to be read from. A full path is noise in a CSV cell, it leaks the
    machine's directory layout into shared output, and it makes a readout
    unreadable.

    But a basename is only a name while it is unique. Corpora organised as
    ``2020/report.txt`` and ``2021/report.txt`` are ordinary, and collapsing
    both to ``report.txt`` would silently merge two documents in any table
    keyed by name. So: the basename when nothing else shares it, otherwise the
    path relative to the corpus's own root, and only then the full path. Where
    even that collides -- the same file listed twice -- the ``doc_id`` is
    appended, because these labels are identifiers and duplicates in an
    identifier are worse than an ugly one.

    Pure and order-independent (R6): the same corpus always gets the same
    labels, whichever tool asks for them.
    """
    counts: dict[str, int] = {}
    for doc in docs:
        counts[doc.path.name] = counts.get(doc.path.name, 0) + 1
    root = _common_root([doc.path for doc in docs]) if len(docs) > 1 else None

    names: dict[int, str] = {}
    for doc in docs:
        if counts[doc.path.name] == 1:
            names[doc.doc_id] = doc.path.name
            continue
        if root is not None:
            try:
                relative = doc.path.relative_to(root).as_posix()
            except ValueError:  # not actually under the root we computed
                relative = ""
            # "." means the root *is* this path, which happens when the corpus
            # lists one file twice. That is not a label; fall back and let the
            # doc_id suffix below do the disambiguating.
            if relative and relative != ".":
                names[doc.doc_id] = relative
                continue
        names[doc.doc_id] = doc.path.name if root is not None else doc.path.as_posix()

    seen: dict[str, int] = {}
    for label in names.values():
        seen[label] = seen.get(label, 0) + 1
    return {doc_id: (label if seen[label] == 1 else f"{label} (#{doc_id})") for doc_id, label in sorted(names.items())}


def hash_text(text: str) -> str:
    """SHA-256 of a document's text. Stable across platforms and runs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_nul_bytes(text: str) -> tuple[str, bool]:
    """Remove NUL bytes from decoded text, reporting whether any were found.

    NUL bytes are never content: they break CSV artifacts and downstream
    consumers. The legacy suite "handled" them by rewriting the user's input
    file in place (R3 violation). Here the source is never touched; the NULs
    are stripped from the in-memory text and the strip is announced.
    """
    if "\x00" not in text:
        return text, False
    return text.replace("\x00", ""), True


def read_text(path: Path, encoding: str | None = None) -> Result[str]:
    """Read a text file, falling back through `ENCODING_CHAIN`.

    A fallback is a warning, not a silent success: callers can tell that the
    file was not the encoding they expected. Embedded NUL bytes are stripped
    from the returned text (never from the source file) with a
    `NULL_BYTE_STRIPPED` info diagnostic.
    """
    source = Path(path)
    if not source.is_file():
        return Result.failure(Diagnostic.error("FILE_NOT_FOUND", f"{source} is not a file", path=str(source)))

    candidates = (encoding,) if encoding else ENCODING_CHAIN
    attempted: list[str] = []

    for index, candidate in enumerate(candidates):
        try:
            text = source.read_text(encoding=candidate)
        except (UnicodeDecodeError, LookupError):
            attempted.append(candidate)
            continue
        except OSError as exc:
            return Result.failure(
                Diagnostic.error("FILE_UNREADABLE", f"could not read {source}: {exc}", path=str(source))
            )

        cleaned, stripped = strip_nul_bytes(text)
        nul_diag = (
            (
                Diagnostic.info(
                    "NULL_BYTE_STRIPPED",
                    f"{source.name} contained NUL byte(s); stripped from the in-memory text, source untouched",
                    path=str(source),
                ),
            )
            if stripped
            else ()
        )
        if index == 0:
            return Result.success(cleaned, *nul_diag)
        return Result.success(
            cleaned,
            Diagnostic.warning(
                "ENCODING_FALLBACK",
                f"{source.name} was not {candidates[0]}; read as {candidate}",
                path=str(source),
                encoding=candidate,
                attempted=attempted,
            ),
            *nul_diag,
        )

    return Result.failure(
        Diagnostic.error(
            "FILE_UNDECODABLE",
            f"could not decode {source} with any of {list(candidates)}",
            path=str(source),
            attempted=attempted,
        )
    )


def date_from_filename(path: Path) -> date | None:
    """Best-effort date embedded in a filename, or None.

    The legacy date classifier passed its separator and its format string the
    wrong way round, so the feature could never work. This is deliberately
    conservative: an unparseable match yields None rather than a wrong date.
    """
    match = _DATE_IN_NAME.search(Path(path).stem)
    if match is None:
        return None
    token = match.group(1)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(token, fmt).date()
        except ValueError:
            continue
    return None


def corpus_fingerprint(docs: tuple[Document, ...]) -> str:
    """A portable hash over the whole corpus, independent of traversal order.

    Absolute paths made an unchanged restored project a different snapshot.
    Paths are now relative to the documents' shared parent, preserving nested
    corpus distinctions while allowing a byte-identical project to move.
    """
    if not docs:
        return hashlib.sha256(b"").hexdigest()
    root = _common_root([doc.path.parent for doc in docs])

    def identity(doc: Document) -> str:
        if root is not None:
            try:
                location = doc.path.relative_to(root).as_posix()
            except ValueError:
                location = doc.path.name
        else:
            location = doc.path.name
        return f"{location}:{doc.sha256}"

    payload = "\n".join(sorted(identity(doc) for doc in docs))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_corpus(
    root: Path,
    *,
    pattern: str = "*.txt",
    encoding: str | None = None,
    recursive: bool = True,
) -> Result[Corpus]:
    """Read a directory of documents into a Corpus.

    Documents are read in sorted path order and assigned dense ids ``1..N``.
    An unreadable file produces an ERROR diagnostic and is skipped; an empty
    file produces an EMPTY_DOC warning and is kept. Neither aborts the corpus,
    which is what the legacy reader did on the first empty file.
    """
    directory = Path(root)
    if not directory.is_dir():
        return Result.failure(
            Diagnostic.error("CORPUS_DIR_MISSING", f"{directory} is not a directory", path=str(directory))
        )

    globber = directory.rglob(pattern) if recursive else directory.glob(pattern)
    paths = sorted(path for path in globber if path.is_file())
    if not paths:
        return Result.failure(
            Diagnostic.error(
                "CORPUS_EMPTY",
                f"no files matching {pattern!r} under {directory}",
                path=str(directory),
                pattern=pattern,
            )
        )

    diagnostics: list[Diagnostic] = []
    readings: list[tuple[Path, str]] = []

    for path in paths:
        result = read_text(path, encoding)
        diagnostics.extend(result.diagnostics)
        if result.value is None:
            continue
        readings.append((path, result.value))

    docs: list[Document] = []
    for doc_id, (path, text) in enumerate(readings, start=1):
        if not text.strip():
            diagnostics.append(
                Diagnostic.warning(
                    "EMPTY_DOC",
                    f"{path.name} has no content",
                    path=str(path),
                    doc_id=doc_id,
                )
            )
        docs.append(
            Document(
                doc_id=doc_id,
                path=path,
                text=text,
                date=date_from_filename(path),
                sha256=hash_text(text),
            )
        )

    if not docs:
        return (
            Result.failure(*diagnostics)
            if diagnostics
            else Result.failure(Diagnostic.error("CORPUS_EMPTY", "no documents could be read", path=str(directory)))
        )

    return Result.success(Corpus(docs=tuple(docs), sha256=corpus_fingerprint(tuple(docs))), *diagnostics)
