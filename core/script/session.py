"""What one running notebook holds: where its documents come from, and what it produced.

The library's functions are module-level (``nlp.show(table)``), because that is
how a script reads best and how every notebook library people already know
works. They need somewhere to find the corpus and somewhere to put their
outputs, so a kernel opens a :class:`Session` and makes it current for the
duration of a cell. It is held in a :class:`contextvars.ContextVar`, not a
module global: two sessions in one process (a test, and later two kernels in
one worker) cannot see each other's outputs.

A :class:`Source` is the one seam between the library and the app. The kernel
hands in one backed by a project (documents from the workspace, parses from
the shared cache); :class:`FolderSource` reads a directory of ``.txt`` files,
for tests and for running a notebook with the suite installed but no app.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from core.io.reader import Corpus
    from core.research.phrase import Tokenization
    from core.result import Diagnostic

__all__ = [
    "FigureOutput",
    "FolderSource",
    "HtmlOutput",
    "NoteOutput",
    "Output",
    "ParsedTable",
    "Session",
    "Source",
    "SuiteError",
    "TableOutput",
    "current",
    "open_session",
]


class SuiteError(Exception):
    """A library call that could not do what was asked, said in words.

    Carries the engine's diagnostics, whose messages were written for the
    person reading them, instead of a stack of ``Result.unwrap`` frames that
    only a maintainer could decode.
    """

    def __init__(self, message: str, diagnostics: Sequence[Diagnostic] = ()) -> None:
        super().__init__(message)
        self.diagnostics = tuple(diagnostics)


@dataclass(frozen=True)
class ParsedTable:
    """A corpus's token table and the tokenizer that produced it."""

    table: pd.DataFrame
    tokenizer: Tokenization | None
    diagnostics: tuple[Diagnostic, ...] = ()


class Source(Protocol):
    """Where a session's documents, parses and earlier results come from."""

    #: What to call this collection in a reference or a provenance record.
    name: str

    def documents(self) -> list[dict[str, Any]]:
        """Every document: ``id``, ``name`` and, optionally, ``fields``."""
        ...

    def corpus(self, ids: Sequence[str] | None) -> Corpus:
        """The documents with these ids (all when None), numbered 1..N in that order."""
        ...

    def parse(self, corpus: Corpus) -> ParsedTable:
        """The corpus parsed, from a cache when one exists."""
        ...

    def tables(self) -> list[dict[str, Any]]:
        """Result tables of earlier runs this source can read (may be empty)."""
        ...

    def read_table(self, job: str, index: int) -> pd.DataFrame:
        """One earlier result table."""
        ...


# ---------------------------------------------------------------- outputs --


@dataclass(frozen=True)
class TableOutput:
    """A table a cell showed, with the chart drawn beside it (if any)."""

    name: str
    frame: pd.DataFrame
    title: str = ""
    #: Chart settings in the workbench's own shape (kind, x, y, group, agg,
    #: top_n), or None when nothing worth drawing was found.
    chart: dict[str, Any] | None = None
    #: The rows the chart reads, when they differ from ``frame`` (several y
    #: columns are drawn from a long table with a "Series" column).
    chart_frame: pd.DataFrame | None = None
    #: Why no chart was drawn, when none was.
    chart_note: str = ""
    #: True for ``nlp.save``: data kept for later, not something to look at now.
    saved: bool = False


@dataclass(frozen=True)
class FigureOutput:
    """A matplotlib figure, rendered to PNG (and SVG, when it can be made).

    ``svg`` is None when the SVG writer was unavailable: a packaged runtime
    can lack matplotlib's lazily-imported SVG backend, and a figure kept
    without its SVG is still the figure the reader came to see.
    """

    name: str
    png: bytes
    svg: bytes | None = None


@dataclass(frozen=True)
class HtmlOutput:
    """A self-contained interactive HTML fragment a cell asked to show.

    ``html`` is a fragment (no ``<html>``/``<body>``) meant to be shown in a
    sandboxed frame under the cell and, when the notebook is saved as a run,
    written beside the tables as ``<name>.html``. Produced by
    :func:`core.script.api.chart` (the engine's plotly renderer), so it carries
    the same interactive chart the Analyze page draws.
    """

    name: str
    html: str


@dataclass(frozen=True)
class NoteOutput:
    """A sentence the script wants kept with its results."""

    text: str


Output = TableOutput | FigureOutput | HtmlOutput | NoteOutput


_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass
class Call:
    """One library call, for the run's provenance."""

    function: str
    arguments: dict[str, str]
    seconds: float
    rows: int | None = None


@dataclass
class Session:
    """One notebook's connection to its documents, and everything it produced."""

    source: Source
    on_output: Callable[[Output], None] | None = None
    outputs: list[Output] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    _parses: dict[str, ParsedTable] = field(default_factory=dict)
    _names: set[str] = field(default_factory=set)
    #: One-off sentences already said (a missing SVG backend is worth saying
    #: once, not once for every figure a notebook draws).
    _said_once: set[str] = field(default_factory=set)

    def parse(self, corpus: Corpus) -> ParsedTable:
        """Parse once per corpus per session; the source may also have it on disk.

        Keyed by the documents *in order*: a parse is a table of positions
        (``Document ID``, ``Sentence ID``, ``Record ID``), so the same
        documents in a different order is a different table. The order-free
        :attr:`Corpus.sha256` would let one ordering's ids be read back for
        another.
        """
        from core.io.reader import ordered_fingerprint

        key = ordered_fingerprint(corpus.docs)
        found = self._parses.get(key)
        if found is None:
            found = self.source.parse(corpus)
            self._parses[key] = found
        return found

    def start_cell(self) -> None:
        """Forget earlier cells' outputs and names (a kernel, between cells).

        A kernel lives for many cells, and a cell run twice would otherwise
        name its table ``rates_3`` the third time, while every table ever
        shown stayed in memory. Its files are kept apart by the kernel's own
        numbering. "Run and save" never calls this: there, one session is one
        notebook, and names must be unique across it.
        """
        self.outputs.clear()
        self._names.clear()
        self.calls.clear()

    def name_for(self, wanted: str | None, fallback: str) -> str:
        """A file-safe name no other output of this session has taken."""
        stem = _UNSAFE_NAME.sub("_", (wanted or fallback).strip()).strip("_") or fallback
        name, counter = stem, 2
        while name.lower() in self._names:
            name = f"{stem}_{counter}"
            counter += 1
        self._names.add(name.lower())
        return name

    def emit(self, output: Output) -> None:
        self.outputs.append(output)
        if self.on_output is not None:
            self.on_output(output)

    def note_once(self, text: str) -> None:
        """Emit *text* as a note the first time it is raised in this session."""
        if text not in self._said_once:
            self._said_once.add(text)
            self.emit(NoteOutput(text))

    def record(self, function: str, arguments: dict[str, Any], started: float, rows: int | None = None) -> None:
        shown = {key: _short(value) for key, value in arguments.items()}
        self.calls.append(Call(function, shown, round(time.perf_counter() - started, 3), rows))


#: How much of an argument a provenance record keeps (a word list can be long).
_SHOWN_ARGUMENT = 200


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= _SHOWN_ARGUMENT else text[: _SHOWN_ARGUMENT - 3] + "..."


_CURRENT: ContextVar[Session | None] = ContextVar("nlpsuite_session", default=None)


def current() -> Session:
    """The session this code is running in, or a refusal that says how to get one."""
    session = _CURRENT.get()
    if session is None:
        raise SuiteError(
            "This notebook is not connected to any documents. Inside the NLP Suite that happens by itself; "
            "outside it, call nlp.use_folder('path/to/texts') first."
        )
    return session


@contextmanager
def open_session(session: Session) -> Iterator[Session]:
    """Make *session* current for the block (a cell, a test, a whole run)."""
    token = _CURRENT.set(session)
    try:
        yield session
    finally:
        _CURRENT.reset(token)


def set_current(session: Session | None) -> None:
    """Make *session* current until replaced. For a kernel's whole lifetime, or ``use_folder``."""
    _CURRENT.set(session)


# ----------------------------------------------------------- folder source --


class FolderSource:
    """A directory of ``.txt`` files, parsed with spaCy, no app needed.

    For tests, and for running an exported notebook on a machine that has the
    suite installed as a Python package but not the desktop app. Parses are
    kept for the session only; the app's own source reads the shared cache.
    """

    def __init__(self, directory: Path, *, parser: str = "spacy") -> None:
        from core.io.reader import read_corpus

        self.directory = Path(directory)
        self.name = self.directory.name
        self.parser = parser
        read = read_corpus(self.directory)
        if read.value is None:
            raise SuiteError(f"Could not read texts from {self.directory}.", read.diagnostics)
        self._corpus = read.unwrap()
        self._pipeline: Any = None

    def documents(self) -> list[dict[str, Any]]:
        return [{"id": str(doc.doc_id), "name": doc.name} for doc in self._corpus.docs]

    def corpus(self, ids: Sequence[str] | None) -> Corpus:
        from dataclasses import replace

        from core.io.reader import Corpus, corpus_fingerprint

        by_id = {str(doc.doc_id): doc for doc in self._corpus.docs}
        chosen = [by_id[i] for i in ids] if ids is not None else list(self._corpus.docs)
        docs = tuple(
            replace(doc, doc_id=position, source_id=str(doc.doc_id), label=doc.name)
            for position, doc in enumerate(chosen, 1)
        )
        return Corpus(docs, corpus_fingerprint(docs))

    def parse(self, corpus: Corpus) -> ParsedTable:
        from core.config import NLPConfig
        from core.pipelines.cache import PipelineCache
        from core.pipelines.resolve import resolve_pipeline
        from core.research.phrase import Tokenization

        if self._pipeline is None:
            resolved = resolve_pipeline(PipelineCache(), NLPConfig(parser=self.parser, language="en"))  # type: ignore[arg-type]
            if resolved.value is None:
                raise SuiteError("No English parser is installed.", resolved.diagnostics)
            self._pipeline = resolved.unwrap()
        parsed = self._pipeline.parse(corpus)
        if parsed.value is None:
            raise SuiteError("The documents could not be parsed.", parsed.diagnostics)
        backend = str(getattr(self._pipeline, "backend", self.parser))
        return ParsedTable(parsed.unwrap(), Tokenization(self._pipeline.tokenize, backend), tuple(parsed.diagnostics))

    def tables(self) -> list[dict[str, Any]]:
        return []

    def read_table(self, job: str, index: int) -> pd.DataFrame:
        raise SuiteError("A folder of texts has no earlier runs to read.")
