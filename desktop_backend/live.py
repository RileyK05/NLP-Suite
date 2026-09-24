"""Live analysis: the same engine, without the round trip.

An analysis costs two very different things. On a twenty-speech corpus --
113,000 words -- parsing takes **thirty seconds** and computing readability
over the result takes **170 milliseconds**. The desktop charged for both on
every run, so changing an n-gram's ``n`` from 2 to 3 meant submitting a job,
waiting half a minute, and opening a file to see a different bar chart. At that
price people do not explore; they run the analysis they already decided on.

So the expensive half is paid once. :meth:`Bench.warm` parses a chosen set of
documents and keeps the token table; :meth:`Bench.analyse` runs any analysis
over that table and hands back rows. After warming, changing a parameter costs
what the analysis itself costs -- a sixth of a second for readability, half a
second for collocations -- which is fast enough that the controls can drive the
picture directly.

Three things this is careful about.

**A live result is not a published one.** Nothing here writes a run directory,
records provenance or produces an artifact. It is a way of looking, exactly as
the chart workbench's preview is, and publishing is still a real run through
the ordinary path. What makes that safe is that both come from the same call:
:func:`execute` returns the frames a publish would write, so the rows on screen
are the rows that would be published, not an approximation of them.

**The parse is cached against everything that could change it.** A cached
parse reused after a model upgrade would silently answer today's question with
last month's annotations. The key carries the corpus fingerprint, the backend
that actually ran, the language, the model and its installed version, and a
schema version bumped by hand when the token table's columns change.

**The cache is a cache.** It lives beside the workspace, never inside a
project's ``runs/``, and deleting all of it costs time and nothing else. R3
keeps the core library from writing; this is the desktop storing something it
can always recompute, and it is written through a temporary file so a killed
process leaves no half-parsed table behind.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import contextlib
from dataclasses import dataclass, field, replace
import hashlib
import importlib.metadata
import os
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from core.io.reader import Corpus
    from core.research.phrase import Tokenization
    from core.result import Diagnostic

# The parsers the desktop offers. Named once: the runner rejects anything else
# at submission and this module has to turn the same string into the engine's
# Literal, and two lists of two names is two chances to add a third to one.
Parser = Literal["spacy", "stanza"]
PARSERS: tuple[Parser, ...] = ("spacy", "stanza")


def as_parser(name: str) -> Parser:
    """The engine's parser name, or a refusal a reader can act on."""
    if name not in PARSERS:
        raise ValueError("Choose spaCy or Stanza.")
    return name


# Bumped by hand when the parsed token table's columns or their meaning change.
# A cached table from before such a change is not the same table, and nothing
# about the file itself would say so.
SCHEMA_VERSION = 1

# One warmed corpus is held in memory at a time. A 113,000-word parse is about
# 68 MB as a frame; holding several would trade a lot of memory for the 750 ms
# it costs to read one back from disk.
HELD_CORPORA = 1


class Stage(Protocol):
    """Somewhere to report progress while a long parse is running."""

    def __call__(self, stage: str) -> None: ...


def _installed(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def parser_identity(pipeline: Any) -> dict[str, str]:
    """What produced a parse, in enough detail to know a later one differs.

    ``pipeline.backend`` rather than the requested parser: ``resolve_pipeline``
    falls back to another backend when a model is missing, and a table parsed
    by the fallback must not be reused as though the configured parser had
    produced it.
    """
    model = str(getattr(pipeline, "model_name", "") or "")
    return {
        "backend": str(getattr(pipeline, "backend", "unknown")),
        "language": str(getattr(pipeline, "language", "")),
        "model": model,
        "backend_version": _installed(str(getattr(pipeline, "backend", ""))),
        "model_version": _installed(model) if model else "absent",
        "schema": str(SCHEMA_VERSION),
    }


def annotation_key(fingerprint: str, identity: dict[str, str]) -> str:
    """A filename for one corpus parsed one particular way.

    Every part of *identity* is in the digest, so a model upgrade produces a
    different key rather than a stale hit.
    """
    parts = [fingerprint, *(f"{name}={identity[name]}" for name in sorted(identity))]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


class Annotations:
    """Parsed token tables on disk, keyed by what produced them.

    Small enough to keep: the 113,000-word table above is 68 MB in memory and
    1.7 MB as parquet.
    """

    def __init__(self, root: Path) -> None:
        self.root = root / "annotations"

    def path(self, key: str) -> Path:
        return self.root / f"{key}.parquet"

    def load(self, key: str) -> pd.DataFrame | None:
        import pandas as pd

        target = self.path(key)
        if not target.is_file():
            return None
        try:
            return pd.read_parquet(target)
        except (OSError, ValueError):
            # A truncated or unreadable cache file is a reason to parse again,
            # not a reason to fail. Drop it so the next run does not retry it.
            target.unlink(missing_ok=True)
            return None

    def store(self, key: str, frame: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.path(key)
        staging = target.with_suffix(f".{os.getpid()}.partial")
        try:
            frame.to_parquet(staging)
            staging.replace(target)
        except (OSError, ValueError):
            staging.unlink(missing_ok=True)
            raise

    def forget(self) -> int:
        """Delete every cached parse. Costs time to recompute and nothing else."""
        removed = 0
        for stale in self.root.glob("*.parquet"):
            stale.unlink(missing_ok=True)
            removed += 1
        return removed


@dataclass(frozen=True)
class Warm:
    """A corpus that has been parsed, and is ready to be asked questions."""

    corpus: Corpus
    key: str
    identity: dict[str, str]
    table: pd.DataFrame | None
    diagnostics: tuple[Diagnostic, ...] = ()
    #: Where this parse came from. "parsed" cost the full wait, "disk" cost a
    #: read of the cache, "memory" cost nothing. A boolean could not tell the
    #: last two apart, and they are the two the interface most wants to
    #: distinguish: one is instant and one is most of a second.
    source: Literal["parsed", "disk", "memory"] = "parsed"
    parse_ms: int = 0
    #: How the parser that produced this table splits text into tokens. A
    #: question about words has to use it: the query and the table must agree
    #: on what a word is, or a phrase that is plainly in the corpus comes back
    #: with no occurrences and nothing on screen says why.
    tokenize: Callable[[str], Sequence[str]] | None = None
    tokenizer_name: str = ""

    @property
    def tokenizer(self) -> Tokenization | None:
        """This parse's tokenizer, for the tools that have to split a question.

        ``None`` when the table came from somewhere that did not record one,
        which an answer reports rather than papering over.
        """
        from core.research.phrase import Tokenization

        return None if self.tokenize is None else Tokenization(self.tokenize, self.tokenizer_name)

    @property
    def cached(self) -> bool:
        """True when nothing had to be parsed to get here."""
        return self.source != "parsed"

    @property
    def documents(self) -> int:
        return len(self.corpus.docs)

    @property
    def tokens(self) -> int:
        return 0 if self.table is None else len(self.table)


@dataclass(frozen=True)
class LiveResult:
    """What an analysis produced, and how long it took to produce it."""

    tool: str
    ok: bool
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()
    elapsed_ms: int = 0

    def table(self) -> tuple[str, pd.DataFrame] | None:
        """The frame worth charting: the first one the tool produced."""
        for name, frame in self.frames.items():
            return name, frame
        return None


#: How many abandoned request ids to remember (R-C8). The set holds only
#: questions nobody is waiting for, so it may not grow with every question
#: ever asked: past twice this it is halved to the newest half.
_CANCELLED_KEEP = 32
#: How many live answers to keep, so a figure can be redrawn with other
#: settings without re-running the analysis. Small: an answer is a few
#: frames, and a reader redraws the one they are looking at.
_ANSWERS_KEPT = 4


class Bench:
    """One warmed corpus, and analyses run against it.

    Deliberately not a pool. A reader looks at one corpus at a time, and a
    second warmed corpus costs 68 MB to save 750 ms.
    """

    def __init__(self, root: Path) -> None:
        self.annotations = Annotations(root)
        self._held: list[Warm] = []

    def release(self) -> None:
        """Let go of the parse in memory. The copy on disk is untouched, so
        coming back to this corpus costs a read rather than a parse."""
        self._held = []

    def forget(self, key: str) -> None:
        """Let go of one parse, named.

        For the parse that finished after nobody wanted it any more. Releasing
        everything would be wrong -- a newer load may be held by now -- and
        releasing nothing leaves the 68 MB a cooled session reports having
        freed. The session forgetting a snapshot is not the same as the bench
        forgetting its table.
        """
        self._held = [held for held in self._held if held.key != key]

    def held(self, key: str) -> Warm | None:
        return next((warm for warm in self._held if warm.key == key), None)

    def _hold(self, warm: Warm) -> Warm:
        self._held = [warm, *(held for held in self._held if held.key != warm.key)][:HELD_CORPORA]
        return warm

    def warm(self, corpus: Corpus, parser: str, *, stage: Stage | None = None) -> Warm:
        """Parse this corpus, or recover the parse from an earlier one.

        The corpus fingerprint covers the documents and their text, so a
        different selection of documents is a different key and gets its own
        parse rather than a wrong hit.
        """
        from core.config import NLPConfig
        from core.pipelines.cache import PipelineCache
        from core.pipelines.resolve import resolve_pipeline

        def report(text: str) -> None:
            if stage is not None:
                stage(text)

        report("Preparing the parser")
        resolved = resolve_pipeline(PipelineCache(), NLPConfig(parser=as_parser(parser), language="en"))
        if resolved.value is None:
            raise ValueError("; ".join(d.message for d in resolved.diagnostics))
        pipeline = resolved.unwrap()
        identity = parser_identity(pipeline)
        key = annotation_key(corpus.sha256, identity)
        # Captured here, where the pipeline is already resolved, and carried on
        # the Warm. Re-resolving it at question time would reload the model.
        tokenizer_name = f"{identity['backend']}/{identity['model'] or identity['language']}"

        def tokenize(text: str) -> Sequence[str]:
            return pipeline.tokenize(text)

        already = self.held(key)
        if already is not None:
            return replace(already, source="memory")

        cached = self.annotations.load(key)
        if cached is not None:
            report("Reading the parse from an earlier run")
            return self._hold(
                Warm(
                    corpus=corpus,
                    key=key,
                    identity=identity,
                    table=cached,
                    diagnostics=tuple(resolved.diagnostics),
                    source="disk",
                    tokenize=tokenize,
                    tokenizer_name=tokenizer_name,
                )
            )

        report(f"Parsing {len(corpus.docs)} document{'' if len(corpus.docs) == 1 else 's'}")
        started = time.perf_counter()
        parsed = pipeline.parse(corpus)
        if parsed.value is None:
            raise ValueError("; ".join(d.message for d in parsed.diagnostics))
        table = parsed.unwrap()
        elapsed = int((time.perf_counter() - started) * 1000)

        report("Keeping the parse so the next question is quick")
        # Failing to cache is not failing to parse. The reader still gets their
        # analysis; the next warm just pays for the parse again.
        with contextlib.suppress(OSError, ValueError):
            self.annotations.store(key, table)
        return self._hold(
            Warm(
                corpus=corpus,
                key=key,
                identity=identity,
                table=table,
                diagnostics=(*resolved.diagnostics, *parsed.diagnostics),
                parse_ms=elapsed,
                tokenize=tokenize,
                tokenizer_name=tokenizer_name,
            )
        )

    def analyse(self, warm: Warm, tool: str, params: dict[str, Any]) -> LiveResult:
        """Run one analysis over an already-parsed corpus.

        The same :func:`execute` a published run calls, given the same corpus
        and the same table, so the rows returned are the rows that would be
        published. That equality is the point: a preview worth acting on has
        to be the thing itself, not a likeness of it.
        """
        from core.profiler.executor import execute
        from core.profiler.plan import build_plan

        # Keyed by tool, exactly as the published path keys it: build_plan
        # takes {tool: {param: value}}, and a bare {param: value} is read as a
        # request for a tool named after the parameter. It fails cleanly rather
        # than loudly -- "params given for unselected tool(s): n" -- so every
        # analysis with a parameter came back empty while analyses without one
        # worked, which looks like the tool's fault and not the caller's.
        planned = build_plan([tool], {tool: params})
        if planned.value is None:
            return LiveResult(tool=tool, ok=False, diagnostics=tuple(planned.diagnostics))
        plan = planned.unwrap()
        if plan.needs_parse and warm.table is None:
            raise ValueError(f"{tool} needs parsed documents, and this corpus has not been parsed.")

        started = time.perf_counter()
        batch = execute(
            plan,
            corpus=warm.corpus,
            table=warm.table if plan.needs_parse else None,
            parse_diagnostics=warm.diagnostics if plan.needs_parse else (),
            tokenizer=warm.tokenizer if plan.needs_parse else None,
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        outcome = batch.outcomes[0]
        return LiveResult(
            tool=tool,
            ok=outcome.ok,
            frames=dict(outcome.frames or {}),
            diagnostics=tuple(outcome.diagnostics),
            elapsed_ms=elapsed,
        )


# How many rows of a live result travel to the browser. The same bound the
# artifact preview uses, so "drawn from N rows on this page" means the same
# thing whether the table came from a finished run or from this second.
LIVE_ROWS = 500


def as_table(frame: pd.DataFrame, limit: int = LIVE_ROWS) -> dict[str, Any]:
    """A result frame in the shape the desktop already reads.

    Identical to what the artifact preview returns, so the workbench draws a
    live table with no idea that it never touched a file.
    """
    head = frame.head(limit)
    return {
        "columns": [str(column) for column in frame.columns],
        "rows": [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in head.to_dict("records")],
        "total": len(frame),
        "filtered_total": len(frame),
        "offset": 0,
        "truncated": len(frame) > len(head),
    }


class StaleSnapshot(ValueError):
    """A question was asked of documents that are no longer loaded.

    Its own class because the interface has to do something specific about it
    -- offer to load the corpus again and re-ask -- rather than show the text
    of a validation failure it can do nothing with.
    """


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One resolved corpus: its parse, its parser, and the selection that
    produced it, committed as a single value.

    Kept together because they were not. Loading set the parser and selection
    before parsing and the parse afterwards, so two overlapping loads left the
    slower one's table sitting behind the faster one's labels: real evidence
    under a false description of where it came from, which no reader could
    detect from the answer.
    """

    warm: Warm
    parser: str
    selection: dict[str, Any] | None
    #: Which load produced this. A load that finishes after a newer one has
    #: started is not allowed to commit.
    generation: int

    @property
    def id(self) -> str:
        """What a question cites. Content-addressed, so the same documents
        parsed the same way are the same snapshot."""
        return self.warm.key


class Session:
    """The corpus a project is exploring right now, and how warm it is.

    One per project. Warming is slow and everything after it is not, so the
    state a reader needs is small: what is being parsed, how far it has got,
    and whether questions can be asked yet.

    Ownership runs through a generation counter rather than a lock held across
    the parse. Holding the lock for thirty seconds would serialise loads but
    also freeze every request that reads the state; counting generations lets
    a superseded load finish harmlessly and discard itself.
    """

    def __init__(self, bench: Bench) -> None:
        self.bench = bench
        self.lock = threading.Lock()
        self._snapshot: Snapshot | None = None
        #: The request currently being parsed, if any. Not a snapshot: until it
        #: commits it describes an intention rather than any evidence.
        self._loading: dict[str, Any] | None = None
        self._generation = 0
        self._stage = ""
        self._error = ""
        #: Live request ids the reader abandoned (R-C8). Answers for these are
        #: discarded rather than delivered to a question that is no longer asked.
        self._cancelled: set[str] = set()
        #: Recent answers by id, newest last, each with the snapshot it was
        #: computed from and the settings that produced it.
        self._answers: dict[str, tuple[str, LiveResult, dict[str, Any]]] = {}

    def remember(self, result: LiveResult, settings: dict[str, Any], snapshot_id: str) -> str:
        """Keep *result* for redrawing its figures; return the id to cite."""
        answer_id = hashlib.sha256(f"{snapshot_id}:{time.perf_counter_ns()}:{id(result)}".encode()).hexdigest()[:24]
        with self.lock:
            self._answers[answer_id] = (snapshot_id, result, dict(settings))
            while len(self._answers) > _ANSWERS_KEPT:
                self._answers.pop(next(iter(self._answers)))
        return answer_id

    def answer(self, answer_id: str) -> tuple[LiveResult, dict[str, Any]] | None:
        """A remembered answer, or None if it was dropped or its corpus was
        reloaded since: a figure drawn from it would describe documents that
        are no longer the ones loaded."""
        with self.lock:
            held = self._answers.get(answer_id)
            current = self._snapshot.id if self._snapshot else ""
        if held is None or held[0] != current:
            return None
        return held[1], held[2]

    def _state(self) -> dict[str, Any]:
        """The state as the interface reads it. The caller holds the lock."""
        snapshot = self._snapshot
        warm = snapshot.warm if snapshot else None
        busy = self._loading is not None
        loading = self._loading or {}
        return {
            "state": "warming" if busy else "failed" if self._error else "ready" if warm else "cold",
            "stage": self._stage,
            "error": self._error,
            "documents": warm.documents if warm else 0,
            "tokens": warm.tokens if warm else 0,
            "parsed": bool(warm and warm.table is not None),
            "cached": bool(warm and warm.cached),
            "source": warm.source if warm else "",
            "parse_ms": warm.parse_ms if warm else 0,
            # While loading, the selection being loaded; otherwise the one the
            # committed snapshot was actually built from. Never a mixture.
            "selection": loading.get("selection") if busy else (snapshot.selection if snapshot else None),
            "parser": str(loading.get("parser", "")) if busy else (snapshot.parser if snapshot else ""),
            "snapshot_id": snapshot.id if snapshot else "",
            "generation": self._generation,
        }

    def state(self) -> dict[str, Any]:
        with self.lock:
            return self._state()

    def warm(self, corpus: Corpus, parser: str, *, selection: dict[str, Any] | None = None) -> dict[str, Any]:
        """Parse this corpus. Blocking: the caller runs it off the event loop."""
        with self.lock:
            self._generation += 1
            mine = self._generation
            self._error, self._stage = "", "Starting"
            self._loading = {"selection": selection, "parser": parser}
        try:
            warm = self.bench.warm(corpus, parser, stage=lambda text: self._report(mine, text))
        except Exception as exc:
            with self.lock:
                if mine != self._generation:
                    # Superseded: this failure belongs to a load nobody waits
                    # for, and reporting it would blame the current one.
                    return self._state()
                self._loading, self._stage = None, ""
                self._error = str(exc)
                self._snapshot = None
                return self._state()
        with self.lock:
            if mine != self._generation:
                # A newer load started, or the corpus was cooled, while this was
                # parsing. The table is correct and describes documents nobody
                # is asking about; committing it would restore a released corpus
                # or stand behind the newer request's labels.
                #
                # The bench is already holding it, though: warming holds before
                # it returns. Forgetting it here is what makes a cooled session
                # actually cold, rather than a session that says "cold" while
                # the parse it released sits in memory behind it.
                committed = self._snapshot.warm.key if self._snapshot else ""
                if warm.key != committed:
                    self.bench.forget(warm.key)
                return self._state()
            self._snapshot = Snapshot(warm=warm, parser=parser, selection=selection, generation=mine)
            self._loading, self._stage = None, ""
            return self._state()

    def _report(self, generation: int, stage: str) -> None:
        with self.lock:
            # A superseded load's progress is not this session's progress.
            if generation == self._generation:
                self._stage = stage

    def resolved(self, snapshot_id: str = "") -> Snapshot:
        """The snapshot a question may be asked of, or why it may not be.

        *snapshot_id* is what the caller believes it is asking about. A
        mismatch is refused rather than quietly answered from whatever happens
        to be loaded now, because that answer looks exactly like the one that
        was asked for.
        """
        with self.lock:
            if self._loading is not None:
                raise ValueError("Still reading the documents. This will be ready in a moment.")
            snapshot = self._snapshot
            if snapshot is None:
                # Nothing loaded is not a stale reference: there is no answer to
                # carry on from, and the reader has simply not started yet.
                raise ValueError("Choose the documents to explore first.")
            if snapshot_id and snapshot_id != snapshot.id:
                raise StaleSnapshot(
                    "These are not the documents that answer was built from. Load them again to continue from here."
                )
            return snapshot

    def analyse(self, tool: str, params: dict[str, Any]) -> LiveResult:
        return self.bench.analyse(self.resolved().warm, tool, params)

    def cancel_analysis(self, request_id: str) -> dict[str, Any]:
        """Record that nobody is waiting for this answer any more (R-C8).

        This is "discard the answer", not "stop the CPU": a gensim or sklearn
        fit is not interruptible mid-call, and pretending otherwise would be a
        lie the reader could measure as a machine that keeps heating up after
        they walked away. What it does buy: the answer is dropped instead of
        landing on a question that is gone, and an answer not yet started is
        never computed at all.

        The set is bounded — it holds only the ids of abandoned questions, so
        it may not grow with every question ever asked.
        """
        with self.lock:
            self._cancelled.add(request_id)
            if len(self._cancelled) > _CANCELLED_KEEP * 2:
                self._cancelled = set(sorted(self._cancelled)[-_CANCELLED_KEEP:])
        return {"ok": True, "request_id": request_id, "cancelled": True}

    def analysis_cancelled(self, request_id: str) -> bool:
        """Whether *request_id* was abandoned while it was being answered."""
        with self.lock:
            return request_id in self._cancelled

    def forget_cancelled(self, request_id: str) -> None:
        """Drop a cancellation once it has been honoured (id is one-use)."""
        with self.lock:
            self._cancelled.discard(request_id)

    def track_phrase(self, request: dict[str, Any]) -> dict[str, Any]:
        """Answer one phrase-distribution question from the warmed evidence."""
        from core.research.phrase import EvidenceRequest, PhraseQuery, track_phrase

        snapshot = self.resolved(str(request.get("snapshot_id") or ""))
        warm = snapshot.warm
        if warm.table is None:
            raise ValueError("Phrase tracking needs parsed documents.")
        return track_phrase(
            warm.table,
            warm.corpus,
            PhraseQuery(
                text=str(request["text"]),
                case_sensitive=bool(request.get("case_sensitive", False)),
                position_bins=int(request.get("position_bins", 10)),
                normalize=bool(request.get("normalize", False)),
                match_lemma=bool(request.get("match_lemma", False)),
                match_nominalization=bool(request.get("match_nominalization", False)),
            ),
            snapshot_id=snapshot.id,
            evidence=EvidenceRequest(
                offset=int(request.get("evidence_offset", 0)),
                limit=int(request.get("evidence_limit", 100)),
                year=request.get("evidence_year"),
                document_id=request.get("evidence_document_id"),
                position_start=request.get("evidence_position_start"),
                position_end=request.get("evidence_position_end"),
            ),
            tokenize=warm.tokenize,
            tokenizer_name=warm.tokenizer_name,
        )

    def cool(self) -> None:
        """Forget the warmed corpus. The cached parse on disk survives."""
        with self.lock:
            # Bumped so a load still parsing cannot commit afterwards and
            # quietly restore what was just released.
            self._generation += 1
            self._snapshot = None
            self._loading = None
            self._stage, self._error = "", ""
        # Forgetting the corpus has to actually free it, or "cool" only hides
        # the 68 MB it was holding.
        self.bench.release()


__all__ = [
    "HELD_CORPORA",
    "LIVE_ROWS",
    "PARSERS",
    "SCHEMA_VERSION",
    "Annotations",
    "Bench",
    "LiveResult",
    "Session",
    "Snapshot",
    "StaleSnapshot",
    "Warm",
    "annotation_key",
    "as_parser",
    "parser_identity",
]
