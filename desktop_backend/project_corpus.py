"""A project's documents as a :class:`Corpus`, and that corpus parsed once.

Three callers build a corpus from a project and parse it: a published run
(:mod:`desktop_backend.runner`), the live bench (:mod:`desktop_backend.live`)
and a notebook's kernel (:mod:`desktop_backend.kernel`). Each used to spell
out the reading, the hash check and the parse for itself, and only the bench
ever looked in the parse cache. So a run over twenty speeches parsed them
again every time it was asked -- about fifty seconds, most of the run -- while
the bench, one page away, had the same table on disk.

Built here once, the three agree on two things that must not drift apart:
what counts as "the same corpus" (the fingerprint), and what counts as "the
same parse" (the annotation key). The first run pays for the parse; the
second, from any of the three doors, reads it back.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import contextlib
from dataclasses import dataclass
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Literal

from core.io.cleaning import strip_stage_directions
from core.io.document_fields import document_date
from core.io.reader import Corpus, Document, corpus_fingerprint, hash_file, hash_text, ordered_fingerprint, read_text
from core.result import Diagnostic, Result
from desktop_backend.fields import TextCleaning, cleaning as project_cleaning

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from core.research.phrase import Tokenization
    from desktop_backend.live import Annotations
    from desktop_backend.store import Workspace

__all__ = ["Parsed", "load_corpus", "parse_cached", "resolve_parser"]


def load_corpus(
    workspace: Workspace,
    project_id: str,
    items: Sequence[dict[str, Any]],
    *,
    text_cleaning: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Corpus, list[Diagnostic]]:
    """Read the documents *items* name, checking each is still what was imported.

    *items* are document rows (a run's frozen request, or a resolved
    selection). A row may carry its own ``project_id``, which a comparison
    across projects needs; otherwise it belongs to *project_id*.

    *text_cleaning* is what each project leaves out of the analysis (stage
    directions like "(Applause.)"), frozen into the run at submission so a
    result says what it read; a request from before the setting existed falls
    back to the project's current setting. The imported file is never changed
    (R3): the clean is of the text the tools see.

    A changed file is an error, not a warning: a run records what it read, and
    reading something other than what was imported would make that record
    false. An unreadable one costs its own document and a diagnostic.
    """
    documents: list[Document] = []
    diagnostics: list[Diagnostic] = []
    rules: dict[str, TextCleaning] = {}
    removed_total = 0
    removed_docs = 0

    def rule_for(home: str) -> TextCleaning:
        if home not in rules:
            frozen = (text_cleaning or {}).get(home)
            rules[home] = TextCleaning(**frozen) if frozen is not None else project_cleaning(workspace, home)
        return rules[home]

    for index, item in enumerate(items, 1):
        home = str(item.get("project_id") or project_id)
        path = workspace.project_dir(home) / "corpus" / str(item["stored_name"])
        if hash_file(path) != item["sha256"]:
            raise ValueError(f"Imported document has changed since import: {item['name']}. Reimport it before running.")
        text = read_text(path)
        diagnostics.extend(text.diagnostics)
        if text.value is None:
            continue
        raw = text.unwrap()
        rule = rule_for(home)
        body, removed = strip_stage_directions(raw, rule.extra_terms) if rule.stage_directions else (raw, 0)
        if removed:
            removed_total += removed
            removed_docs += 1
        # A request frozen before documents had details has no "fields": it
        # still reads its dates from the names, exactly as it did.
        details = item.get("fields") or {}
        documents.append(
            Document(
                doc_id=index,
                path=path,
                text=body,
                sha256=hash_text(body),
                date=document_date(details, str(item["name"])),
                source_id=str(item["id"]),
                source_project_id=str(item.get("project_id") or ""),
                label=str(item["name"]),
                fields=tuple(sorted((str(k), str(v)) for k, v in details.items())),
            )
        )
    if not documents:
        raise ValueError("No readable documents remain in this selection.")
    if removed_total:
        diagnostics.append(
            Diagnostic.info(
                "STAGE_DIRECTIONS_REMOVED",
                f"left out {removed_total} bracketed stage direction(s) in {removed_docs} document(s), "
                "as the project's text cleaning asks (plan 5.2)",
                removed=removed_total,
                documents=removed_docs,
            )
        )
    docs = tuple(documents)
    return Corpus(docs, corpus_fingerprint(docs)), diagnostics


def resolve_parser(parser: str) -> Result[Any]:
    """The English pipeline for *parser*, with the CLI's fallback policy.

    A backend whose model is missing or damaged does not fail the caller while
    another installed backend can parse; the substitution rides along in the
    diagnostics, and :func:`parse_cached` keys the cache by the backend that
    actually ran, so a fallback's table is never mistaken for the requested one.
    """
    from core.config import NLPConfig
    from core.pipelines.cache import PipelineCache
    from core.pipelines.resolve import resolve_pipeline
    from desktop_backend.live import as_parser

    return resolve_pipeline(PipelineCache(), NLPConfig(parser=as_parser(parser), language="en"))


@dataclass(frozen=True)
class Parsed:
    """A corpus's token table, and where it came from."""

    table: pd.DataFrame
    key: str
    identity: dict[str, str]
    diagnostics: tuple[Diagnostic, ...]
    #: "parsed" paid for the parse; "disk" read an earlier one back.
    source: Literal["parsed", "disk"]
    parse_ms: int
    tokenize: Callable[[str], Sequence[str]]
    tokenizer_name: str

    @property
    def tokenizer(self) -> Tokenization:
        """How this table's parser splits text, for tools that split a question."""
        from core.research.phrase import Tokenization

        return Tokenization(self.tokenize, self.tokenizer_name)


def _store_parse(annotations: Annotations, key: str, table: pd.DataFrame) -> None:
    """A cache write is best-effort after parsing has already succeeded."""
    with contextlib.suppress(OSError, ValueError):
        annotations.store(key, table)


def parse_cached(
    root: Path,
    corpus: Corpus,
    pipeline: Any,
    *,
    stage: Callable[[str], None] | None = None,
) -> Parsed:
    """Parse *corpus* with *pipeline*, or read back the parse an earlier caller kept.

    The key is the corpus fingerprint plus everything about the parser that
    could change its output (backend, model, versions, table schema). See
    :func:`desktop_backend.live.annotation_key`. Failing to *store* the parse
    is not failing to parse: the caller gets its table either way.

    Below the corpus key sits a per-document one (plan 5.1): the documents a
    comparison joins are the same documents its two projects already paid for,
    so a run over twenty speeches against thirty-one inaugurals parses the
    speeches once and the inaugurals once, not the joined fifty-one twice. The
    missing documents are parsed in one call and split apart; each slice is
    kept with its own ids, and the table handed back is the slices joined in
    corpus order with Record ID offset to run on (Sentence ID restarts per
    document, as it always has). Splitting that way is safe because a parse is
    per-document independent -- ``tests/test_parse_per_document.py`` proves
    parsing apart equals parsing together.
    """
    from desktop_backend.live import Annotations, annotation_key, parser_identity

    def report(text: str) -> None:
        if stage is not None:
            stage(text)

    identity = parser_identity(pipeline)
    # The parse key is order-SENSITIVE: Document/Sentence/Record IDs are
    # positions, so two orderings of the same documents are two tables. The
    # per-document keys below stay order-free (a document's parse does not
    # depend on its neighbours), which is what keeps 5.1's reuse intact.
    key = annotation_key(ordered_fingerprint(corpus.docs), identity)
    tokenizer_name = f"{identity['backend']}/{identity['model'] or identity['language']}"

    def tokenize(text: str) -> Sequence[str]:
        tokens: Sequence[str] = pipeline.tokenize(text)
        return tokens

    annotations = Annotations(root)
    cached = annotations.load(key)
    if cached is not None:
        report("Reading the parse from an earlier run")
        return Parsed(cached, key, identity, (), "disk", 0, tokenize, tokenizer_name)

    def doc_key(doc: Any) -> str:
        return annotation_key(doc.sha256, identity)

    report(f"Parsing {len(corpus.docs)} document{'' if len(corpus.docs) == 1 else 's'}")
    started = time.perf_counter()
    slices: dict[str, Any] = {}
    missing = []
    for doc in corpus.docs:
        one = annotations.load(doc_key(doc))
        if one is None:
            missing.append(doc)
        else:
            slices[str(doc.doc_id)] = one
    notes: tuple[Any, ...] = ()
    if missing:
        parsed = pipeline.parse(Corpus(tuple(missing), corpus_fingerprint(tuple(missing))))
        if parsed.value is None:
            raise ValueError("; ".join(d.message for d in parsed.diagnostics))
        batch = parsed.unwrap()
        notes = tuple(parsed.diagnostics)
        for doc in missing:
            one = batch[batch["Document ID"].astype(str) == str(doc.doc_id)].reset_index(drop=True)
            if one.empty:
                one = batch[batch["Document"].astype(str) == doc.name].reset_index(drop=True)
            # The slice keeps its own numbering: ids are per-document and
            # rewritable, and a cache entry must not carry another corpus's
            # Record ID offsets with it.
            one["Record ID"] = list(range(1, len(one) + 1))
            _store_parse(annotations, doc_key(doc), one)
            slices[str(doc.doc_id)] = one
    offset = 0
    rows = []
    for doc in corpus.docs:
        one = slices[str(doc.doc_id)]
        one["Document ID"] = str(doc.doc_id)
        one["Document"] = doc.name
        one["Record ID"] = one["Record ID"].astype("int64") + offset
        offset += len(one)
        rows.append(one)
    import pandas as pd

    table = pd.concat(rows, ignore_index=True) if len(rows) > 1 else rows[0].reset_index(drop=True)
    elapsed = int((time.perf_counter() - started) * 1000)
    _store_parse(annotations, key, table)
    return Parsed(
        table,
        key,
        identity,
        notes,
        "parsed" if missing else "disk",
        elapsed if missing else 0,
        tokenize,
        tokenizer_name,
    )
