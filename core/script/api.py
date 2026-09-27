"""The functions a notebook calls: ``import nlpsuite as nlp``.

Every function here is a thin, named door onto something the suite already
does. :func:`run` builds the same plan and calls the same
:func:`core.profiler.executor.execute` a published run calls, so a table a
script gets is the table the Analyze page would publish -- not a likeness of
it. The convenience functions (:func:`term_rates`, :func:`passages`,
:func:`keyness`, ...) are :func:`run` or an existing analysis function plus the
reshaping a script would otherwise write by hand. Where a script needed
behaviour no tool had (counting words in the raw text, the way the hand-written
research scripts did), it went into ``core.analysis`` first and is wrapped here.

Rules the functions keep:

* They return pandas DataFrames. People know pandas, and every table a tool
  writes already is one.
* They never write files. Output goes through :func:`show`, :func:`figure` and
  :func:`save`, which hand it to the session; the app decides where it lands.
* A failure is a :class:`SuiteError` whose message is the engine's own
  sentence, never a traceback into the engine.
* Every call is recorded in the session, so a saved run can say which suite
  tool produced each table.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import difflib
import io
import re
import time
from typing import TYPE_CHECKING, Any

import pandas as pd

from core.script.session import (
    FigureOutput,
    FolderSource,
    NoteOutput,
    Session,
    SuiteError,
    TableOutput,
    current,
    set_current,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from core.io.reader import Corpus as CoreCorpus

__all__ = [
    "Corpus",
    "RunResult",
    "SuiteError",
    "corpus",
    "describe",
    "entities",
    "figure",
    "keyness",
    "load",
    "measures",
    "note",
    "passages",
    "run",
    "save",
    "sentiment",
    "show",
    "similar",
    "tables",
    "term_rates",
    "tools",
    "topics",
    "use_folder",
]

#: Tools a script may run: every tool with a batch adapter, less the ones that
#: only make sense behind their own page.
_PRIVATE_TOOLS = frozenset({"phrase_distribution"})
#: Chart kinds the app's chart canvas draws (desktop/src/chartLayout.ts LIVE_KINDS).
_DRAWN_KINDS = ("bar", "line", "scatter", "bubble", "histogram", "box", "heatmap")
_MATCHES = ("lemma", "form", "exact-lowercase")
#: What counts may be grouped along. Beyond these, ``by="field:<detail name>"``
#: groups by any document detail ("field:Village") - the same vocabulary the
#: lexicon tool speaks, so one grouping question has one answer everywhere.
_BY = ("document", "year", "decade", "order", "period")


# ------------------------------------------------------------------ corpus --


class Corpus:
    """A set of documents a script works on.

    ``documents`` is a table with one row per document: ``Document ID`` (the
    id every result table uses), ``Document``, ``Date`` and ``Year`` (when the
    file name holds a date), ``Words``, and any details the documents carry.
    Narrow it with :meth:`filter` or :meth:`where`; every narrowing is a new
    corpus, so the original stays as it was.
    """

    def __init__(self, session: Session, ids: Sequence[str] | None) -> None:
        self._session = session
        self._ids = None if ids is None else tuple(ids)
        self._core: CoreCorpus | None = None
        self._documents: pd.DataFrame | None = None

    @property
    def core(self) -> CoreCorpus:
        """The engine's corpus object (for advanced use; most scripts never need it)."""
        if self._core is None:
            self._core = self._session.source.corpus(self._ids)
        return self._core

    @property
    def documents(self) -> pd.DataFrame:
        if self._documents is None:
            fields = {str(row["id"]): row.get("fields") or {} for row in self._session.source.documents()}
            rows = []
            for doc in self.core.docs:
                row: dict[str, Any] = {
                    "Document ID": str(doc.doc_id),
                    "Document": doc.name,
                    "Date": doc.date.isoformat() if doc.date else None,
                    "Year": doc.date.year if doc.date else None,
                    "Words": len(doc.text.split()),
                }
                row.update(fields.get(doc.source_id, {}))
                rows.append(row)
            frame = pd.DataFrame(rows)
            if frame["Year"].notna().all():
                frame["Year"] = frame["Year"].astype(int)
            self._documents = frame
        return self._documents.copy()

    def __len__(self) -> int:
        return len(self.core.docs)

    def __repr__(self) -> str:
        years = [doc.date.year for doc in self.core.docs if doc.date]
        span = f", {min(years)}-{max(years)}" if years else ""
        return f"<Corpus: {len(self)} documents{span}>"

    def _narrowed(self, keep: Sequence[bool], described: str) -> Corpus:
        chosen = [doc.source_id for doc, wanted in zip(self.core.docs, keep, strict=True) if wanted]
        if not chosen:
            raise SuiteError(f"No documents match {described}. Look at corpus.documents to see what there is.")
        return Corpus(self._session, chosen)

    def filter(self, **conditions: Any) -> Corpus:
        """The documents whose columns match every condition.

        A condition is a value (``Year=1946``), a list or range of values
        (``Year=range(1950, 1990)``), or a function of the value
        (``Words=lambda n: n > 5000``). Names are ``documents`` columns.
        """
        docs = self.documents
        keep = pd.Series(True, index=docs.index)
        for column, wanted in conditions.items():
            if column not in docs.columns:
                raise SuiteError(
                    f"The documents have no column {column!r}. Columns: {', '.join(map(str, docs.columns))}."
                )
            values = docs[column]
            if callable(wanted):
                keep &= values.map(lambda v, test=wanted: bool(test(v)))
            elif isinstance(wanted, (list, tuple, set, frozenset, range)):
                keep &= values.isin(list(wanted))
            else:
                keep &= values == wanted
        return self._narrowed(keep.tolist(), ", ".join(f"{k}={v!r}" for k, v in conditions.items()))

    def where(self, predicate: Callable[[pd.Series], bool]) -> Corpus:
        """The documents for which ``predicate(row)`` is true (a row of ``documents``)."""
        docs = self.documents
        keep = [bool(predicate(row)) for _, row in docs.iterrows()]
        return self._narrowed(keep, "that condition")

    def text(self, document: int | str) -> str:
        """One document's text, by ``Document ID`` or by name."""
        for doc in self.core.docs:
            if str(doc.doc_id) == str(document) or doc.name == document:
                return doc.text
        raise SuiteError(f"No document {document!r} in this corpus.")

    def tokens(self) -> pd.DataFrame:
        """The parsed token table (one row per word; parsed once, then reused)."""
        return self._session.parse(self.core).table

    def sentences(self) -> pd.DataFrame:
        """One row per parsed sentence: ``Document ID``, ``Document``, ``Sentence ID``, ``Text``."""
        from core.analysis.detokenize import detokenize

        table = self.tokens()
        rows = [
            {"Document ID": str(doc_id), "Sentence ID": sentence, "Text": detokenize(group["Form"].astype(str))}
            for (doc_id, sentence), group in table.groupby(["Document ID", "Sentence ID"], sort=False)
        ]
        return _with_documents(pd.DataFrame(rows), self)


def _with_documents(frame: pd.DataFrame, corpus: Corpus) -> pd.DataFrame:
    """*frame* with each document's name, date and year beside its ``Document ID``."""
    docs = corpus.documents[["Document ID", "Document", "Date", "Year"]]
    rest = [c for c in frame.columns if c not in ("Document ID", "Document", "Date", "Year")]
    return docs.merge(frame[["Document ID", *rest]], on="Document ID", how="right")[
        ["Document ID", "Document", "Date", "Year", *rest]
    ]


def corpus(selection: Mapping[str, Any] | None = None) -> Corpus:
    """This notebook's documents (all of them, or those matching *selection*).

    *selection* is the same as :meth:`Corpus.filter`'s conditions, as a dict:
    ``nlp.corpus({"Year": range(1950, 1990)})``.
    """
    session = current()
    started = time.perf_counter()
    whole = Corpus(session, None)
    chosen = whole.filter(**dict(selection)) if selection else whole
    session.record("corpus", {"selection": selection}, started, len(chosen))
    return chosen


def use_folder(directory: str, *, parser: str = "spacy") -> Corpus:
    """Outside the app: read a folder of ``.txt`` files and make it this notebook's corpus."""
    from pathlib import Path

    set_current(Session(FolderSource(Path(directory), parser=parser)))
    return corpus()


# -------------------------------------------------------------------- tools --


class RunResult:
    """What one tool produced: every table it wrote, and its diagnostics."""

    def __init__(self, tool: str, tables: dict[str, pd.DataFrame], diagnostics: list[dict[str, str]]) -> None:
        self.tool = tool
        self.tables = tables
        self.diagnostics = diagnostics

    @property
    def table(self) -> pd.DataFrame:
        """The tool's main table (the first CSV it declares)."""
        for name, frame in self.tables.items():
            if name.endswith(".csv"):
                return frame
        raise SuiteError(f"{self.tool} produced no table.")

    def __getitem__(self, name: str) -> pd.DataFrame:
        for candidate in (name, f"{name}.csv"):
            if candidate in self.tables:
                return self.tables[candidate]
        raise SuiteError(f"{self.tool} produced no table {name!r}. It produced: {', '.join(self.tables)}.")

    def __repr__(self) -> str:
        return f"<RunResult {self.tool}: {', '.join(self.tables)}>"


def _tool_names() -> list[str]:
    from core.profiler.executor import ADAPTERS
    from core.profiler.registry import get_tool

    return sorted(name for name in ADAPTERS if name not in _PRIVATE_TOOLS and get_tool(name) is not None)


def tools() -> pd.DataFrame:
    """Every tool :func:`run` accepts: ``Tool``, ``Name``, ``What it does``, ``Needs parsing``, ``Tables``."""
    from core.profiler.labels import tool_description, tool_label
    from core.profiler.registry import get_tool

    rows = []
    for name in _tool_names():
        spec = get_tool(name)
        assert spec is not None  # noqa: S101 - filtered in _tool_names
        rows.append(
            {
                "Tool": name,
                "Name": tool_label(name),
                "What it does": tool_description(name, spec.description),
                "Needs parsing": spec.requires_parse,
                "Tables": ", ".join(o for o in spec.outputs if o.endswith(".csv")),
            }
        )
    return pd.DataFrame(rows)


def _python_name(name: str) -> str:
    return name.replace("-", "_")


def describe(tool: str) -> str:
    """A tool's parameters (Python names, types, defaults, choices) and the tables it writes."""
    from core.profiler.labels import tool_description, tool_label

    spec = _spec(tool)
    lines = [f"{tool}: {tool_label(tool)}", tool_description(tool, spec.description), "", "Parameters:"]
    if not spec.params:
        lines.append("  (none)")
    for param in spec.params:
        detail = f"{param.type}, default {param.default!r}"
        if param.choices:
            detail += f", one of {list(param.choices)}"
        lines.append(f"  {_python_name(param.name)} ({detail}): {param.help}")
    lines += ["", "Tables: " + ", ".join(spec.outputs)]
    return "\n".join(lines)


def _spec(tool: str) -> Any:
    from core.profiler.registry import get_tool

    names = _tool_names()
    if tool not in names:
        close = difflib.get_close_matches(tool, names, n=3)
        hint = f" Did you mean {', '.join(close)}?" if close else " nlp.tools() lists them."
        raise SuiteError(f"There is no tool called {tool!r}.{hint}")
    return get_tool(tool)


def run(tool: str, corpus: Corpus, **params: Any) -> RunResult:
    """Run one suite tool over *corpus*, exactly as a published run would.

    Parameters use Python spelling (``top_n=20``); :func:`describe` lists a
    tool's parameters. The corpus is parsed once per notebook and reused.
    """
    from core.profiler.executor import execute
    from core.profiler.plan import build_plan

    session = current()
    started = time.perf_counter()
    spec = _spec(tool)
    known = {param.name for param in spec.params}
    supplied = {(key if key in known else key.replace("_", "-")): value for key, value in params.items()}
    planned = build_plan([tool], {tool: supplied})
    if planned.value is None:
        messages = [d.message for d in planned.diagnostics]
        unknown = [key for key in supplied if key not in known]
        for key in unknown:
            close = difflib.get_close_matches(key, sorted(known), n=1)
            if close:
                messages.append(f"did you mean {_python_name(close[0])}?")
        raise SuiteError(f"{tool}: " + "; ".join(messages), planned.diagnostics)
    plan = planned.unwrap()
    parsed = session.parse(corpus.core) if plan.needs_parse else None
    batch = execute(
        plan,
        corpus=corpus.core,
        table=parsed.table if parsed else None,
        parse_diagnostics=parsed.diagnostics if parsed else (),
        tokenizer=parsed.tokenizer if parsed else None,
    )
    outcome = batch.outcomes[0]
    frames = {name: frame for name, frame in (outcome.frames or {}).items() if not name.endswith(".html")}
    diagnostics = [{"severity": d.severity.value, "message": d.message} for d in outcome.diagnostics]
    if not frames:
        reasons = "; ".join(d.message for d in outcome.diagnostics) or "no reason given"
        raise SuiteError(f"{tool} produced no table: {reasons}", outcome.diagnostics)
    result = RunResult(tool, frames, diagnostics)
    session.record("run", {"tool": tool, **params}, started, len(result.table))
    return result


# ------------------------------------------------------------- word counts --


def _lexicon(groups: Mapping[str, Iterable[str]] | Iterable[str] | str) -> dict[str, tuple[str, ...]]:
    from core.analysis.lexicon_series import parse_lexicon

    if isinstance(groups, str):
        parsed = parse_lexicon(groups)
        if parsed.value is None:
            raise SuiteError("; ".join(d.message for d in parsed.diagnostics), parsed.diagnostics)
        return parsed.unwrap()
    if isinstance(groups, Mapping):
        lexicon = {
            str(name): tuple(dict.fromkeys(str(t).strip().lower() for t in terms if str(t).strip()))
            for name, terms in groups.items()
        }
    else:
        lexicon = {"terms": tuple(dict.fromkeys(str(t).strip().lower() for t in groups if str(t).strip()))}
    empty = [name for name, terms in lexicon.items() if not terms]
    if not lexicon or empty:
        raise SuiteError(f"Every word group needs at least one word; empty: {', '.join(empty) or 'all'}.")
    return lexicon


def _check_match(match: str) -> None:
    if match not in _MATCHES:
        raise SuiteError(f"match must be one of {', '.join(_MATCHES)}, not {match!r}.")


def _per_label(per: float) -> str:
    return f"{per:g}" if float(per) != int(per) else str(int(per))


def _facet_groups(corpus: Corpus, by: str) -> tuple[str, dict[str, str]]:
    """The group column's name and each document's label, for ``by`` past year.

    The labels come from :func:`core.analysis.lexicon_series.facet_labels` -
    the same computation the lexicon tool uses - and ``"period"`` therefore
    means the axis's blocks ("Documents 1-15") exactly as it does there. Every
    document must get a label: one dropped document is a silent hole in a rate
    table, so the refusals name what is missing instead.
    """
    from core.analysis.lexicon_series import BY_FIELD_PREFIX, facet_labels
    from core.io.document_fields import document_order

    docs = corpus.core.docs
    labeled = facet_labels(
        [(doc.name, doc.date) for doc in docs],
        by,
        details={doc.name: dict(doc.fields) for doc in docs},
        positions={doc.name: value for doc in docs if (value := document_order(doc.details)) is not None},
    )
    if labeled.value is None or labeled.diagnostics:
        why = "; ".join(d.message for d in labeled.diagnostics) or "the documents cannot be grouped that way"
        raise SuiteError(f"Counting by {by} needs a value for every document: {why}")
    field_name = by[len(BY_FIELD_PREFIX) :].strip() if by.startswith(BY_FIELD_PREFIX) else ""
    key = field_name or ("Order" if by == "order" else "Period")
    return key, dict(labeled.value)


def _pool(per_doc: pd.DataFrame, corpus: Corpus, by: str) -> pd.DataFrame:
    """One row per group along ``by``: documents counted, words and counts summed."""
    counts = [c for c in per_doc.columns if c.endswith(" count")]
    pooled: dict[str, Any] = {
        "Documents": ("Document ID", "count"),
        **{"Words counted": ("Words counted", "sum")},
        **{c: (c, "sum") for c in counts},
    }
    if by in ("year", "decade"):
        if per_doc["Year"].isna().any():
            raise SuiteError(f"Counting by {by} needs a date for every document; some have none.")
        key = "Year" if by == "year" else "Decade"
        if by == "decade":
            per_doc["Decade"] = (per_doc["Year"].astype(int) // 10 * 10).astype(str) + "s"
        return per_doc.groupby(key, as_index=False).agg(**pooled)
    from core.corpus_axis import period_key

    key, labels = _facet_groups(corpus, by)
    if key in per_doc.columns:
        key = f"Group: {key}"
    grouped = per_doc.copy()
    grouped[key] = grouped["Document"].map(labels)
    out = grouped.groupby(key, as_index=False, sort=False).agg(**pooled)
    # "2" before "10": text order put chapter 10 second.
    return out.sort_values(key, key=lambda column: column.map(period_key)).reset_index(drop=True)


def term_rates(
    corpus: Corpus,
    groups: Mapping[str, Iterable[str]] | Iterable[str] | str,
    *,
    per: float = 1000,
    by: str = "document",
    match: str = "lemma",
) -> pd.DataFrame:
    """How often each group of words occurs, as a rate per *per* words.

    *groups* is a dict of named word lists (``{"immigration": ["immigrant",
    "border"]}``), one list (named "terms"), or the text form ``"Name: word,
    word; Other: word"``. A multi-word term matches as a phrase.

    *match* says what counts as a word, and the numbers differ between them,
    so say in your write-up which you used:

    * ``"lemma"`` (default): the parser's dictionary forms, so "immigrants"
      counts as "immigrant". List dictionary forms.
    * ``"form"``: the parser's words as written, lowercased.
    * ``"exact-lowercase"``: runs of letters in the raw text, lowercased, no
      parser. The rule hand-written research scripts used; use it to reproduce
      their numbers. List every inflection you want counted.

    *by* is ``"document"`` (one row per document, with Date and Year),
    ``"year"``, ``"decade"``, ``"order"`` (one row per Order value - the
    chapters of a book), ``"period"`` (the axis cut into blocks, "Documents
    1-15"), or ``"field:<detail name>"`` for any document detail
    (``by="field:Village"``). By anything but document, words are pooled: the
    rate is all matches over all words of that group. For the average of
    per-document rates instead, count by document and group the rows yourself.
    Every document must carry the grouping's value (a date for year, an Order
    for order, the detail for field:...) - one missing document is a hole in
    the rates, so it is refused rather than silently dropped.

    Columns: the document (or period) columns, ``Words counted``, and for each
    group ``<group> count`` and ``<group> per <per>``.
    """
    from core.analysis.lexicon_series import lexicon_series, raw_text_series
    from core.conll.schema import Col

    session = current()
    started = time.perf_counter()
    _check_match(match)
    from core.analysis.lexicon_series import BY_FIELD_PREFIX

    if by not in _BY and not (by.startswith(BY_FIELD_PREFIX) and by[len(BY_FIELD_PREFIX) :].strip()):
        raise SuiteError(
            f"by must be one of {', '.join(_BY)}, or '{BY_FIELD_PREFIX}<detail name>' (like 'field:Village'), not {by!r}."
        )
    if per <= 0:
        raise SuiteError("per must be a positive number, such as 1000 or 10000.")
    lexicon = _lexicon(groups)
    if match == "exact-lowercase":
        counted = raw_text_series({str(doc.doc_id): doc.text for doc in corpus.core.docs}, lexicon)
    else:
        table = session.parse(corpus.core).table
        # Facets keyed by Document ID: the table's Document column is the
        # stored file name, which is not always the name a reader sees.
        ids = dict(zip(table["Document"].astype(str), table["Document ID"].astype(str), strict=False))
        counted = lexicon_series(table, lexicon, ids, field=Col.LEMMA if match == "lemma" else Col.FORM)
    if counted.value is None:
        raise SuiteError("; ".join(d.message for d in counted.diagnostics), counted.diagnostics)
    long = counted.unwrap().rename(columns={"Facet": "Document ID"})
    wide = long.pivot_table(index="Document ID", columns="Category", values="Occurrences", aggfunc="sum")
    wide = wide.reindex(columns=list(lexicon)).fillna(0).astype(int)
    wide.columns = [f"{name} count" for name in wide.columns]
    tokens = long.groupby("Document ID")["Tokens"].first().rename("Words counted")
    per_doc = pd.concat([tokens, wide], axis=1).reset_index()
    per_doc = _with_documents(per_doc, corpus)
    out = per_doc if by == "document" else _pool(per_doc, corpus, by)
    label = _per_label(per)
    for name in lexicon:
        words = out["Words counted"].replace(0, pd.NA)
        out[f"{name} per {label}"] = (out[f"{name} count"] * float(per) / words).astype(float).fillna(0.0)
    session.record("term_rates", {"groups": lexicon, "per": per, "by": by, "match": match}, started, len(out))
    return out.reset_index(drop=True)


def passages(
    corpus: Corpus,
    terms: Mapping[str, Iterable[str]] | Iterable[str] | str,
    *,
    context: int = 1,
    match: str = "lemma",
    limit: int = 500,
) -> pd.DataFrame:
    """Every sentence that contains one of *terms*, with *context* sentences either side.

    Found by the same rule :func:`term_rates` counts with (*match* means the
    same there and here), so every counted occurrence has a passage.

    Columns: ``Document ID``, ``Document``, ``Date``, ``Year``, ``Sentence ID``
    (its number in the document, from 1, as in the token table), ``Groups``, ``Terms``, ``Before``,
    ``Text``, ``After``. At most *limit* rows, in document order.
    """
    from core.analysis.detokenize import detokenize
    from core.analysis.lexicon_series import RAW_WORD, raw_sentences, term_hits

    session = current()
    started = time.perf_counter()
    _check_match(match)
    lexicon = _lexicon(terms)
    by_term = {term: [term] for words in lexicon.values() for term in words}
    group_of = {term: name for name, words in lexicon.items() for term in words}
    documents: list[tuple[str, list[str], list[list[str]]]] = []
    if match == "exact-lowercase":
        for doc in corpus.core.docs:
            sentences = raw_sentences(doc.text)
            documents.append((str(doc.doc_id), sentences, [RAW_WORD.findall(s.lower()) for s in sentences]))
    else:
        table = corpus.tokens()
        column = "Lemma" if match == "lemma" else "Form"
        for doc_id, rows in table.groupby("Document ID", sort=False):
            groups = list(rows.groupby("Sentence ID", sort=False))
            documents.append(
                (
                    str(doc_id),
                    [detokenize(group["Form"].astype(str)) for _, group in groups],
                    [group[column].fillna("").astype(str).str.lower().tolist() for _, group in groups],
                )
            )
    found: list[dict[str, Any]] = []
    for doc_id, texts, words in documents:
        for position, sentence_words in enumerate(words):
            hits = term_hits(sentence_words, by_term)
            if not hits:
                continue
            found.append(
                {
                    "Document ID": doc_id,
                    "Sentence ID": position + 1,
                    "Groups": ", ".join(dict.fromkeys(group_of[t] for t in hits)),
                    "Terms": ", ".join(hits),
                    "Before": " ".join(texts[max(0, position - context) : position]),
                    "Text": texts[position],
                    "After": " ".join(texts[position + 1 : position + 1 + context]),
                }
            )
            if len(found) >= limit:
                break
        if len(found) >= limit:
            break
    columns = ["Document ID", "Sentence ID", "Groups", "Terms", "Before", "Text", "After"]
    out = _with_documents(pd.DataFrame(found, columns=columns), corpus)
    session.record("passages", {"terms": lexicon, "context": context, "match": match}, started, len(out))
    # A table of sentences has no measure to draw; how many of them fall in
    # each year does answer something.
    return _prefer(
        out,
        kind="bar",
        x="Year",
        y="Passages",
        prepare={"count": {"by": "Year", "name": "Passages"}},
        question="How many passages fall in each year?",
    )


# --------------------------------------------------------- tool shortcuts --


def measures(corpus: Corpus, which: Sequence[str] = ("readability", "lexical_diversity")) -> pd.DataFrame:
    """Per-document measures from several tools, side by side (one row per document)."""
    merged: pd.DataFrame | None = None
    for tool in which:
        table = run(tool, corpus).table
        if "Document ID" not in table.columns:
            raise SuiteError(f"{tool} does not give one row per document, so it cannot be put side by side.")
        table = table.assign(**{"Document ID": table["Document ID"].astype(str)})
        if merged is None:
            merged = table
        else:
            new = [c for c in table.columns if c not in merged.columns]
            merged = merged.merge(table[["Document ID", *new]], on="Document ID", how="outer")
    if merged is None:
        raise SuiteError("Name at least one tool, for example which=('readability',).")
    return merged


def sentiment(corpus: Corpus, *, model: str = "vader", unit: str = "document") -> pd.DataFrame:
    """Tone per document (or per sentence with ``unit="sentence"``).

    ``model="vader"`` (a word-list method, fast, always available) or
    ``"distilbert-sst2"`` (a language model, positive/negative only, slower).
    """
    if unit not in ("document", "sentence"):
        raise SuiteError("unit must be 'document' or 'sentence'.")
    if model == "vader":
        result = run("sentiment_vader_anew", corpus, analysis="vader")
        # Compound is VADER's overall score (-1 to 1); Neg/Neu/Pos are its parts,
        # and a chart of the first of them answers a question nobody asked.
        return _prefer(
            result["vader_sentences" if unit == "sentence" else "vader"],
            kind="line",
            x="Date",
            y="Compound",
            agg="mean",
            question="How does the overall tone (VADER compound, -1 to 1) change over time?",
        )
    if model == "distilbert-sst2":
        result = run("sentiment_neural_bert", corpus, model=model)
        return result["sentiment_sentences" if unit == "sentence" else "sentiment_documents"]
    raise SuiteError("model must be 'vader' or 'distilbert-sst2'.")


def entities(corpus: Corpus) -> pd.DataFrame:
    """Named people, places and organisations, per document (the NER tool's timeline table)."""
    return run("ner", corpus)["entity_timeline"]


def topics(corpus: Corpus, *, k: int = 10, seed: int = 100) -> RunResult:
    """An LDA topic model with *k* topics; ``.tables`` holds topics, dominant topics and more."""
    return run("lda_gensim", corpus, topics=k, seed=seed)


def keyness(corpus_a: Corpus, corpus_b: Corpus, *, top_n: int = 200, field: str = "lemma") -> pd.DataFrame:
    """Words that set *corpus_a* apart from *corpus_b* (G2 log-likelihood, log ratio).

    Both must come from this notebook's documents; a document in both is refused.
    """
    session = current()
    a = [doc.source_id for doc in corpus_a.core.docs]
    b = [doc.source_id for doc in corpus_b.core.docs]
    if set(a) & set(b):
        raise SuiteError("A document is in both corpora; keyness needs two separate groups.")
    joined = Corpus(session, [*a, *b])
    table = joined.tokens()
    names = table.drop_duplicates("Document ID").set_index("Document ID")["Document"].astype(str)
    group_a = [names[str(i)] for i in range(1, len(a) + 1) if str(i) in names.index]
    pattern = "^(?:" + "|".join(re.escape(name) for name in group_a) + ")$"
    out = run("keyness", joined, group_pattern=pattern, top_n=top_n, field=field).table
    renamed = out.rename(
        columns=lambda c: str(c).replace("Group A (pattern docs)", "A").replace("Group B (other docs)", "B")
    )
    return _prefer(
        renamed,
        kind="bar",
        x="Word",
        y="G2 (log-likelihood)",
        group="Overrepresented in",
        top_n=30,
        question="Which words most set one group apart, and which group uses them more?",
    )


def similar(
    corpus: Corpus, query: str, *, unit: str = "sentence", model: str = "granite-embedding-english-r2"
) -> pd.DataFrame:
    """Sentences (or documents) closest in meaning to *query*, best first."""
    return run("doc_embeddings", corpus, query=query, unit=unit, model=model)["search_results"]


# ------------------------------------------------------------------ output --


def _as_frame(table: Any) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table
    if isinstance(table, pd.Series):
        return table.reset_index()
    if isinstance(table, RunResult):
        return table.table
    raise SuiteError(f"show() and save() take a table (a pandas DataFrame), not {type(table).__name__}.")


def _looks_ordered(values: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(values):
        return True
    return bool(values.astype(str).str.fullmatch(r"\d{4}(-\d{2}){0,2}").all())


#: Where a library function says which chart its table is for (``DataFrame.attrs``).
_CHART_ATTR = "nlpsuite_chart"


def _prefer(frame: pd.DataFrame, **chart: Any) -> pd.DataFrame:
    """Tell :func:`show` which chart *frame* is for.

    The generic recommender reads columns, not meaning: it charted VADER's
    ``Neg`` rather than its overall score, and summed sentence numbers for a
    table of passages. A library function knows what its table is for, so it
    says so. The advice holds only while the columns it names are still there;
    a table reshaped since falls back to the recommender.

    ``prepare`` describes a step between the table and the chart, so the
    "Copy matplotlib code" button can repeat it from the CSV:
    ``{"count": {"by": column, "name": label}}`` counts rows per value.
    """
    frame.attrs[_CHART_ATTR] = {"group": "", "agg": "", "top_n": 0, **chart}
    return frame


def _preferred_chart(frame: pd.DataFrame, kind: str | None) -> tuple[dict[str, Any], pd.DataFrame | None] | None:
    chart = frame.attrs.get(_CHART_ATTR)
    if not isinstance(chart, dict) or (kind is not None and chart["kind"] != kind):
        return None
    count = (chart.get("prepare") or {}).get("count")
    needed = [chart["x"], *([chart["group"]] if chart["group"] else []), *([] if count else [chart["y"]])]
    if not all(column in frame.columns for column in needed) or frame.empty:
        return None
    if count:
        data = frame.groupby(count["by"], as_index=False).size().rename(columns={"size": count["name"]})
        return dict(chart), data
    return dict(chart), None


def _chart_for(
    frame: pd.DataFrame, x: str | None, y: str | Sequence[str] | None, kind: str | None, group: str | None
) -> tuple[dict[str, Any] | None, pd.DataFrame | None, str]:
    if kind is not None and kind not in _DRAWN_KINDS:
        raise SuiteError(f"kind must be one of {', '.join(_DRAWN_KINDS)}, not {kind!r}.")
    if x is None and y is None:
        preferred = _preferred_chart(frame, kind)
        if preferred is not None:
            return preferred[0], preferred[1], ""
        from core.insight.recommend import recommend_charts

        for recommendation in recommend_charts(frame):
            spec = recommendation.spec
            if spec.kind in _DRAWN_KINDS and (kind is None or spec.kind == kind):
                chart = {
                    "kind": spec.kind,
                    "x": spec.x,
                    "y": spec.y,
                    "group": spec.group or "",
                    "agg": spec.agg or "",
                    "top_n": spec.top_n or 0,
                    "question": recommendation.question,
                }
                return chart, None, ""
        return None, None, "No chart is worth drawing from these columns on their own; pass x= and y=."
    ys = [y] if isinstance(y, str) else list(y or [])
    for column in [x, *ys, group]:
        if column is not None and column not in frame.columns:
            raise SuiteError(f"The table has no column {column!r}. Columns: {', '.join(map(str, frame.columns))}.")
    if x is None:
        raise SuiteError("Give x= as well as y=.")
    if not ys:
        return {"kind": kind or "histogram", "x": x, "y": x, "group": group or "", "agg": "", "top_n": 0}, None, ""
    chosen = kind or ("line" if _looks_ordered(frame[x]) else "bar")
    data: pd.DataFrame | None = None
    prepare: dict[str, Any] | None = None
    y_column, group_column = ys[0], group or ""
    if len(ys) > 1:
        if group:
            raise SuiteError("Give either several y columns or group=, not both.")
        data = frame.melt(id_vars=[x], value_vars=ys, var_name="Series", value_name="Value")
        y_column, group_column = "Value", "Series"
        # The CSV keeps the columns side by side; the copied matplotlib code
        # has to melt them the same way before it draws.
        prepare = {"melt": {"id": x, "values": ys}}
    source = frame if data is None else data
    repeated = bool(source.duplicated([x, *([group_column] if group_column else [])]).any())
    agg = "mean" if repeated and chosen in ("bar", "line") else ""
    note = f"Several rows share each {x}; the chart shows their mean." if agg else ""
    chart = {"kind": chosen, "x": x, "y": y_column, "group": group_column, "agg": agg, "top_n": 0}
    if prepare:
        chart["prepare"] = prepare
    return chart, data, note


def show(
    table: Any,
    *,
    x: str | None = None,
    y: str | Sequence[str] | None = None,
    kind: str | None = None,
    group: str | None = None,
    title: str | None = None,
    name: str | None = None,
) -> None:
    """Show a table under the cell, with a chart beside it.

    With no x/y the app picks the chart most worth drawing (the same advice
    the Explore page gives). Give ``x=`` and ``y=`` (one column or a list)
    to choose; ``kind=`` is bar, line, scatter, bubble, histogram, box or
    heatmap. When the notebook is saved as a run, the table is kept as a CSV.
    """
    session = current()
    frame = _as_frame(table)
    chart, data, chart_note = _chart_for(frame, x, y, kind, group)
    count = sum(isinstance(o, TableOutput) and not o.saved for o in session.outputs) + 1
    session.emit(
        TableOutput(session.name_for(name or title, f"table_{count}"), frame, title or "", chart, data, chart_note)
    )


def save(table: Any, name: str) -> None:
    """Keep a table as ``<name>.csv`` with the notebook's results (no chart, nothing shown)."""
    session = current()
    session.emit(TableOutput(session.name_for(name, "data"), _as_frame(table), saved=True))


def figure(fig: Any = None, name: str | None = None) -> None:
    """Keep a matplotlib figure (PNG and SVG) and show it under the cell.

    With no figure, the current one (``plt.gcf()``). The figure is closed
    afterwards, so a long notebook does not keep every figure in memory.
    """
    import matplotlib.pyplot as plt

    session = current()
    fig = fig if fig is not None else plt.gcf()
    if not hasattr(fig, "savefig"):
        raise SuiteError("figure() takes a matplotlib figure, such as the fig from plt.subplots().")
    png, svg = io.BytesIO(), io.BytesIO()
    fig.savefig(png, format="png", dpi=200, bbox_inches="tight")
    fig.savefig(svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    count = sum(isinstance(o, FigureOutput) for o in session.outputs) + 1
    session.emit(FigureOutput(session.name_for(name, f"figure_{count}"), png.getvalue(), svg.getvalue()))


def note(text: str) -> None:
    """Keep a sentence with the results (a caveat, a definition, what a number means)."""
    current().emit(NoteOutput(str(text)))


def tables() -> pd.DataFrame:
    """Result tables of earlier runs in this project: ``Run``, ``Tool``, ``Table``, ``Created``."""
    rows = [
        {
            "Run": item["job"],
            "Index": item["index"],
            "Tool": item["tool"],
            "Table": item["path"],
            "Created": item["created"],
        }
        for item in current().source.tables()
    ]
    return pd.DataFrame(rows, columns=["Run", "Index", "Tool", "Table", "Created"])


def load(run_or_tool: str, table: str | None = None) -> pd.DataFrame:
    """A table from an earlier run: by run id, or by tool name (its newest run).

    *table* is the file name (``"readability.csv"``); without it, the run's first table.
    """
    session = current()
    found = [
        item for item in session.source.tables() if item["job"].startswith(run_or_tool) or item["tool"] == run_or_tool
    ]
    if table is not None:
        found = [
            item for item in found if item["path"] in (table, f"{table}.csv") or item["path"].endswith("/" + table)
        ]
    if not found:
        raise SuiteError(
            f"No earlier table matches {run_or_tool!r}{f' / {table!r}' if table else ''}. nlp.tables() lists them."
        )
    return session.source.read_table(found[0]["job"], int(found[0]["index"]))
