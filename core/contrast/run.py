"""One comparison, every chosen method, and the notes a reader needs first (plan 3.4, 3.6).

:func:`contrast` is the single entry the executor's ``contrast`` adapter and
the scripting library call. It never parses: it receives the joined corpus's
token table, and runs the suite's own tools (readability, VADER, LDA ...)
through *run_tool*, the executor's adapter table, so a measure in a comparison
is the measure those tools publish.

Notes come first in the output (``contrast_notes.csv``) because they change
how everything after them reads: two sides from different collections differ
in setting as well as in whatever was meant to be compared, a side five times
the size of the other dominates keyness by chance, and an alignment that left
out half the groups answers a smaller question than it seems to.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from core.conll.schema import Col
from core.contrast.carryover import CarryoverSpec, carried_over
from core.contrast.focus import focus_rates
from core.contrast.measures import MEASURES, TONE, measure_table, measure_tests
from core.contrast.sides import Alignment, align, side_of
from core.contrast.topics import topic_prevalence
from core.contrast.words import distinctive_words
from core.io.document_fields import field_value
from core.io.reader import Corpus
from core.models.vector_cache import VectorCache
from core.result import Diagnostic, Result

__all__ = ["CONTRAST_METHODS", "PROJECT", "ContrastSpec", "RunTool", "contrast"]

#: The methods, in the order the Compare page lists them.
CONTRAST_METHODS = ("measures", "keyness", "focus", "topics", "tone", "meaning", "carryover")
#: Method that needs a sentence map, which remains outside this vertical slice.
_LATER = frozenset({"meaning"})
#: The detail naming the project a document came from (set by the runner).
PROJECT = "Project"
#: A side with more than this many times the other's words triggers the size note.
_SIZE_RATIO = 5.0
_TWO_SIDES = 2

RunTool = Callable[[str, dict[str, object]], Result[dict[str, pd.DataFrame]]]


@dataclass(frozen=True)
class ContrastSpec:
    """What to compare: the sides (in order), how to line them up, which methods."""

    sides: tuple[str, ...]
    alignment: str = "none"
    methods: tuple[str, ...] = ("measures", "keyness", "tone")
    focus: Mapping[str, Sequence[str]] = field(default_factory=dict)
    match: str = "lemma"
    topics: int = 8


def _words(text: str) -> int:
    """Words as the rest of the app counts them (the Corpus page, the Compare preview): split on spaces."""
    return len(text.split())


def _unit(alignment: str) -> str:
    """What an alignment's groups are called in a sentence: "Speaker groups", "decades"."""
    if alignment == "period":
        return "decades"
    return f"{alignment.removeprefix('field:')} groups"


def _sides_table(corpus: Corpus, spec: ContrastSpec, alignment: Alignment) -> pd.DataFrame:
    rows = []
    for side in spec.sides:
        docs = [doc for doc in corpus.docs if side_of(doc) == side]
        dates = sorted(doc.date for doc in docs if doc.date is not None)
        projects = sorted({field_value(doc.details, PROJECT) for doc in docs} - {""})
        rows.append(
            {
                "Side": side,
                "Project": ", ".join(projects),
                "Documents": len(docs),
                "Words": sum(_words(doc.text) for doc in docs),
                "First date": dates[0].isoformat() if dates else "",
                "Last date": dates[-1].isoformat() if dates else "",
                "Compared documents": sum(1 for doc in docs if doc.doc_id in alignment.groups),
            }
        )
    return pd.DataFrame(rows)


def _notes(sides: pd.DataFrame, spec: ContrastSpec, alignment: Alignment) -> list[str]:
    notes = []
    projects = {value for value in sides["Project"] if value}
    if len(projects) > 1:
        notes.append(
            "These sides come from different collections, so they differ in setting as well as in whatever you "
            "meant to compare (a written address and a spoken one, a speech and a debate). Differences in sentence "
            "length, pronouns or tone may be the setting."
        )
    words = dict(zip(sides["Side"], sides["Words"], strict=True))
    largest, smallest = max(words.values()), min(words.values())
    if smallest and largest / smallest > _SIZE_RATIO:
        big = max(words, key=lambda side: words[side])
        small = min(words, key=lambda side: words[side])
        notes.append(
            f"{big} has {largest / smallest:.0f} times the words of {small} ({largest:,} against {smallest:,}). "
            "Rates are used throughout, but distinctive words on the smaller side are dominated by chance."
        )
    coverage = alignment.coverage(spec.alignment)
    if coverage:
        notes.append(coverage)
    return notes


def contrast(
    table: pd.DataFrame | None,
    corpus: Corpus,
    spec: ContrastSpec,
    run_tool: RunTool,
    *,
    vector_cache: object | None = None,
) -> Result[dict[str, pd.DataFrame]]:
    """Every chosen method's tables, plus ``contrast_sides.csv`` and ``contrast_notes.csv``."""
    refusal = _refusal(corpus, spec)
    if refusal is not None:
        return Result.failure(refusal)
    alignment = align(corpus, spec.sides, spec.alignment)
    if not alignment.shared:
        return Result.failure(
            Diagnostic.error(
                "CONTRAST_NO_SHARED_GROUPS",
                f"no value of the alignment is on every side. {alignment.coverage(spec.alignment)}",
            )
        )
    diagnostics: list[Diagnostic] = []
    sides = _sides_table(corpus, spec, alignment)
    notes = _notes(sides, spec, alignment)
    out: dict[str, pd.DataFrame] = {
        "contrast_sides.csv": sides,
        "contrast_notes.csv": pd.DataFrame({"Note": notes}, columns=["Note"]),
    }
    diagnostics.extend(Diagnostic.info("CONTRAST_NOTE", note) for note in notes)

    def tool(name: str, params: dict[str, object] | None = None) -> dict[str, pd.DataFrame]:
        result = run_tool(name, params or {})
        diagnostics.extend(result.diagnostics)
        return result.value or {}

    tables, said = _methods(table, corpus, spec, alignment, tool, vector_cache)
    out.update(tables)
    diagnostics.extend(said)
    for method in spec.methods:
        if method in _LATER:
            diagnostics.append(
                Diagnostic.warning(
                    "CONTRAST_NOT_YET",
                    "The meaning map is not available in this build yet.",
                )
            )
    return Result.success(out, *diagnostics)


def _refusal(corpus: Corpus, spec: ContrastSpec) -> Diagnostic | None:
    """Why this comparison cannot run at all: a side with no documents, or a method nobody knows."""
    present = {side_of(doc) for doc in corpus.docs}
    missing = [side for side in spec.sides if side not in present]
    if len(spec.sides) < 2 or missing:  # noqa: PLR2004 - a comparison has two sides at least
        return Diagnostic.error(
            "CONTRAST_SIDES", f"every side needs documents; none for {', '.join(missing) or 'the second side'}"
        )
    unknown = [method for method in spec.methods if method not in CONTRAST_METHODS]
    if unknown:
        return Diagnostic.error("CONTRAST_METHOD", f"unknown method(s): {', '.join(unknown)}")
    return None


def _methods(  # noqa: PLR0912,PLR0913,PLR0917 -- dispatches independent selected methods
    table: pd.DataFrame | None,
    corpus: Corpus,
    spec: ContrastSpec,
    alignment: Alignment,
    tool: Callable[..., dict[str, pd.DataFrame]],
    vector_cache: object | None,
) -> tuple[dict[str, pd.DataFrame], list[Diagnostic]]:
    """The tables of each chosen method that runs in this build, and what the methods said."""
    out: dict[str, pd.DataFrame] = {}
    diagnostics: list[Diagnostic] = []
    if "measures" in spec.methods:
        frames: dict[str, pd.DataFrame] = {}
        for tool_name in dict.fromkeys(measure.tool for measure in MEASURES):
            frames.update(tool(tool_name))
        measured = measure_table(corpus, frames, MEASURES, alignment)
        out["contrast_measures.csv"] = measured
        out["contrast_measure_tests.csv"] = measure_tests(measured, MEASURES, spec.sides, _unit(spec.alignment))

    if "tone" in spec.methods:
        frames = tool(TONE.tool, {"analysis": "vader"})
        toned = measure_table(corpus, frames, (TONE,), alignment)
        out["contrast_tone.csv"] = toned
        out["contrast_tone_tests.csv"] = measure_tests(toned, (TONE,), spec.sides, _unit(spec.alignment))

    if "keyness" in spec.methods:
        if table is None:
            diagnostics.append(Diagnostic.warning("CONTRAST_NEEDS_PARSE", "distinctive words need the parse"))
        else:
            compared = _compared_rows(table, corpus, alignment)
            words = distinctive_words(compared, corpus, spec.sides, field=Col.LEMMA)
            diagnostics.extend(words.diagnostics)
            if words.value is not None:
                out["contrast_keyness.csv"] = words.unwrap()

    if "focus" in spec.methods:
        rates = focus_rates(table, corpus, spec.focus, alignment, match=spec.match)
        diagnostics.extend(rates.diagnostics)
        if rates.value is not None:
            per_document, per_group = rates.unwrap()
            out["contrast_focus_rates.csv"] = per_document
            out["contrast_focus_by_group.csv"] = per_group

    if "topics" in spec.methods:
        frames = tool("lda_gensim", {"topics": spec.topics})
        prevalence = topic_prevalence(frames, corpus, spec.sides)
        diagnostics.extend(prevalence.diagnostics)
        if prevalence.value is not None:
            words_table, shares = prevalence.unwrap()
            out["contrast_topics.csv"] = words_table
            out["contrast_topic_prevalence.csv"] = shares

    if "carryover" in spec.methods:
        if table is None:
            diagnostics.append(Diagnostic.warning("CONTRAST_NEEDS_PARSE", "carried-over passages need the parse"))
        elif len(spec.sides) != _TWO_SIDES:
            diagnostics.append(
                Diagnostic.warning("CONTRAST_CARRYOVER_SIDES", "carried-over passages compare two sides only")
            )
        else:
            cache = vector_cache if isinstance(vector_cache, VectorCache) else None
            found = carried_over(
                table,
                corpus,
                alignment,
                CarryoverSpec((spec.sides[0], spec.sides[1]), spec.focus, spec.match),
                cache=cache,
            )
            diagnostics.extend(found.diagnostics)
            if found.value is not None:
                out["contrast_carryover.csv"] = found.unwrap()
    return out, diagnostics


def _compared_rows(table: pd.DataFrame, corpus: Corpus, alignment: Alignment) -> pd.DataFrame:
    """The token rows of documents inside the alignment (keyness ignores the rest)."""
    kept = {doc.path.name for doc in corpus.docs if doc.doc_id in alignment.groups}
    kept |= {doc.name for doc in corpus.docs if doc.doc_id in alignment.groups}
    return table[table[Col.DOCUMENT.value].astype(str).isin(kept)]
