"""WordNet aggregation (FR-4.4) — real synset/hypernym traversal.

Ports the traversal semantics of legacy
``semantic_aggregation_WordNet_util.py`` (first sense wins, BFS hypernym
climb to a top supersense or a user anchor, transitive hyponym expansion)
onto an injectable backend. The production backend reads Princeton WordNet
3.0 through NLTK (optional ``wordnet`` extra, lazy import); tests inject a
hand-built fake. No baked lemma map lives here: an unknown word is a
``Not found`` row, never a silent guess.

Intentional differences from the legacy (all defect-grade, none silent):

- the legacy wrote nothing when every word was unknown; this returns the
  ``Not found`` rows with a ``WORDNET_ALL_NOT_FOUND`` warning so the
  envelope shows what was attempted;
- unresolved anchors warn (``WORDNET_ANCHOR_UNRESOLVED``) instead of opening
  a Tkinter messagebox;
- non-English input is the caller's contract (the legacy showed a reminder
  popup); WordNet itself is English-only and unknown words surface as
  ``Not found``;
- relation visit order is normalized by synset name (``_children``): NLTK
  yields hypernyms/hyponyms nondeterministically across processes, so the
  legacy's run-to-run row shuffle is intentionally not reproduced.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from core.result import Diagnostic, Result

__all__ = [
    "NOUN_TOP_SYNSETS",
    "VERB_TOP_SYNSETS",
    "NltkWordNet",
    "WordNetBackend",
    "aggregate_up",
    "category_counts",
    "default_backend",
    "expand_down",
]

# Legacy top-level supersenses (the 25 noun / 15 verb lexicographer names the
# legacy climbs toward). A synset whose own lexname is in its set classifies
# in one step; only ``noun.Tops`` synsets (person, organism, …) climb.
NOUN_TOP_SYNSETS: frozenset[str] = frozenset(
    {
        "act",
        "animal",
        "artifact",
        "attribute",
        "body",
        "cognition",
        "communication",
        "event",
        "feeling",
        "food",
        "group",
        "location",
        "motive",
        "object",
        "person",
        "phenomenon",
        "plant",
        "possession",
        "process",
        "quantity",
        "relation",
        "shape",
        "state",
        "substance",
        "time",
    }
)

VERB_TOP_SYNSETS: frozenset[str] = frozenset(
    {
        "body",
        "change",
        "cognition",
        "communication",
        "competition",
        "consumption",
        "contact",
        "creation",
        "emotion",
        "motion",
        "perception",
        "possession",
        "social",
        "stative",
        "weather",
    }
)

_POS_VALUES: tuple[str, str] = ("NOUN", "VERB")

_UP_COLUMNS: tuple[str, str] = ("Word", "WordNet Category")
_DOWN_COLUMNS: tuple[str, str, str, str, str] = ("Term", "WordNet Category", "Definition", "Frequency", "Examples")


class LemmaView(Protocol):
    """One lemma of a synset."""

    def name(self) -> str: ...
    def count(self) -> int: ...


class SynsetView(Protocol):
    """One WordNet synset (NLTK object or test fake)."""

    def name(self) -> str: ...
    def lexname(self) -> str: ...
    def lemmas(self) -> list[LemmaView]: ...
    def hypernyms(self) -> list[SynsetView]: ...
    def hyponyms(self) -> list[SynsetView]: ...
    def definition(self) -> str: ...
    def examples(self) -> list[str]: ...


class WordNetBackend(Protocol):
    """Minimal WordNet surface the traversal needs."""

    def synsets(self, word: str, pos: str) -> list[SynsetView]: ...
    def synset(self, name: str) -> SynsetView: ...


def _tops(pos: str) -> frozenset[str]:
    return VERB_TOP_SYNSETS if pos == "VERB" else NOUN_TOP_SYNSETS


def _check_pos(pos: str) -> Diagnostic | None:
    if pos not in _POS_VALUES:
        return Diagnostic.error("WORDNET_BAD_POS", f"pos must be NOUN or VERB, got {pos!r}", pos=pos)
    return None


class _NltkLemma:
    """NLTK lemma behind the LemmaView protocol."""

    def __init__(self, lemma: Any) -> None:
        self._lemma = lemma

    def name(self) -> str:
        return str(self._lemma.name())

    def count(self) -> int:
        return int(self._lemma.count())


class _NltkSynset:
    """NLTK synset behind the SynsetView protocol."""

    def __init__(self, synset: Any) -> None:
        self._synset = synset

    def name(self) -> str:
        return str(self._synset.name())

    def lexname(self) -> str:
        return str(self._synset.lexname())

    def lemmas(self) -> list[LemmaView]:
        return [_NltkLemma(lemma) for lemma in self._synset.lemmas()]

    def hypernyms(self) -> list[SynsetView]:
        return [_NltkSynset(parent) for parent in self._synset.hypernyms()]

    def hyponyms(self) -> list[SynsetView]:
        return [_NltkSynset(child) for child in self._synset.hyponyms()]

    def definition(self) -> str:
        return str(self._synset.definition())

    def examples(self) -> list[str]:
        return [str(example) for example in self._synset.examples()]


class NltkWordNet:
    """Production backend: Princeton WordNet 3.0 via NLTK (lazy, offline)."""

    def __init__(self, wordnet: Any) -> None:
        self._wn = wordnet

    def synsets(self, word: str, pos: str) -> list[SynsetView]:
        native_pos: Any = self._wn.VERB if pos == "VERB" else self._wn.NOUN
        return [_NltkSynset(syn) for syn in self._wn.synsets(word, pos=native_pos)]

    def synset(self, name: str) -> SynsetView:
        return _NltkSynset(self._wn.synset(name))


def default_backend(data_dir: str | Path | None = None) -> Result[NltkWordNet]:
    """Build the NLTK backend, or fail with the exact install command.

    ``data_dir`` is an NLTK data root holding ``corpora/wordnet`` (used by
    tests); when omitted the ambient ``nltk.data.path`` applies
    (``NLTK_DATA`` env var, ``~/nltk_data``).
    """
    try:
        import nltk
    except ImportError:
        return Result.failure(
            Diagnostic.error(
                "WORDNET_BACKEND_MISSING",
                "NLTK is not installed; WordNet traversal needs the optional 'wordnet' extra",
                fix="python -m pip install nlp-suite-ng[wordnet]",
            )
        )
    try:
        if data_dir is not None:
            nltk.data.path.insert(0, str(data_dir))
        from nltk.corpus import wordnet as wn

        # A real query, not nltk.data.find: the loader also serves the corpus
        # from wordnet.zip, which find() does not resolve.
        wn.synsets("dog", pos=wn.NOUN)
    except LookupError:
        target = str(Path(data_dir) / "corpora" / "wordnet") if data_dir is not None else "<NLTK_DATA>/corpora/wordnet"
        return Result.failure(
            Diagnostic.error(
                "WORDNET_DATA_MISSING",
                f"WordNet 3.0 corpus not found at {target}",
                fix="python -m nltk.downloader -d <NLTK_DATA> wordnet",
            )
        )
    return Result.success(NltkWordNet(wn))


def _resolve_backend(backend: WordNetBackend | None) -> tuple[WordNetBackend | None, list[Diagnostic]]:
    if backend is not None:
        return backend, []
    resolved = default_backend()
    if resolved.value is None:
        return None, list(resolved.diagnostics)
    return resolved.unwrap(), list(resolved.diagnostics)


def _children(node: SynsetView, which: str) -> list[SynsetView]:
    """Related synsets in synset-name order.

    NLTK returns relation order nondeterministically (hash-ordered internals),
    so the legacy DOWN/UP row order varied run to run. Sorting by name keeps
    BFS structure (anchor first, then depth levels) while making every run
    byte-identical. This is an intentional, documented stability fix.
    """
    related = node.hypernyms() if which == "hypernyms" else node.hyponyms()
    return sorted(related, key=lambda syn: syn.name())


def _climb_to_top(synset: SynsetView, tops: frozenset[str]) -> tuple[str, list[str]]:
    """BFS hypernym climb to the synset's top supersense (legacy verbatim)."""
    visited: set[str] = set()
    queue: list[tuple[list[str], SynsetView]] = [([synset.name()], synset)]
    while queue:
        path, current = queue.pop(0)
        if current.name() in visited:
            continue
        visited.add(current.name())
        lexname = current.lexname().split(".")[-1]
        if lexname in tops:
            return lexname, path
        for parent in _children(current, "hypernyms"):
            queue.append(([*path, parent.name()], parent))
    return "unknown", [synset.name()]


def _resolve_anchors(anchor_terms: Sequence[str], pos: str, backend: WordNetBackend) -> tuple[set[str], list[str]]:
    """Resolve anchor terms to synset names (words: first sense; else explicit)."""
    import re

    anchors: set[str] = set()
    unresolved: list[str] = []
    for term in anchor_terms:
        cleaned = str(term).strip()
        if not cleaned:
            continue
        key = cleaned.replace(" ", "_")
        if re.match(r"^[\w\-]+\.[a-z]\.\d+$", key):
            try:
                anchors.add(backend.synset(key).name())
                continue
            except Exception:
                unresolved.append(cleaned)
                continue
        senses = backend.synsets(key, pos)
        if senses:
            anchors.add(senses[0].name())
        else:
            unresolved.append(cleaned)
    return anchors, unresolved


def _climb_to_target(synset: SynsetView, targets: set[str], tops: frozenset[str]) -> tuple[str, list[str]]:
    """BFS climb to the nearest anchor synset, else '(other) <supersense>'."""
    visited: set[str] = set()
    queue: list[tuple[list[str], SynsetView]] = [([synset.name()], synset)]
    while queue:
        path, current = queue.pop(0)
        if current.name() in visited:
            continue
        visited.add(current.name())
        if current.name() in targets:
            return current.lemmas()[0].name().replace("_", " "), path
        for parent in _children(current, "hypernyms"):
            queue.append(([*path, parent.name()], parent))
    category, path = _climb_to_top(synset, tops)
    return "(other) " + category, path


def aggregate_up(
    words: Sequence[str | float | None],
    *,
    pos: str = "NOUN",
    anchors: Sequence[str] = (),
    backend: WordNetBackend | None = None,
) -> Result[pd.DataFrame]:
    """Aggregate words UP to top supersenses (legacy ``aggregate_GoingUP``).

    Returns one row per input word in input order with columns ``Word``,
    ``WordNet Category``, ``Intermediate synset 1..N``. Unknown words become
    ``Not found`` rows; they are never dropped.
    """
    bad = _check_pos(pos)
    if bad is not None:
        return Result.failure(bad)
    resolved, diags = _resolve_backend(backend)
    if resolved is None:
        return Result.failure(*diags)
    tops = _tops(pos)

    anchor_names: set[str] = set()
    if anchors:
        anchor_names, unresolved = _resolve_anchors(anchors, pos, resolved)
        if unresolved:
            diags.append(
                Diagnostic.warning(
                    "WORDNET_ANCHOR_UNRESOLVED",
                    f"anchor synset(s) not found in WordNet for {pos} and ignored: {', '.join(unresolved)}",
                    anchors=list(unresolved),
                )
            )

    entries: list[tuple[str, str, list[str]]] = []
    for raw in words:
        if raw is None:
            continue
        if not isinstance(raw, str):
            # Non-string cells (NaN from CSV reads): skip like the legacy dropna.
            try:
                if bool(pd.isna(raw)):
                    continue
            except (TypeError, ValueError):
                pass
        cleaned = str(raw).strip().lower()
        senses = resolved.synsets(cleaned, pos) if cleaned else []
        if not senses:
            entries.append((cleaned, "Not found", []))
            continue
        first = senses[0]
        if anchor_names:
            category, path = _climb_to_target(first, anchor_names, tops)
        else:
            category, path = _climb_to_top(first, tops)
        entries.append((cleaned, category, path))

    width = max((len(path) for _, _, path in entries), default=0)
    columns = list(_UP_COLUMNS) + [f"Intermediate synset {i + 1}" for i in range(width)]
    frame = pd.DataFrame(
        [
            {
                "Word": word,
                "WordNet Category": category,
                **{f"Intermediate synset {i + 1}": step for i, step in enumerate(path)},
            }
            for word, category, path in entries
        ],
        columns=columns,
    )
    if not entries:
        diags.append(Diagnostic.warning("WORDNET_EMPTY_INPUT", "no words to aggregate"))
    elif all(category == "Not found" for _, category, _ in entries):
        diags.append(
            Diagnostic.warning(
                "WORDNET_ALL_NOT_FOUND", f"WordNet found none of the {len(entries)} word(s); all rows are 'Not found'"
            )
        )
    return Result.success(frame, *diags)


def expand_down(keyword: str, *, pos: str = "NOUN", backend: WordNetBackend | None = None) -> Result[pd.DataFrame]:
    """Expand a keyword DOWN to all transitive hyponyms (legacy ``disaggregate``).

    Columns ``Term, WordNet Category, Definition, Frequency, Examples`` in
    BFS order, starting with the anchor synset's own lemmas.
    """
    bad = _check_pos(pos)
    if bad is not None:
        return Result.failure(bad)
    resolved, diags = _resolve_backend(backend)
    if resolved is None:
        return Result.failure(*diags)
    senses = resolved.synsets(keyword, pos)
    if not senses:
        return Result.failure(
            Diagnostic.error(
                "WORDNET_KEYWORD_NOT_FOUND",
                f"WordNet has no {pos} synset for {keyword!r}; nothing to expand",
                keyword=keyword,
            )
        )
    anchor = senses[0]
    rows: list[dict[str, object]] = []
    queue: list[SynsetView] = [anchor]
    seen: set[str] = set()
    while queue:
        current = queue.pop(0)
        if current.name() in seen:
            continue
        seen.add(current.name())
        # Legacy computes one frequency per synset (the summed lemma counts)
        # and repeats it on every lemma row — not a per-lemma count.
        synset_frequency = sum(lemma.count() for lemma in current.lemmas())
        for lemma in current.lemmas():
            rows.append(
                {
                    "Term": lemma.name().replace("_", " "),
                    "WordNet Category": keyword,
                    "Definition": current.definition(),
                    "Frequency": int(synset_frequency),
                    "Examples": "; ".join(current.examples()),
                }
            )
        queue.extend(_children(current, "hyponyms"))
    frame = pd.DataFrame(rows, columns=list(_DOWN_COLUMNS))
    return Result.success(frame, *diags)


def category_counts(up: pd.DataFrame) -> Result[pd.DataFrame]:
    """Frequency table over an UP frame (legacy second CSV), highest first.

    ``Not found`` rows are excluded, matching the legacy frequency file.
    """
    if "WordNet Category" not in up.columns:
        return Result.failure(Diagnostic.error("WORDNET_BAD_FRAME", "UP frame needs a 'WordNet Category' column"))
    found = up.loc[up["WordNet Category"] != "Not found", "WordNet Category"].astype(str)
    tallies = found.value_counts()
    frame = pd.DataFrame(
        [{"WordNet Category": category, "Frequency": int(count)} for category, count in tallies.items()],
        columns=["WordNet Category", "Frequency"],
    )
    return Result.success(frame)
