"""Symbolic space + social actor typologies (FR-4.6).

Ports the two legacy ``GIS_symbolic_*_typology_util`` classifiers onto
injectable backends:

- space: a location noun -> one of ten symbolic space types
  (domestic_interior, field_labor, wild_forest, threshold_liminal,
  royal_court, sacred, market_public, water_passage, subterranean,
  tower_height);
- actor: a person noun -> one of twelve social types (kin_family,
  child_youth, laborer, professional, authority_official,
  military_police, clergy, merchant_trade, elite_landowner,
  crowd_collective, criminal_accused, generic_person).

Both resolve exact curated-lexicon hits first, then a WordNet hypernym
fallback over injectable graph primitives, then ``unclassified``. The
curated CSVs (``term,category``) are user-supplied files — no bytes
vendored. Category lists, anchor synsets, stopword/generic guards, and
the traversal rules below are the ported method, documented here:

- space: exact hit -> last-token hit (``throne room`` -> ``room``) ->
  generic-abstract guard -> WordNet climb over every noun sense in
  order, skipping instance synsets (named entities are geocoding
  business, not symbolic business), nearest anchor wins;
- actor: proper nouns (``PROPN``/``NNP``/``NNPS``) are never classified
  (WordNet knows no Harrys — name them in the CSV) -> exact hit ->
  generic-person bucket -> first-sense-only climb capped at depth 12.

Intentional differences from the legacy (all defect-grade, none silent):

- the space fallback climbs breadth-first from each sense instead of
  scanning ``hypernym_paths`` lists: same nearest-anchor contract
  without depending on NLTK's path order;
- unknown lexicon categories are skipped with a warning instead of
  silently becoming output buckets;
- no module-global lexicon cache: loaders return fresh mappings, so
  tests cannot pollute each other.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from core.result import Diagnostic, Result

__all__ = [
    "ACTOR_CATEGORIES",
    "SPACE_CATEGORIES",
    "UNCLASSIFIED",
    "NltkTypology",
    "TypologyBackend",
    "aggregate_actors",
    "aggregate_spaces",
    "category_counts",
    "classify_actor",
    "classify_space",
    "default_backend",
    "load_actor_lexicon",
    "load_space_lexicon",
]

UNCLASSIFIED = "unclassified"

SPACE_CATEGORIES: tuple[str, ...] = (
    "domestic_interior",
    "field_labor",
    "wild_forest",
    "threshold_liminal",
    "royal_court",
    "sacred",
    "market_public",
    "water_passage",
    "subterranean",
    "tower_height",
)

ACTOR_CATEGORIES: tuple[str, ...] = (
    "kin_family",
    "child_youth",
    "laborer",
    "professional",
    "authority_official",
    "military_police",
    "clergy",
    "merchant_trade",
    "elite_landowner",
    "crowd_collective",
    "criminal_accused",
    "generic_person",
)

#: WordNet anchor synsets per space category (the ported fallback map).
SPACE_ANCHORS: dict[str, tuple[str, ...]] = {
    "domestic_interior": ("room.n.01", "dwelling.n.01", "house.n.01", "housing.n.01"),
    "field_labor": ("tract.n.01", "farm.n.01", "field.n.01"),
    "wild_forest": ("forest.n.01", "wood.n.01", "geological_formation.n.01"),
    "royal_court": ("castle.n.02", "palace.n.01"),
    "sacred": ("place_of_worship.n.01", "religious_residence.n.01"),
    "market_public": ("mercantile_establishment.n.01", "municipality.n.01"),
    "water_passage": ("body_of_water.n.01", "way.n.06"),
    "subterranean": ("cave.n.01", "cellar.n.01"),
    "tower_height": ("tower.n.01",),
    # threshold_liminal has no clean WordNet anchor; the lexicon carries it.
}

#: WordNet anchor synsets per actor category (the ported fallback map).
ACTOR_ANCHORS: dict[str, tuple[str, ...]] = {
    "kin_family": ("relative.n.01", "parent.n.01", "spouse.n.01", "sibling.n.01"),
    "child_youth": ("child.n.01", "juvenile.n.01"),
    "laborer": ("worker.n.01", "laborer.n.01", "peasant.n.01", "servant.n.01", "slave.n.01"),
    "professional": ("professional.n.01", "health_professional.n.01", "educator.n.01", "intellectual.n.01"),
    "authority_official": (
        "official.n.01",
        "head_of_state.n.01",
        "administrator.n.01",
        "magistrate.n.01",
        "legislator.n.01",
    ),
    "military_police": ("serviceman.n.01", "lawman.n.01", "military_officer.n.01"),
    "clergy": ("clergyman.n.01", "religious_person.n.01"),
    "merchant_trade": ("merchant.n.01", "businessman.n.01", "trader.n.01"),
    "elite_landowner": ("aristocrat.n.01", "landowner.n.01", "capitalist.n.02"),
    "crowd_collective": ("gathering.n.01", "social_group.n.01"),
    "criminal_accused": ("criminal.n.01", "wrongdoer.n.01", "prisoner.n.01"),
}

#: Generic/abstract nouns that must never classify via the space fallback.
SPACE_STOPWORDS: frozenset[str] = frozenset(
    {
        "way",
        "place",
        "line",
        "part",
        "thing",
        "side",
        "area",
        "point",
        "bit",
        "lot",
        "kind",
        "sort",
        "number",
        "matter",
        "deal",
        "course",
        "rest",
        "world",
        "one",
        "end",
        "top",
        "bottom",
        "front",
        "back",
        "middle",
        "edge",
        "spot",
        "space",
        "position",
        "location",
        "distance",
        "direction",
        "moment",
        "time",
        "day",
    }
)

#: Person words too generic for a social type; they form their own bucket.
ACTOR_GENERIC: frozenset[str] = frozenset(
    {
        "man",
        "woman",
        "men",
        "women",
        "person",
        "people",
        "boy",
        "girl",
        "one",
        "someone",
        "somebody",
        "anyone",
        "anybody",
        "everyone",
        "everybody",
        "no one",
        "nobody",
        "they",
        "other",
        "others",
        "individual",
        "human",
        "being",
        "folk",
        "folks",
        "party",
        "figure",
        "fellow",
        "guy",
        "body",
        "soul",
        "creature",
    }
)

#: BFS depth cap for the actor climb (WordNet's person hierarchy is shallow).
_ACTOR_MAX_DEPTH = 12

_SPACE_COLUMNS: tuple[str, str] = ("Word", "Space Type")
_ACTOR_COLUMNS: tuple[str, str] = ("Actor", "Actor Type")


class TypologyBackend(Protocol):
    """WordNet graph primitives the fallback climbs need."""

    def noun_synsets(self, word: str) -> list[str]:
        """Noun synset names for a word, first sense first."""
        ...

    def hypernyms(self, synset: str) -> list[str]:
        """Direct hypernym names of a synset."""
        ...

    def is_instance(self, synset: str) -> bool:
        """Whether the synset is a named-entity instance rather than a class."""
        ...

    def resolve(self, name: str) -> str | None:
        """Canonical synset name for an anchor, or None when absent."""
        ...


class NltkTypology:
    """Production backend: NLTK WordNet 3.0 graph (lazy, offline)."""

    def __init__(self, wordnet: Any) -> None:
        self._wn = wordnet

    def noun_synsets(self, word: str) -> list[str]:
        try:
            return [str(syn.name()) for syn in self._wn.synsets(word, pos=self._wn.NOUN)]
        except Exception:
            return []

    def hypernyms(self, synset: str) -> list[str]:
        try:
            return [str(parent.name()) for parent in self._wn.synset(synset).hypernyms()]
        except Exception:
            return []

    def is_instance(self, synset: str) -> bool:
        try:
            return bool(self._wn.synset(synset).instance_hypernyms())
        except Exception:
            return False

    def resolve(self, name: str) -> str | None:
        try:
            return str(self._wn.synset(name).name())
        except Exception:
            return None


def default_backend(data_dir: str | Path | None = None) -> Result[NltkTypology]:
    """Build the NLTK typology backend, or fail with the exact install command."""
    try:
        import nltk
    except ImportError:
        return Result.failure(
            Diagnostic.error(
                "SYMBOLIC_BACKEND_MISSING",
                "NLTK is not installed; symbolic typologies need the optional 'wordnet' extra",
                fix="python -m pip install nlp-suite-ng[wordnet]",
            )
        )
    try:
        if data_dir is not None:
            nltk.data.path.insert(0, str(data_dir))
        from nltk.corpus import wordnet as wn

        wn.synsets("dog", pos=wn.NOUN)
    except LookupError:
        target = str(Path(data_dir) / "corpora" / "wordnet") if data_dir is not None else "<NLTK_DATA>/corpora/wordnet"
        return Result.failure(
            Diagnostic.error(
                "SYMBOLIC_DATA_MISSING",
                f"WordNet 3.0 corpus not found at {target}",
                fix="python -m nltk.downloader -d <NLTK_DATA> wordnet",
            )
        )
    return Result.success(NltkTypology(wn))


def _load_lexicon(path: str | Path, categories: tuple[str, ...], kind: str) -> Result[dict[str, str]]:
    import csv

    try:
        handle = open(path, encoding="utf-8-sig", newline="")  # noqa: SIM115
    except OSError as exc:
        return Result.failure(Diagnostic.error("SYMBOLIC_LEXICON_UNREADABLE", f"cannot read {kind} CSV {path}: {exc}"))
    table: dict[str, str] = {}
    skipped = 0
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "term" not in reader.fieldnames or "category" not in reader.fieldnames:
            return Result.failure(
                Diagnostic.error(
                    "SYMBOLIC_LEXICON_BAD_COLUMNS",
                    f"{kind} CSV needs 'term' and 'category', got {reader.fieldnames}",
                )
            )
        for row in reader:
            term = str(row["term"] or "").strip().lower()
            category = str(row["category"] or "").strip()
            if not term or not category:
                skipped += 1
                continue
            if category not in categories:
                skipped += 1
                continue
            table[term] = category
    diags = [Diagnostic.warning("SYMBOLIC_LEXICON_SKIPPED_ROWS", f"skipped {skipped} row(s)")] if skipped else []
    return Result.success(table, *diags)


def load_space_lexicon(path: str | Path) -> Result[dict[str, str]]:
    """Load a ``term,category`` space typology CSV."""
    return _load_lexicon(path, SPACE_CATEGORIES, "space typology")


def load_actor_lexicon(path: str | Path) -> Result[dict[str, str]]:
    """Load a ``term,category`` social actor typology CSV."""
    return _load_lexicon(path, ACTOR_CATEGORIES, "actor typology")


def _resolve_asset(name: str, override: object, legacy_lib: str) -> tuple[Path | None, list[Diagnostic]]:
    if override is not None:
        if not isinstance(override, (str, Path)):
            return None, [
                Diagnostic.error(
                    "SYMBOLIC_LEXICON_NOT_A_PATH", f"lexicon override must be a path, got {type(override).__name__}"
                )
            ]
        target = Path(override)
        if not target.is_file():
            return None, [
                Diagnostic.error("SYMBOLIC_LEXICON_NOT_FOUND", f"lexicon file not found at {target}", path=str(target))
            ]
        return target, []
    from core.assets.registry import default_registry

    resolved = default_registry().path(name)
    if resolved.value is None:
        hint = next((d for d in resolved.diagnostics if d.code == "ASSET_NOT_USABLE"), None)
        status = hint.context.get("status", "missing") if hint is not None else "missing"
        return None, [
            Diagnostic.error(
                f"SYMBOLIC_{name.upper().replace('-', '_')}_{status}",
                f"{name} asset is {status}; copy it from the legacy {legacy_lib} "
                "or install the documented source (see the asset registry)",
                asset=name,
            )
        ]
    return resolved.unwrap(), list(resolved.diagnostics)


def _anchor_map(backend: TypologyBackend, anchors: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """Resolve anchor synset names to categories (unresolvable anchors dropped)."""
    mapping: dict[str, str] = {}
    for category, names in anchors.items():
        for name in names:
            resolved = backend.resolve(name)
            if resolved is not None:
                mapping[resolved] = category
    return mapping


def _climb_senses(
    senses: Sequence[str], anchors: Mapping[str, str], backend: TypologyBackend, max_depth: int | None
) -> str | None:
    """BFS hypernym climb over senses in order; the nearest anchor wins."""
    for sense in senses:
        queue: list[tuple[str, int]] = [(sense, 0)]
        seen: set[str] = {sense}
        while queue:
            current, depth = queue.pop(0)
            if current in anchors:
                return anchors[current]
            if max_depth is not None and depth >= max_depth:
                continue
            for parent in backend.hypernyms(current):
                if parent not in seen:
                    seen.add(parent)
                    queue.append((parent, depth + 1))
    return None


def classify_space(
    word: str | None,
    *,
    lexicon: Mapping[str, str] | str | Path | None = None,
    backend: TypologyBackend | None = None,
    use_wordnet: bool = True,
) -> Result[str]:
    """Classify one location noun to a space type (or ``unclassified``)."""
    table, diags = _resolve_lexicon(lexicon, "space-typology", "lib/symbolic_space_typology.csv", load_space_lexicon)
    if table is None:
        return Result.failure(*diags)
    cleaned = str(word or "").strip().lower()
    if not cleaned:
        return Result.success(UNCLASSIFIED, *diags)
    if cleaned in table:
        return Result.success(table[cleaned], *diags)
    if " " in cleaned and cleaned.split()[-1] in table:
        return Result.success(table[cleaned.split()[-1]], *diags)
    if cleaned in SPACE_STOPWORDS:
        return Result.success(UNCLASSIFIED, *diags)
    if use_wordnet:
        resolved_backend, backend_diags = _resolve_backend(backend)
        diags.extend(backend_diags)
        if resolved_backend is None:
            return Result.failure(*diags)
        anchors = _anchor_map(resolved_backend, SPACE_ANCHORS)
        # Named-entity instances climb through real-world classes (London is
        # an instance of city); skipping them keeps geocodable places out of
        # the symbolic table, as in the legacy.
        senses = [s for s in resolved_backend.noun_synsets(cleaned) if not resolved_backend.is_instance(s)]
        hit = _climb_senses(senses, anchors, resolved_backend, None)
        if hit is not None:
            return Result.success(hit, *diags)
    return Result.success(UNCLASSIFIED, *diags)


def classify_actor(
    actor: str | None,
    *,
    pos: str = "",
    lexicon: Mapping[str, str] | str | Path | None = None,
    backend: TypologyBackend | None = None,
    use_wordnet: bool = True,
) -> Result[str]:
    """Classify one person noun to a social type (or ``unclassified``)."""
    if str(pos).strip().upper() in ("PROPN", "NNP", "NNPS"):
        return Result.success(UNCLASSIFIED)
    table, diags = _resolve_lexicon(lexicon, "actor-typology", "lib/social_actor_typology.csv", load_actor_lexicon)
    if table is None:
        return Result.failure(*diags)
    cleaned = str(actor or "").strip().lower()
    if not cleaned:
        return Result.success(UNCLASSIFIED, *diags)
    if cleaned in table:
        return Result.success(table[cleaned], *diags)
    if cleaned in ACTOR_GENERIC:
        return Result.success("generic_person", *diags)
    if use_wordnet:
        resolved_backend, backend_diags = _resolve_backend(backend)
        diags.extend(backend_diags)
        if resolved_backend is None:
            return Result.failure(*diags)
        anchors = _anchor_map(resolved_backend, ACTOR_ANCHORS)
        # First sense only: later senses of a common word wander far from the
        # reading a narrative intends ("minister" -> diplomat, "mother" -> abbess).
        hit = _climb_senses(resolved_backend.noun_synsets(cleaned)[:1], anchors, resolved_backend, _ACTOR_MAX_DEPTH)
        if hit is not None:
            return Result.success(hit, *diags)
    return Result.success(UNCLASSIFIED, *diags)


def _resolve_lexicon(
    lexicon: Mapping[str, str] | str | Path | None,
    asset: str,
    legacy_lib: str,
    loader: Callable[[str | Path], Result[dict[str, str]]],
) -> tuple[dict[str, str] | None, list[Diagnostic]]:
    if isinstance(lexicon, Mapping):
        return dict(lexicon), []
    target, diags = _resolve_asset(asset, lexicon, legacy_lib)
    if target is None:
        return None, diags
    loaded = loader(target)
    if loaded.value is None:
        return None, [*diags, *loaded.diagnostics]
    return loaded.unwrap(), [*diags, *loaded.diagnostics]


def _resolve_backend(backend: TypologyBackend | None) -> tuple[TypologyBackend | None, list[Diagnostic]]:
    if backend is not None:
        return backend, []
    resolved = default_backend()
    if resolved.value is None:
        return None, list(resolved.diagnostics)
    return resolved.unwrap(), list(resolved.diagnostics)


def _aggregate(
    words: Sequence[str | float | None],
    columns: tuple[str, str],
    classify: Callable[[str], Result[str]],
) -> tuple[pd.DataFrame, list[Diagnostic]]:
    rows: list[dict[str, object]] = []
    diags: list[Diagnostic] = []
    for raw in words:
        if raw is None:
            continue
        if not isinstance(raw, str):
            try:
                if bool(pd.isna(raw)):
                    continue
            except (TypeError, ValueError):
                pass
        cleaned = str(raw).strip()
        if not cleaned:
            continue
        result = classify(cleaned)
        if result.value is None:
            return pd.DataFrame(columns=list(columns)), list(result.diagnostics)
        diags.extend(result.diagnostics)
        rows.append({columns[0]: cleaned.lower(), columns[1]: result.unwrap()})
    return pd.DataFrame(rows, columns=list(columns)), diags


def aggregate_spaces(
    words: Sequence[str | float | None],
    *,
    lexicon: Mapping[str, str] | str | Path | None = None,
    backend: TypologyBackend | None = None,
    use_wordnet: bool = True,
) -> Result[pd.DataFrame]:
    """Classify location nouns to space types, in input order."""
    frame, diags = _aggregate(
        words,
        _SPACE_COLUMNS,
        lambda word: classify_space(word, lexicon=lexicon, backend=backend, use_wordnet=use_wordnet),
    )
    if frame.empty and not diags:
        diags.append(Diagnostic.warning("SYMBOLIC_EMPTY_INPUT", "no words to classify"))
    return Result.success(frame, *diags)


def aggregate_actors(
    words: Sequence[str | float | None],
    *,
    lexicon: Mapping[str, str] | str | Path | None = None,
    backend: TypologyBackend | None = None,
    use_wordnet: bool = True,
) -> Result[pd.DataFrame]:
    """Classify person nouns to social types, in input order."""
    frame, diags = _aggregate(
        words,
        _ACTOR_COLUMNS,
        lambda word: classify_actor(word, lexicon=lexicon, backend=backend, use_wordnet=use_wordnet),
    )
    if frame.empty and not diags:
        diags.append(Diagnostic.warning("SYMBOLIC_EMPTY_INPUT", "no words to classify"))
    return Result.success(frame, *diags)


def category_counts(frame: pd.DataFrame, column: str) -> Result[pd.DataFrame]:
    """Frequency table over a classification frame, highest first (``unclassified`` excluded)."""
    if column not in frame.columns:
        return Result.failure(Diagnostic.error("SYMBOLIC_BAD_FRAME", f"classification frame needs a {column!r} column"))
    found = frame.loc[frame[column] != UNCLASSIFIED, column].astype(str)
    tallies = found.value_counts()
    counted = pd.DataFrame(
        [{column: category, "Frequency": int(count)} for category, count in tallies.items()],
        columns=[column, "Frequency"],
    )
    return Result.success(counted)
