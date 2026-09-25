"""What word vectors say about meaning: axes, groups, neighbourhoods, change.

A list of nearest neighbours says which words sit close together; it does
not say what the closeness is *about*. The views built on this module each
answer a question a reader can check against the text:

* **An axis between two ideas** (``war`` ... ``peace``): the direction from
  one set of pole words to the other, and where every other word falls on
  it. On the State of the Union, *welfare* and *school* lean towards *poor*
  on the rich-poor axis, and *soldier* towards *war*. This is SemAxis (An,
  Kwak and Ahn, 2018), the difference-of-poles construction.
* **Groups of words that mean alike**: k-means over the vectors, each group
  named by its most central words, so the vocabulary reads as themes
  ("warfare, aggression, ally ..."; "payment, borrowing, finance ...").
* **A word's neighbourhood**: its nearest words and which of them are near
  each other, which shows when one word's company splits in two.
* **Change between periods** (contextual vectors only): a word's vector
  averaged within each decade, compared with the whole corpus's words.

**Centring.** Every view reads centred vectors: the mean vector of the
vocabulary is subtracted before the cosine. Contextual type vectors share a
large common direction (on BERT base over 87 speeches the average cosine
between two random words is 0.46), which squeezes every similarity towards
one number and tilts every axis towards frequent words; after centring the
average is 0.00 and the neighbours are unchanged in order (Mu and
Viswanath, 2018, "All-but-the-top"). The saved vectors are not changed; the
centring happens here, where it is read.

Nothing here trains or embeds: it reads a ``Word/Count/Vector`` table that a
run already published, so changing a pole word redraws in a moment.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
import re

import numpy as np
import pandas as pd

from core.result import Diagnostic, Result

__all__ = [
    "CHANGE_COLUMNS",
    "OVER_TIME_COLUMNS",
    "WORD_CLASS",
    "Group",
    "Space",
    "Spoke",
    "axis_scores",
    "candidates",
    "meaning_groups",
    "meaning_over_time",
    "neighbourhood",
    "parse_words",
    "read_space",
    "space_of",
    "word_classes",
]

#: The column naming each word's usual part of speech, when a run wrote one.
WORD_CLASS = "Word class"
_SPLIT = re.compile(r"[,;\s]+")
_FEWEST_WORDS = 2


@dataclass(frozen=True, slots=True)
class Space:
    """A vocabulary's centred, unit-length vectors.

    ``unit`` rows are the words' vectors with the vocabulary mean removed,
    scaled to length one, so ``unit @ unit[i]`` is the centred cosine of
    every word with word *i*. ``mean`` is what was removed, for placing
    other vectors (a word's vector in one decade) in the same space.
    """

    words: tuple[str, ...]
    counts: np.ndarray
    classes: tuple[str, ...]
    unit: np.ndarray
    mean: np.ndarray
    dropped: int = 0
    index: dict[str, int] = field(default_factory=dict)

    @property
    def has_classes(self) -> bool:
        return any(self.classes)

    def place(self, vector: np.ndarray) -> np.ndarray:
        """Another vector from the same model, centred and scaled like the rest."""
        centred: np.ndarray = np.asarray(vector, dtype=float) - self.mean
        norm = float(np.linalg.norm(centred))
        return np.asarray(centred / norm) if norm else centred


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(matrix / norms)


def _parse(raw: object) -> np.ndarray | None:
    try:
        vector = np.asarray([float(value) for value in str(raw).split(",")], dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.size == 0 or not np.isfinite(vector).all():
        return None
    return vector


def read_space(frame: pd.DataFrame) -> Result[Space]:
    """The centred space of a ``Word/Count/Vector`` table.

    Rows whose vector is malformed, non-finite or of another dimension are
    left out and counted (``dropped``). Words are compared case-insensitively,
    so a table holding "Freedom" and "freedom" is refused as ambiguous
    rather than answering for whichever came first.
    """
    words = frame["Word"].astype(str).str.strip()
    folded = words.str.casefold()
    duplicated = folded.duplicated(keep=False)
    if duplicated.any():
        examples = sorted(set(words.loc[duplicated].tolist()))[:5]
        return Result.failure(
            Diagnostic.error(
                "PANEL_AMBIGUOUS_VOCABULARY",
                f"case-insensitive duplicate words make vector lookup ambiguous: {examples}",
                words=examples,
            )
        )
    parsed = [_parse(raw) for raw in frame["Vector"]]
    sizes = pd.Series([vector.size for vector in parsed if vector is not None])
    if sizes.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_VECTORS", "vectors.csv has no usable numeric vectors"))
    width = int(sizes.mode().max())
    usable = [(i, vector) for i, vector in enumerate(parsed) if vector is not None and vector.size == width]
    keep = [i for i, _ in usable]
    if len(keep) < _FEWEST_WORDS:
        return Result.failure(Diagnostic.error("PANEL_NO_VECTORS", "vectors.csv needs at least two usable vectors"))
    matrix = np.vstack([vector for _, vector in usable])
    counts = pd.to_numeric(frame["Count"], errors="coerce").fillna(0).to_numpy()[keep].astype(int)
    classes: list[str] = []
    if WORD_CLASS in frame.columns:
        column = frame[WORD_CLASS].fillna("").astype(str).str.strip().str.lower()
        classes = [column.iloc[i] for i in keep]
    space = space_of([folded.iloc[i] for i in keep], counts, matrix, classes)
    return Result.success(replace(space, dropped=len(parsed) - len(keep)))


def space_of(words: Sequence[str], counts: Sequence[int], matrix: np.ndarray, classes: Sequence[str] = ()) -> Space:
    """The centred space of vectors already in hand (a model just built)."""
    matrix = np.asarray(matrix, dtype=float)
    mean = matrix.mean(axis=0)
    kept = tuple(str(word).casefold() for word in words)
    return Space(
        words=kept,
        counts=np.asarray(counts, dtype=int),
        classes=tuple(classes) if classes else tuple("" for _ in kept),
        unit=_unit_rows(matrix - mean),
        mean=mean,
        index={word: position for position, word in enumerate(kept)},
    )


def parse_words(text: object) -> list[str]:
    """Words typed into a box: split on commas, semicolons or spaces, case-folded, in order."""
    seen: list[str] = []
    for part in _SPLIT.split(str(text or "")):
        word = part.strip().casefold()
        if word and word not in seen:
            seen.append(word)
    return seen


def candidates(
    space: Space,
    *,
    word_class: str = "all",
    exclude: Iterable[str] = (),
    min_count: int = 1,
) -> np.ndarray:
    """Indexes of the words a view may show, most frequent first.

    ``word_class`` keeps one part of speech when the run recorded them (and
    is ignored when it did not: a figure of nothing would say less than a
    figure of every class). ``exclude`` names words to leave out, such as
    function words.
    """
    excluded = {str(word).casefold() for word in exclude}
    use_class = word_class != "all" and space.has_classes
    chosen = [
        i
        for i, word in enumerate(space.words)
        if word not in excluded
        and word.isalpha()
        and int(space.counts[i]) >= min_count
        and (not use_class or space.classes[i] == word_class)
    ]
    order = sorted(chosen, key=lambda i: (-int(space.counts[i]), space.words[i]))
    return np.asarray(order, dtype=int)


def axis_scores(
    space: Space, low: Sequence[str], high: Sequence[str]
) -> Result[tuple[np.ndarray, list[str], list[str]]]:
    """Every word's position on the axis from the *low* words to the *high* words.

    The axis is the difference of the two poles' mean centred vectors, at
    unit length; a word's score is its cosine with that direction, so
    positive leans towards *high* and negative towards *low*. Returns the
    scores and the pole words actually found; pole words not in the
    vocabulary are named in a warning, and a pole with none found is refused.
    """
    found_low = [word for word in low if word in space.index]
    found_high = [word for word in high if word in space.index]
    missing = [word for word in (*low, *high) if word not in space.index]
    diagnostics: list[Diagnostic] = []
    if missing:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_POLE_NOT_FOUND",
                f"not in this run's vocabulary, so left off the axis: {', '.join(missing)}",
                words=missing,
            )
        )
    for side, found, typed in (("first", found_low, low), ("second", found_high, high)):
        if not found:
            hint = ", ".join(space.words[i] for i in candidates(space)[:8])
            return Result.failure(
                Diagnostic.error(
                    "PANEL_POLE_EMPTY",
                    f"none of the {side} end's words ({', '.join(typed) or 'none given'}) are in this run's "
                    f"vocabulary; try frequent words such as {hint}",
                ),
                *diagnostics,
            )
    if set(found_low) == set(found_high):
        return Result.failure(
            Diagnostic.error("PANEL_POLE_SAME", "the two ends name the same words; an axis needs two different ideas")
        )
    low_mean = space.unit[[space.index[word] for word in found_low]].mean(axis=0)
    high_mean = space.unit[[space.index[word] for word in found_high]].mean(axis=0)
    direction = high_mean - low_mean
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        return Result.failure(
            Diagnostic.error("PANEL_POLE_SAME", "the two ends have the same vector; choose other words")
        )
    return Result.success((space.unit @ (direction / norm), found_low, found_high), *diagnostics)


@dataclass(frozen=True, slots=True)
class Group:
    """One group of words that mean alike, most central first."""

    name: str
    members: tuple[int, ...]
    closeness: tuple[float, ...]


def meaning_groups(space: Space, among: np.ndarray, *, groups: int, seed: int = 42, name_words: int = 3) -> list[Group]:
    """*among* split into *groups* by meaning, largest group first.

    Seeded k-means over the centred unit vectors (so the split follows
    cosine, not length). Each group is named by its *name_words* members
    closest to the group's centre; ``closeness`` is each member's cosine with
    that centre. Asks for fewer groups when there are too few distinct words
    to fill them.
    """
    from sklearn.cluster import KMeans

    points = space.unit[among]
    distinct = len(np.unique(points.round(6), axis=0))
    k = max(1, min(groups, distinct // 2))
    if k == 1:
        labels = np.zeros(len(among), dtype=int)
    else:
        labels = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(points)
    found: list[Group] = []
    for label in range(k):
        rows = np.flatnonzero(labels == label)
        if not rows.size:
            continue
        centre = points[rows].mean(axis=0)
        centre = centre / (float(np.linalg.norm(centre)) or 1.0)
        close = points[rows] @ centre
        order = np.argsort(-close, kind="stable")
        members = tuple(int(among[rows[i]]) for i in order)
        found.append(
            Group(
                name=", ".join(space.words[i] for i in members[:name_words]),
                members=members,
                closeness=tuple(float(close[i]) for i in order),
            )
        )
    found.sort(key=lambda group: (-len(group.members), group.name))
    return found


@dataclass(frozen=True, slots=True)
class Spoke:
    """One neighbour of the anchor word, placed for a radial drawing."""

    word: int
    similarity: float
    x: float
    y: float
    nearest: int
    nearest_similarity: float


def neighbourhood(space: Space, anchor: int, among: np.ndarray, *, size: int) -> list[Spoke]:
    """The anchor's *size* nearest words, laid out around it.

    Distance from the centre is honest: it grows as similarity to the anchor
    falls, on the range the drawn neighbours span. Angle is not a measured
    quantity: neighbours are ordered around the circle so that words near
    each other sit side by side (by the angle of a two-dimensional scaling of
    their own similarities), then spaced evenly so no two overlap. Each spoke
    names the other neighbour it is nearest to, which the drawing links.
    """
    pool = np.asarray([i for i in among.tolist() if i != anchor], dtype=int)
    if not pool.size:
        return []
    similarity = space.unit[pool] @ space.unit[anchor]
    top = pool[np.argsort(-similarity, kind="stable")[:size]]
    sims = space.unit[top] @ space.unit[anchor]
    if len(top) < 2:
        return [Spoke(int(top[0]), float(sims[0]), 0.5, 0.2, int(top[0]), float(sims[0]))] if len(top) else []
    mutual = space.unit[top] @ space.unit[top].T
    # Classical scaling of the neighbours' own distances, for their order round the circle.
    distance = np.clip(1.0 - mutual, 0.0, None) ** 2
    n = len(top)
    centring = np.eye(n) - np.ones((n, n)) / n
    inner = -0.5 * centring @ distance @ centring
    values, vectors = np.linalg.eigh(inner)
    lead = vectors[:, np.argsort(-values)[:2]] * np.sqrt(np.clip(np.sort(values)[::-1][:2], 0.0, None))
    # Fix each axis's sign so the same space always draws the same way round.
    for column in range(lead.shape[1]):
        if lead[int(np.argmax(np.abs(lead[:, column]))), column] < 0:
            lead[:, column] *= -1
    angles = np.arctan2(lead[:, 1], lead[:, 0])
    ordered = np.argsort(angles, kind="stable")
    high, low = float(sims.max()), float(sims.min())
    spread = (high - low) or 1.0
    np.fill_diagonal(mutual, -np.inf)
    spokes: list[Spoke] = []
    for slot, position in enumerate(ordered.tolist()):
        theta = 2 * np.pi * slot / n - np.pi / 2
        radius = 0.16 + 0.30 * (high - float(sims[position])) / spread
        nearest = int(np.argmax(mutual[position]))
        spokes.append(
            Spoke(
                word=int(top[position]),
                similarity=float(sims[position]),
                x=0.5 + radius * float(np.cos(theta)),
                y=0.5 + radius * float(np.sin(theta)),
                nearest=int(top[nearest]),
                nearest_similarity=float(mutual[position, nearest]),
            )
        )
    return spokes


def word_classes(frame: pd.DataFrame, column: str) -> dict[str, str]:
    """Each word type's most frequent word class in a parsed table (ties: alphabetical)."""
    from core.analysis.postags import word_class

    if column not in frame.columns or "POS" not in frame.columns:
        return {}
    words = frame[column].astype(str).str.strip().str.lower()
    classes = frame["POS"].map(word_class)
    counted = pd.DataFrame({"word": words, "class": classes})
    counted = counted[counted["class"] != ""]
    if counted.empty:
        return {}
    tally = counted.groupby(["word", "class"]).size().reset_index(name="n")
    tally = tally.sort_values(["word", "n", "class"], ascending=[True, False, True], kind="stable")
    first = tally.drop_duplicates("word")
    return dict(zip(first["word"], first["class"], strict=True))


#: Columns of the two change tables, in order.
OVER_TIME_COLUMNS = ["Word", "Period", "Uses", "Neighbor", "Similarity", "Rank"]
CHANGE_COLUMNS = ["Word", "Uses", "Periods", "From", "To", "Change", "Company before", "Company after"]


def meaning_over_time(
    space: Space,
    by_period: Sequence[tuple[str, str, int, Sequence[float]]],
    *,
    min_uses: int = 5,
    top_n: int = 6,
    exclude: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How each word's company changed from period to period.

    For every word used at least *min_uses* times in two or more periods,
    its vector in each period is compared with the whole corpus's words (the
    fixed reference: everyone else's overall vectors). ``over_time`` holds,
    for the union of its *top_n* nearest words in any period, their
    similarity in every period, so a figure can show a neighbour rising and
    falling rather than only where it made the list. ``change`` holds one row
    per word: one minus the cosine between its first and last period vectors
    (``Change``), with its nearest words then and now.

    Periods sort as labels ("1930s" < "1940s"; years likewise).
    """
    excluded = {str(word).casefold() for word in exclude}
    pool = candidates(space, exclude=excluded)
    if not pool.size:
        return pd.DataFrame(columns=OVER_TIME_COLUMNS), pd.DataFrame(columns=CHANGE_COLUMNS)
    reference = space.unit[pool]
    per_word: dict[str, list[tuple[str, int, np.ndarray]]] = {}
    for word, period, uses, vector in by_period:
        if uses >= min_uses and word not in excluded and word.isalpha() and word in space.index:
            per_word.setdefault(word, []).append((period, int(uses), space.place(np.asarray(vector, dtype=float))))
    timeline: list[dict[str, object]] = []
    changes: list[dict[str, object]] = []
    for word in sorted(per_word):
        rows = sorted(per_word[word], key=lambda row: row[0])
        if len(rows) < 2:
            continue
        own = int(np.flatnonzero(pool == space.index[word])[0]) if space.index[word] in pool else -1
        sims = np.vstack([reference @ vector for _, _, vector in rows])
        if own >= 0:
            sims[:, own] = -np.inf
        ranks = np.argsort(-sims, axis=1, kind="stable")
        union: list[int] = []
        for period_ranks in ranks:
            for column in period_ranks[:top_n].tolist():
                if column not in union:
                    union.append(column)
        place = np.argsort(ranks, axis=1)
        for p, (period, uses, _) in enumerate(rows):
            for column in union:
                timeline.append(
                    {
                        "Word": word,
                        "Period": period,
                        "Uses": uses,
                        "Neighbor": space.words[int(pool[column])],
                        "Similarity": round(float(sims[p, column]), 4),
                        "Rank": int(place[p, column]) + 1,
                    }
                )
        first, last = rows[0][2], rows[-1][2]
        changes.append(
            {
                "Word": word,
                "Uses": int(sum(uses for _, uses, _ in rows)),
                "Periods": len(rows),
                "From": rows[0][0],
                "To": rows[-1][0],
                "Change": round(float(1.0 - np.dot(first, last)), 4),
                "Company before": ", ".join(space.words[int(pool[c])] for c in ranks[0][:top_n]),
                "Company after": ", ".join(space.words[int(pool[c])] for c in ranks[-1][:top_n]),
            }
        )
    over_time = pd.DataFrame(timeline, columns=OVER_TIME_COLUMNS)
    change = pd.DataFrame(changes, columns=CHANGE_COLUMNS)
    if not change.empty:
        change = change.sort_values(["Change", "Word"], ascending=[False, True], kind="stable").reset_index(drop=True)
    return over_time, change
