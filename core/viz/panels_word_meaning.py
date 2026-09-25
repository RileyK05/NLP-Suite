"""Figures that say what word vectors mean, not only what sits near what.

The neighbour list and the t-SNE map answer "which words are close?". A
reader of a corpus asks something else: close *in what way*, and what does
that tell me about these texts? Each figure here answers one such question
and leads back to the sentences:

* **Between two ideas** -- an axis from one set of words to another (bad ...
  good, war ... peace), with the words that lean furthest each way and any
  words the reader places on it.
* **A map with named axes** -- two such axes as a plane, in place of t-SNE's
  unlabelled dimensions: every position on it means something.
* **Groups that mean alike** -- the vocabulary in themes, each named by its
  most central words.
* **A word's neighbourhood** -- a word, its nearest words, and which of them
  are near each other; a word whose company splits shows it here.
* **Meaning over time** (BERT) -- a word's nearest words in each decade, and
  the words whose company changed most.
* **Two senses of a word** (word senses) -- the words that tell a word's two
  uses apart, with the sentences behind each.

All read centred vectors (:mod:`core.analysis.word_meaning` says why) from
a run's saved tables, so every control redraws without re-running a model.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
from typing import Any

import numpy as np
import pandas as pd

from core.analysis import word_meaning
from core.analysis.postags import WORD_CLASSES
from core.analysis.word_meaning import Space
from core.result import Diagnostic, Result
from core.viz.panel_helpers import FUNCTION_WORDS
from core.viz.panelspec import (
    Annotation,
    Evidence,
    PanelDefinition,
    PanelEdge,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
)

__all__ = ["WORD_MEANING_PANELS"]

WORD = "Word"
_TOOLS = {"word2vec_gensim": "Word2Vec", "word2vec_bert": "BERT"}
_LABEL_MAX = 60
_MAX_BARS = 60
_MIN_CONTEXT_SENTENCES = 2
_FEWEST_TO_GROUP = 4
#: The default word for the senses figure: well used, and the first of the clearest splits to show both sides.
_STEADY_USES = 20
_DEFAULT_TRIES = 25

# ------------------------------------------------------------------ params --

_CLASS_CHOICES = ("any", *WORD_CLASSES)


def _word_class(default: str) -> PanelParam:
    return PanelParam(
        name="word-class",
        type="choice",
        default=default,
        choices=_CLASS_CHOICES,
        label="Part of speech",
        help="Keep to nouns (themes, things), verbs (actions) or adjectives (qualities). Runs made before this "
        "release did not record parts of speech; then every word is used.",
    )


_HIDE = PanelParam(
    name="hide-function-words",
    type="bool",
    default=True,
    label="Hide function words",
    help="Leave out the, of, and ... : they carry grammar, not meaning.",
)
_MIN_USES = PanelParam(
    name="min-uses",
    type="int",
    default=5,
    minimum=1,
    maximum=1000,
    label="Fewest uses",
    help="Leave out words used fewer times than this: a word seen twice has a vector made from two sentences.",
)
_PLACE = PanelParam(
    name="place",
    type="str",
    default="",
    label="Words to place",
    help="Words you want to see on the figure whatever their position, separated by commas: tax, welfare, army.",
)


def _pole(name: str, label: str, default: str, help_text: str) -> PanelParam:
    return PanelParam(name=name, type="str", default=default, label=label, help=help_text)


_ONE_END = _pole(
    "one-end",
    "One end",
    "bad",
    "Words for one idea, separated by commas (war, conflict). The axis runs from their average to the other end's.",
)
_OTHER_END = _pole("other-end", "Other end", "good", "Words for the opposite idea (peace, cooperation).")

# ------------------------------------------------------------------ shared --

_SPACES: dict[tuple[Any, ...], Space] = {}


def _fingerprint(frame: pd.DataFrame) -> tuple[Any, ...]:
    """Enough of a vectors table to tell it from another: its size, ends and middle."""
    vectors = frame["Vector"].astype(str)
    picks = sorted({0, len(frame) // 2, len(frame) - 1}) if len(frame) else []
    return (
        len(frame),
        tuple(frame.columns),
        tuple(str(frame["Word"].iloc[i]) for i in picks),
        tuple(hash(vectors.iloc[i]) for i in picks),
        int(pd.to_numeric(frame["Count"], errors="coerce").fillna(0).sum()),
    )


def _space(frame: pd.DataFrame) -> Result[Space]:
    """The table's space, parsed once per table (a 4,500-word BERT table takes seconds)."""
    key = _fingerprint(frame)
    if key in _SPACES:
        return Result.success(_SPACES[key])
    parsed = word_meaning.read_space(frame)
    if parsed.value is not None:
        if len(_SPACES) >= 4:
            _SPACES.pop(next(iter(_SPACES)))
        _SPACES[key] = parsed.value
    return parsed


def _dropped_note(space: Space) -> list[Diagnostic]:
    if not space.dropped:
        return []
    return [
        Diagnostic.warning(
            "PANEL_BAD_VECTORS",
            f"{space.dropped} vector row(s) were malformed, non-finite, or had a different dimension and were omitted.",
            dropped=space.dropped,
        )
    ]


def _lemma(provenance: Provenance) -> bool:
    return str(provenance.settings.get("field", "lemma")) == "lemma"


def _pool(
    space: Space, params: Mapping[str, Any], *, word_class: str | None = None
) -> tuple[np.ndarray, list[Diagnostic]]:
    """The words a figure may draw under its settings, most frequent first."""
    chosen = str(params.get("word-class", "any")) if word_class is None else word_class
    exclude = FUNCTION_WORDS if bool(params.get("hide-function-words", True)) else frozenset()
    pool = word_meaning.candidates(
        space,
        word_class="all" if chosen == "any" else chosen,
        exclude=exclude,
        min_count=int(params.get("min-uses", 1)),
    )
    notes: list[Diagnostic] = []
    if chosen != "any" and not space.has_classes:
        notes.append(
            Diagnostic.info(
                "PANEL_NO_WORD_CLASS",
                "This run did not record parts of speech (runs before this release), so every word is used. "
                "Re-run the tool to keep to one part of speech.",
            )
        )
    return pool, notes


def _class_text(params: Mapping[str, Any], space: Space) -> str:
    chosen = str(params.get("word-class", "any"))
    return f" · word class: {chosen}" if chosen != "any" and space.has_classes else ""


def _words_text(words: Sequence[str]) -> str:
    return ", ".join(words) if len(words) <= 3 else ", ".join(words[:3]) + " ..."


def _clip(text: str, limit: int = _LABEL_MAX) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip(" ,") + "..."


def _phrase_evidence(word: str, describe: str, provenance: Provenance, *, count: int = 1) -> Evidence:
    return Evidence(
        scope="terms",
        filters=((WORD, word),),
        count=max(0, count),
        describe=describe,
        phrase=word,
        lemma=_lemma(provenance),
    )


# -------------------------------------------------------------------- axis --

_AXIS_NOTES = (
    "The axis runs from the average of one end's words to the average of the other's; each word's score is the "
    "cosine of its vector with that direction, after removing the vocabulary's average vector. Positive leans "
    "towards the other end, negative towards the first.",
    "A lean says the word is used in company more like one end's than the other's, not that the texts call it "
    "good or bad. Click a word to read how it is used.",
    "The pole words themselves are left off the ranking; an axis built from one word per end is noisier than "
    "one built from three or four.",
)


def _axis_or_fail(space: Space, low_text: object, high_text: object) -> Result[tuple[np.ndarray, list[str], list[str]]]:
    return word_meaning.axis_scores(space, word_meaning.parse_words(low_text), word_meaning.parse_words(high_text))


def _meaning_axis(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, panel: str
) -> Result[PreparedPanel]:
    parsed = _space(frame)
    if parsed.value is None:
        return Result.failure(*parsed.diagnostics)
    space = parsed.value
    scored = _axis_or_fail(space, params["one-end"], params["other-end"])
    if scored.value is None:
        return Result.failure(*scored.diagnostics)
    scores, low, high = scored.value
    diagnostics = [*_dropped_note(space), *scored.diagnostics]
    pool, notes = _pool(space, params)
    diagnostics += notes
    poles = set(low) | set(high)
    pool = np.asarray([i for i in pool.tolist() if space.words[i] not in poles], dtype=int)
    per_end = int(params["words-per-end"])
    if len(pool) < 2:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", "no words are left to place under these settings"), *diagnostics
        )
    order = pool[np.argsort(scores[pool], kind="stable")]
    lows = order[: min(per_end, len(order) // 2)]
    highs = order[::-1][: min(per_end, len(order) // 2)]
    placed_words = word_meaning.parse_words(params.get("place", ""))
    missing = [word for word in placed_words if word not in space.index]
    if missing:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_WORD_NOT_FOUND", f"not in this run's vocabulary: {', '.join(missing)}", words=missing
            )
        )
    shown_ends = {int(i) for i in (*lows.tolist(), *highs.tolist())}
    placed = [space.index[word] for word in placed_words if word in space.index and space.index[word] not in shown_ends]
    low_name, high_name = _words_text(low), _words_text(high)
    low_group, high_group, placed_group = (
        f"toward {_clip(low_name, 40)}",
        f"toward {_clip(high_name, 40)}",
        "your words",
    )
    rows = sorted({*shown_ends, *placed}, key=lambda i: (float(scores[i]), space.words[i]))
    marks: list[PanelMark] = []
    records: list[dict[str, Any]] = []
    for rank, i in enumerate(rows):
        word, score, uses = space.words[i], float(scores[i]), int(space.counts[i])
        group = placed_group if i in placed else (low_group if score < 0 else high_group)
        lean = low_name if score < 0 else high_name
        marks.append(
            PanelMark(
                key=f"axis:{word}",
                label=word,
                x=score,
                y=float(rank),
                group=group,
                evidence=_phrase_evidence(
                    word,
                    f"{word} ({uses:,} uses) leans toward {lean}: {score:+.3f} on the {low_name} to {high_name} axis",
                    provenance,
                    count=uses,
                ),
            )
        )
        records.append(
            {
                WORD: word,
                "Uses": uses,
                "Score": round(score, 4),
                "Leans toward": lean,
                word_meaning.WORD_CLASS: space.classes[i],
            }
        )
    groups = tuple(g for g in (low_group, high_group, placed_group) if any(m.group == g for m in marks))
    return Result.success(
        PreparedPanel(
            panel=panel,
            shape="ranked_bars",
            title=_clip(f"Words between {low_name} and {high_name}"),
            subtitle=f"the {len(lows)} words leaning furthest toward each end"
            + (f", and {len(placed)} you placed" if placed else "")
            + _class_text(params, space),
            marks=tuple(marks),
            x_label=f"toward {low_name}  <  0  >  toward {high_name}  (cosine with the axis)",
            y_label="",
            provenance=provenance,
            data=pd.DataFrame(records),
            groups=groups,
            notes=_AXIS_NOTES,
        ),
        *diagnostics,
    )


# ------------------------------------------------------------------- plane --

_PLANE_NOTES = (
    "Each axis is a direction between two sets of words, as in Between two ideas; a word's position is its cosine "
    "with each direction, after removing the vocabulary's average vector.",
    "Unlike a t-SNE map, both axes mean something and distances along them are measured. The two directions are "
    "not forced to be independent: related ideas (bad-good and past-future) can tilt the cloud.",
    "Labels go to the words furthest from the centre, and to any you placed.",
)


def _meaning_plane(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, panel: str
) -> Result[PreparedPanel]:
    parsed = _space(frame)
    if parsed.value is None:
        return Result.failure(*parsed.diagnostics)
    space = parsed.value
    across = _axis_or_fail(space, params["x-one-end"], params["x-other-end"])
    if across.value is None:
        return Result.failure(*across.diagnostics)
    up = _axis_or_fail(space, params["y-one-end"], params["y-other-end"])
    if up.value is None:
        return Result.failure(*up.diagnostics)
    xs, x_low, x_high = across.value
    ys, y_low, y_high = up.value
    diagnostics = [*_dropped_note(space), *across.diagnostics, *up.diagnostics]
    pool, notes = _pool(space, params)
    diagnostics += notes
    shown = pool[: int(params["words"])].tolist()
    placed_words = word_meaning.parse_words(params.get("place", ""))
    missing = [word for word in placed_words if word not in space.index]
    if missing:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_WORD_NOT_FOUND", f"not in this run's vocabulary: {', '.join(missing)}", words=missing
            )
        )
    placed = [space.index[word] for word in placed_words if word in space.index]
    shown += [i for i in placed if i not in shown]
    if len(shown) < 3:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", "fewer than three words to map under these settings"), *diagnostics
        )
    points = np.asarray(shown, dtype=int)
    spread_x, spread_y = float(np.std(xs[points])) or 1.0, float(np.std(ys[points])) or 1.0
    reach = np.maximum(np.abs(xs[points]) / spread_x, np.abs(ys[points]) / spread_y)
    by_reach = [int(points[j]) for j in np.argsort(-reach, kind="stable")]
    budget = int(params["label-count"])
    labelled = set(placed) | set(by_reach[: max(0, budget - len(placed))])
    x_name, x_other = _words_text(x_low), _words_text(x_high)
    y_name, y_other = _words_text(y_low), _words_text(y_high)
    marks: list[PanelMark] = []
    records: list[dict[str, Any]] = []
    for i in shown:
        word, uses = space.words[i], int(space.counts[i])
        x, y = float(xs[i]), float(ys[i])
        marks.append(
            PanelMark(
                key=f"plane:{word}",
                label=word,
                x=x,
                y=y,
                size=float(uses),
                labelled=i in labelled,
                group="your words" if i in placed else "(other words)" if placed else "",
                evidence=_phrase_evidence(
                    word,
                    f"{word} ({uses:,} uses): {x:+.3f} from {x_name} to {x_other}, {y:+.3f} from {y_name} to {y_other}",
                    provenance,
                    count=uses,
                ),
            )
        )
        records.append(
            {
                WORD: word,
                "Uses": uses,
                "Across": round(x, 4),
                "Up": round(y, 4),
                word_meaning.WORD_CLASS: space.classes[i],
            }
        )
    return Result.success(
        PreparedPanel(
            panel=panel,
            shape="scatter_labelled",
            title=_clip(f"{x_name} vs {x_other} (across), {y_name} vs {y_other} (up)"),
            subtitle=f"the {min(len(pool), int(params['words']))} most frequent words"
            + (f" and {len(placed)} you placed" if placed else "")
            + " · point size is frequency"
            + _class_text(params, space),
            marks=tuple(marks),
            x_label=f"toward {x_name}  <  0  >  toward {x_other}",
            y_label=f"toward {y_name}  <  0  >  toward {y_other}",
            provenance=provenance,
            data=pd.DataFrame(records),
            groups=("your words", "(other words)") if placed else (),
            annotations=(
                Annotation(kind="vline", value=0.0, label=""),
                Annotation(kind="hline", value=0.0, label=""),
            ),
            notes=_PLANE_NOTES,
            height=620,
        ),
        *diagnostics,
    )


# ------------------------------------------------------------------ groups --

_GROUP_NOTES = (
    "Groups are k-means clusters of the words' vectors (seeded, after removing the vocabulary's average vector); "
    "each is named by its three words closest to its centre, and bar length is that closeness.",
    "A group is what the model puts together, which is often a theme (war, allies, invasion) and sometimes a "
    "grammatical kind (verbs of doing). Keeping to nouns gives themes most clearly.",
    "Each group's words are its most central ones; the group's size counts every word in it.",
)


def _meaning_groups(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, panel: str
) -> Result[PreparedPanel]:
    parsed = _space(frame)
    if parsed.value is None:
        return Result.failure(*parsed.diagnostics)
    space = parsed.value
    diagnostics = _dropped_note(space)
    pool, notes = _pool(space, params)
    diagnostics += notes
    among = pool[: int(params["vocabulary"])]
    wanted = int(params["groups"])
    if len(among) < _FEWEST_TO_GROUP:
        return Result.failure(
            Diagnostic.error(
                "PANEL_NO_DATA",
                f"{len(among)} word(s) under these settings; grouping needs at least {_FEWEST_TO_GROUP}",
            ),
            *diagnostics,
        )
    if len(among) < 2 * wanted:
        diagnostics.append(
            Diagnostic.info(
                "PANEL_FEWER_GROUPS",
                f"{len(among)} word(s) under these settings make at most {len(among) // 2} groups of two or more",
            )
        )
        wanted = len(among) // 2
    found = word_meaning.meaning_groups(space, among, groups=wanted, seed=42)
    per_group = int(params["words-per-group"])
    fits = max(1, _MAX_BARS // max(1, len(found)))
    if per_group > fits:
        diagnostics.append(
            Diagnostic.info(
                "PANEL_WORDS_CAPPED",
                f"{len(found)} groups fit {fits} words each in one figure; ask for fewer groups to see more words.",
            )
        )
        per_group = fits
    names = [_clip(f"{n + 1}. {group.name} ({len(group.members)})", 48) for n, group in enumerate(found)]
    marks: list[PanelMark] = []
    records: list[dict[str, Any]] = []
    rank = 0
    for name, group in zip(names, found, strict=True):
        for i, close in zip(group.members[:per_group], group.closeness[:per_group], strict=False):
            word, uses = space.words[i], int(space.counts[i])
            marks.append(
                PanelMark(
                    key=f"group:{word}",
                    label=word,
                    x=close,
                    y=float(rank),
                    group=name,
                    evidence=_phrase_evidence(
                        word,
                        f"{word} ({uses:,} uses) is in group {name}; closeness to its centre {close:.2f}",
                        provenance,
                        count=uses,
                    ),
                )
            )
            rank += 1
        for i, close in zip(group.members, group.closeness, strict=True):
            records.append(
                {
                    "Group": name,
                    WORD: space.words[i],
                    "Closeness": round(close, 4),
                    "Uses": int(space.counts[i]),
                    "Group size": len(group.members),
                }
            )
    chosen = str(params.get("word-class", "noun"))
    kind = f"{chosen}s" if chosen != "any" and space.has_classes else "words"
    return Result.success(
        PreparedPanel(
            panel=panel,
            shape="ranked_bars",
            title=f"Groups of {kind} that mean alike",
            subtitle=f"{len(found)} groups among the {len(among)} most frequent {kind} · the {per_group} most central of each"
            + _class_text(params, space),
            marks=tuple(marks),
            x_label="Closeness to the group's centre (cosine)",
            y_label="",
            provenance=provenance,
            data=pd.DataFrame(records),
            groups=tuple(names),
            notes=_GROUP_NOTES,
        ),
        *diagnostics,
    )


# ----------------------------------------------------------------- network --

_NETWORK_NOTES = (
    "The chosen word is at the centre and its nearest words around it. Distance from the centre is measured: the "
    "nearer a word, the more similar its vector (cosine, after removing the vocabulary's average), on the range "
    "the drawn words span.",
    "Order around the circle is not measured: words near each other are put side by side, and each is linked to "
    "the other drawn word it is nearest to. Colour is a group of neighbours that are near one another; two groups "
    "can mean the word is used in two ways.",
)


def _default_word(space: Space, params: Mapping[str, Any]) -> int | None:
    pool, _ = _pool(space, params, word_class="noun" if space.has_classes else "any")
    if not pool.size:
        pool, _ = _pool(space, params)
    return int(pool[0]) if pool.size else None


def _word_network(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance, *, panel: str
) -> Result[PreparedPanel]:
    parsed = _space(frame)
    if parsed.value is None:
        return Result.failure(*parsed.diagnostics)
    space = parsed.value
    diagnostics = _dropped_note(space)
    typed = str(params.get("word", "")).strip().casefold()
    if typed and typed not in space.index:
        return Result.failure(
            Diagnostic.error("PANEL_QUERY_NOT_FOUND", f"{params['word']!r} is not in this run's vocabulary")
        )
    anchor = space.index[typed] if typed else _default_word(space, params)
    if anchor is None:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no words to draw under these settings"))
    pool, notes = _pool(space, params)
    diagnostics += notes
    spokes = word_meaning.neighbourhood(space, anchor, pool, size=int(params["neighbours"]))
    if len(spokes) < 2:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", "fewer than two neighbours under these settings"), *diagnostics
        )
    centre = space.words[anchor]
    if not typed:
        provenance = replace(provenance, params={**provenance.params, "word": centre})
    parts = max(1, min(3, len(spokes) // 4))
    members = np.asarray([spoke.word for spoke in spokes], dtype=int)
    clusters = word_meaning.meaning_groups(space, members, groups=parts, seed=42, name_words=2) if parts > 1 else []
    group_of = {member: _clip(f"near {group.name}", 40) for group in clusters for member in group.members}
    center_group = f"{centre} (chosen)"
    marks = [
        PanelMark(
            key=f"node:{centre}",
            label=centre,
            x=0.5,
            y=0.5,
            size=float(space.counts[anchor]),
            labelled=True,
            group=center_group,
            evidence=_phrase_evidence(
                centre,
                f"{centre}: the chosen word, {int(space.counts[anchor]):,} uses",
                provenance,
                count=int(space.counts[anchor]),
            ),
        )
    ]
    edges: list[PanelEdge] = []
    records: list[dict[str, Any]] = []
    seen_links: set[frozenset[str]] = set()
    for spoke in spokes:
        word, uses = space.words[spoke.word], int(space.counts[spoke.word])
        nearest = space.words[spoke.nearest]
        marks.append(
            PanelMark(
                key=f"node:{word}",
                label=word,
                x=spoke.x,
                y=spoke.y,
                size=float(uses),
                labelled=True,
                group=group_of.get(spoke.word, ""),
                evidence=_phrase_evidence(
                    word,
                    f"{word} ({uses:,} uses): similarity {spoke.similarity:.2f} to {centre}; nearest drawn word {nearest}",
                    provenance,
                    count=uses,
                ),
            )
        )
        edges.append(
            PanelEdge(
                key=f"{centre}::{word}",
                source=f"node:{centre}",
                target=f"node:{word}",
                weight=max(0.0, spoke.similarity),
                evidence=Evidence(
                    scope="rows",
                    filters=((WORD, word),),
                    count=1,
                    describe=f"{centre} and {word}: similarity {spoke.similarity:.2f}",
                ),
            )
        )
        link = frozenset((word, nearest))
        if nearest != word and link not in seen_links:
            seen_links.add(link)
            edges.append(
                PanelEdge(
                    key=f"{word}::{nearest}",
                    source=f"node:{word}",
                    target=f"node:{nearest}",
                    weight=max(0.0, spoke.nearest_similarity),
                    evidence=Evidence(
                        scope="rows",
                        filters=((WORD, word),),
                        count=1,
                        describe=f"{word} and {nearest}: similarity {spoke.nearest_similarity:.2f}",
                    ),
                )
            )
        records.append(
            {
                WORD: word,
                f"Similarity to {centre}": round(spoke.similarity, 4),
                "Uses": uses,
                "Group": group_of.get(spoke.word, ""),
                "Nearest drawn word": nearest,
                "Similarity to it": round(spoke.nearest_similarity, 4),
            }
        )
    groups = (center_group, *dict.fromkeys(g for g in group_of.values()))
    return Result.success(
        PreparedPanel(
            panel=panel,
            shape="network",
            title=f"The neighbourhood of {centre}",
            subtitle=f"its {len(spokes)} nearest words"
            + (" · defaulted to the most frequent noun" if not typed else "")
            + _class_text(params, space),
            marks=tuple(marks),
            x_label=f"Nearer the centre: more similar to {centre}. Each word links to the drawn word it is nearest to.",
            y_label="",
            provenance=provenance,
            data=pd.DataFrame(records),
            groups=groups,
            edges=tuple(edges),
            notes=_NETWORK_NOTES,
            height=620,
        ),
        *diagnostics,
    )


# ----------------------------------------------------------- over the tsne --


def _definitions(tool: str) -> tuple[PanelDefinition, ...]:
    model = _TOOLS[tool]

    def axis(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
        return _meaning_axis(frame, params, provenance, panel=f"{tool}_meaning_axis")

    def plane(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
        return _meaning_plane(frame, params, provenance, panel=f"{tool}_meaning_plane")

    def groups(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
        return _meaning_groups(frame, params, provenance, panel=f"{tool}_meaning_groups")

    def network(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
        return _word_network(frame, params, provenance, panel=f"{tool}_word_network")

    requires = (WORD, "Count", "Vector")
    return (
        PanelDefinition(
            name=f"{tool}_meaning_groups",
            title=f"Groups of words that mean alike ({model})",
            question="What themes does the vocabulary fall into?",
            tool=tool,
            shape="ranked_bars",
            summary="The most frequent words clustered by meaning, each group named by its most central words: "
            "the corpus's vocabulary read as themes.",
            requires=requires,
            params=(
                PanelParam(
                    name="groups",
                    type="int",
                    default=8,
                    minimum=2,
                    maximum=20,
                    label="Groups",
                    help="How many groups to split the words into.",
                ),
                PanelParam(
                    name="words-per-group",
                    type="int",
                    default=6,
                    minimum=2,
                    maximum=15,
                    label="Words per group",
                    help="How many of each group's most central words to show.",
                ),
                PanelParam(
                    name="vocabulary",
                    type="int",
                    default=1500,
                    minimum=50,
                    maximum=10000,
                    label="Words to group",
                    help="Group this many of the most frequent words.",
                ),
                _word_class("noun"),
                _MIN_USES,
                _HIDE,
            ),
            build=groups,
            notes=_GROUP_NOTES,
        ),
        PanelDefinition(
            name=f"{tool}_meaning_axis",
            title=f"Between two ideas ({model})",
            question="Which words lean toward one idea or its opposite?",
            tool=tool,
            shape="ranked_bars",
            summary="Name two ideas with a few words each (war ... peace); see the words that lean furthest toward "
            "each, and where words you choose fall between them.",
            requires=requires,
            params=(
                _ONE_END,
                _OTHER_END,
                _PLACE,
                PanelParam(
                    name="words-per-end",
                    type="int",
                    default=15,
                    minimum=3,
                    maximum=30,
                    label="Words per end",
                    help="How many of the words leaning furthest toward each end to show.",
                ),
                _word_class("any"),
                _MIN_USES,
                _HIDE,
            ),
            build=axis,
            notes=_AXIS_NOTES,
        ),
        PanelDefinition(
            name=f"{tool}_word_network",
            title=f"A word's neighbourhood ({model})",
            question="Which words surround this one, and do they form separate groups?",
            tool=tool,
            shape="network",
            summary="A word at the centre, its nearest words around it at a distance that shows how near, and links "
            "between the neighbours that are near each other.",
            requires=requires,
            params=(
                PanelParam(
                    name="word",
                    type="str",
                    default="",
                    label="Word",
                    help="The word at the centre. Left empty, the most frequent noun.",
                ),
                PanelParam(
                    name="neighbours",
                    type="int",
                    default=14,
                    minimum=4,
                    maximum=30,
                    label="Neighbours",
                    help="How many of its nearest words to draw.",
                ),
                _word_class("any"),
                _MIN_USES,
                _HIDE,
            ),
            build=network,
            notes=_NETWORK_NOTES,
        ),
        PanelDefinition(
            name=f"{tool}_meaning_plane",
            title=f"Map on two meaning axes ({model})",
            question="Where do words fall on two ideas at once?",
            tool=tool,
            shape="scatter_labelled",
            summary="Two axes you name (bad ... good across, past ... future up) as a map of the vocabulary, "
            "where every position means something.",
            requires=requires,
            params=(
                _pole("x-one-end", "Across: one end", "bad", "Words for the left of the map."),
                _pole("x-other-end", "Across: other end", "good", "Words for the right of the map."),
                _pole("y-one-end", "Up: one end", "past", "Words for the bottom of the map."),
                _pole("y-other-end", "Up: other end", "future", "Words for the top of the map."),
                _PLACE,
                PanelParam(
                    name="words",
                    type="int",
                    default=200,
                    minimum=20,
                    maximum=1000,
                    label="Words to map",
                    help="Map this many of the most frequent words.",
                ),
                PanelParam(
                    name="label-count",
                    type="int",
                    default=30,
                    minimum=0,
                    maximum=40,
                    label="Words to label",
                    help="Label this many words: those furthest from the centre, and any you placed.",
                ),
                _word_class("any"),
                _MIN_USES,
                _HIDE,
            ),
            build=plane,
            notes=_PLANE_NOTES,
        ),
    )


# ------------------------------------------------------- meaning over time --

_OVER_TIME_NOTES = (
    "Each column is one period: the chosen word's vector averaged over its uses in that period's documents, "
    "compared with every word's vector over the whole corpus. A cell is how near the row's word was to the chosen "
    "word then (cosine, after removing the vocabulary's average vector).",
    "BERT reads every period with the same model, so the periods share one space and can be compared directly.",
    "A period with few uses gives a noisier vector; the uses are in each column's label. Click a cell to read the "
    "chosen word in the corpus.",
)
_CHANGE_NOTES = (
    "Change is one minus the cosine between the word's vector in the first period it was used in and in the last "
    "(after removing the vocabulary's average vector). 0 is used in the same company; higher is more different.",
    "Words with few uses in a period have noisier vectors and can look changed by chance; raise Fewest uses to "
    "keep to words read in many sentences. Open a word in Meaning over time to see which neighbours came and went.",
)


def _over_time(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    table = frame.copy()
    table["Similarity"] = pd.to_numeric(table["Similarity"], errors="coerce")
    table["Rank"] = pd.to_numeric(table["Rank"], errors="coerce")
    table["Uses"] = pd.to_numeric(table["Uses"], errors="coerce").fillna(0).astype(int)
    table = table.dropna(subset=["Similarity", "Rank"])
    table["Word"] = table["Word"].astype(str)
    table["Period"] = table["Period"].astype(str)
    if table.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no usable rows in meaning_over_time.csv"))
    typed = str(params.get("word", "")).strip().casefold()
    words = set(table["Word"])
    if typed and typed not in words:
        return Result.failure(
            Diagnostic.error(
                "PANEL_QUERY_NOT_FOUND",
                f"{params['word']!r} was not used often enough in two periods to compare; choose a word from "
                "Words whose company changed most",
            )
        )
    word = typed or _most_changed(table)
    if not typed:
        provenance = replace(provenance, params={**provenance.params, "word": word})
    rows = table[table["Word"] == word]
    k = int(params["neighbours"])
    periods = sorted(rows["Period"].unique())
    uses = rows.drop_duplicates("Period").set_index("Period")["Uses"].to_dict()
    top = rows[rows["Rank"] <= k]
    first_seen: dict[str, tuple[int, float]] = {}
    for period_index, period in enumerate(periods):
        for _, row in top[top["Period"] == period].sort_values("Rank").iterrows():
            first_seen.setdefault(str(row["Neighbor"]), (period_index, float(row["Rank"])))
    neighbours = sorted(first_seen, key=lambda n: (first_seen[n], n))
    columns = [f"{period} ({int(uses.get(period, 0))} uses)" for period in periods]
    marks: list[PanelMark] = []
    lookup = {(str(r["Period"]), str(r["Neighbor"])): r for _, r in rows.iterrows()}
    for y, neighbour in enumerate(neighbours):
        for x, period in enumerate(periods):
            row = lookup.get((period, neighbour))
            if row is None:
                continue
            similarity, rank = float(row["Similarity"]), int(row["Rank"])
            marks.append(
                PanelMark(
                    key=f"{period}:{neighbour}",
                    label=neighbour,
                    x=float(x),
                    y=float(y),
                    value=round(similarity, 3),
                    evidence=Evidence(
                        scope="terms",
                        filters=(("Period", period), ("Neighbor", neighbour)),
                        count=int(uses.get(period, 0)),
                        describe=f"In the {period}, {neighbour} was number {rank} among the words nearest {word} "
                        f"(similarity {similarity:.2f}, {int(uses.get(period, 0))} uses)",
                        phrase=word,
                        lemma=_lemma(provenance),
                    ),
                )
            )
    return Result.success(
        PreparedPanel(
            panel="word2vec_bert_meaning_over_time",
            shape="heatmap",
            title=f"The company {word} kept, period by period",
            subtitle=f"its {k} nearest words in each of {len(periods)} periods"
            + (" · defaulted to the most changed frequent word" if not typed else ""),
            marks=tuple(marks),
            x_label="Period",
            y_label="Nearby word (in the order it first entered the top)",
            provenance=provenance,
            data=rows[rows["Neighbor"].isin(neighbours)].reset_index(drop=True),
            x_categories=tuple(columns),
            y_categories=tuple(neighbours),
            color_scale="sequential",
            value_label="Similarity",
            notes=_OVER_TIME_NOTES,
        )
    )


def _most_changed(table: pd.DataFrame) -> str:
    """The frequent word whose nearest words turned over most from first to last period."""
    best: tuple[float, int, str] | None = None
    totals = table.drop_duplicates(["Word", "Period"]).groupby("Word")["Uses"].sum()
    floor = float(totals.median()) if len(totals) else 0.0
    for word, rows in table.groupby("Word", sort=True):
        if float(totals.get(word, 0)) < floor:
            continue
        periods = sorted(rows["Period"].unique())
        before = set(rows[(rows["Period"] == periods[0]) & (rows["Rank"] <= 6)]["Neighbor"])
        after = set(rows[(rows["Period"] == periods[-1]) & (rows["Rank"] <= 6)]["Neighbor"])
        union = before | after
        turnover = 1.0 - (len(before & after) / len(union) if union else 1.0)
        key = (turnover, int(totals.get(word, 0)), str(word))
        if best is None or key[:2] > best[:2]:
            best = key
    return best[2] if best else str(table["Word"].iloc[0])


def _change(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    table = frame.copy()
    table["Change"] = pd.to_numeric(table["Change"], errors="coerce")
    table["Uses"] = pd.to_numeric(table["Uses"], errors="coerce").fillna(0).astype(int)
    table = table.dropna(subset=["Change"])
    table = table[table["Uses"] >= int(params["min-uses"])]
    chosen = str(params.get("word-class", "any"))
    classed = (
        chosen != "any" and word_meaning.WORD_CLASS in table.columns and table[word_meaning.WORD_CLASS].notna().any()
    )
    if classed:
        table = table[table[word_meaning.WORD_CLASS].fillna("").astype(str) == chosen]
    if table.empty:
        return Result.failure(
            Diagnostic.error(
                "PANEL_NO_DATA",
                f"no word has {int(params['min-uses'])} or more uses across two periods; lower Fewest uses",
            )
        )
    drawn = table.sort_values(["Change", "Word"], ascending=[False, True], kind="stable").head(int(params["top-n"]))
    marks: list[PanelMark] = []
    for rank, (_, row) in enumerate(drawn.iterrows()):
        word, change = str(row["Word"]), float(row["Change"])
        span = f"{row['From']} to {row['To']}"
        marks.append(
            PanelMark(
                key=f"change:{word}",
                label=_clip(f"{word} ({span})"),
                x=change,
                y=float(rank),
                evidence=Evidence(
                    scope="terms",
                    filters=(("Word", word),),
                    count=int(row["Uses"]),
                    describe=f"{word}, {row['From']}: {row['Company before']}. {row['To']}: {row['Company after']}",
                    phrase=word,
                    lemma=_lemma(provenance),
                ),
            )
        )
    return Result.success(
        PreparedPanel(
            panel="word2vec_bert_meaning_change",
            shape="ranked_bars",
            title="Words whose company changed most",
            subtitle=f"the {len(drawn)} words most different between their first and last period · {int(params['min-uses'])}+ uses"
            + (f" · word class: {chosen}" if classed else ""),
            marks=tuple(marks),
            x_label="Change: 1 - cosine between the first and last period's vectors",
            y_label="",
            provenance=provenance,
            data=drawn.reset_index(drop=True),
            notes=_CHANGE_NOTES,
        )
    )


WORD2VEC_BERT_MEANING_OVER_TIME = PanelDefinition(
    name="word2vec_bert_meaning_over_time",
    title="Meaning over time (BERT)",
    question="What company did this word keep in each period?",
    tool="word2vec_bert",
    shape="heatmap",
    summary="A word's nearest words in each decade (or year) of a dated corpus, and how near each stayed in the "
    "other periods: new company arriving and old company leaving.",
    requires=("Word", "Period", "Uses", "Neighbor", "Similarity", "Rank"),
    params=(
        PanelParam(
            name="word",
            type="str",
            default="",
            label="Word",
            help="The word to follow. Left empty, the frequent word whose company changed most.",
        ),
        PanelParam(
            name="neighbours",
            type="int",
            default=5,
            minimum=2,
            maximum=6,
            label="Nearest words per period",
            help="How many of the word's nearest words to take from each period.",
        ),
    ),
    build=_over_time,
    notes=_OVER_TIME_NOTES,
)

WORD2VEC_BERT_MEANING_CHANGE = PanelDefinition(
    name="word2vec_bert_meaning_change",
    title="Words whose company changed most (BERT)",
    question="Which words are used differently now than at the start?",
    tool="word2vec_bert",
    shape="ranked_bars",
    summary="Words ranked by how different their company was in the last period they were used in against the "
    "first, with the words near them then and now.",
    requires=("Word", "Uses", "From", "To", "Change", "Company before", "Company after"),
    params=(
        PanelParam(
            name="top-n",
            type="int",
            default=20,
            minimum=5,
            maximum=40,
            label="Words to show",
            help="How many words to rank.",
        ),
        PanelParam(
            name="min-uses",
            type="int",
            default=30,
            minimum=2,
            maximum=100000,
            label="Fewest uses",
            help="Leave out words used fewer times than this across the periods compared: few uses, noisy vectors.",
        ),
        _word_class("noun"),
    ),
    build=_change,
    notes=_CHANGE_NOTES,
)


# -------------------------------------------------------------- two senses --

_SENSE_NOTES = (
    "Word senses splits a word's uses in two when their BERT vectors fall clearly into two groups (a silhouette of "
    "0.2 or more, the smaller group holding at least 15% of the uses). Sense 1 is the more frequent.",
    "A context word's score is the log-odds of its appearing in a sentence of one sense rather than the other; "
    "words in the sentences of both senses score near zero and are not shown.",
    "The split is the model's; read the sentences before calling it two meanings. It can also be two topics, or "
    "two grammatical uses of one meaning.",
)


def _tokens(sentence: str, lemma: str) -> set[str]:
    stem = lemma[: max(3, len(lemma) - 2)]
    return {
        token
        for token in (part.strip(".,;:!?\"'()[]").lower() for part in str(sentence).split())
        if token.isalpha() and token not in FUNCTION_WORDS and not token.startswith(stem) and len(token) > 2
    }


def _contexts(rows: pd.DataFrame, lemma: str) -> dict[int, list[tuple[str, float]]]:
    """Each sense's most distinctive context words, by smoothed log-odds, best first."""
    sets = {sense: [_tokens(text, lemma) for text in group["Sentence"]] for sense, group in rows.groupby("Sense")}
    if len(sets) < 2:
        return {}
    one, two = sets.get(1, []), sets.get(2, [])
    in_one, in_two = Counter(t for s in one for t in s), Counter(t for s in two for t in s)
    out: dict[int, list[tuple[str, float]]] = {1: [], 2: []}
    for token in set(in_one) | set(in_two):
        a, b = in_one[token], in_two[token]
        if a + b < _MIN_CONTEXT_SENTENCES:
            continue
        score = math.log((b + 0.5) / (len(two) - b + 0.5)) - math.log((a + 0.5) / (len(one) - a + 0.5))
        if score > 0 and b >= _MIN_CONTEXT_SENTENCES:
            out[2].append((token, score))
        elif score < 0 and a >= _MIN_CONTEXT_SENTENCES:
            out[1].append((token, score))
    out[1].sort(key=lambda pair: (pair[1], pair[0]))
    out[2].sort(key=lambda pair: (-pair[1], pair[0]))
    return out


def _uses(frame: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    """The uses of words that split, under the figure's part-of-speech setting."""
    table = frame.copy()
    table["Sense"] = pd.to_numeric(table["Sense"], errors="coerce")
    table["Separation"] = pd.to_numeric(table["Separation"], errors="coerce")
    table = table.dropna(subset=["Sense", "Separation"])
    table["Sense"] = table["Sense"].astype(int)
    table["Lemma"] = table["Lemma"].astype(str)
    table = table[table["Separation"] > 0]
    chosen = str(params.get("word-class", "noun"))
    if chosen != "any" and _classed(table):
        table = table[table[word_meaning.WORD_CLASS].fillna("").astype(str) == chosen]
    return table


def _classed(table: pd.DataFrame) -> bool:
    return (
        word_meaning.WORD_CLASS in table.columns and table[word_meaning.WORD_CLASS].fillna("").astype(str).ne("").any()
    )


def _sense_class_text(params: Mapping[str, Any], frame: pd.DataFrame) -> str:
    chosen = str(params.get("word-class", "noun"))
    return f" · word class: {chosen}" if chosen != "any" and _classed(frame) else ""


def _split_words(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    table = _uses(frame, params)
    counts = table.groupby("Lemma").size()
    table = table[table["Lemma"].map(counts) >= int(params["min-uses"])]
    if table.empty:
        return Result.failure(
            Diagnostic.error(
                "PANEL_NO_DATA",
                f"no word with two senses has {int(params['min-uses'])} or more uses; lower Fewest uses",
            )
        )
    ranked = table.drop_duplicates("Lemma").sort_values(["Separation", "Lemma"], ascending=[False, True], kind="stable")
    marks: list[PanelMark] = []
    records: list[dict[str, Any]] = []
    for rank, (_, head) in enumerate(ranked.head(int(params["top-n"])).iterrows()):
        lemma, separation = str(head["Lemma"]), float(head["Separation"])
        rows = table[table["Lemma"] == lemma]
        contexts = _contexts(rows, lemma)
        first = ", ".join(word for word, _ in contexts.get(1, [])[:2]) or "-"
        second = ", ".join(word for word, _ in contexts.get(2, [])[:2]) or "-"
        share = float((rows["Sense"] == 1).mean())
        marks.append(
            PanelMark(
                key=f"split:{lemma}",
                label=_clip(f"{lemma}: {first} | {second}"),
                x=separation,
                y=float(rank),
                evidence=Evidence(
                    scope="rows",
                    filters=(("Lemma", lemma),),
                    count=len(rows),
                    describe=f"{lemma}: {len(rows)} uses, sense 1 ({share:.0%}) near {first}; sense 2 near {second}; "
                    f"separation {separation:.2f}",
                ),
            )
        )
        records.append(
            {
                "Lemma": lemma,
                "Uses": len(rows),
                "Sense 1 share": round(share, 3),
                "Separation": round(separation, 4),
                "Sense 1 context": ", ".join(word for word, _ in contexts.get(1, [])[:6]),
                "Sense 2 context": ", ".join(word for word, _ in contexts.get(2, [])[:6]),
            }
        )
    return Result.success(
        PreparedPanel(
            panel="word_sense_induction_split_words",
            shape="ranked_bars",
            title="Words used in two senses",
            subtitle=f"the {len(marks)} clearest splits · each label: the word, then context words of sense 1 | sense 2"
            + _sense_class_text(params, frame),
            marks=tuple(marks),
            x_label="How clearly the two senses separate (silhouette)",
            y_label="",
            provenance=provenance,
            data=pd.DataFrame(records),
            notes=_SENSE_NOTES,
        )
    )


def _clearest(table: pd.DataFrame) -> str:
    """The clearest split of a well-used word that has words to show on both sides.

    Well used is twenty uses (or the most any word has): below that the
    clearest splits are four- and six-sentence words whose "senses" are one
    odd sentence each.
    """
    counts = table.groupby("Lemma").size()
    steady = table[table["Lemma"].map(counts) >= min(_STEADY_USES, int(counts.max()))].drop_duplicates("Lemma")
    ranked = steady.sort_values(["Separation", "Lemma"], ascending=[False, True])
    lemmas = [str(lemma) for lemma in ranked["Lemma"]]
    for lemma in lemmas[:_DEFAULT_TRIES]:
        contexts = _contexts(table[table["Lemma"] == lemma], lemma)
        if contexts.get(1) and contexts.get(2):
            return lemma
    return lemmas[0]


def _senses(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    table = _uses(frame, params)
    if table.empty:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", "no word of this part of speech splits into two senses in this run")
        )
    typed = str(params.get("word", "")).strip().casefold()
    lemmas = {lemma.casefold(): lemma for lemma in table["Lemma"].unique()}
    if typed and typed not in lemmas:
        return Result.failure(
            Diagnostic.error(
                "PANEL_QUERY_NOT_FOUND",
                f"{params['word']!r} does not split into two senses in this run; see Words used in two senses",
            )
        )
    lemma = lemmas[typed] if typed else _clearest(table)
    if not typed:
        provenance = replace(provenance, params={**provenance.params, "word": lemma})
    rows = table[table["Lemma"] == lemma]
    contexts = _contexts(rows, lemma)
    k = int(params["context-words"])
    picked = {1: contexts.get(1, [])[:k], 2: contexts.get(2, [])[:k]}
    if not picked[1] and not picked[2]:
        return Result.failure(
            Diagnostic.error(
                "PANEL_NO_DATA", f"the sentences of {lemma}'s two senses share no distinctive words to show"
            )
        )
    diagnostics = [
        Diagnostic.info(
            "PANEL_ONE_SIDED",
            f"no word appears in two or more sentences of {lemma}'s sense {sense} and not the other's; "
            "read that sense's sentences in the table",
        )
        for sense in (1, 2)
        if not picked[sense]
    ]
    share = {sense: float((rows["Sense"] == sense).mean()) for sense in (1, 2)}
    names = {sense: f"sense {sense} ({share[sense]:.0%} of {len(rows)} uses)" for sense in (1, 2)}
    ordered = [*picked[1], *reversed(picked[2])]
    marks: list[PanelMark] = []
    evidence_rows: list[dict[str, Any]] = []
    for rank, (token, score) in enumerate(ordered):
        sense = 2 if score > 0 else 1
        sentences = rows[
            (rows["Sense"] == sense) & rows["Sentence"].map(lambda text, t=token: t in _tokens(text, lemma))
        ]
        marks.append(
            PanelMark(
                key=f"context:{sense}:{token}",
                label=token,
                x=float(score),
                y=float(rank),
                group=names[sense],
                evidence=Evidence(
                    scope="rows",
                    filters=(("Context word", token),),
                    count=len(sentences),
                    describe=f"{token} appears in {len(sentences)} sentence(s) of {lemma}'s sense {sense}; log-odds {score:+.2f}",
                ),
            )
        )
        for _, use in sentences.iterrows():
            evidence_rows.append(
                {
                    "Context word": token,
                    "Sense": sense,
                    "Document": use.get("Document", ""),
                    "Sentence": use["Sentence"],
                }
            )
    return Result.success(
        PreparedPanel(
            panel="word_sense_induction_senses",
            shape="ranked_bars",
            title=f"The two senses of {lemma}",
            subtitle=f"words that tell the senses apart · separation {float(rows['Separation'].iloc[0]):.2f}"
            + (" · defaulted to the clearest split of a well-used word" if not typed else "")
            + _sense_class_text(params, frame),
            marks=tuple(marks),
            x_label="more typical of sense 1  <  0  >  more typical of sense 2  (log-odds)",
            y_label="",
            provenance=provenance,
            data=pd.DataFrame(evidence_rows, columns=["Context word", "Sense", "Document", "Sentence"]),
            groups=(names[1], names[2]),
            notes=_SENSE_NOTES,
        ),
        *diagnostics,
    )


_SENSE_REQUIRES = ("Lemma", "Sense", "Separation", "Sentence")

WORD_SENSE_SPLIT_WORDS = PanelDefinition(
    name="word_sense_induction_split_words",
    title="Words used in two senses",
    question="Which words does the corpus use in two different ways?",
    tool="word_sense_induction",
    shape="ranked_bars",
    summary="The words whose uses split most clearly into two senses, each with the context words that mark "
    "each sense.",
    requires=_SENSE_REQUIRES,
    params=(
        PanelParam(
            name="top-n",
            type="int",
            default=20,
            minimum=5,
            maximum=40,
            label="Words to show",
            help="How many words to rank.",
        ),
        PanelParam(
            name="min-uses",
            type="int",
            default=20,
            minimum=4,
            maximum=10000,
            label="Fewest uses",
            help="Leave out words with fewer uses than this: a split of four sentences is easily chance.",
        ),
        _word_class("noun"),
    ),
    build=_split_words,
    notes=_SENSE_NOTES,
)

WORD_SENSE_SENSES = PanelDefinition(
    name="word_sense_induction_senses",
    title="The two senses of a word",
    question="What tells this word's two uses apart?",
    tool="word_sense_induction",
    shape="ranked_bars",
    summary="For one word, the words that appear with each of its two senses; click one to read those sentences.",
    requires=_SENSE_REQUIRES,
    params=(
        PanelParam(
            name="word",
            type="str",
            default="",
            label="Word",
            help="A word from Words used in two senses. Left empty, the clearest split among frequent words.",
        ),
        PanelParam(
            name="context-words",
            type="int",
            default=10,
            minimum=3,
            maximum=20,
            label="Context words per sense",
            help="How many of each sense's most distinctive words to show.",
        ),
        _word_class("noun"),
    ),
    build=_senses,
    notes=_SENSE_NOTES,
)


WORD_MEANING_PANELS: tuple[PanelDefinition, ...] = (
    *_definitions("word2vec_gensim"),
    *_definitions("word2vec_bert"),
    WORD2VEC_BERT_MEANING_OVER_TIME,
    WORD2VEC_BERT_MEANING_CHANGE,
    WORD_SENSE_SPLIT_WORDS,
    WORD_SENSE_SENSES,
)
