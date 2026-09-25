"""What word vectors mean: axes, groups, neighbourhoods, change, senses.

The figures were built against real BERT vectors of the State of the Union
(war ... peace put *soldier* toward war and *trade* toward peace; *energy*
moved from strength and spirit in the 1940s to fuel and oil from the 1970s).
These tests pin the arithmetic on small spaces whose answers are known by
construction, so a regression names itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from core.analysis import word_meaning
from core.analysis.contextual import sense_uses
from core.analysis.postags import word_class
from core.analysis.word2vec_bert import train_bert
from core.viz.panels import get_panel
from core.viz.panelspec import Provenance

# ------------------------------------------------------------------ spaces --


def _vectors(rows: dict[str, tuple[int, list[float]]], classes: dict[str, str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {"Word": word, "Count": count, "Vector": ",".join(f"{v:.4f}" for v in vector)}
            for word, (count, vector) in rows.items()
        ]
    )
    if classes is not None:
        frame[word_meaning.WORD_CLASS] = frame["Word"].map(classes).fillna("")
    return frame


#: Two themes in four dimensions, plus a shared offset every contextual model has.
_OFFSET = [3.0, 3.0, 3.0, 3.0]


def _theme(axis: int, wobble: float) -> list[float]:
    vector = [0.0, 0.0, 0.0, 0.0]
    vector[axis] = 1.0
    vector[(axis + 2) % 4] = wobble
    return [a + b for a, b in zip(vector, _OFFSET, strict=True)]


WORDS = {
    "war": (40, _theme(0, 0.10)),
    "battle": (20, _theme(0, 0.20)),
    "army": (18, _theme(0, 0.05)),
    "invasion": (9, _theme(0, 0.15)),
    "peace": (35, _theme(1, 0.10)),
    "treaty": (15, _theme(1, 0.20)),
    "harmony": (8, _theme(1, 0.05)),
    "friendship": (7, _theme(1, 0.15)),
    "soldier": (12, [a + b for a, b in zip([0.8, 0.2, 0.0, 0.0], _OFFSET, strict=True)]),
    "trade": (12, [a + b for a, b in zip([0.2, 0.8, 0.0, 0.0], _OFFSET, strict=True)]),
    "the": (500, [a + b for a, b in zip([0.5, 0.5, 0.5, 0.5], _OFFSET, strict=True)]),
}
CLASSES = {word: "noun" for word in WORDS} | {"the": ""}


def space() -> word_meaning.Space:
    return word_meaning.read_space(_vectors(WORDS, CLASSES)).unwrap()


class TestTheSpace:
    def test_centring_removes_the_shared_direction(self) -> None:
        """Raw cosines crowd near one (the offset); centred ones spread out."""
        raw = np.asarray([vector for _, vector in WORDS.values()], dtype=float)
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        centred = space().unit
        assert (raw @ raw.T).min() > 0.9
        assert (centred @ centred.T).min() < 0

    def test_words_are_folded_and_duplicates_refused(self) -> None:
        frame = _vectors({"Freedom": (3, [1.0, 0.0]), "freedom": (2, [0.0, 1.0])})
        result = word_meaning.read_space(frame)
        assert result.value is None
        assert result.diagnostics[0].code == "PANEL_AMBIGUOUS_VOCABULARY"

    def test_malformed_rows_are_dropped_and_counted(self) -> None:
        frame = pd.concat(
            [_vectors(WORDS), pd.DataFrame([{"Word": "broken", "Count": 3, "Vector": "1,nan"}])], ignore_index=True
        )
        parsed = word_meaning.read_space(frame).unwrap()
        assert parsed.dropped == 1
        assert "broken" not in parsed.index

    def test_typed_words_split_on_commas_and_spaces(self) -> None:
        assert word_meaning.parse_words("War, conflict;  WAR peace") == ["war", "conflict", "peace"]

    def test_candidates_keep_one_class_and_drop_excluded_words(self) -> None:
        found = word_meaning.candidates(space(), word_class="noun", exclude={"war"}, min_count=10)
        words = [space().words[i] for i in found]
        assert words[0] == "peace"  # most frequent first
        assert "war" not in words and "the" not in words and "harmony" not in words


class TestAnAxisBetweenTwoIdeas:
    def test_words_lean_toward_the_pole_they_resemble(self) -> None:
        scores, low, high = word_meaning.axis_scores(space(), ["war"], ["peace"]).unwrap()
        s = space()
        assert (low, high) == (["war"], ["peace"])
        assert scores[s.index["soldier"]] < 0 < scores[s.index["trade"]]
        assert scores[s.index["army"]] < scores[s.index["soldier"]]

    def test_a_missing_pole_word_is_named(self) -> None:
        result = word_meaning.axis_scores(space(), ["war", "zeppelin"], ["peace"])
        assert result.value is not None
        assert [d.code for d in result.diagnostics] == ["PANEL_POLE_NOT_FOUND"]

    def test_an_end_with_no_known_word_is_refused_with_suggestions(self) -> None:
        result = word_meaning.axis_scores(space(), ["zeppelin"], ["peace"])
        assert result.value is None
        refusal = next(d for d in result.diagnostics if d.code == "PANEL_POLE_EMPTY")
        assert "the" in refusal.message or "war" in refusal.message

    def test_the_same_words_at_both_ends_are_refused(self) -> None:
        result = word_meaning.axis_scores(space(), ["war"], ["war"])
        assert [d.code for d in result.diagnostics] == ["PANEL_POLE_SAME"]


class TestGroupsAndNeighbourhoods:
    def test_groups_find_the_two_themes_and_name_them_by_their_centre(self) -> None:
        s = space()
        among = word_meaning.candidates(s, word_class="noun")
        groups = word_meaning.meaning_groups(s, among, groups=2)
        members = [{s.words[i] for i in group.members} for group in groups]
        assert {"war", "battle", "army", "invasion"} <= members[0] | members[1]
        assert any({"war", "battle", "army"} <= m for m in members)
        assert any({"peace", "treaty", "harmony"} <= m for m in members)
        for group in groups:
            assert list(group.closeness) == sorted(group.closeness, reverse=True)
            assert group.name.split(", ")[0] == s.words[group.members[0]]

    def test_too_few_distinct_words_ask_for_fewer_groups(self) -> None:
        s = space()
        among = np.asarray([s.index["war"], s.index["peace"], s.index["army"]])
        assert len(word_meaning.meaning_groups(s, among, groups=5)) == 1

    def test_nearer_neighbours_sit_nearer_the_centre(self) -> None:
        s = space()
        spokes = word_meaning.neighbourhood(s, s.index["war"], word_meaning.candidates(s, word_class="noun"), size=6)
        by_similarity = sorted(spokes, key=lambda spoke: -spoke.similarity)
        radius = [np.hypot(spoke.x - 0.5, spoke.y - 0.5) for spoke in by_similarity]
        assert radius == sorted(radius)
        assert s.words[by_similarity[0].word] in {"army", "battle", "invasion"}
        assert all(spoke.nearest != spoke.word for spoke in spokes)


# -------------------------------------------------------------------- time --


class ContextBackend:
    """A word's vector is the theme of its sentence: 'strength' or 'fuel' contexts."""

    def dimension(self) -> int:
        return 4

    def embed(self, words: Sequence[str], contexts: Sequence[str]) -> list[list[float]]:
        out = []
        for word, context in zip(words, contexts, strict=True):
            if word == "energy":
                out.append(_theme(0, 0.1) if "spirit" in context else _theme(1, 0.1))
            else:
                out.append(_theme(0 if word in {"spirit", "strength", "vigor"} else 1, 0.1 + 0.01 * len(word)))
        return out


def _conll(sentences: list[tuple[str, int, str]]) -> pd.DataFrame:
    rows = []
    for record, (doc, sid, text) in enumerate(sentences):
        for position, form in enumerate(text.split(), start=1):
            rows.append(
                {
                    "ID": position,
                    "Form": form,
                    "Lemma": form,
                    "POS": "NN",
                    "NER": "O",
                    "Head": 0,
                    "DepRel": "root",
                    "Deps": "",
                    "Clause Tag": "",
                    "Record ID": record,
                    "Sentence ID": sid,
                    "Document ID": doc,
                    "Document": f"{doc}.txt",
                }
            )
    return pd.DataFrame(rows)


def _energy_corpus() -> pd.DataFrame:
    early = [("1", i, "energy spirit strength vigor") for i in range(6)]
    late = [("2", 10 + i, "energy fuel oil electric") for i in range(6)]
    return _conll(early + late)


class TestMeaningOverTime:
    def test_train_bert_averages_each_period_without_embedding_again(self) -> None:
        trained = train_bert(
            _energy_corpus(), field="lemma", backend=ContextBackend(), periods={"1": "1940s", "2": "1980s"}
        ).unwrap()
        periods = {(word, period): uses for word, period, uses, _ in trained.by_period}
        assert periods[("energy", "1940s")] == 6
        assert periods[("energy", "1980s")] == 6
        assert ("fuel", "1940s") not in periods

    def test_no_periods_no_period_vectors(self) -> None:
        assert train_bert(_energy_corpus(), field="lemma", backend=ContextBackend()).unwrap().by_period == ()

    def test_a_word_whose_company_moves_is_the_one_that_changed(self) -> None:
        trained = train_bert(
            _energy_corpus(), field="lemma", backend=ContextBackend(), periods={"1": "1940s", "2": "1980s"}
        ).unwrap()
        s = word_meaning.space_of(trained.words, trained.counts, np.asarray(trained.vectors))
        over_time, change = word_meaning.meaning_over_time(s, trained.by_period, min_uses=3, top_n=2)
        assert list(change.columns) == word_meaning.CHANGE_COLUMNS
        energy = change.set_index("Word").loc["energy"]
        assert (energy["From"], energy["To"]) == ("1940s", "1980s")
        assert energy["Change"] > 1.0
        assert set(energy["Company before"].split(", ")) <= {"spirit", "strength", "vigor"}
        assert set(energy["Company after"].split(", ")) <= {"fuel", "oil", "electric"}
        rows = over_time[over_time["Word"] == "energy"]
        # Every neighbour that made either period's list is scored in both.
        assert rows.groupby("Neighbor")["Period"].nunique().eq(2).all()

    def test_words_below_the_floor_are_not_compared(self) -> None:
        trained = train_bert(
            _energy_corpus(), field="lemma", backend=ContextBackend(), periods={"1": "1940s", "2": "1980s"}
        ).unwrap()
        s = word_meaning.space_of(trained.words, trained.counts, np.asarray(trained.vectors))
        _, change = word_meaning.meaning_over_time(s, trained.by_period, min_uses=7)
        assert change.empty


# ------------------------------------------------------------ word classes --


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("NN", "noun"),
        ("NNPS", "noun"),
        ("PROPN", "noun"),
        ("VBD", "verb"),
        ("VERB", "verb"),
        ("JJR", "adjective"),
        ("ADJ", "adjective"),
        ("RB", "adverb"),
        ("AUX", ""),
        ("MD", ""),
        (None, ""),
    ],
)
def test_word_class_reads_both_tagsets(tag: object, expected: str) -> None:
    assert word_class(tag) == expected


def test_each_word_takes_its_most_frequent_class() -> None:
    frame = pd.DataFrame({"Lemma": ["Run", "run", "run", "cat"], "POS": ["VB", "VBD", "NN", "NN"]})
    assert word_meaning.word_classes(frame, "Lemma") == {"run": "verb", "cat": "noun"}


# ------------------------------------------------------------------ senses --


def test_sense_uses_joins_each_sentence_once_with_its_share() -> None:
    frame = _conll([("1", 1, "the bank of the river"), ("1", 2, "the bank lent money"), ("1", 3, "a bank loan")])
    senses = pd.DataFrame(
        {
            "Lemma": ["bank", "bank", "bank", "loan"],
            "Sense": [1, 1, 2, 1],
            "Separation": [0.4, 0.4, 0.4, 0.0],
            "Sentence ID": [2, 3, 1, 3],
            "Document ID": ["1", "1", "1", "1"],
        }
    )
    uses = sense_uses(frame, senses, names={"1": "speech.txt"})
    assert set(uses["Lemma"]) == {"bank"}
    assert uses.set_index("Sentence ID").loc["1", "Sentence"] == "the bank of the river"
    assert sorted(uses["Share"].round(2).tolist()) == [0.33, 0.67, 0.67]
    assert set(uses["Document"]) == {"speech.txt"}


# ------------------------------------------------------------------ panels --


def _build(name: str, frame: pd.DataFrame, **params: Any):  # type: ignore[no-untyped-def]
    definition = get_panel(name)
    assert definition is not None
    return definition.build(
        frame, definition.defaults() | params, Provenance(tool=definition.tool, panel=name, settings={"field": "lemma"})
    )


class TestTheWordVectorFigures:
    @pytest.mark.parametrize("tool", ["word2vec_gensim", "word2vec_bert"])
    def test_every_word_vector_tool_has_the_meaning_figures(self, tool: str) -> None:
        for view in ("meaning_groups", "meaning_axis", "word_network", "meaning_plane"):
            assert get_panel(f"{tool}_{view}") is not None

    def test_the_axis_places_chosen_words_and_leaves_the_poles_off(self) -> None:
        prepared = _build(
            "word2vec_bert_meaning_axis",
            _vectors(WORDS, CLASSES),
            **{"one-end": "war", "other-end": "peace", "place": "trade", "words-per-end": 3},
        ).unwrap()
        labels = [mark.label for mark in prepared.marks]
        assert "war" not in labels and "peace" not in labels
        assert labels[0] in {"army", "battle", "invasion"}
        placed = [mark for mark in prepared.marks if mark.group == "your words"]
        assert [mark.label for mark in placed] == ["trade"]
        assert placed[0].x > 0
        assert all(mark.evidence.phrase == mark.label and mark.evidence.lemma for mark in prepared.marks)

    def test_the_plane_names_its_axes_and_labels_placed_words(self) -> None:
        prepared = _build(
            "word2vec_bert_meaning_plane",
            _vectors(WORDS, CLASSES),
            **{
                "x-one-end": "war",
                "x-other-end": "peace",
                "y-one-end": "army",
                "y-other-end": "treaty",
                "place": "soldier",
                "label-count": 2,
            },
        ).unwrap()
        assert "war" in prepared.x_label and "peace" in prepared.x_label
        soldier = next(mark for mark in prepared.marks if mark.label == "soldier")
        assert soldier.labelled and soldier.group == "your words"

    def test_groups_stay_within_one_readable_figure(self) -> None:
        many = {
            f"w{chr(97 + i)}{chr(97 + j)}": (10, list(np.random.default_rng(i * 26 + j).normal(size=6)))
            for i in range(10)
            for j in range(10)
        }
        result = _build(
            "word2vec_gensim_meaning_groups", _vectors(many), groups=12, **{"words-per-group": 15, "word-class": "any"}
        )
        prepared = result.unwrap()
        assert len(prepared.marks) <= 60
        assert "PANEL_WORDS_CAPPED" in [d.code for d in result.diagnostics]

    def test_nouns_asked_of_a_run_without_classes_says_so(self) -> None:
        result = _build("word2vec_gensim_meaning_groups", _vectors(WORDS), groups=2)
        assert "PANEL_NO_WORD_CLASS" in [d.code for d in result.diagnostics]
        assert result.unwrap().title == "Groups of words that mean alike"

    def test_the_network_links_neighbours_and_defaults_to_a_frequent_noun(self) -> None:
        result = _build("word2vec_bert_word_network", _vectors(WORDS, CLASSES), neighbours=5)
        prepared = result.unwrap()
        assert prepared.provenance.params["word"] == "war"
        keys = {mark.key for mark in prepared.marks}
        assert all(edge.source in keys and edge.target in keys for edge in prepared.edges)
        assert sum(edge.source == "node:war" for edge in prepared.edges) == 5

    def test_an_unknown_word_is_refused(self) -> None:
        result = _build("word2vec_bert_word_network", _vectors(WORDS), word="zeppelin")
        assert [d.code for d in result.diagnostics] == ["PANEL_QUERY_NOT_FOUND"]

    def test_the_tsne_map_is_coloured_by_named_group(self) -> None:
        tsne = pd.DataFrame(
            {
                "Word": ["war", "army", "peace", "the"],
                "X": [0, 1, 5, 3],
                "Y": [0, 1, 5, 2],
                "Count": [9, 8, 7, 100],
                "Group": ["war, army", "war, army", "peace", ""],
            }
        )
        prepared = _build("word2vec_bert_tsne", tsne, **{"hide-function-words": False}).unwrap()
        assert prepared.groups == ("war, army", "peace", "(other words)")
        assert {m.label: m.group for m in prepared.marks}["the"] == "(other words)"


class TestTheChangeFigures:
    def _tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        trained = train_bert(
            _energy_corpus(), field="lemma", backend=ContextBackend(), periods={"1": "1940s", "2": "1980s"}
        ).unwrap()
        s = word_meaning.space_of(trained.words, trained.counts, np.asarray(trained.vectors))
        return word_meaning.meaning_over_time(s, trained.by_period, min_uses=3, top_n=3)

    def test_the_heatmap_follows_one_word_through_its_periods(self) -> None:
        over_time, _ = self._tables()
        prepared = _build("word2vec_bert_meaning_over_time", over_time, word="energy", neighbours=2).unwrap()
        assert prepared.x_categories == ("1940s (6 uses)", "1980s (6 uses)")
        assert set(prepared.y_categories) <= {"spirit", "strength", "vigor", "fuel", "oil", "electric"}
        early = prepared.y_categories[0]
        assert early in {"spirit", "strength", "vigor"}
        assert all(mark.evidence.phrase == "energy" for mark in prepared.marks)

    def test_a_word_not_compared_is_refused_with_the_way_forward(self) -> None:
        over_time, _ = self._tables()
        result = _build("word2vec_bert_meaning_over_time", over_time, word="zeppelin")
        assert "Words whose company changed most" in result.diagnostics[0].message

    def test_the_change_ranking_names_each_word_span(self) -> None:
        _, change = self._tables()
        prepared = _build("word2vec_bert_meaning_change", change, **{"min-uses": 2, "word-class": "any"}).unwrap()
        assert prepared.marks[0].label == "energy (1940s to 1980s)"
        assert "spirit" in prepared.marks[0].evidence.describe or "strength" in prepared.marks[0].evidence.describe


class TestTheSenseFigures:
    def _uses(self) -> pd.DataFrame:
        money = [f"the bank lent money loan interest {i}" for i in range(12)]
        river = [f"the bank of the river water shore {i}" for i in range(10)]
        rows = [
            {
                "Lemma": "bank",
                "Sense": 1,
                "Separation": 0.45,
                "Share": 0.55,
                "Document": "a.txt",
                "Sentence ID": str(i),
                "Document ID": "1",
                "Sentence": text,
                word_meaning.WORD_CLASS: "noun",
            }
            for i, text in enumerate(money)
        ] + [
            {
                "Lemma": "bank",
                "Sense": 2,
                "Separation": 0.45,
                "Share": 0.45,
                "Document": "b.txt",
                "Sentence ID": str(100 + i),
                "Document ID": "2",
                "Sentence": text,
                word_meaning.WORD_CLASS: "noun",
            }
            for i, text in enumerate(river)
        ]
        return pd.DataFrame(rows)

    def test_split_words_are_labelled_by_the_context_of_each_sense(self) -> None:
        prepared = _build("word_sense_induction_split_words", self._uses(), **{"min-uses": 5}).unwrap()
        label = prepared.marks[0].label
        assert label.startswith("bank: ")
        first, second = label.removeprefix("bank: ").split(" | ")
        assert set(first.split(", ")) <= {"lent", "money", "loan", "interest"}
        assert set(second.split(", ")) <= {"river", "water", "shore"}

    def test_each_context_word_opens_the_sentences_of_its_sense_that_hold_it(self) -> None:
        prepared = _build("word_sense_induction_senses", self._uses()).unwrap()
        assert prepared.provenance.params["word"] == "bank"
        river = next(mark for mark in prepared.marks if mark.label == "river")
        assert river.x > 0
        assert river.evidence.filters == (("Context word", "river"),)
        behind = prepared.data[prepared.data["Context word"] == "river"]
        assert len(behind) == river.evidence.count == 10
        assert behind["Sentence"].str.contains("river").all()

    def test_another_part_of_speech_finds_none_and_says_so(self) -> None:
        result = _build("word_sense_induction_senses", self._uses(), **{"word-class": "verb"})
        assert result.value is None
        assert "part of speech" in result.diagnostics[0].message


# --------------------------------------------------------------- adapters --


def test_periods_are_decades_or_years_when_there_are_two() -> None:
    from core.profiler.executor import _periods

    class Doc:
        def __init__(self, doc_id: int, when: date | None) -> None:
            self.doc_id, self.date = doc_id, when

    class Corpus:
        def __init__(self, *docs: Doc) -> None:
            self.docs = list(docs)

    assert _periods(Corpus(Doc(1, date(1941, 1, 6)), Doc(2, date(1985, 2, 6)))) == {"1": "1940s", "2": "1980s"}  # type: ignore[arg-type]
    assert _periods(Corpus(Doc(1, date(1981, 1, 6)), Doc(2, date(1985, 2, 6)))) == {"1": "1981", "2": "1985"}  # type: ignore[arg-type]
    assert _periods(Corpus(Doc(1, date(1981, 1, 6)), Doc(2, None))) == {}  # type: ignore[arg-type]
