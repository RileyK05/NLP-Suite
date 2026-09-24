"""FR-5.7 C2 — Gensim LDA contract tests (oracle: tests/fixtures/topics/).

Randomized algorithm, so the evidence is structural + recorded-reference,
not byte-pinned weights: theme partition, dominant alignment, determinism,
distribution sums, and the guard rails. Gensim-backed tests are marked
model_integration; guard tests run offline (guards fire before any import).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from core.analysis import lda as lda_mod
from core.io.reader import Corpus, Document

FIX = Path(__file__).resolve().parent / "fixtures" / "topics"
CATS = {"cat", "kitten", "whisker", "purr", "meow", "feline"}
CARS = {"car", "engine", "wheel", "drive", "road", "vehicle"}


def _texts() -> tuple[list[str], list[str]]:
    docs: list[str] = []
    names: list[str] = []
    with (FIX / "cats_cars_texts.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            names.append(row["Document"])
            docs.append(row["Text"])
    return names, docs


def _doc_tokens() -> dict[str, list[str]]:
    names, docs = _texts()
    return {name: text.split() for name, text in zip(names, docs, strict=True)}


def _stub_plan() -> Any:
    """A SegmentPlan-shaped object for the guard test: the guard must fire on
    the missing bags, not on plan contents."""
    from core.analysis.topic_flow import SegmentPlan

    return SegmentPlan(labels={}, spans={}, rules={}, unaligned=())


def _frame(doc_tokens: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    rec = 1
    for doc_id, (name, toks) in enumerate(doc_tokens.items(), start=1):
        for tok in toks:
            rows.append(
                {
                    "ID": rec,
                    "Form": tok,
                    "Lemma": tok,
                    "POS": "NN",
                    "NER": "O",
                    "Head": 0,
                    "DepRel": "root",
                    "Sentence ID": 1,
                    "Document ID": doc_id,
                    "Document": name,
                    "Deps": "",
                    "Record ID": rec,
                    "Clause Tag": "",
                }
            )
            rec += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Offline: guards and pure helpers (no gensim import on these paths)
# ---------------------------------------------------------------------------
class TestGuards:
    def test_no_documents_is_error(self) -> None:
        res = lda_mod.fit_lda({}, n_topics=2)
        assert res.value is None
        assert any(d.code == "TOPIC_NO_DOCUMENTS" for d in res.diagnostics)

    def test_single_document_is_error(self) -> None:
        res = lda_mod.fit_lda({"only.txt": ["cat", "kitten", "whisker"]}, n_topics=2)
        assert res.value is None
        assert any(d.code == "TOPIC_SINGLE_DOCUMENT" for d in res.diagnostics)

    def test_stopwords_only_is_error(self) -> None:
        toks = {f"d{i}.txt": ["the", "and", "of", "the", "and"] for i in range(3)}
        res = lda_mod.fit_lda(toks, n_topics=2)
        assert res.value is None
        assert any(d.code == "TOPIC_EMPTY_VOCABULARY" for d in res.diagnostics)

    def test_bad_topic_count_fails(self) -> None:
        res = lda_mod.fit_lda(_doc_tokens(), n_topics=0)
        assert res.value is None
        assert any(d.code == "TOPIC_BAD_K" for d in res.diagnostics)

    def test_corpus_size_advice_table(self) -> None:
        assert lda_mod.corpus_size_advice(0)[0] == "error"
        assert lda_mod.corpus_size_advice(1)[0] == "error"
        level, message = lda_mod.corpus_size_advice(6)
        assert level == "advice" and "6" in message
        assert lda_mod.corpus_size_advice(50) == ("", "")
        assert lda_mod.corpus_size_advice(200) == ("", "")

    def test_tokens_from_frame(self) -> None:
        toks = lda_mod.tokens_from_frame(_frame(_doc_tokens()))
        names, _ = _texts()
        assert set(toks) == set(names)
        assert toks["cats1.txt"] == ["cat", "kitten", "whisker", "purr", "meow", "feline"]

    def test_tokens_from_frame_drops_stopwords(self) -> None:
        toks = lda_mod.tokens_from_frame(_frame({"d.txt": ["the", "cat", "and", "a", "kitten"]}))
        assert toks["d.txt"] == ["cat", "kitten"]

    @staticmethod
    def _tagged(words: list[str], tags: list[str]) -> pd.DataFrame:
        frame = _frame({"d.txt": words})
        frame["POS"] = tags
        return frame

    def test_nouns_only_keeps_penn_nouns(self) -> None:
        """The regression. spaCy's parse carries Penn tags; the filter used to
        accept Universal tags only, so on the real corpus it kept nothing and
        the topic model failed with "no documents to model"."""
        frame = self._tagged(
            ["president", "speaks", "congress", "quickly", "america"], ["NN", "VBZ", "NNP", "RB", "NNP"]
        )
        toks = lda_mod.tokens_from_frame(frame, nouns_only=True)
        assert toks["d.txt"] == ["president", "congress", "america"]

    def test_nouns_only_keeps_universal_nouns(self) -> None:
        frame = self._tagged(["president", "speaks", "congress"], ["NOUN", "VERB", "PROPN"])
        assert lda_mod.tokens_from_frame(frame, nouns_only=True)["d.txt"] == ["president", "congress"]

    def test_nouns_only_drops_every_non_noun_under_both_tagsets(self) -> None:
        frame = self._tagged(["speaks", "quickly", "green", "run"], ["VBZ", "RB", "ADJ", "VERB"])
        assert lda_mod.tokens_from_frame(frame, nouns_only=True) == {}

    def test_without_nouns_only_the_tag_never_matters(self) -> None:
        frame = self._tagged(["president", "speaks"], ["NN", "VBZ"])
        assert lda_mod.tokens_from_frame(frame)["d.txt"] == ["president", "speaks"]

    def test_missing_and_non_alphabetic_cells_are_skipped(self) -> None:
        frame = _frame({"d.txt": ["cat", "x", "dog", "1934", "wo-man"]})
        frame.loc[1, "Lemma"] = None
        toks = lda_mod.tokens_from_frame(frame)
        assert toks["d.txt"] == ["cat", "dog"], "None, digits and hyphenated tokens are not alphabetic words"

    def test_documents_keep_the_order_they_first_appear_in(self) -> None:
        frame = _frame({"b.txt": ["cat"], "a.txt": ["dog"], "c.txt": ["car"]})
        assert list(lda_mod.tokens_from_frame(frame)) == ["b.txt", "a.txt", "c.txt"]

    def test_a_document_with_no_kept_tokens_is_absent(self) -> None:
        """Absent, not present-and-empty: fit_lda's own guard reports it."""
        frame = _frame({"a.txt": ["cat"], "b.txt": ["the", "and"]})
        assert set(lda_mod.tokens_from_frame(frame)) == {"a.txt"}

    def test_missing_gensim_is_actionable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "gensim", None)
        monkeypatch.setitem(sys.modules, "gensim.corpora", None)
        monkeypatch.setitem(sys.modules, "gensim.models", None)
        res = lda_mod.fit_lda(_doc_tokens(), n_topics=2)
        assert res.value is None
        assert any(d.code == "TOPIC_BACKEND_MISSING" for d in res.diagnostics)


# ---------------------------------------------------------------------------
# Gensim-backed: recorded-reference properties (oracle: ORACLE_NOTES.md)
# ---------------------------------------------------------------------------
pytestmark_gensim = pytest.mark.model_integration
gensim = pytest.importorskip("gensim")


@pytestmark_gensim
class TestLdaReference:
    def test_theme_partition(self) -> None:
        res = lda_mod.fit_lda(_doc_tokens(), n_topics=2, top_n=6, seed=100)
        assert res.ok, res.diagnostics
        topics = res.unwrap().topics
        assert list(topics.columns) == ["Topic", "Word", "Weight"]
        assert len(topics) == 12
        by_topic = {t: set(g["Word"]) for t, g in topics.groupby("Topic")}
        assert set(by_topic) == {0, 1}
        families = [CATS if next(iter(words)) in CATS else CARS for words in by_topic.values()]
        assert sorted(map(sorted, by_topic.values())) == [sorted(CARS), sorted(CATS)]
        assert families[0] != families[1]

    def test_dominant_alignment(self) -> None:
        res = lda_mod.fit_lda(_doc_tokens(), n_topics=2, top_n=6, seed=100)
        assert res.ok, res.diagnostics
        out = res.unwrap()
        dom = out.dominant
        assert list(dom.columns) == ["Document ID", "Document", "Dominant topic", "Contribution", "Topic keywords"]
        top_by_topic = {t: set(g["Word"]) for t, g in out.topics.groupby("Topic")}
        for _, row in dom.iterrows():
            family = CATS if "cats" in str(row["Document"]) else CARS
            assert top_by_topic[int(row["Dominant topic"])] == family
            assert float(row["Contribution"]) > 0.5

    def test_determinism(self) -> None:
        toks = _doc_tokens()
        first = lda_mod.fit_lda(toks, n_topics=2, top_n=6, seed=100).unwrap()
        second = lda_mod.fit_lda(toks, n_topics=2, top_n=6, seed=100).unwrap()
        pd.testing.assert_frame_equal(first.topics, second.topics)
        pd.testing.assert_frame_equal(first.dominant, second.dominant)

    def test_metadata(self) -> None:
        res = lda_mod.fit_lda(_doc_tokens(), n_topics=2, top_n=6, seed=100)
        assert res.ok, res.diagnostics
        out = res.unwrap()
        assert out.seed == 100 and out.n_topics == 2
        assert out.coherence is not None and -1.0 <= out.coherence <= 1.0
        assert out.perplexity < 0  # log-perplexity is negative
        assert any(d.code == "TOPIC_FEW_DOCUMENTS" for d in res.diagnostics)  # 6 < 50: advice, non-blocking

    def test_segments_without_tokens_is_a_caller_bug(self) -> None:
        """The segment bags must come from the training filter. Asking for
        scoring without them would otherwise return a flow of empty rows."""
        res = lda_mod.fit_lda(_doc_tokens(), n_topics=2, segments=_stub_plan())
        assert res.value is None
        assert any(d.code == "TOPIC_SEGMENTS_INCOMPLETE" for d in res.diagnostics)

    def test_flow_scores_each_paragraph_and_reports_the_rule(self) -> None:
        """Engine-derived: plan the segments, keep the tokens through the
        training filter, and check the flow table against the model itself."""
        from core.analysis.topic_flow import plan_segments

        names, texts = _texts()
        corpus = Corpus(
            docs=tuple(
                Document(doc_id=i, path=Path(f"d{i}.txt"), text=text)
                for i, (name, text) in enumerate(zip(names, texts, strict=True), start=1)
            ),
            sha256="test",
        )
        frame = _frame(_doc_tokens())
        plan = plan_segments(frame, corpus)
        segment_bags = lda_mod.segment_tokens_for(frame, plan.labels, remove_stopwords=False)
        res = lda_mod.fit_lda(_doc_tokens(), n_topics=2, top_n=6, seed=100, segments=plan, segment_tokens=segment_bags)
        assert res.ok, res.diagnostics
        flow = res.unwrap().flow
        assert flow is not None
        assert list(flow.columns) == list(lda_mod.FLOW_COLUMNS)
        # Every document is present, and each carries its own rule.
        assert set(flow["Document"]) == set(names)
        assert flow["Rule"].isin({"blank-line", "line", "sentences"}).all()
        # Segments cover 1..Segments with no gaps.
        for (_name, total), group in flow.groupby(["Document", "Segments"]):
            assert sorted(group["Segment"]) == list(range(1, int(total) + 1))
        # Scored paragraphs carry a real topic and a real share.
        scored = flow[flow["Dominant topic"].notna()]
        assert not scored.empty
        assert scored["Dominant topic"].between(0, 1).all(), "two topics were fitted"
        assert ((scored["Contribution"] > 0) & (scored["Contribution"] <= 1)).all()
        # A paragraph with no kept tokens is blank, not the prior.
        unscored = flow[flow["Tokens"] == 0]
        assert unscored["Dominant topic"].isna().all()
        assert unscored["Topic keywords"].eq("").all()

    def test_each_kept_token_lands_in_its_own_rows_paragraph(self) -> None:
        """Regression: with stopwords removed the kept tokens are a subset of
        the rows, and pairing them with the row labels by position moved every
        word forward -- here both content words of paragraph 2 were counted in
        paragraph 1, and paragraph 2 came out empty. On the 87-speech corpus
        the executor's strict zip crashed the whole LDA run instead."""
        from core.analysis.topic_flow import SegmentPlan

        tokens = _doc_tokens() | {"d.txt": ["the", "of", "and", "cat", "kitten", "dog", "puppy"]}
        frame = _frame(tokens)
        rows = frame.index[frame["Document"] == "d.txt"]
        labels = pd.Series([1, 1, 1, 1, 1, 2, 2], index=rows)
        plan = SegmentPlan(
            labels={"d.txt": labels}, spans={"d.txt": [None, None]}, rules={"d.txt": "line"}, unaligned=()
        )
        bags = lda_mod.segment_tokens_for(frame, plan.labels, remove_stopwords=True)
        assert bags["d.txt"].tolist() == ["cat", "kitten", "dog", "puppy"]
        flow = lda_mod.fit_lda(tokens, n_topics=2, seed=100, segments=plan, segment_tokens=bags).unwrap().flow
        assert flow is not None
        assert flow.set_index("Segment")["Tokens"].to_dict() == {1: 2, 2: 2}

    def test_segments_do_not_change_the_document_level_fit(self) -> None:
        """Scoring segments must not perturb the fit itself: same seed, same
        tokens, the topics table is identical with and without the ask."""
        from core.analysis.topic_flow import plan_segments

        names, texts = _texts()
        corpus = Corpus(
            docs=tuple(
                Document(doc_id=i, path=Path(f"d{i}.txt"), text=text)
                for i, (name, text) in enumerate(zip(names, texts, strict=True), start=1)
            ),
            sha256="test",
        )
        frame = _frame(_doc_tokens())
        plan = plan_segments(frame, corpus)
        segment_bags = lda_mod.segment_tokens_for(frame, plan.labels, remove_stopwords=False)
        without = lda_mod.fit_lda(_doc_tokens(), n_topics=2, top_n=6, seed=100).unwrap()
        with_flow = lda_mod.fit_lda(
            _doc_tokens(), n_topics=2, top_n=6, seed=100, segments=plan, segment_tokens=segment_bags
        ).unwrap()
        pd.testing.assert_frame_equal(without.topics, with_flow.topics)
        assert with_flow.flow is not None and without.flow is None
