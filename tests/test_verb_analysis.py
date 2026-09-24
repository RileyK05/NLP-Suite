"""verb_analysis: modality, tense and voice (FR-5.10, CAP-CONLL-10).

Offline contract tests over hand-built CoNLL frames: morphology-first tense,
the POS and suffix fallbacks with their degradation warning, the will/shall
future heuristic, passive detection (parse and degraded), modal mapping, the
summary's facet contract, the empty-input contract and every loud failure.
"""

from __future__ import annotations

import pandas as pd

from core.analysis.verb_analysis import (
    SUMMARY_COLUMNS,
    VERB_COLUMNS,
    analyze_verbs,
    summarize_verbs,
)

_COLUMNS = ["ID", "Form", "Lemma", "POS", "NER", "Head", "DepRel", "Sentence ID", "Document ID", "Document"]


def _frame(rows: list[dict[str, object]], *, feats: bool = False) -> pd.DataFrame:
    """(id, form, lemma, pos, head, deprel[, feats]) per token -> canonical frame."""
    out = []
    for row in rows:
        item = {
            "ID": row["id"],
            "Form": row["form"],
            "Lemma": row["lemma"],
            "POS": row["pos"],
            "NER": "O",
            "Head": row.get("head", 0),
            "DepRel": row.get("deprel", ""),
            "Sentence ID": row.get("sent", 1),
            "Document ID": row.get("doc_id", 1),
            "Document": row.get("doc", "d.txt"),
        }
        if feats:
            item["feats"] = row.get("feats", "")
        out.append(item)
    columns = [*_COLUMNS, "feats"] if feats else _COLUMNS
    return pd.DataFrame(out, columns=columns)


class TestTense:
    def test_morphology_wins_when_present(self) -> None:
        frame = _frame(
            [{"id": 1, "form": "walked", "lemma": "walk", "pos": "VBZ", "feats": "Tense=Past|VerbForm=Fin"}],
            feats=True,
        )
        result = analyze_verbs(frame)
        assert result.ok, result.diagnostics
        assert result.unwrap().iloc[0]["Tense"] == "past"

    def test_verbform_gerund_and_participle_and_infinitive(self) -> None:
        frame = _frame(
            [
                {"id": 1, "form": "walking", "lemma": "walk", "pos": "VB", "feats": "VerbForm=Ger"},
                {"id": 2, "form": "walked", "lemma": "walk", "pos": "VB", "feats": "VerbForm=Part"},
                {"id": 3, "form": "walk", "lemma": "walk", "pos": "VB", "feats": "VerbForm=Inf"},
            ],
            feats=True,
        )
        result = analyze_verbs(frame)
        assert result.ok, result.diagnostics
        assert result.unwrap()["Tense"].tolist() == ["gerund", "participle", "infinitive"]

    def test_pos_fallback_without_morphology_warns(self) -> None:
        frame = _frame(
            [
                {"id": 1, "form": "walked", "lemma": "walk", "pos": "VBD"},
                {"id": 2, "form": "walks", "lemma": "walk", "pos": "VBZ"},
                {"id": 3, "form": "walk", "lemma": "walk", "pos": "VB"},
                {"id": 4, "form": "taken", "lemma": "take", "pos": "VBN"},
            ]
        )
        result = analyze_verbs(frame)
        assert result.ok, result.diagnostics
        assert any(d.code == "VERB_NO_MORPHOLOGY" for d in result.diagnostics)
        assert result.unwrap()["Tense"].tolist() == ["past", "present", "infinitive", "participle"]

    def test_form_suffix_fallback_when_pos_is_universal(self) -> None:
        frame = _frame(
            [
                {"id": 1, "form": "walked", "lemma": "walk", "pos": "VERB"},
                {"id": 2, "form": "walking", "lemma": "walk", "pos": "VERB"},
                {"id": 3, "form": "go", "lemma": "go", "pos": "VERB"},
            ]
        )
        result = analyze_verbs(frame)
        assert result.ok, result.diagnostics
        assert result.unwrap()["Tense"].tolist() == ["past", "gerund", "unknown"]

    def test_will_or_shall_marks_future_first(self) -> None:
        frame = _frame(
            [
                {"id": 1, "form": "will", "lemma": "will", "pos": "AUX", "head": 2, "deprel": "aux"},
                {"id": 2, "form": "go", "lemma": "go", "pos": "VBN", "head": 0, "deprel": "root"},
            ]
        )
        result = analyze_verbs(frame)
        assert result.ok, result.diagnostics
        frame_out = result.unwrap()
        go = frame_out[frame_out["Verb"] == "go"].iloc[0]
        assert go["Tense"] == "future"
        assert go["Modality"] == "none"
        assert go["Modal"] == "will"


class TestVoice:
    def test_passive_marker_on_a_dependent(self) -> None:
        frame = _frame(
            [
                {"id": 1, "form": "cake", "lemma": "cake", "pos": "NN", "head": 3, "deprel": "nsubj:pass"},
                {"id": 2, "form": "was", "lemma": "be", "pos": "AUX", "head": 3, "deprel": "aux:pass"},
                {"id": 3, "form": "eaten", "lemma": "eat", "pos": "VBN", "head": 0, "deprel": "root"},
            ]
        )
        result = analyze_verbs(frame)
        assert result.ok, result.diagnostics
        eaten = result.unwrap().query("Verb == 'eaten'").iloc[0]
        assert eaten["Voice"] == "passive"

    def test_degraded_be_plus_participle_without_markers(self) -> None:
        frame = _frame(
            [
                {"id": 1, "form": "was", "lemma": "be", "pos": "VBD", "head": 2, "deprel": ""},
                {"id": 2, "form": "eaten", "lemma": "eat", "pos": "VBN", "head": 0, "deprel": ""},
            ]
        )
        result = analyze_verbs(frame)
        assert result.ok, result.diagnostics
        eaten = result.unwrap().query("Verb == 'eaten'").iloc[0]
        assert eaten["Voice"] == "passive"

    def test_plain_clause_is_active(self) -> None:
        frame = _frame(
            [
                {"id": 1, "form": "they", "lemma": "they", "pos": "PRON", "head": 2, "deprel": "nsubj"},
                {"id": 2, "form": "ate", "lemma": "eat", "pos": "VBD", "head": 0, "deprel": "root"},
            ]
        )
        result = analyze_verbs(frame)
        assert result.ok, result.diagnostics
        assert result.unwrap().iloc[0]["Voice"] == "active"


class TestModality:
    def _modal(self, modal: str) -> pd.Series:
        frame = _frame(
            [
                {"id": 1, "form": modal, "lemma": modal, "pos": "AUX", "head": 2, "deprel": "aux"},
                {"id": 2, "form": "swim", "lemma": "swim", "pos": "VB", "head": 0, "deprel": "root"},
            ]
        )
        result = analyze_verbs(frame)
        assert result.ok, result.diagnostics
        return result.unwrap().query("Verb == 'swim'").iloc[0]

    def test_can_is_ability(self) -> None:
        row = self._modal("can")
        assert row["Modality"] == "ability"
        assert row["Modal"] == "can"

    def test_may_is_possibility_and_keeps_the_modal_word(self) -> None:
        row = self._modal("may")
        assert row["Modality"] == "possibility"
        assert row["Modal"] == "may"

    def test_must_should_ought_are_obligation(self) -> None:
        for modal in ("must", "should", "ought"):
            assert self._modal(modal)["Modality"] == "obligation"

    def test_dare_is_unknown_modality(self) -> None:
        row = self._modal("dare")
        assert row["Modality"] == "unknown"
        assert row["Modal"] == "dare"

    def test_no_modal_means_none(self) -> None:
        frame = _frame([{"id": 1, "form": "swim", "lemma": "swim", "pos": "VB", "head": 0, "deprel": "root"}])
        row = analyze_verbs(frame).unwrap().iloc[0]
        assert row["Modality"] == "none"
        assert row["Modal"] == ""


class TestSummary:
    def _raw(self) -> pd.DataFrame:
        return _frame(
            [
                {"id": 1, "form": "can", "lemma": "can", "pos": "AUX", "head": 2, "deprel": "aux"},
                {"id": 2, "form": "walk", "lemma": "walk", "pos": "VB", "head": 0, "deprel": "root"},
                {"id": 3, "form": "eaten", "lemma": "eat", "pos": "VBN", "head": 0, "deprel": "root", "sent": 2},
            ]
        )

    def _annotated(self) -> pd.DataFrame:
        return analyze_verbs(self._raw()).unwrap()

    def test_counts_and_columns(self) -> None:
        result = summarize_verbs(self._annotated())
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert list(frame.columns) == SUMMARY_COLUMNS
        row = frame.iloc[0]
        assert row["Verbs"] == 3
        assert row["Active"] == 3
        assert row["Ability"] == 1
        assert row["No modality"] == 2

    def test_analysis_zeroes_the_columns_outside_its_facet(self) -> None:
        frame = summarize_verbs(self._annotated(), analysis="voice").unwrap().iloc[0]
        assert frame["Verbs"] == 3
        assert frame["Active"] == 3
        assert frame["Ability"] == 0
        assert frame["Past"] == 0

    def test_analyze_verbs_keeps_every_row_whatever_analysis_says(self) -> None:
        result = analyze_verbs(self._raw(), analysis="voice")
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert list(frame.columns) == VERB_COLUMNS
        assert len(frame) == 3


class TestContract:
    def test_empty_input_is_an_empty_table_not_a_failure(self) -> None:
        result = analyze_verbs(pd.DataFrame(columns=_COLUMNS))
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert frame.empty
        assert list(frame.columns) == VERB_COLUMNS

    def test_bad_analysis_fails_loudly(self) -> None:
        result = analyze_verbs(_frame([{"id": 1, "form": "go", "lemma": "go", "pos": "VB"}]), analysis="mood")
        assert not result.ok
        assert result.diagnostics[0].code == "VERB_BAD_ANALYSIS"

    def test_summarize_bad_analysis_fails_loudly(self) -> None:
        result = summarize_verbs(pd.DataFrame(columns=VERB_COLUMNS), analysis="mood")
        assert not result.ok
        assert result.diagnostics[0].code == "VERB_BAD_ANALYSIS"

    def test_missing_column_fails_loudly(self) -> None:
        frame = _frame([{"id": 1, "form": "go", "lemma": "go", "pos": "VB"}]).drop(columns=["Lemma"])
        result = analyze_verbs(frame)
        assert not result.ok
        assert result.diagnostics[0].code == "VERB_MISSING_COLUMN"

    def test_other_missing_required_columns_fail_loudly(self) -> None:
        frame = _frame([{"id": 1, "form": "go", "lemma": "go", "pos": "VB"}]).drop(columns=["NER"])
        result = analyze_verbs(frame)
        assert not result.ok
        assert result.diagnostics[0].code == "CONLL_MISSING_COLUMN"
