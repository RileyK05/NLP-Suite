"""gender_guess: writing-style gender guessing (FR-2.10, CAP-STYLE-10).

Offline contract tests: the documented threshold table on its public seam,
the empty-input contract, the two loud failures, and one hand-built frame per
document. No live parser and no gender-labelled oracle: the scores are
transparent linear sums, so the tests pin the CONTRACT (columns, thresholds,
agreement rules), not any theory of gender.
"""

from __future__ import annotations

import pandas as pd

from core.analysis.gender_guess import (
    GUESS_COLUMNS,
    SUMMARY_COLUMNS,
    guess_from_counts,
    guess_gender,
    label_for,
    score_from_counts,
    summarize_labels,
)

_CANONICAL = ["ID", "Form", "Lemma", "POS", "NER", "Head", "DepRel", "Sentence ID", "Document ID", "Document"]


def _frame(docs: dict[str, list[str]]) -> pd.DataFrame:
    """doc name -> token forms, as a canonical parse frame."""
    rows: list[dict[str, object]] = []
    record = 1
    for name, forms in docs.items():
        for index, form in enumerate(forms, start=1):
            rows.append(
                {
                    "ID": index,
                    "Form": form,
                    "Lemma": form.lower(),
                    "POS": "NN",
                    "NER": "O",
                    "Head": 0,
                    "DepRel": "root",
                    "Sentence ID": 1,
                    "Document ID": record,
                    "Document": name,
                }
            )
        record += 1
    return pd.DataFrame(rows, columns=_CANONICAL)


class TestLabelThresholds:
    def test_documented_bands(self) -> None:
        assert label_for(0.51) == "male"
        assert label_for(0.5) == "mostly_male"
        assert label_for(0.16) == "mostly_male"
        assert label_for(0.15) == "androgynous"
        assert label_for(0.0) == "androgynous"
        assert label_for(-0.15) == "androgynous"
        assert label_for(-0.16) == "mostly_female"
        assert label_for(-0.5) == "mostly_female"
        assert label_for(-0.51) == "female"


class TestGuessFromCounts:
    def test_under_fifty_words_is_unknown_on_both_axes(self) -> None:
        assert guess_from_counts({}, 49) == ("unknown", "unknown", "unknown")
        assert guess_from_counts({}, 50)[0] != "unknown"

    def test_informal_features_move_the_informal_score_toward_male(self) -> None:
        lean, _ = score_from_counts({"pronouns_i": 200, "weak": 400, "question_marks": 40}, 1000)
        neutral, _ = score_from_counts({}, 1000)
        assert lean > 0.5 > neutral
        assert guess_from_counts({"pronouns_i": 200, "weak": 400, "question_marks": 40}, 1000)[1] == "male"

    def test_formal_features_move_the_formal_score_toward_female(self) -> None:
        _, lean = score_from_counts({"prepositions": 300, "articles": 200, "certainty": 30}, 1000)
        assert lean > 0.5
        assert guess_from_counts({"prepositions": 300, "articles": 200, "certainty": 30}, 1000)[0] == "female"

    def test_he_and_she_split_counts_feed_the_combined_pronoun_rate(self) -> None:
        split = {"pronouns_he": 50, "pronouns_she": 50}
        combined = {"pronouns_he_she": 100}
        assert score_from_counts(split, 1000) == score_from_counts(combined, 1000)

    def test_disagreeing_axes_report_androgynous(self) -> None:
        # Loud informal AND loud formal: each axis votes a different way.
        counts = {
            "pronouns_i": 150,
            "weak": 300,
            "question_marks": 30,
            "prepositions": 250,
            "articles": 150,
            "numbers": 20,
            "certainty": 20,
            "avg_word_length": 5.4,
        }
        formal_label, informal_label, label = guess_from_counts(counts, 1000)
        assert informal_label == "male"
        assert formal_label == "female"
        assert label == "androgynous"

    def test_agreeing_axes_report_the_more_extreme_label(self) -> None:
        counts = {"pronouns_i": 150, "weak": 300, "question_marks": 30}
        formal_label, informal_label, label = guess_from_counts(counts, 1000)
        assert informal_label == "male"
        assert formal_label == "male"
        assert label == "male"

    def test_mode_selects_the_axis_and_leaves_the_other_label_empty(self) -> None:
        counts = {"pronouns_i": 150, "weak": 300, "question_marks": 30}
        formal_only = guess_from_counts(counts, 1000, mode="formal")
        assert formal_only[1] == ""
        assert formal_only[2] == formal_only[0]
        informal_only = guess_from_counts(counts, 1000, mode="informal")
        assert informal_only[0] == ""
        assert informal_only[2] == informal_only[1]


class TestGuessGender:
    def test_per_document_table_has_the_declared_columns(self) -> None:
        result = guess_gender(_frame({"a.txt": ["the"] * 60, "b.txt": ["I", "think", "you"] * 20}))
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert list(frame.columns) == GUESS_COLUMNS
        assert len(frame) == 2
        assert set(frame["Label"]) <= {"male", "mostly_male", "androgynous", "mostly_female", "female", "unknown"}
        assert frame["Words"].tolist() == [60, 60]

    def test_empty_input_is_an_empty_table_not_a_failure(self) -> None:
        result = guess_gender(pd.DataFrame(columns=_CANONICAL))
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert frame.empty
        assert list(frame.columns) == GUESS_COLUMNS

    def test_bad_mode_fails_loudly(self) -> None:
        result = guess_gender(_frame({"a.txt": ["the"] * 60}), mode="guess")
        assert not result.ok
        assert result.diagnostics[0].code == "GENDER_BAD_MODE"

    def test_missing_form_fails_loudly(self) -> None:
        frame = _frame({"a.txt": ["the"] * 60}).drop(columns=["Form"])
        result = guess_gender(frame)
        assert not result.ok
        assert result.diagnostics[0].code == "GENDER_MISSING_COLUMN"

    def test_other_missing_required_columns_fail_loudly(self) -> None:
        frame = _frame({"a.txt": ["the"] * 60}).drop(columns=["NER"])
        result = guess_gender(frame)
        assert not result.ok
        assert result.diagnostics[0].code == "CONLL_MISSING_COLUMN"

    def test_short_documents_are_unknown(self) -> None:
        result = guess_gender(_frame({"a.txt": ["the"] * 40}))
        assert result.ok, result.diagnostics
        row = result.unwrap().iloc[0]
        assert row["Formal label"] == "unknown"
        assert row["Informal label"] == "unknown"
        assert row["Label"] == "unknown"


class TestSummary:
    def test_counts_documents_per_label(self) -> None:
        annotated = pd.DataFrame({"Label": ["male", "male", "female"], "Words": [60, 60, 60]})
        result = summarize_labels(annotated)
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert list(frame.columns) == SUMMARY_COLUMNS
        assert dict(zip(frame["Label"], frame["Documents"], strict=True)) == {"female": 1, "male": 2}

    def test_empty_annotated_is_an_empty_table(self) -> None:
        result = summarize_labels(pd.DataFrame())
        assert result.ok
        assert result.unwrap().empty
