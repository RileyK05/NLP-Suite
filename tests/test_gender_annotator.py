"""CAP-ANNO-10 — name-gender annotation over four name dictionaries.

Offline: injected name sets stand in for the dictionaries (the ``names``
seam) and frames are canonical CoNLL built from Col.*.value columns. The
live NLTK names corpus is environment-backed and deliberately not exercised
here; the contract tested is the one every dictionary shares -- loud
resolution failures, case-insensitive lookup, "both"/unknown honesty, and
the empty-input shape.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.analysis.gender_annotator import annotate_names, load_name_dictionary, summarize_names
from core.conll.schema import Col

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gender"

FAKE_NAMES: dict[str, set[str]] = {"male": {"john", "harry"}, "female": {"mary", "alice"}}


def _conll(rows: list[tuple[str, str, str, int, int, str]]) -> pd.DataFrame:
    """(form, pos, ner, sentence_id, document_id, document) -> canonical CoNLL frame."""
    out: list[dict[str, object]] = []
    for i, (form, pos, ner, sent, doc_id, doc) in enumerate(rows, start=1):
        out.append(
            {
                Col.ID.value: i,
                Col.FORM.value: form,
                Col.LEMMA.value: form.lower(),
                Col.POS.value: pos,
                Col.NER.value: ner,
                Col.HEAD.value: 0,
                Col.DEPREL.value: "dep",
                Col.DEPS.value: "",
                Col.CLAUSE_TAG.value: "",
                Col.RECORD_ID.value: i,
                Col.SENTENCE_ID.value: sent,
                Col.DOCUMENT_ID.value: doc_id,
                Col.DOCUMENT.value: doc,
            }
        )
    return pd.DataFrame(out)


def _person_doc() -> pd.DataFrame:
    # "John Smith met Mary ." + "John waved ."
    return _conll(
        [
            ("John", "PROPN", "PERSON", 1, 1, "a.txt"),
            ("Smith", "PROPN", "PERSON", 1, 1, "a.txt"),
            ("met", "VERB", "O", 1, 1, "a.txt"),
            ("Mary", "PROPN", "PERSON", 1, 1, "a.txt"),
            (".", "PUNCT", "O", 1, 1, "a.txt"),
            ("John", "PROPN", "PERSON", 2, 1, "a.txt"),
            ("waved", "VERB", "O", 2, 1, "a.txt"),
            (".", "PUNCT", "O", 2, 1, "a.txt"),
        ]
    )


class TestAnnotate:
    def test_person_span_classifies_by_given_name(self) -> None:
        result = annotate_names(_person_doc(), names=FAKE_NAMES)
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert list(frame.columns) == ["Document", "Document ID", "Name", "Gender", "Dictionary", "Mentions"]
        john = frame[frame["Name"] == "John Smith"]
        assert list(john["Gender"]) == ["male"]
        assert list(john["Mentions"]) == [1]

    def test_mentions_aggregate_per_document(self) -> None:
        frame = _conll(
            [
                ("Mary", "PROPN", "PERSON", 1, 1, "a.txt"),
                ("met", "VERB", "O", 1, 1, "a.txt"),
                ("mary", "PROPN", "PERSON", 2, 1, "a.txt"),
            ]
        )
        result = annotate_names(frame, names=FAKE_NAMES)
        rows = result.unwrap()
        assert len(rows) == 1
        assert rows.iloc[0]["Name"] == "Mary"  # first-seen surface form
        assert rows.iloc[0]["Mentions"] == 2  # case-folded key merges the two

    def test_lookup_is_case_insensitive(self) -> None:
        frame = _conll([("JOHN", "PROPN", "PERSON", 1, 1, "a.txt")])
        result = annotate_names(frame, names={"male": {"John"}, "female": set()})
        row = result.unwrap().iloc[0]
        assert row["Gender"] == "male"

    def test_unknown_name_is_unknown_not_guessed(self) -> None:
        frame = _conll([("Zzqx", "PROPN", "PERSON", 1, 1, "a.txt")])
        result = annotate_names(frame, names=FAKE_NAMES)
        row = result.unwrap().iloc[0]
        assert row["Gender"] == "unknown"

    def test_name_in_both_lists_is_both_with_a_warning(self) -> None:
        frame = _conll([("Alex", "PROPN", "PERSON", 1, 1, "a.txt")])
        result = annotate_names(frame, names={"male": {"alex"}, "female": {"alex"}})
        assert result.ok
        assert result.unwrap().iloc[0]["Gender"] == "both"
        assert any(d.code == "GENDER_NAME_BOTH" for d in result.diagnostics)
        assert all(d.severity.value == "WARNING" for d in result.diagnostics)

    def test_propn_source_takes_single_tokens(self) -> None:
        frame = _conll(
            [
                ("John", "PROPN", "PERSON", 1, 1, "a.txt"),
                ("Smith", "PROPN", "PERSON", 1, 1, "a.txt"),
            ]
        )
        result = annotate_names(frame, source="propn", names=FAKE_NAMES)
        frame = result.unwrap()
        assert set(frame["Name"]) == {"John", "Smith"}
        assert list(frame[frame["Name"] == "John"]["Gender"]) == ["male"]
        assert list(frame[frame["Name"] == "Smith"]["Gender"]) == ["unknown"]

    def test_dictionary_label_is_recorded(self) -> None:
        frame = _conll([("Mary", "PROPN", "PERSON", 1, 1, "a.txt")])
        result = annotate_names(frame, dictionary="census", names=FAKE_NAMES)
        assert result.unwrap().iloc[0]["Dictionary"] == "census"

    def test_bio_prefixed_person_tags_are_persons(self) -> None:
        frame = _conll(
            [
                ("John", "PROPN", "B-PERSON", 1, 1, "a.txt"),
                ("Smith", "PROPN", "I-PERSON", 1, 1, "a.txt"),
            ]
        )
        result = annotate_names(frame, names=FAKE_NAMES)
        assert list(result.unwrap()["Name"]) == ["John Smith"]


class TestEmptyContract:
    def test_empty_frame_is_empty_success_with_columns(self) -> None:
        empty = _person_doc().iloc[:0]
        result = annotate_names(empty, names=FAKE_NAMES)
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert list(frame.columns) == ["Document", "Document ID", "Name", "Gender", "Dictionary", "Mentions"]
        assert len(frame) == 0

    def test_empty_summary_is_empty_success_with_columns(self) -> None:
        empty = pd.DataFrame(columns=["Document", "Document ID", "Name", "Gender", "Dictionary", "Mentions"])
        result = summarize_names(empty)
        assert result.ok
        frame = result.unwrap()
        assert list(frame.columns) == ["Document", "Document ID", "Male", "Female", "Both", "Unknown", "Coverage"]
        assert len(frame) == 0


class TestLoudFailures:
    def test_missing_dictionary_without_names_dir_fails(self) -> None:
        result = annotate_names(_person_doc(), dictionary="census", names=None, names_dir=None)
        assert not result.ok
        diag = result.diagnostics[0]
        assert diag.code == "GENDER_DICT_MISSING"
        assert "<names-dir>/census/male.txt" in str(diag.context.get("fix", ""))

    def test_missing_dictionary_is_never_substituted(self) -> None:
        # nltk may well be installed; the census list must still not fall back.
        result = load_name_dictionary("social_security", names_dir=None)
        assert not result.ok
        assert result.diagnostics[0].code == "GENDER_DICT_MISSING"

    def test_absent_folder_under_names_dir_fails(self) -> None:
        result = load_name_dictionary("carnegie_mellon", names_dir=FIXTURES)
        assert not result.ok
        diag = result.diagnostics[0]
        assert diag.code == "GENDER_DICT_MISSING"
        assert "carnegie_mellon" in str(diag.context.get("path", ""))

    def test_bad_source_is_loud(self) -> None:
        result = annotate_names(_person_doc(), source="noun", names=FAKE_NAMES)
        assert not result.ok
        assert result.diagnostics[0].code == "GENDER_BAD_SOURCE"

    def test_malformed_names_override_is_loud(self) -> None:
        result = annotate_names(_person_doc(), names={"male": {"john"}})  # type: ignore[arg-type]
        assert not result.ok
        assert result.diagnostics[0].code == "GENDER_BAD_NAMES"

    def test_missing_tool_column_is_loud(self) -> None:
        frame = _person_doc().drop(columns=[Col.NER.value])
        result = annotate_names(frame, names=FAKE_NAMES)
        assert not result.ok
        assert result.diagnostics[0].code == "GENDER_MISSING_COLUMN"

    def test_noncanonical_frame_is_loud(self) -> None:
        # Tool columns all present, but the CoNLL contract is not: validate_columns speaks.
        frame = pd.DataFrame(
            {
                Col.FORM.value: ["John"],
                Col.NER.value: ["PERSON"],
                Col.SENTENCE_ID.value: [1],
                Col.DOCUMENT_ID.value: [1],
            }
        )
        result = annotate_names(frame, names=FAKE_NAMES)
        assert not result.ok
        assert result.diagnostics[0].code == "CONLL_MISSING_COLUMN"


class TestDictionaryFiles:
    def test_names_dir_files_load_with_comments_skipped(self) -> None:
        result = load_name_dictionary("census", names_dir=FIXTURES)
        assert result.ok, result.diagnostics
        names = result.unwrap()
        assert "john" in names["male"] and "mary" in names["female"]
        assert "alex" in names["male"] and "alex" in names["female"]
        assert not any(word.startswith("#") for word in names["male"] | names["female"])

    def test_annotations_work_against_the_fixture_dictionary(self) -> None:
        result = annotate_names(_person_doc(), dictionary="census", names_dir=FIXTURES)
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        john = frame[frame["Name"] == "John Smith"].iloc[0]
        assert john["Gender"] == "male"
        assert frame[frame["Name"] == "Mary"].iloc[0]["Gender"] == "female"


class TestSummary:
    def test_counts_are_mentions_and_coverage_is_known_over_total(self) -> None:
        result = annotate_names(_person_doc(), names=FAKE_NAMES)
        summary = summarize_names(result.unwrap())
        assert summary.ok, summary.diagnostics
        row = summary.unwrap().iloc[0]
        # John x1 + "John Smith" x1 (male) + Mary x1 (female): 3 mentions, all known.
        assert row["Male"] == 2
        assert row["Female"] == 1
        assert row["Both"] == 0
        assert row["Unknown"] == 0
        assert row["Coverage"] == 1.0

    def test_coverage_counts_both_as_known_and_unknown_as_not(self) -> None:
        frame = _conll(
            [
                ("Alex", "PROPN", "PERSON", 1, 1, "a.txt"),
                ("Zzqx", "PROPN", "PERSON", 1, 1, "a.txt"),
                ("Zzqx", "PROPN", "PERSON", 1, 1, "a.txt"),
            ]
        )
        result = annotate_names(frame, source="propn", names={"male": {"alex"}, "female": {"alex"}})
        row = summarize_names(result.unwrap()).unwrap().iloc[0]
        assert row["Both"] == 1
        assert row["Unknown"] == 2
        assert row["Coverage"] == round(1 / 3, 4)
