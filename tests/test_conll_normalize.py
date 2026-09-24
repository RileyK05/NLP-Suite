"""Tagset conversion and canonical normalization.

The Universal->Penn mapping is a port of the legacy ``universal_to_penn``,
which the defect review called correct. These tests pin its behaviour so the
port cannot silently drift, and pin the three ways the new version differs
from the old one (name addressing, Result returns, convert-once).
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.conll.normalize import (
    detect_source_tagset,
    normalize_table,
    parse_feats,
    universal_to_penn,
)
from core.conll.schema import Col, canonical_columns


class TestParseFeats:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("", {}),
            ("_", {}),
            (None, {}),
            ("Number=Plur", {"Number": "Plur"}),
            ("Tense=Past|VerbForm=Fin", {"Tense": "Past", "VerbForm": "Fin"}),
            ("NoEqualsSign", {}),
            ("Number=Plur|", {"Number": "Plur"}),
        ],
    )
    def test_parsing(self, raw: object, expected: dict[str, str]) -> None:
        assert parse_feats(raw) == expected


class TestUniversalToPenn:
    @pytest.mark.parametrize(
        ("pos", "feats", "expected"),
        [
            ("NOUN", "", "NN"),
            ("NOUN", "Number=Plur", "NNS"),
            ("NOUN", "Number=Sing", "NN"),
            ("PROPN", "", "NNP"),
            ("PROPN", "Number=Plur", "NNPS"),
            ("VERB", "VerbForm=Ger", "VBG"),
            ("VERB", "VerbForm=Part|Tense=Pres", "VBG"),
            ("VERB", "VerbForm=Part|Tense=Past", "VBN"),
            ("VERB", "VerbForm=Inf", "VB"),
            ("VERB", "VerbForm=Fin|Tense=Past", "VBD"),
            ("VERB", "VerbForm=Fin|Person=3|Number=Sing", "VBZ"),
            ("VERB", "VerbForm=Fin", "VBP"),
            ("VERB", "", "VB"),
            ("AUX", "VerbForm=Fin|Tense=Past", "VBD"),
            ("ADJ", "", "JJ"),
            ("ADJ", "Degree=Cmp", "JJR"),
            ("ADJ", "Degree=Sup", "JJS"),
            ("ADV", "Degree=Cmp", "RBR"),
            ("ADV", "Degree=Sup", "RBS"),
            ("ADP", "", "IN"),
            ("DET", "", "DT"),
            ("PRON", "", "PRP"),
            ("PRON", "Poss=Yes", "PRP$"),
            ("PRON", "PronType=Int", "WP"),
            ("PRON", "PronType=Rel|Poss=Yes", "WP$"),
            ("CCONJ", "", "CC"),
            ("SCONJ", "", "IN"),
            ("NUM", "", "CD"),
            ("PART", "", "RP"),
            ("INTJ", "", "UH"),
            ("PUNCT", "", "."),
        ],
    )
    def test_mapping(self, pos: str, feats: str, expected: str) -> None:
        assert universal_to_penn(pos, feats) == expected

    @pytest.mark.parametrize("tag", ["NN", "NNS", "VBZ", "JJR", ".", "$", "WEIRD"])
    def test_penn_and_unknown_tags_pass_through(self, tag: str) -> None:
        """The mapping is idempotent — safe on already-Penn tables."""
        assert universal_to_penn(tag) == tag

    def test_is_idempotent(self) -> None:
        once = universal_to_penn("NOUN", "Number=Plur")
        assert universal_to_penn(once, "Number=Plur") == once

    def test_sym_and_x_pass_through(self) -> None:
        assert universal_to_penn("SYM") == "SYM"
        assert universal_to_penn("X") == "X"

    def test_none_input_is_survivable(self) -> None:
        assert universal_to_penn(None) == ""


class TestDetectSourceTagset:
    def test_feats_column_implies_universal(self) -> None:
        assert detect_source_tagset([*canonical_columns(), "feats"]) == "universal"

    def test_without_feats_assume_penn(self) -> None:
        assert detect_source_tagset(list(canonical_columns())) == "penn"


class TestNormalizeTable:
    def test_universal_tags_become_penn(self, universal_frame: pd.DataFrame) -> None:
        result = normalize_table(universal_frame)
        assert result.ok
        assert list(result.unwrap()[Col.POS.value]) == ["DT", "NNS", "VBD", "RBR"]

    def test_conversion_is_announced(self, universal_frame: pd.DataFrame) -> None:
        result = normalize_table(universal_frame)
        assert any(d.code == "POS_TAGSET_CONVERTED" for d in result.diagnostics)

    def test_penn_table_is_not_double_converted(self, conll_frame: pd.DataFrame) -> None:
        result = normalize_table(conll_frame)
        assert result.ok
        assert not any(d.code == "POS_TAGSET_CONVERTED" for d in result.diagnostics)

    def test_columns_end_up_canonical_then_extras(self, universal_frame: pd.DataFrame) -> None:
        result = normalize_table(universal_frame)
        columns = list(result.unwrap().columns)
        assert columns[: len(canonical_columns())] == list(canonical_columns())
        assert columns[len(canonical_columns()) :] == ["feats"]

    def test_auto_generated_columns_are_filled(self, conll_frame: pd.DataFrame) -> None:
        result = normalize_table(conll_frame)
        frame = result.unwrap()
        for column in ("Deps", "Clause Tag"):
            assert column in frame.columns
            assert (frame[column] == "").all()

    def test_filling_absent_auto_columns_is_silent(self, conll_frame: pd.DataFrame) -> None:
        """Synthesizing Deps / Clause Tag is normal operation, not an event.

        validate_columns has already rejected any table missing a *required*
        column, so a column that still needs filling at synthesis time can only
        be an auto-generated one. This frame is otherwise clean, so the whole
        normalization is silent — pins that no per-column synthesized warning
        is (re)introduced.
        """
        result = normalize_table(conll_frame)
        assert result.ok
        assert "Deps" in result.unwrap().columns
        assert "Clause Tag" in result.unwrap().columns
        assert result.diagnostics == ()

    def test_row_count_is_conserved(self, conll_frame: pd.DataFrame) -> None:
        """Property: normalization never drops or adds rows."""
        result = normalize_table(conll_frame)
        assert len(result.unwrap()) == len(conll_frame)

    def test_token_content_is_untouched(self, conll_frame: pd.DataFrame) -> None:
        result = normalize_table(conll_frame)
        assert list(result.unwrap()[Col.FORM.value]) == list(conll_frame[Col.FORM.value])

    def test_record_id_is_assigned_when_absent(self, conll_frame: pd.DataFrame) -> None:
        without = conll_frame.assign(**{Col.RECORD_ID.value: ""})
        result = normalize_table(without)
        assert result.ok
        assert list(result.unwrap()[Col.RECORD_ID.value]) == [str(i) for i in range(1, 14)]
        assert any(d.code == "RECORD_ID_ASSIGNED" for d in result.diagnostics)

    def test_existing_record_id_is_preserved(self, conll_frame: pd.DataFrame) -> None:
        """What the parser supplied is passed through untouched, including type."""
        result = normalize_table(conll_frame)
        assert list(result.unwrap()[Col.RECORD_ID.value]) == list(conll_frame[Col.RECORD_ID.value])
        assert not any(d.code == "RECORD_ID_ASSIGNED" for d in result.diagnostics)

    def test_missing_required_column_fails_with_a_result(self, conll_frame: pd.DataFrame) -> None:
        """Legacy raised, printed, showed a messagebox, and returned None."""
        broken = conll_frame.drop(columns=[Col.POS.value])
        result = normalize_table(broken)
        assert not result.ok
        assert result.value is None
        assert any(d.code == "CONLL_MISSING_COLUMN" for d in result.errors)

    def test_source_tagset_can_be_declared(self, conll_frame: pd.DataFrame) -> None:
        """A declared Penn source skips conversion even if feats are present."""
        with_feats = conll_frame.assign(feats="Number=Plur")
        result = normalize_table(with_feats, source_tagset="penn")
        assert list(result.unwrap()[Col.POS.value]) == list(conll_frame[Col.POS.value])

    def test_empty_frame_normalizes(self) -> None:
        empty = pd.DataFrame(columns=list(canonical_columns()))
        result = normalize_table(empty)
        assert result.ok
        assert len(result.unwrap()) == 0

    def test_does_not_mutate_the_input(self, universal_frame: pd.DataFrame) -> None:
        before = universal_frame[Col.POS.value].tolist()
        normalize_table(universal_frame)
        assert universal_frame[Col.POS.value].tolist() == before
