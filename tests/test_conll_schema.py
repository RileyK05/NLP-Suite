"""The CoNLL table contract (ARCHITECTURE.md R9, section 2.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.conll.schema import (
    SCHEMA_VERSION,
    Col,
    CoNLLSchema,
    auto_generated_columns,
    canonical_columns,
    read_sidecar,
    required_columns,
    sidecar_path,
    validate_columns,
    write_sidecar,
)
from core.result import Severity


class TestColumns:
    def test_canonical_columns_match_the_spec(self) -> None:
        assert canonical_columns() == (
            "ID",
            "Form",
            "Lemma",
            "POS",
            "NER",
            "Head",
            "DepRel",
            "Deps",
            "Clause Tag",
            "Record ID",
            "Sentence ID",
            "Document ID",
            "Document",
        )

    def test_required_columns_are_a_subset_of_canonical(self) -> None:
        assert required_columns() < set(canonical_columns())

    def test_auto_generated_columns(self) -> None:
        assert auto_generated_columns() == {"Record ID", "Deps", "Clause Tag"}

    def test_col_stringifies_to_its_column_name(self) -> None:
        assert str(Col.CLAUSE_TAG) == "Clause Tag"
        assert Col.SENTENCE_ID.value == "Sentence ID"

    def test_columns_are_addressed_by_name(self) -> None:
        """R9: never by position."""
        assert Col.POS.value == "POS"
        assert canonical_columns().index(Col.POS.value) == 3


class TestValidateColumns:
    def test_valid_header_passes(self) -> None:
        result = validate_columns(list(canonical_columns()))
        assert result.ok

    def test_missing_required_column_is_an_error(self) -> None:
        columns = [c for c in canonical_columns() if c != "POS"]
        result = validate_columns(columns)
        assert not result.ok
        assert any(d.code == "CONLL_MISSING_COLUMN" for d in result.errors)

    def test_missing_auto_generated_column_is_fine(self) -> None:
        """Deps / Clause Tag / Record ID are synthesized downstream."""
        columns = [c for c in canonical_columns() if c not in auto_generated_columns()]
        assert validate_columns(columns).ok

    def test_duplicate_columns_are_an_error(self) -> None:
        result = validate_columns(["ID", "ID", *canonical_columns()])
        assert not result.ok
        assert any(d.code == "CONLL_DUPLICATE_COLUMN" for d in result.errors)

    def test_extra_columns_are_reported_but_not_fatal(self) -> None:
        result = validate_columns([*canonical_columns(), "feats"])
        assert result.ok
        infos = [d for d in result.diagnostics if d.severity is Severity.INFO]
        assert any(d.code == "CONLL_EXTRA_COLUMN" for d in infos)


class TestSchemaObject:
    def test_defaults(self) -> None:
        schema = CoNLLSchema(parser="stanza", parser_version="1.10")
        assert schema.schema_version == SCHEMA_VERSION
        assert schema.pos_tagset == "penn"
        assert schema.created

    def test_unsupported_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported CoNLL schema version"):
            CoNLLSchema(schema_version=999)

    def test_unknown_tagset_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown pos_tagset"):
            CoNLLSchema(pos_tagset="penn-ish")  # type: ignore[arg-type]

    def test_columns_include_extras_after_canonical(self) -> None:
        schema = CoNLLSchema(extra_columns=("feats",))
        assert schema.columns[-1] == "feats"
        assert schema.columns[:-1] == canonical_columns()

    def test_round_trip_through_dict(self) -> None:
        schema = CoNLLSchema(parser="stanza", parser_version="1.10", pos_tagset="penn", extra_columns=("feats",))
        restored = CoNLLSchema.from_dict(schema.to_dict())
        assert restored == schema

    def test_from_dict_ignores_missing_keys(self) -> None:
        schema = CoNLLSchema.from_dict({})
        assert schema.parser == ""
        assert schema.pos_tagset == "penn"


class TestSidecar:
    def test_sidecar_path_appends_schema_json(self) -> None:
        assert sidecar_path(Path("out/table.csv")) == Path("out/table.csv.schema.json")

    def test_round_trip(self, tmp_path: Path) -> None:
        table = tmp_path / "table.csv"
        table.write_text("ID,Form\n1,The\n", encoding="utf-8")
        schema = CoNLLSchema(parser="stanza", parser_version="1.10")

        written = write_sidecar(table, schema)
        assert written.ok
        assert written.unwrap().is_file()

        result = read_sidecar(table)
        assert result.ok
        assert result.unwrap().parser == "stanza"
        assert result.unwrap().parser_version == "1.10"

    def test_missing_sidecar_is_an_error_not_a_guess(self, tmp_path: Path) -> None:
        table = tmp_path / "orphan.csv"
        table.write_text("ID,Form\n1,The\n", encoding="utf-8")

        result = read_sidecar(table)
        assert not result.ok
        assert result.errors[0].code == "SIDECAR_MISSING"

    def test_malformed_json_is_reported(self, tmp_path: Path) -> None:
        table = tmp_path / "table.csv"
        table.write_text("ID,Form\n", encoding="utf-8")
        sidecar_path(table).write_text("{not json", encoding="utf-8")

        result = read_sidecar(table)
        assert not result.ok
        assert result.errors[0].code == "SIDECAR_UNREADABLE"

    def test_non_object_sidecar_is_reported(self, tmp_path: Path) -> None:
        table = tmp_path / "table.csv"
        table.write_text("ID,Form\n", encoding="utf-8")
        sidecar_path(table).write_text("[1, 2, 3]", encoding="utf-8")

        result = read_sidecar(table)
        assert not result.ok
        assert result.errors[0].code == "SIDECAR_MALFORMED"

    def test_schema_with_bad_version_is_reported(self, tmp_path: Path) -> None:
        table = tmp_path / "table.csv"
        table.write_text("ID,Form\n", encoding="utf-8")
        sidecar_path(table).write_text('{"schema_version": 99}', encoding="utf-8")

        result = read_sidecar(table)
        assert not result.ok
        assert result.errors[0].code == "SIDECAR_INVALID"
