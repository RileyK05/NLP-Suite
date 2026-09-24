"""FR-1.6 SPEC_ONLY registry — every un-golden-able capability has a contract.

A SPEC_ONLY entry is the Gate B substitute for a golden: it says why no
valid golden can be produced and what behavior + evidence stands in. An
entry with an empty basis or contract is a placeholder, not evidence, and
these tests reject placeholders.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.result import Severity
from core.spec_only import (
    SPEC_ONLY_REGISTRY,
    SpecCategory,
    check_consistency,
    get_spec,
    spec_capability_ids,
)


class TestRegistryShape:
    def test_covers_exactly_the_owed_s_rows(self) -> None:
        # Scope gate: adding a SPEC_ONLY capability means updating the
        # ledger Oracle column AND this set together, never one alone.
        assert set(spec_capability_ids()) == {"CAP-PARSE-03", "CAP-PARSE-04"}

    def test_ids_are_unique(self) -> None:
        assert len(spec_capability_ids()) == len(SPEC_ONLY_REGISTRY)

    def test_lookup_misses_return_none(self) -> None:
        assert get_spec("CAP-NOPE-00") is None
        assert get_spec("CAP-PARSE-03") is not None

    def test_no_entry_is_a_placeholder(self) -> None:
        for entry in SPEC_ONLY_REGISTRY:
            assert entry.capability_id.startswith("CAP-")
            assert isinstance(entry.category, SpecCategory)
            assert len(entry.reason) > 40, entry.capability_id
            assert len(entry.legacy_refs) >= 1, entry.capability_id
            assert len(entry.basis) >= 1, entry.capability_id
            assert len(entry.contract) > 40, entry.capability_id
            assert len(entry.evidence) >= 1, entry.capability_id

    def test_legacy_refs_look_like_repo_paths(self) -> None:
        for entry in SPEC_ONLY_REGISTRY:
            for ref in entry.legacy_refs:
                assert ref.endswith(".py") or ref.startswith("lib/"), ref


class TestSeededEntries:
    def test_corenlp_is_unreachable_with_server_contract(self) -> None:
        entry = get_spec("CAP-PARSE-03")
        assert entry is not None
        assert entry.category is SpecCategory.CAPTURE_PENDING
        assert "9000" in entry.contract or "server" in entry.contract.lower()
        assert any("Stanford_CoreNLP_util" in ref for ref in entry.legacy_refs)

    def test_srl_is_unreachable_with_isolated_env_contract(self) -> None:
        entry = get_spec("CAP-PARSE-04")
        assert entry is not None
        assert entry.category is SpecCategory.UNREACHABLE
        assert "3.8" in entry.contract or "isolat" in entry.contract.lower()
        assert any("SRL_worker" in ref for ref in entry.legacy_refs)


class TestConsistencyCheck:
    def test_clean_registry_passes(self) -> None:
        diags = check_consistency(SPEC_ONLY_REGISTRY, {"CAP-PARSE-03", "CAP-PARSE-04"})
        assert diags == ()

    def test_unknown_capability_is_an_error(self) -> None:
        diags = check_consistency(SPEC_ONLY_REGISTRY, {"CAP-PARSE-03"})
        assert len(diags) == 1
        assert diags[0].severity is Severity.ERROR
        assert diags[0].code == "SPEC_ONLY_UNKNOWN_CAPABILITY"
        assert "CAP-PARSE-04" in str(diags[0].context)

    def test_missing_entry_for_known_s_row_is_an_error(self) -> None:
        diags = check_consistency(SPEC_ONLY_REGISTRY, {"CAP-PARSE-03", "CAP-PARSE-04", "CAP-NEW-09"})
        assert len(diags) == 1
        assert diags[0].code == "SPEC_ONLY_MISSING_ENTRY"

    def test_duplicate_entries_are_an_error(self) -> None:
        entry = get_spec("CAP-PARSE-03")
        assert entry is not None
        diags = check_consistency(
            (*SPEC_ONLY_REGISTRY, entry),
            {"CAP-PARSE-03", "CAP-PARSE-04"},
        )
        assert any(d.code == "SPEC_ONLY_DUPLICATE" for d in diags)


class TestC616Corrections:
    """C6-16: semantic consistency over prose length; evidence separation."""

    def test_corenlp_reason_records_the_probe(self) -> None:
        entry = get_spec("CAP-PARSE-03")
        assert entry is not None
        # The reason must name the actual environment evidence, not a
        # blanket "no server exists" claim — the reviewer verified Java 21
        # + CoreNLP 4.5.8 exist on the dev machine and a probe returned 200.
        assert "HTTP 200" in entry.reason
        assert "capture" in entry.reason.lower()

    def test_every_cited_legacy_path_exists(self) -> None:
        oracle = Path(__file__).resolve().parent.parent.parent / "NLP-Suite-1.6.38"
        if not oracle.is_dir():
            pytest.skip("oracle checkout absent")
        for entry in SPEC_ONLY_REGISTRY:
            for ref in entry.legacy_refs:
                assert (oracle / ref).exists(), f"{entry.capability_id}: {ref}"

    def test_contract_semantics_not_length(self) -> None:
        for capability in ("CAP-PARSE-03", "CAP-PARSE-04"):
            entry = get_spec(capability)
            assert entry is not None
            assert "PIPELINE_NOT_CONFIGURED" in entry.contract
            assert entry.category is not None and entry.contract != entry.reason
