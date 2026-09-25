"""FR-7.1 declarative tool registry — capability, params, prerequisites, outputs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from core.profiler.registry import (
    SHARED_CAPABILITIES,
    TOOL_REGISTRY,
    TOOL_REGISTRY_EXCLUSIONS,
    ParamSpec,
    ToolSpec,
    get_tool,
    tool_names,
    validate_specs,
)


class TestRegistryShape:
    def test_expected_tools_registered(self) -> None:
        # Scope gate: adding a tool registers here AND updates this set together.
        assert set(tool_names()) == {
            "doc_embeddings",
            "readability",
            "lexical_diversity",
            "doc_similarity",
            "doc_duplicates",
            "spellcheck",
            "search",
            "collocations",
            "tfidf",
            "dispersion",
            "stats_categorical",
            "stats_groups",
            "stats_trends",
            "sentence_complexity",
            "convert",
            "filenames",
            "csv_stats",
            "profiler",
            "wordnet",
            "lda_gensim",
            "lda_mallet",
            "word2vec_gensim",
            "sentiment_vader_anew",
            "sentiment_swn_hedono",
            "nrc",
            "nominalization",
            "style",
            "verbnet",
            "framenet",
            "symbolic",
            "coreference",
            "narrative",
            "charts",
            "panels",
            "lda_stability",
            "wordcloud_gephi",
            "word_sense_induction",
            "clause_svo",
            "ner",
            "bert_extract",
            "ngrams",
            "ngram_cooccurrence",
            "conll_wordlist",
            "corpus_statistics",
            "k_sentences",
            "svo_compare",
            "text_statistics",
            "table_search",
            "semantic",
            "knowledge_graph",
            "shapes",
            "kwic",
            "keyness",
            "lexicon_series",
            "bert_topics",
            "gender_annotator",
            "date_annotator",
            "quote_annotator",
            "gender_guess",
            "verb_analysis",
            "ngram_viewer",
            "sentiment_neural_bert",
            "sentiment_neural_spacy",
            "sentiment_neural_stanza",
            "sentiment_neural_corenlp",
            "shape_hc",
            "shape_svd",
            "shape_nmf",
            "geocode",
            "gis_map",
            "svo_map",
            "word2vec_bert",
        }

    def test_lookup_miss_returns_none(self) -> None:
        assert get_tool("nope") is None
        assert get_tool("readability") is not None

    def test_every_spec_is_complete(self) -> None:
        for spec in TOOL_REGISTRY:
            assert spec.name and spec.description and spec.packet
            # filenames emits stdout + in-place renames, not run-dir artifacts.
            assert spec.outputs or spec.name == "filenames", spec.name
            assert spec.version, spec.name
            for param in spec.params:
                assert param.name and param.help, (spec.name, param.name)
                assert param.type in ("str", "int", "float", "bool", "path"), param.type

    def test_parse_hunger_is_declared(self) -> None:
        raw = get_tool("readability")
        assert raw is not None and not raw.requires_parse
        parsed = get_tool("sentence_complexity")
        assert parsed is not None and parsed.requires_parse


class TestValidation:
    def test_clean_registry_passes(self) -> None:
        assert validate_specs(TOOL_REGISTRY) == ()

    def test_duplicate_names_rejected(self) -> None:
        spec = get_tool("readability")
        assert spec is not None
        diags = validate_specs((*TOOL_REGISTRY, spec))
        assert any(d.code == "REGISTRY_DUPLICATE" for d in diags)

    def test_bad_param_type_rejected(self) -> None:
        bad = ToolSpec(
            name="bad",
            capability_ids=("CAP-X-01",),
            packet="FR-X",
            description="bad",
            requires_parse=False,
            params=(ParamSpec(name="x", type="fancy", default=0, required=False, help="bad"),),
            outputs=("out.csv",),
            version="1",
        )
        assert any(d.code == "REGISTRY_BAD_PARAM" for d in validate_specs((bad,)))

    def test_missing_outputs_rejected(self) -> None:
        bad = ToolSpec(
            name="hollow",
            capability_ids=("CAP-X-01",),
            packet="FR-X",
            description="hollow",
            requires_parse=False,
            params=(),
            outputs=(),
            version="1",
        )
        assert any(d.code == "REGISTRY_NO_OUTPUTS" for d in validate_specs((bad,)))


class TestC618Completion:
    """C6-18: capability IDs, input kinds, prerequisites, metadata."""

    def test_every_tool_has_capability_ids(self) -> None:
        for spec in TOOL_REGISTRY:
            assert spec.capability_ids, spec.name

    def test_input_kinds_declared_and_valid(self) -> None:
        for spec in TOOL_REGISTRY:
            assert spec.input_kind in ("corpus", "csv", "conll", "database", "asset", "document", "directory"), (
                spec.name
            )
        assert get_tool("csv_stats").input_kind == "csv"  # type: ignore[union-attr]
        assert get_tool("profiler").input_kind == "corpus"  # type: ignore[union-attr]

    def test_parse_claim_carries_backend(self) -> None:
        for spec in TOOL_REGISTRY:
            if spec.requires_parse:
                assert spec.parser_backend, spec.name

    def test_profiler_metadata_present(self) -> None:
        for spec in TOOL_REGISTRY:
            assert isinstance(spec.profiler_eligible, bool)
            assert 0 <= spec.execution_phase <= 3, spec.name

    def test_intake_tools_not_profiler_eligible(self) -> None:
        convert = get_tool("convert")
        filenames = get_tool("filenames")
        assert convert is not None and not convert.profiler_eligible
        assert filenames is not None and not filenames.profiler_eligible

    def test_unknown_input_kind_rejected(self) -> None:
        bad = ToolSpec(
            name="odd",
            capability_ids=("CAP-X-01",),
            packet="FR-X",
            description="odd",
            requires_parse=False,
            params=(),
            outputs=("o.csv",),
            version="1",
            input_kind="telepathy",
        )
        assert any(d.code == "REGISTRY_BAD_INPUT_KIND" for d in validate_specs((bad,)))

    def test_parse_without_backend_rejected(self) -> None:
        bad = ToolSpec(
            name="p",
            capability_ids=("CAP-X-01",),
            packet="FR-X",
            description="p",
            requires_parse=True,
            params=(),
            outputs=("o.csv",),
            version="1",
        )
        assert any(d.code == "REGISTRY_PARSE_NO_BACKEND" for d in validate_specs((bad,)))

    def test_capability_ids_resolve_to_ledger(self) -> None:
        """Registry-vs-ledger: declared capability IDs must exist in the ledger."""
        from pathlib import Path

        ledger = Path(__file__).resolve().parent.parent / "docs" / "REPLACEMENT_LEDGER.md"
        text = ledger.read_text(encoding="utf-8")
        for spec in TOOL_REGISTRY:
            for cap in spec.capability_ids:
                assert cap in text, f"{spec.name}: {cap} not in ledger"

    def test_capability_ids_have_one_owner(self) -> None:
        """A capability ID is the ledger join key: two owners reassigns a row.

        Genuine two-entry-point capabilities are named in SHARED_CAPABILITIES
        with a reason, so an accidental copy-paste cannot hide among them.
        """
        owners: dict[str, list[str]] = {}
        for spec in TOOL_REGISTRY:
            for capability in spec.capability_ids:
                owners.setdefault(capability, []).append(spec.name)
        contested = {cap: names for cap, names in owners.items() if len(names) > 1}
        assert set(contested) == set(SHARED_CAPABILITIES), contested
        for capability, reason in SHARED_CAPABILITIES.items():
            assert reason.strip(), capability

    def test_validate_specs_rejects_undeclared_capability_sharing(self) -> None:
        base = get_tool("readability")
        assert base is not None
        # Two distinct specs claiming one ID that is NOT on the allowlist.
        clash = replace(base, name="readability_clone", capability_ids=base.capability_ids)
        codes = [d.code for d in validate_specs([base, clash])]
        assert "REGISTRY_DUPLICATE_CAPABILITY" in codes

    def test_validate_specs_allows_declared_capability_sharing(self) -> None:
        shared = next(iter(SHARED_CAPABILITIES))
        owners = [s for s in TOOL_REGISTRY if shared in s.capability_ids]
        assert len(owners) > 1, shared
        codes = [d.code for d in validate_specs(owners)]
        assert "REGISTRY_DUPLICATE_CAPABILITY" not in codes

    def test_search_modes_modeled(self) -> None:
        search = get_tool("search")
        assert search is not None
        # conll mode requires a parse: the mode is modeled as a parameter with
        # choices, and the registry notes the per-mode requirement honestly.
        modes = [p for p in search.params if p.name == "mode"]
        assert modes and "conll" in str(modes[0].help)

    def test_every_cli_is_registered_or_explicitly_excluded(self) -> None:
        cli_names = {
            path.stem
            for path in (Path(__file__).resolve().parent.parent / "tools").glob("*.py")
            if path.stem not in {"_cli", "__init__"}
        }
        covered = set(tool_names()) | {name for name, _reason in TOOL_REGISTRY_EXCLUSIONS}
        assert cli_names == covered
