"""Pipeline cache, protocol, and spaCy backend.

Covers: protocol is runtime-checkable, cache deduplicates, unsupported
languages are rejected before a model is touched, and the no-silent-degradation
contract: a missing model is a hard failure with a fix command, never a
blank-pipeline fallback.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

try:
    import spacy
except ImportError:
    spacy = None  # type: ignore[assignment]

from core.config import NLPConfig
from core.io.reader import Corpus, Document, hash_text
from core.pipelines.cache import PipelineCache
from core.pipelines.protocol import SUPPORTED_BACKENDS, Pipeline, supports
from core.pipelines.spacy_backend import SpacyPipeline, build_spacy_pipeline, spacy_model_name
from core.result import Result


class _FakePipeline:
    backend = "fake"
    language = "en"
    tasks: frozenset[str] = frozenset()

    def parse(self, corpus: Corpus) -> Result[pd.DataFrame]:
        return Result.success(pd.DataFrame())

    def supports(self, language: str, tasks: object = ()) -> bool:
        return True

    def tokenize(self, text: str) -> tuple[str, ...]:
        return tuple(text.split())


def _corpus() -> Corpus:
    docs = (
        Document(doc_id=1, path=Path("a.txt"), text="Hello world.", sha256=hash_text("Hello world.")),
        Document(
            doc_id=2, path=Path("b.txt"), text="Another sentence here.", sha256=hash_text("Another sentence here.")
        ),
    )
    return Corpus(docs=docs, sha256="x")


class TestProtocol:
    def test_fake_pipeline_satisfies_protocol(self) -> None:
        assert isinstance(_FakePipeline(), Pipeline)

    def test_supported_backends(self) -> None:
        assert "spacy" in SUPPORTED_BACKENDS
        assert "stanza" in SUPPORTED_BACKENDS

    def test_supports_gate(self) -> None:
        assert supports("spacy", "en")
        assert not supports("spacy", "xx")
        assert not supports("unknown", "en")


class TestCache:
    def test_deduplicates(self) -> None:
        cache = PipelineCache()
        calls = {"n": 0}

        def builder(language: str, tasks: frozenset[str]) -> Result[Pipeline]:
            calls["n"] += 1
            return Result.success(_FakePipeline())

        cache.register("spacy", builder)
        first = cache.get("spacy", "en")
        second = cache.get("spacy", "en")
        assert first.ok and second.ok
        assert first.unwrap() is second.unwrap()
        assert calls["n"] == 1
        assert len(cache) == 1

    def test_different_keys_are_distinct(self) -> None:
        cache = PipelineCache()
        cache.register("spacy", lambda lang, tasks: Result.success(_FakePipeline()))
        assert cache.get("spacy", "en").ok
        assert cache.get("spacy", "de").value is not cache.get("spacy", "en").value

    def test_unsupported_language_rejected_without_building(self) -> None:
        cache = PipelineCache()
        built = {"n": 0}

        def builder(language: str, tasks: frozenset[str]) -> Result[Pipeline]:
            built["n"] += 1
            return Result.success(_FakePipeline())

        cache.register("spacy", builder)
        result = cache.get("spacy", "xx")
        assert not result.ok
        assert result.errors[0].code == "PIPELINE_UNSUPPORTED"
        assert built["n"] == 0

    def test_tasks_part_of_key(self) -> None:
        cache = PipelineCache()
        cache.register("spacy", lambda lang, tasks: Result.success(_FakePipeline()))
        assert not cache.contains("spacy", "en", tasks=["ner"])
        cache.get("spacy", "en", tasks=["ner"])
        assert cache.contains("spacy", "en", tasks=["ner"])
        assert not cache.contains("spacy", "en", tasks=["pos"])

    def test_config_mismatch_rejected(self) -> None:
        cache = PipelineCache()
        cache.register("spacy", lambda lang, tasks: Result.success(_FakePipeline()))
        config = NLPConfig(parser="spacy", language="en")
        result = cache.get("stanza", "en", config=config)
        assert not result.ok
        assert result.errors[0].code == "PIPELINE_CONFIG_MISMATCH"

    def test_no_builder_is_an_error(self) -> None:
        cache = PipelineCache()
        result = cache.get("spacy", "en")
        assert not result.ok
        assert result.errors[0].code == "PIPELINE_NO_BUILDER"


def _has_spacy_model() -> bool:
    try:
        spacy.load(spacy_model_name("en"))
        return True
    except Exception:
        return False


_requires_model = pytest.mark.model_integration
_model_skip = pytest.mark.skipif(not _has_spacy_model(), reason="spaCy model not installed on this machine")


class TestSpacyBackend:
    def test_missing_model_is_a_hard_failure_with_fix_command(self) -> None:
        # The no-silent-degradation contract: without a trained model the
        # build fails with PIPELINE_MODEL_MISSING and the exact fix. A blank
        # pipeline would strip POS/lemma/deps from every analysis while the
        # run still exited green — the worst class of failure this suite bans.
        result = build_spacy_pipeline("en", frozenset())
        if result.ok:
            # Model IS installed here: the contract is vacuously satisfied;
            # the doctor test covers the missing-model path directly.
            pytest.skip("model installed; hard-failure path not reachable")
        assert result.errors[0].code == "PIPELINE_MODEL_MISSING"
        fix = result.errors[0].context["fix"]
        assert "python -m spacy download" in fix
        assert spacy_model_name("en") in fix

    @_requires_model
    @_model_skip
    def test_build_succeeds_with_installed_model(self) -> None:
        result = build_spacy_pipeline("en", frozenset())
        assert result.ok
        assert isinstance(result.unwrap(), SpacyPipeline)
        assert result.unwrap().backend == "spacy"

    def test_build_rejects_unsupported_language(self) -> None:
        result = build_spacy_pipeline("xx", frozenset())
        assert not result.ok
        assert result.errors[0].code == "PIPELINE_UNSUPPORTED"

    @_requires_model
    @_model_skip
    def test_parse_produces_canonical_table(self) -> None:
        pipeline = build_spacy_pipeline("en", frozenset()).unwrap()
        corpus = _corpus()
        result = pipeline.parse(corpus)
        assert result.ok
        frame = result.unwrap()
        for col in ["ID", "Form", "Lemma", "POS", "Sentence ID", "Document ID", "Document"]:
            assert col in frame.columns
        assert len(frame) > 0

    @_requires_model
    @_model_skip
    def test_parse_empty_doc_is_skipped_with_diagnostic(self) -> None:
        pipeline = build_spacy_pipeline("en", frozenset()).unwrap()
        corpus = Corpus(
            docs=(
                Document(doc_id=1, path=Path("a.txt"), text="Hello.", sha256="a"),
                Document(doc_id=2, path=Path("b.txt"), text="   ", sha256="b"),
            ),
            sha256="x",
        )
        result = pipeline.parse(corpus)
        assert any(d.code == "EMPTY_DOC" for d in result.diagnostics)
        assert len(result.unwrap()) > 0

    @_requires_model
    @_model_skip
    def test_supports_delegates_to_config_gate(self) -> None:
        pipeline = build_spacy_pipeline("en", frozenset()).unwrap()
        assert pipeline.supports("en")
        assert not pipeline.supports("xx")
