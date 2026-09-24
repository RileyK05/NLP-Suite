"""FR-5.1 Stanza production backend — tokenize/POS/lemma/depparse/NER on English.

Needs the downloaded stanza ``en`` model; skips (never downloads) without
it. Other config-listed languages stay unverified until their models are
backed the same way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.io.reader import Corpus, Document, hash_text
from core.pipelines.stanza_backend import build_stanza_pipeline


def _model_dir() -> Path | None:
    try:
        import stanza

        from core.pipelines.stanza_backend import build_stanza_pipeline

        if build_stanza_pipeline("en", frozenset()).value is None:
            return None
        return Path(stanza.resources.common.DEFAULT_MODEL_DIR) / "en"
    except ImportError:
        return None  # stanza absent means no model present either
    except Exception:
        return None


needs_stanza_en = pytest.mark.model_integration
_stanza_skip = pytest.mark.skipif(_model_dir() is None, reason="stanza en model not downloaded here")


def _corpus(text: str) -> Corpus:
    doc = Document(doc_id=1, path=Path("a.txt"), text=text, sha256=hash_text(text))
    return Corpus(docs=(doc,), sha256="x")


def _parse(text: str):  # type: ignore[no-untyped-def]
    pipeline = build_stanza_pipeline("en", frozenset()).unwrap()
    return pipeline.parse(_corpus(text)).unwrap()


class TestEnglishPipeline:
    @needs_stanza_en
    @_stanza_skip
    def test_tokenize_pos_lemma_dependencies(self) -> None:
        frame = _parse("The president went to Italy.")
        forms = frame["Form"].tolist()
        assert forms == ["The", "president", "went", "to", "Italy", "."]
        italy = frame[frame["Form"] == "Italy"].iloc[0]
        assert italy["Lemma"] == "Italy"
        assert italy["POS"] != ""
        assert int(italy["Head"]) == 3  # headed by "went"

    @needs_stanza_en
    @_stanza_skip
    def test_ner_tags_entities_canonical(self) -> None:
        """C6-13: NER is canonical bare-type (spaCy convention), not BIOES."""
        frame = _parse("Barack Obama visited Italy in 2009.")
        tags = dict(zip(frame["Form"].tolist(), frame["NER"].tolist(), strict=True))
        assert tags["Barack"] == "PERSON" and tags["Obama"] == "PERSON"
        assert tags["Italy"] == "GPE"
        assert tags["visited"] == "O"

    @needs_stanza_en
    @_stanza_skip
    def test_empty_doc_warns_and_continues(self) -> None:
        pipeline = build_stanza_pipeline("en", frozenset()).unwrap()
        result = pipeline.parse(_corpus("   "))
        assert any(d.code == "EMPTY_DOC" for d in result.diagnostics)

    def test_unsupported_language_fails_without_network(self) -> None:
        result = build_stanza_pipeline("xx", frozenset())
        assert result.value is None
        assert any(d.code == "PIPELINE_UNSUPPORTED" for d in result.diagnostics)

    def test_unknown_task_is_rejected(self) -> None:
        result = build_stanza_pipeline("en", frozenset({"sentiment"}))
        assert result.value is None
        assert any(d.code == "PIPELINE_BAD_TASK" for d in result.diagnostics)

    @needs_stanza_en
    @_stanza_skip
    def test_task_restricted_builds_include_prerequisites(self) -> None:
        """depparse needs pos+lemma; every processor needs tokenize.

        A task-restricted build that drops Stanza's processor prerequisites
        fails with a pipeline-requirements error and was previously
        misdiagnosed as PIPELINE_MODEL_CORRUPT.
        """
        cases = {
            frozenset({"dependency"}): {"tokenize", "pos", "lemma", "depparse"},
            frozenset({"lemma"}): {"tokenize", "pos", "lemma"},
            frozenset({"pos"}): {"tokenize", "pos"},
            frozenset({"ner"}): {"tokenize", "ner"},
        }
        for tasks, expected_processors in cases.items():
            result = build_stanza_pipeline("en", tasks)
            assert result.ok, [str(d) for d in result.diagnostics]
            processors = set(result.unwrap()._nlp.processors.keys())
            assert expected_processors <= processors, (tasks, processors)


class TestFixtureParses:
    @needs_stanza_en
    @_stanza_skip
    def test_probe_fixtures_parse(self) -> None:
        probes = Path(__file__).resolve().parent / "fixtures" / "probes"
        texts = [
            (probes / "17_dialogue.txt").read_text(encoding="utf-8"),
            (probes / "04_doc.txt").read_text(encoding="utf-8"),
        ]
        docs = tuple(
            Document(doc_id=i + 1, path=Path(f"d{i}.txt"), text=text, sha256=hash_text(text))
            for i, text in enumerate(texts)
        )
        pipeline = build_stanza_pipeline("en", frozenset()).unwrap()
        frame = pipeline.parse(Corpus(docs=docs, sha256="x")).unwrap()
        assert len(frame) > 0
        assert set(frame["Document ID"].tolist()) == {"1", "2"}
