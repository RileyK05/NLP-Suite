"""The model registry, where model files live, and the ONNX backends.

Every backend test runs real ONNX Runtime code against the tiny
random-weight models in tests/fixtures/models (``tiny_models`` installs one
under a real registry id). The weights mean nothing, so these tests check
plumbing — shapes, alignment, pooling, batching, truncation, caching and
the diagnostics a reader sees — never what a vector says. What the real
models say is checked in test_models_real.py, against real exports.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from core.models import locate, onnx_backend, registry
from core.models.align import plan_requests

pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

CONTEXT = "the bank raised rates and the river bank flooded"


def conll(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """Canonical parse frame from (doc_id, sent_id, form, lemma)."""
    records = []
    for index, (doc, sent, form, lemma) in enumerate(rows, 1):
        records.append(
            {
                "ID": index,
                "Form": form,
                "Lemma": lemma,
                "POS": "NN",
                "NER": "O",
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": sent,
                "Document ID": doc,
                "Document": f"doc{doc}.txt",
            }
        )
    return pd.DataFrame(records)


def words(doc: str, sent: str, text: str, lemmas: dict[str, str] | None = None) -> list[tuple[str, str, str, str]]:
    return [(doc, sent, form, (lemmas or {}).get(form, form)) for form in text.split()]


class TestRegistry:
    def test_ids_and_aliases_resolve(self) -> None:
        assert registry.get_model("bert-base-uncased").id == "bert-base-uncased"  # type: ignore[union-attr]
        assert registry.get_model("google-bert/bert-base-uncased").id == "bert-base-uncased"  # type: ignore[union-attr]
        # Saved runs carry the old Hugging Face name of the sentiment model.
        old = registry.get_model("distilbert-base-uncased-finetuned-sst-2-english")
        assert old is not None and old.id == "distilbert-sst2"
        assert registry.get_model("Qwen/Qwen3-Embedding-0.6B").id == "qwen3-embedding-0.6b"  # type: ignore[union-attr]
        assert registry.get_model("no-such-model") is None

    def test_every_spec_is_pinned_and_licensed(self) -> None:
        for spec in registry.MODELS:
            assert re.fullmatch(r"[0-9a-f]{40}", spec.revision), spec.id
            assert spec.license == "Apache-2.0", spec.id
            assert spec.description and spec.display_name
            if spec.kind == "classifier":
                assert len(spec.labels) >= 2
            if spec.kind == "token_embeddings":
                assert spec.hidden_states > 1

    def test_kinds_filter(self) -> None:
        kinds = {spec.kind for spec in registry.models_of_kind("sentence_embeddings")}
        assert kinds == {"sentence_embeddings"}
        assert len(registry.models_of_kind()) == len(registry.MODELS)

    def test_release_assets_are_flat_names(self) -> None:
        spec = registry.get_model("qwen3-embedding-0.6b")
        assert spec is not None
        url = registry.release_url(spec, "model.onnx")
        assert url.endswith("/models-1/qwen3-embedding-0.6b--model.onnx")
        assert url.startswith("https://github.com/")

    def test_published_models_record_every_file(self) -> None:
        for spec in registry.MODELS:
            if not spec.published:
                continue
            names = {item.name for item in spec.files}
            assert {"model.onnx", "tokenizer.json", "LICENSE", "NOTICE", "spec.json"} <= names, spec.id
            assert all(re.fullmatch(r"[0-9a-f]{64}", item.sha256) for item in spec.files)
            assert spec.precision in ("q4", "q8e4", "int8", "q8", "fp16")


class TestLocate:
    def test_override_is_the_only_place_searched(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("NLP_SUITE_MODELS", str(tmp_path))
        assert locate.search_roots() == [tmp_path]
        monkeypatch.delenv("NLP_SUITE_MODELS")
        monkeypatch.setenv("NLP_SUITE_MODELS_DIR", str(tmp_path / "user"))
        roots = locate.search_roots()
        assert roots[-1] == tmp_path / "user"
        assert roots[0] == locate.bundled_models_dir()

    def test_status_reads_files_and_sizes(self, tiny_models: Any, tmp_path: Path) -> None:
        spec = tiny_models("bert-base-uncased")
        assert locate.status(spec) == "ready"
        folder = locate.model_dir(spec)
        assert folder is not None and (folder / "model.onnx").is_file()
        # A file of the wrong size (an interrupted copy) is corrupt, not ready.
        (folder / "tokenizer.json").write_text("{}", encoding="utf-8")
        assert locate.status(spec) == "corrupt"
        assert locate.model_dir(spec) is None
        # Nothing there at all is simply not downloaded.
        for path in folder.iterdir():
            path.unlink()
        assert locate.status(spec) == "not_downloaded"

    def test_unexported_model_is_unpublished(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.models import _files

        spec = registry.get_model("qwen3-embedding-0.6b")
        assert spec is not None
        monkeypatch.delitem(_files.FILES, spec.id, raising=False)
        assert locate.status(spec) == "unpublished"
        assert locate.model_dir(spec) is None


class TestAlignment:
    def test_nth_request_reads_nth_occurrence(self) -> None:
        plan = plan_requests(["bank", "bank", "bank"], [CONTEXT, CONTEXT, CONTEXT])
        assert len(plan.contexts) == 1
        # "bank" is words 1 and 7; a third ask wraps to the first.
        assert [position for _, position in plan.requests] == [1, 7, 1]

    def test_absent_word_reads_the_sentence(self) -> None:
        plan = plan_requests(["be"], ["he was here"])
        assert plan.requests == [(0, None)]

    def test_case_insensitive_and_grouped(self) -> None:
        plan = plan_requests(["Bank", "river"], [CONTEXT, "the river"])
        assert plan.requests == [(0, 1), (1, 1)]
        assert plan.by_slot == {0: [0], 1: [1]}

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="pair up"):
            plan_requests(["a"], [])


class TestTokenBackend:
    def backend(self, tiny_models: Any) -> onnx_backend.OnnxTokenBackend:
        tiny_models("bert-base-uncased")
        opened = onnx_backend.open_model("bert-base-uncased", "token_embeddings")
        assert opened.ok, opened.diagnostics
        backend = opened.unwrap()
        assert isinstance(backend, onnx_backend.OnnxTokenBackend)
        return backend

    def test_unit_vectors_of_the_model_width(self, tiny_models: Any) -> None:
        backend = self.backend(tiny_models)
        vectors = np.array(backend.embed(["bank", "river"], [CONTEXT, CONTEXT]))
        assert vectors.shape == (2, 32) and backend.dimension() == 32
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_each_occurrence_reads_its_own_pieces(self, tiny_models: Any) -> None:
        backend = self.backend(tiny_models)
        first, second = np.array(backend.embed(["bank", "bank"], [CONTEXT, CONTEXT]))
        assert float(first @ second) < 0.999

    def test_batching_and_padding_do_not_change_a_vector(self, tiny_models: Any) -> None:
        backend = self.backend(tiny_models)
        alone = np.array(backend.embed(["bank"], ["the bank"]))[0]
        batched = np.array(backend.embed(["river", "bank", "rates"], [CONTEXT, "the bank", CONTEXT]))[1]
        assert float(alone @ batched) > 0.9999

    def test_layers_are_selectable_and_bounded(self, tiny_models: Any) -> None:
        backend = self.backend(tiny_models)
        first = np.array(backend.embed_layer(["bank"], [CONTEXT], 0))[0]
        last = np.array(backend.embed_layer(["bank"], [CONTEXT], -1))[0]
        assert not np.allclose(first, last)
        assert np.allclose(last, np.array(backend.embed(["bank"], [CONTEXT]))[0])
        with pytest.raises(ValueError, match="hidden states"):
            backend.embed_layer(["bank"], [CONTEXT], 3)

    def test_long_sentences_are_counted_as_truncated(self, tiny_models: Any) -> None:
        backend = self.backend(tiny_models)
        long = " ".join(["the bank"] * 50)
        vectors = backend.embed(["bank", "bank"], [long, "the bank"])
        assert len(vectors) == 2 and backend.truncated == 1

    def test_texts_embed_as_one_vector_each(self, tiny_models: Any) -> None:
        backend = self.backend(tiny_models)
        texts = backend.embed_texts(["the bank", CONTEXT, "rates"])
        assert texts.shape == (3, 32)
        assert backend.embed_texts([]).shape == (0, 32)
        assert backend.embed([], []) == []

    def test_sessions_are_cached(self, tiny_models: Any) -> None:
        one = self.backend(tiny_models)
        two = onnx_backend.open_model("bert-base-uncased").unwrap()
        assert one._session is two._session


class TestSentenceBackend:
    def test_pooling_batching_and_query_prompt(self, tiny_models: Any) -> None:
        spec = tiny_models("qwen3-embedding-0.6b")
        assert spec.pooling == "last_token" and spec.query_prompt
        backend = onnx_backend.open_model("qwen3-embedding-0.6b").unwrap()
        both = backend.embed_texts(["the bank", "we sat on the river bank"])
        alone = backend.embed_texts(["the bank"])
        assert both.shape == (2, 32)
        assert float(both[0] @ alone[0]) > 0.9999
        # A query is read with the model's instruction in front of it.
        assert float(backend.embed_texts(["the bank"], query=True)[0] @ alone[0]) < 0.999

    def test_cls_pooling(self, tiny_models: Any) -> None:
        tiny_models("granite-embedding-english-r2")
        backend = onnx_backend.open_model("granite").unwrap()
        vectors = backend.embed_texts(["the bank", "rates"])
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


class TestClassifier:
    def test_labels_probabilities_and_pipeline_shape(self, tiny_models: Any) -> None:
        tiny_models("distilbert-sst2")
        classifier = onnx_backend.open_model("distilbert-base-uncased-finetuned-sst-2-english", "classifier").unwrap()
        verdicts = classifier.classify(["i love this movie", "terrible film", "fine"])
        assert len(verdicts) == 3
        assert all(label in ("NEGATIVE", "POSITIVE") and 0.5 <= score <= 1.0 for label, score in verdicts)
        answer = classifier("i love this movie")
        assert answer[0]["label"] in ("NEGATIVE", "POSITIVE") and "score" in answer[0]


class TestOpenModelDiagnostics:
    def test_unknown_model(self) -> None:
        opened = onnx_backend.open_model("no-such-model")
        assert opened.diagnostics[0].code == "MODEL_UNKNOWN"

    def test_wrong_kind(self, tiny_models: Any) -> None:
        tiny_models("granite-embedding-english-r2")
        opened = onnx_backend.open_model("granite-embedding-english-r2", "classifier")
        assert opened.diagnostics[0].code == "MODEL_WRONG_KIND"

    def test_not_installed_says_where_to_get_it(self) -> None:
        spec = registry.get_model("qwen3-embedding-0.6b")
        assert spec is not None
        opened = onnx_backend.open_model(spec.id)
        diag = opened.diagnostics[0]
        assert diag.code == "MODEL_NOT_INSTALLED"
        assert "Open Models" in diag.message
        assert "Models" in str(diag.context.get("fix", ""))
        assert "pip" not in diag.message

    def test_corrupt_install(self, tiny_models: Any) -> None:
        spec = tiny_models("bert-base-uncased")
        folder = locate.model_dir(spec)
        assert folder is not None
        (folder / "model.onnx").write_bytes(b"broken")
        assert onnx_backend.open_model(spec.id).diagnostics[0].code == "MODEL_CORRUPT"

    def test_missing_runtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(onnx_backend, "runtime_missing", lambda: ["onnxruntime"])
        opened = onnx_backend.open_model("bert-base-uncased")
        assert opened.diagnostics[0].code == "MODEL_RUNTIME_MISSING"


class TestAnalysesRunOnOnnx:
    """The analyses reach the ONNX backend on their own, with no backend injected."""

    FRAME = conll(
        words("1", "1", "the bank raised rates", {"raised": "raise", "rates": "rate"})
        + words("1", "2", "we sat on the river bank", {"sat": "sit"})
        + words("2", "1", "the bank was big", {"was": "be"})
        + words("2", "2", "the river was long", {"was": "be"})
    )

    def test_default_backend_prefers_the_installed_model(self, tiny_models: Any) -> None:
        from core.analysis.contextual import default_backend

        tiny_models("bert-base-uncased")
        resolved = default_backend("bert-base-uncased")
        assert isinstance(resolved.unwrap(), onnx_backend.OnnxTokenBackend)

    def test_missing_model_without_transformers_names_the_models_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.analysis.contextual import default_backend

        monkeypatch.setitem(sys.modules, "transformers", None)
        resolved = default_backend("bert-base-uncased")
        assert not resolved.ok
        assert resolved.diagnostics[0].code == "MODEL_NOT_INSTALLED"
        assert "Open Models" in resolved.diagnostics[0].message

    def test_contextual_vectors_and_senses(self, tiny_models: Any) -> None:
        from core.analysis.contextual import contextual_vectors, wsi_senses
        from core.conll.schema import Col

        tiny_models("bert-base-uncased")
        vectors = contextual_vectors(self.FRAME, field=Col.LEMMA, model="bert-base-uncased")
        assert vectors.ok, vectors.diagnostics
        table = vectors.unwrap()
        assert len(table) == len(self.FRAME)
        # "be" never occurs verbatim; it is read at "was", its surface form,
        # not smeared over the whole sentence.
        forms = contextual_vectors(self.FRAME, field=Col.FORM, model="bert-base-uncased").unwrap()
        be = table[table["Word"] == "be"]["Vector"].tolist()
        was = forms[forms["Word"] == "was"]["Vector"].tolist()
        assert len(be) == 2 and be == was
        senses = wsi_senses(self.FRAME, vectors=table)
        # Content words get senses; function words ("be", "the") have none to find.
        assert senses.ok and set(senses.unwrap()["Lemma"]) == {"bank", "river"}

    def test_word2vec_bert_reads_forms_for_lemmas_and_any_layer(self, tiny_models: Any) -> None:
        from core.analysis.word2vec_bert import train_bert

        tiny_models("bert-base-uncased")
        last = train_bert(self.FRAME, field="lemma", min_count=1)
        assert last.ok, last.diagnostics
        second = train_bert(self.FRAME, field="lemma", min_count=1, layers=1)
        assert second.ok, second.diagnostics
        space = last.unwrap()
        assert "be" in space.words and space.vector_size == 32
        assert space.vectors != second.unwrap().vectors

    def test_bert_sentiment_scores_in_batches(self, tiny_models: Any) -> None:
        from core.analysis.sentiment_neural import bert_sentences

        tiny_models("distilbert-sst2")
        scored = bert_sentences(self.FRAME)
        assert scored.ok, scored.diagnostics
        table = scored.unwrap()
        assert len(table) == 4
        assert set(table["Label"]) <= {"NEGATIVE", "POSITIVE"}
        assert table["Compound"].abs().le(1).all()

    def test_bert_sentiment_names_the_models_page_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.analysis.sentiment_neural import bert_sentences

        monkeypatch.setitem(sys.modules, "transformers", None)
        scored = bert_sentences(self.FRAME)
        assert not scored.ok
        assert "Open Models" in scored.diagnostics[0].message

    @pytest.mark.parametrize("model", ["bert-base-uncased", "granite-embedding-english-r2"])
    def test_topics_and_summaries_take_token_or_sentence_models(self, tiny_models: Any, model: str) -> None:
        from core.analysis.bert_extract import summarize
        from core.analysis.bert_topics import bert_topics

        tiny_models(model)
        summary = summarize(self.FRAME, sentences=1, model=model)
        assert summary.ok, summary.diagnostics
        assert len(summary.unwrap()) == 2
        topics = bert_topics(self.FRAME, n_topics=2, model=model)
        assert topics.ok, topics.diagnostics
        assert set(topics.unwrap()[1]["Topic"]) <= {0, 1}

    def test_summaries_refuse_a_classifier(self, tiny_models: Any) -> None:
        from core.analysis.bert_extract import summarize

        tiny_models("distilbert-sst2")
        result = summarize(self.FRAME, model="distilbert-sst2")
        assert not result.ok
        assert result.diagnostics[0].code == "MODEL_WRONG_KIND"


class TestAvailability:
    """What the tool list tells the app about the BERT tools."""

    BERT_TOOLS = ("word_sense_induction", "bert_extract", "bert_topics", "sentiment_neural_bert", "word2vec_bert")
    #: A frozen app: a parser and the model runtime, no torch or transformers.
    FROZEN: ClassVar[dict[str, bool]] = {"spacy": True, "en_core_web_sm": True, "onnxruntime": True, "tokenizers": True}

    def state(self, tool: str, found: dict[str, bool]) -> dict[str, Any]:
        from core.profiler.registry import get_tool
        from desktop_backend.catalog import desktop_spec
        from desktop_backend.environment import availability

        spec = get_tool(tool)
        assert spec is not None
        return availability(desktop_spec(spec), found)

    def test_every_bert_tool_is_available_with_its_model(self, tiny_models: Any) -> None:
        for model in ("bert-base-uncased", "distilbert-sst2"):
            tiny_models(model)
        for tool in self.BERT_TOOLS:
            assert self.state(tool, self.FROZEN)["state"] == "available", tool

    def test_a_missing_model_is_a_download_not_a_dead_end(self) -> None:
        answer = self.state("bert_topics", self.FROZEN)
        assert answer["state"] == "needs_model"
        assert answer["model_id"] == "bert-base-uncased"
        assert "Download it from Models" in answer["message"]

    def test_a_missing_runtime_is_setup(self) -> None:
        answer = self.state("bert_topics", {"spacy": True, "en_core_web_sm": True})
        assert answer["state"] == "needs_setup"
        assert set(answer["missing"]) == {"onnxruntime", "tokenizers"}

    def test_a_source_checkout_with_transformers_needs_nothing_more(self) -> None:
        found = {"spacy": True, "en_core_web_sm": True, "torch": True, "transformers": True}
        assert self.state("bert_topics", found)["state"] == "available"

    def test_the_bert_tools_are_published_on_the_desktop(self) -> None:
        from desktop_backend.catalog import INTERNAL_TOOLS
        from desktop_backend.runner import DESKTOP_TOOLS

        for tool in self.BERT_TOOLS:
            assert tool in DESKTOP_TOOLS and tool not in INTERNAL_TOOLS, tool

    def test_old_model_names_still_validate(self) -> None:
        from core.profiler.plan import validate_parameters
        from core.profiler.registry import get_tool

        spec = get_tool("sentiment_neural_bert")
        assert spec is not None
        filled = validate_parameters(spec, {"model": "distilbert-base-uncased-finetuned-sst-2-english"})
        assert filled.unwrap()["model"] == "distilbert-sst2"


class TestLiveBudget:
    """A model tool over a whole corpus is refused on the bench, with the way forward."""

    @staticmethod
    def sentences(count: int) -> pd.DataFrame:
        return conll([("1", str(index), "word", "word") for index in range(count)])

    def test_a_small_selection_runs_live(self) -> None:
        from desktop_backend.live import LIVE_MODEL_SENTENCES, model_budget

        assert model_budget("bert_topics", self.sentences(LIVE_MODEL_SENTENCES)) is None

    def test_a_whole_corpus_is_sent_to_a_background_run(self) -> None:
        from desktop_backend.live import LIVE_MODEL_SENTENCES, model_budget

        refused = model_budget("sentiment_neural_bert", self.sentences(LIVE_MODEL_SENTENCES + 1))
        assert refused is not None and refused.code == "LIVE_MODEL_TOO_LARGE"
        assert "Analyze & visualize" in refused.message

    def test_other_tools_are_not_limited(self) -> None:
        from desktop_backend.live import model_budget

        assert model_budget("ngrams", self.sentences(20_000)) is None
