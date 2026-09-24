"""Falling back to a working parser when the configured one's model is not.

Found by running the doctor on this machine: Stanza's model directory was
partially downloaded (an interrupted ``stanza.download`` leaves exactly this),
so the default parse path failed -- and with it 24 of the 29 desktop tools --
while a complete, working spaCy model sat installed and unused.

The risk in fixing that is quietly swapping the parser under someone and
letting them compare results produced by two different tokenizers without
knowing. So the tests here are as much about what must *not* happen: no
fallback on a non-model failure, no silent substitution, and no run record
that names a parser which never touched the text.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.config import NLPConfig
from core.io.reader import Corpus
from core.pipelines.cache import PipelineCache
from core.pipelines.resolve import FALLBACK_ORDER, available_backends, register_builders, resolve_pipeline
from core.result import Diagnostic, Result


class FakePipeline:
    """Shaped like the Pipeline protocol, with no model behind it."""

    def __init__(self, backend: str, language: str = "en") -> None:
        self._backend = backend
        self._language = language

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def language(self) -> str:
        return self._language

    @property
    def tasks(self) -> frozenset[str]:
        return frozenset()

    def parse(self, corpus: Corpus) -> Result[pd.DataFrame]:
        return Result.success(pd.DataFrame({"Document": [d.path.name for d in corpus.docs]}))

    def supports(self, language: str, tasks: object = ()) -> bool:
        return language == self._language


def _works(backend: str):  # type: ignore[no-untyped-def]
    def build(language: str, tasks: frozenset[str]) -> Result[FakePipeline]:
        return Result.success(FakePipeline(backend, language))

    return build


def _model_missing(backend: str, code: str = "PIPELINE_MODEL_MISSING"):  # type: ignore[no-untyped-def]
    def build(language: str, tasks: frozenset[str]) -> Result[FakePipeline]:
        return Result.failure(
            Diagnostic.error(code, f"{backend} model for {language!r} is not installed", fix=f"install {backend}")
        )

    return build


def _other_failure(backend: str):  # type: ignore[no-untyped-def]
    def build(language: str, tasks: frozenset[str]) -> Result[FakePipeline]:
        return Result.failure(Diagnostic.error("PIPELINE_BUILD_FAILED", f"{backend} blew up for another reason"))

    return build


def _cache(**builders: object) -> PipelineCache:
    cache = PipelineCache()
    for backend, builder in builders.items():
        cache.register(backend, builder)  # type: ignore[arg-type]
    return cache


CONFIG = NLPConfig(parser="stanza", language="en")


class TestFallbackHappens:
    def test_a_damaged_default_model_falls_back_to_a_working_backend(self) -> None:
        result = resolve_pipeline(_cache(stanza=_model_missing("stanza"), spacy=_works("spacy")), CONFIG)
        assert result.ok
        assert result.unwrap().backend == "spacy"

    def test_the_substitution_is_announced_not_silent(self) -> None:
        result = resolve_pipeline(_cache(stanza=_model_missing("stanza"), spacy=_works("spacy")), CONFIG)
        warning = next(d for d in result.diagnostics if d.code == "PARSER_FALLBACK")
        assert warning.context["requested"] == "stanza"
        assert warning.context["used"] == "spacy"
        assert "may differ" in warning.message

    def test_the_warning_carries_the_fix_for_the_broken_backend(self) -> None:
        """A fallback keeps work moving; it must not make the repair invisible."""
        result = resolve_pipeline(_cache(stanza=_model_missing("stanza"), spacy=_works("spacy")), CONFIG)
        warning = next(d for d in result.diagnostics if d.code == "PARSER_FALLBACK")
        assert warning.context["fix"] == "install stanza"

    def test_a_successful_fallback_is_a_success(self) -> None:
        """The failed backend's ERROR diagnostics must not ride along.

        Carrying them would make every caller that checks ``ok`` treat a
        completed run as a failed one.
        """
        result = resolve_pipeline(_cache(stanza=_model_missing("stanza"), spacy=_works("spacy")), CONFIG)
        assert result.ok
        assert not result.has_errors

    @pytest.mark.parametrize("code", ["PIPELINE_MODEL_MISSING", "PIPELINE_MODEL_UNREADABLE", "PIPELINE_MODEL_CORRUPT"])
    def test_every_model_problem_triggers_it(self, code: str) -> None:
        result = resolve_pipeline(_cache(stanza=_model_missing("stanza", code), spacy=_works("spacy")), CONFIG)
        assert result.ok and result.unwrap().backend == "spacy"


class TestFallbackDoesNotHappen:
    def test_a_working_backend_is_used_without_comment(self) -> None:
        result = resolve_pipeline(_cache(stanza=_works("stanza"), spacy=_works("spacy")), CONFIG)
        assert result.unwrap().backend == "stanza"
        assert not [d for d in result.diagnostics if d.code == "PARSER_FALLBACK"]

    def test_a_non_model_failure_is_an_answer_not_a_reason_to_switch(self) -> None:
        """If the backend broke for some other reason, switching hides it."""
        result = resolve_pipeline(_cache(stanza=_other_failure("stanza"), spacy=_works("spacy")), CONFIG)
        assert result.value is None
        assert any(d.code == "PIPELINE_BUILD_FAILED" for d in result.diagnostics)

    def test_strict_mode_refuses_to_substitute(self) -> None:
        result = resolve_pipeline(
            _cache(stanza=_model_missing("stanza"), spacy=_works("spacy")),
            CONFIG,
            allow_fallback=False,
        )
        assert result.value is None
        assert any(d.code == "PIPELINE_MODEL_MISSING" for d in result.diagnostics)
        assert not [d for d in result.diagnostics if d.code == "PARSER_FALLBACK"]

    def test_when_nothing_works_the_failure_names_what_was_tried(self) -> None:
        result = resolve_pipeline(_cache(stanza=_model_missing("stanza"), spacy=_model_missing("spacy")), CONFIG)
        assert result.value is None
        no_backend = next(d for d in result.diagnostics if d.code == "PARSER_NO_BACKEND")
        assert no_backend.context["tried"] == ["spacy"]
        # The original fix command survives, because that is the repair.
        assert any(d.code == "PIPELINE_MODEL_MISSING" for d in result.diagnostics)


class TestRegistration:
    def test_a_supplied_builder_is_never_displaced(self) -> None:
        """Registering the standard builders must not overwrite a caller's."""
        cache = _cache(spacy=_works("spacy"))
        register_builders(cache)
        assert resolve_pipeline(cache, NLPConfig(parser="spacy", language="en")).unwrap().backend == "spacy"

    def test_available_backends_is_a_subset_of_the_fallback_order(self) -> None:
        assert set(available_backends()) <= set(FALLBACK_ORDER)

    def test_the_fallback_order_prefers_the_easiest_model_to_install(self) -> None:
        assert FALLBACK_ORDER[0] == "spacy"


class TestTheDoctorReportsThisHonestly:
    """What the doctor calls *required*, across every combination of models.

    The probes are stubbed rather than run. ``_stanza_model_ok`` builds a real
    Stanza pipeline and can reach the network for its resource index, which is
    slow, machine-dependent, and nothing to do with the logic under test --
    which is simply "which of these checks can stop analysis".
    """

    @staticmethod
    def _checks(monkeypatch: pytest.MonkeyPatch, *, stanza: bool, spacy: bool) -> dict[str, object]:
        from tools import doctor

        monkeypatch.setattr(doctor, "_stanza_model_ok", lambda lang: (stanza, "stubbed"))
        monkeypatch.setattr(doctor, "_spacy_model_ok", lambda lang: (spacy, "stubbed"))
        monkeypatch.setattr(doctor, "_has_module", lambda name: True)
        return {c.name: c for c in doctor._check_parsers(["en"])}

    def test_only_having_a_parser_at_all_is_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        checks = self._checks(monkeypatch, stanza=True, spacy=True)
        required = {name for name, c in checks.items() if c.required}  # type: ignore[attr-defined]
        assert required == {"parser for en"}, required

    @pytest.mark.parametrize(
        ("stanza", "spacy", "usable"),
        [(True, True, "spacy, stanza"), (True, False, "stanza"), (False, True, "spacy")],
    )
    def test_either_backend_satisfies_the_requirement(
        self, monkeypatch: pytest.MonkeyPatch, stanza: bool, spacy: bool, usable: str
    ) -> None:
        check = self._checks(monkeypatch, stanza=stanza, spacy=spacy)["parser for en"]
        assert check.ok  # type: ignore[attr-defined]
        assert check.detail == f"usable backend(s): {usable}"  # type: ignore[attr-defined]

    def test_with_no_model_at_all_analysis_is_reported_as_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        check = self._checks(monkeypatch, stanza=False, spacy=False)["parser for en"]
        assert not check.ok  # type: ignore[attr-defined]
        assert "no installed backend" in check.detail  # type: ignore[attr-defined]
        assert "spacy download" in check.fix  # type: ignore[attr-defined]

    def test_a_damaged_default_is_shown_but_does_not_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exactly this machine's state: Stanza half-downloaded, spaCy fine."""
        checks = self._checks(monkeypatch, stanza=False, spacy=True)
        assert not checks["stanza model (en)"].ok  # type: ignore[attr-defined]
        assert not checks["stanza model (en)"].required  # type: ignore[attr-defined]
        assert checks["parser for en"].ok  # type: ignore[attr-defined]


class TestTheRunRecordTellsTheTruth:
    """R8: a run record that names a parser which never ran is worse than none.

    ``params`` in ``result.json`` comes from ``vars(args)``, so the requested
    backend has to be corrected to the one that actually parsed -- and the
    substitution itself has to reach the envelope, or it would be visible on
    the console and invisible in the record that outlives the console.
    """

    @staticmethod
    def _args(tmp_path):  # type: ignore[no-untyped-def]
        import argparse

        return argparse.Namespace(
            corpus=tmp_path / "corpus",
            output=tmp_path / "out",
            parser="stanza",
            language="en",
            allow_parser_fallback=True,
        )

    @staticmethod
    def _corpus(tmp_path):  # type: ignore[no-untyped-def]
        from core.io.reader import Document, hash_text

        root = tmp_path / "corpus"
        root.mkdir(parents=True, exist_ok=True)
        path = root / "a.txt"
        path.write_text("Some words here.", encoding="utf-8")
        docs = (Document(doc_id=1, path=path, text="Some words here.", sha256=hash_text("a")),)
        return Corpus(docs=docs, sha256="c")

    def test_params_name_the_backend_that_actually_parsed(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from tools._cli import get_pipeline

        args = self._args(tmp_path)
        cache = _cache(stanza=_model_missing("stanza"), spacy=_works("spacy"))
        pipeline = get_pipeline(cache, CONFIG, args)
        assert pipeline.backend == "spacy"
        assert args.parser == "spacy", "the record would have claimed stanza parsed this run"

    def test_every_parse_carries_the_substitution(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The route into the record: tools all forward parse diagnostics.

        Attaching the warning to ``parse`` rather than to each tool is what
        makes this work for the tools that build their own ``OutputWriter``
        instead of going through ``make_writer`` -- roughly a third of them.
        """
        from tools._cli import get_pipeline

        args = self._args(tmp_path)
        pipeline = get_pipeline(_cache(stanza=_model_missing("stanza"), spacy=_works("spacy")), CONFIG, args)
        parsed = pipeline.parse(self._corpus(tmp_path))
        assert parsed.ok
        assert any(d.code == "PARSER_FALLBACK" for d in parsed.diagnostics)

    def test_a_pipeline_that_did_not_fall_back_adds_nothing(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from tools._cli import get_pipeline

        args = self._args(tmp_path)
        pipeline = get_pipeline(_cache(stanza=_works("stanza"), spacy=_works("spacy")), CONFIG, args)
        parsed = pipeline.parse(self._corpus(tmp_path))
        assert not [d for d in parsed.diagnostics if d.code == "PARSER_FALLBACK"]

    def test_the_substitution_reaches_the_envelope(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """End to end, the way a tool actually assembles its record."""
        import json

        from tools._cli import get_pipeline, make_writer

        args = self._args(tmp_path)
        corpus = self._corpus(tmp_path)
        pipeline = get_pipeline(_cache(stanza=_model_missing("stanza"), spacy=_works("spacy")), CONFIG, args)
        parsed = pipeline.parse(corpus)

        writer = make_writer(args, corpus, tool="demo", params=vars(args))
        writer.write_table(pd.DataFrame({"Document": ["a.txt"], "Count": [1]}), "demo.csv", kind="table")
        writer.add_diagnostics(*parsed.diagnostics)
        writer.finalize()

        envelope = json.loads((writer.run_dir / "result.json").read_text(encoding="utf-8"))
        assert envelope["params"]["parser"] == "spacy"
        assert "PARSER_FALLBACK" in {d["code"] for d in envelope["diagnostics"]}

    def test_the_wrapper_is_transparent_in_every_other_respect(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A wrapped pipeline must still look exactly like a pipeline."""
        from tools._cli import get_pipeline

        args = self._args(tmp_path)
        pipeline = get_pipeline(_cache(stanza=_model_missing("stanza"), spacy=_works("spacy")), CONFIG, args)
        assert pipeline.backend == "spacy"
        assert pipeline.language == "en"
        assert pipeline.tasks == frozenset()
        assert pipeline.supports("en")
        assert not pipeline.supports("de")
        assert pipeline.parse(self._corpus(tmp_path)).unwrap()["Document"].tolist() == ["a.txt"]

    def test_the_run_parameters_stay_json_serializable(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A regression: stashing diagnostics on the namespace crashed every
        tool that passes ``vars(args)`` straight to its own ``OutputWriter``."""
        import json

        from tools._cli import get_pipeline

        args = self._args(tmp_path)
        get_pipeline(_cache(stanza=_model_missing("stanza"), spacy=_works("spacy")), CONFIG, args)
        json.dumps({k: str(v) for k, v in vars(args).items()})  # no non-serializable leftovers
        assert all(not key.startswith("_") for key in vars(args))

    def test_strict_parser_still_exits_rather_than_substituting(
        self,
        tmp_path,  # type: ignore[no-untyped-def]
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from tools import doctor
        from tools._cli import get_pipeline

        # The failure path prints the doctor report, and the real report builds
        # a Stanza pipeline to probe it -- seconds of work, and a network call
        # for the resource index, neither of which this test is about.
        printed: list[str] = []
        monkeypatch.setattr(doctor, "run_doctor", lambda langs: (printed.append("doctor"), (1, []))[1])

        args = self._args(tmp_path)
        args.allow_parser_fallback = False
        with pytest.raises(SystemExit):
            get_pipeline(_cache(stanza=_model_missing("stanza"), spacy=_works("spacy")), CONFIG, args)
        assert printed == ["doctor"], "a blocked run must still show the environment state"

    def test_no_fallback_leaves_the_requested_parser_alone(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from tools._cli import get_pipeline

        args = self._args(tmp_path)
        get_pipeline(_cache(stanza=_works("stanza"), spacy=_works("spacy")), CONFIG, args)
        assert args.parser == "stanza"


class TestDesktopAvailability:
    """Either backend makes a parse-needing tool available, not one of them.

    The desktop asked for spaCy specifically, so a machine with a complete
    Stanza install and no spaCy was told that 24 of its 29 analyses needed
    setup -- while every one of them would have run.
    """

    @staticmethod
    def _missing(**found: bool) -> list[str]:
        from desktop_backend.environment import _parser_missing

        return _parser_missing(found)

    def test_spacy_alone_is_enough(self) -> None:
        assert self._missing(spacy=True, en_core_web_sm=True) == []

    def test_stanza_alone_is_enough(self) -> None:
        assert self._missing(stanza=True, stanza_model_en=True) == []

    def test_a_package_without_its_model_is_not_enough(self) -> None:
        assert self._missing(spacy=True, en_core_web_sm=False) != []
        assert self._missing(stanza=True, stanza_model_en=False) != []

    def test_with_nothing_installed_the_advice_is_the_cheap_route(self) -> None:
        """spaCy's small model is one command; Stanza's is a much larger pull."""
        assert self._missing() == ["spacy", "en_core_web_sm"]

    def test_a_parse_tool_is_available_on_a_stanza_only_machine(self) -> None:
        from core.profiler.registry import TOOL_REGISTRY
        from desktop_backend.environment import availability

        spec = next(s for s in TOOL_REGISTRY if s.requires_parse and not s.optional_package and not s.assets)
        state = availability(spec, {"stanza": True, "stanza_model_en": True})
        assert state["state"] != "needs_setup", state


class TestTheConsoleSaysItOnce:
    """A substitution is worth one line, not two.

    The pipeline attaches the warning to every parse, and every tool prints
    its parse diagnostics. ``get_pipeline`` printing it as well meant every
    run on a machine with a damaged model opened with the same paragraph
    twice.
    """

    @staticmethod
    def _args(tmp_path):  # type: ignore[no-untyped-def]
        import argparse

        return argparse.Namespace(
            corpus=tmp_path / "corpus",
            output=tmp_path / "out",
            parser="stanza",
            language="en",
            allow_parser_fallback=True,
        )

    def test_choosing_the_pipeline_prints_nothing_on_success(
        self,
        tmp_path,  # type: ignore[no-untyped-def]
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from tools._cli import get_pipeline

        get_pipeline(_cache(stanza=_model_missing("stanza"), spacy=_works("spacy")), CONFIG, self._args(tmp_path))
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_the_warning_is_carried_by_the_parse_exactly_once(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from core.io.reader import Document, hash_text
        from tools._cli import get_pipeline

        root = tmp_path / "corpus"
        root.mkdir(parents=True, exist_ok=True)
        path = root / "a.txt"
        path.write_text("words", encoding="utf-8")
        corpus = Corpus(docs=(Document(doc_id=1, path=path, text="words", sha256=hash_text("a")),), sha256="c")

        pipeline = get_pipeline(
            _cache(stanza=_model_missing("stanza"), spacy=_works("spacy")), CONFIG, self._args(tmp_path)
        )
        codes = [d.code for d in pipeline.parse(corpus).diagnostics]
        assert codes.count("PARSER_FALLBACK") == 1
