"""Choosing a parser backend, including when the configured one cannot load.

The suite defaults to Stanza. A Stanza install whose model directory is
partially downloaded -- a common outcome of an interrupted ``stanza.download``
-- fails at build time, and until now that failed the run outright. On a
machine with a complete, working spaCy model sitting right there, that is a
hard failure with a working answer available, across the 24 of 29 desktop
tools that need a parse.

So: when the *configured* backend cannot load **because its model is missing
or damaged**, and another installed backend has a model for the language, the
run continues on that backend and says so loudly.

Four rules keep this from becoming the kind of silent magic that makes results
untrustworthy:

1. **Only a model problem triggers it.** A configuration error, an unsupported
   language, or a parse that fails on the text is not a reason to change
   backend -- those are answers, and switching would hide them.
2. **It is always announced**, as a warning carrying both backends and the fix
   command for the one that failed, so the environment still gets repaired.
3. **The run records the backend that did the work**, never the one that was
   asked for. A provenance record that names the wrong parser is worse than no
   record (R8).
4. **It can be refused.** ``allow_fallback=False`` reproduces the old hard
   failure, for anyone who needs a specific backend or nothing.

Backends differ in tokenization and tagging, so a fallback can change results.
That is exactly why it is a warning rather than a quiet convenience.
"""

from __future__ import annotations

import importlib.util

from core.config import NLPConfig, supports_language
from core.pipelines.cache import PipelineCache
from core.pipelines.protocol import Pipeline
from core.result import Diagnostic, Result

__all__ = ["FALLBACK_ORDER", "available_backends", "register_builders", "resolve_pipeline"]

#: Tried in this order when the configured backend cannot load. spaCy first:
#: its models are a single pip install, so it is the one most likely to be
#: both present and complete.
FALLBACK_ORDER: tuple[str, ...] = ("spacy", "stanza")

#: Diagnostic codes that mean "the backend is installed but its model is not
#: usable". Only these justify trying a different backend.
_MODEL_CODES = frozenset({"PIPELINE_MODEL_MISSING", "PIPELINE_MODEL_UNREADABLE", "PIPELINE_MODEL_CORRUPT"})


def _is_model_problem(diagnostics: tuple[Diagnostic, ...]) -> bool:
    return any(diagnostic.code in _MODEL_CODES for diagnostic in diagnostics)


def available_backends() -> tuple[str, ...]:
    """Backends whose Python package is installed, in fallback order.

    ``find_spec`` rather than an import: importing Stanza pulls in torch, which
    costs seconds on every tool startup and is pure waste when the configured
    backend is spaCy and loads fine. Presence is all that is needed to decide
    whether a backend is worth *trying*; whether its model loads is answered by
    building it, which is what the caller does next.
    """
    return tuple(backend for backend in FALLBACK_ORDER if importlib.util.find_spec(backend) is not None)


def register_builders(cache: PipelineCache, backends: tuple[str, ...] | None = None) -> None:
    """Register the builder for each importable backend on *cache*.

    A backend that already has a builder is left alone, so adding the standard
    ones never displaces a deliberately supplied builder -- a fake in a test,
    or a preconfigured pipeline in an embedding application.

    Imports are local so that registering spaCy never costs a Stanza import,
    and so a broken optional backend cannot stop the others being usable.
    """
    for backend in backends if backends is not None else available_backends():
        if cache.has_builder(backend):
            continue
        if backend == "spacy":
            from core.pipelines.spacy_backend import build_spacy_pipeline

            cache.register("spacy", build_spacy_pipeline)  # type: ignore[arg-type]
        elif backend == "stanza":
            from core.pipelines.stanza_backend import build_stanza_pipeline

            cache.register("stanza", build_stanza_pipeline)  # type: ignore[arg-type]


class _SubstitutedPipeline:
    """The fallback pipeline, carrying its own explanation into every parse.

    The substitution has to reach the run record, and the run record is
    assembled from the diagnostics each tool already collects -- above all the
    ones its ``parse`` call returns. Attaching the warning here means every
    tool records it through plumbing it already has, rather than each of the
    thirty-odd tools needing to learn about parser fallback.

    Delegation, not inheritance: the wrapped object is whatever the backend
    built, and only ``parse`` behaves differently.
    """

    def __init__(self, inner: Pipeline, warning: Diagnostic) -> None:
        self._inner = inner
        self._warning = warning

    @property
    def backend(self) -> str:
        return self._inner.backend

    @property
    def language(self) -> str:
        return self._inner.language

    @property
    def tasks(self) -> frozenset[str]:
        return self._inner.tasks

    def supports(self, language: str, tasks: object = ()) -> bool:
        return self._inner.supports(language, tasks)  # type: ignore[arg-type]

    def tokenize(self, text: str) -> tuple[str, ...]:
        # The substituted backend is the one that parsed, so it is also the one
        # whose opinion about token boundaries a query has to share.
        return self._inner.tokenize(text)

    def parse(self, corpus: object) -> object:
        parsed = self._inner.parse(corpus)  # type: ignore[arg-type]
        return type(parsed)(parsed.value, (self._warning, *parsed.diagnostics))


def _fallback_warning(requested: str, used: str, failure: tuple[Diagnostic, ...]) -> Diagnostic:
    """One warning that says what happened, why, and how to put it back."""
    reason = next((d.message for d in failure if d.code in _MODEL_CODES), "the model could not be loaded")
    fix = next(
        (d.context["fix"] for d in failure if isinstance(d.context, dict) and d.context.get("fix")),
        "",
    )
    return Diagnostic.warning(
        "PARSER_FALLBACK",
        f"the {requested} model could not be loaded, so this run was parsed with {used} instead. "
        f"Reason: {reason} Results may differ from a {requested} run, and this run's record names "
        f"{used} as its parser.",
        requested=requested,
        used=used,
        fix=fix,
    )


def resolve_pipeline(
    cache: PipelineCache,
    config: NLPConfig,
    *,
    allow_fallback: bool = True,
) -> Result[Pipeline]:
    """The pipeline to parse with, falling back only on a model problem.

    On success the diagnostics carry a ``PARSER_FALLBACK`` warning if the
    backend used is not the one configured; callers must surface it and record
    ``pipeline.backend`` rather than ``config.parser``.

    On failure the diagnostics are the configured backend's own -- its fix
    command is the one the user needs -- followed by a note naming each other
    backend that was tried, so "it didn't work" never means "nothing else was
    attempted".
    """
    register_builders(cache)

    first = cache.get(config.parser, config.language)
    if first.value is not None:
        return first
    if not allow_fallback or not _is_model_problem(first.diagnostics):
        return first

    tried: list[str] = []
    for backend in FALLBACK_ORDER:
        if backend == config.parser:
            continue
        if not supports_language(backend, config.language):
            continue
        tried.append(backend)
        attempt = cache.get(backend, config.language)
        if attempt.value is not None:
            # The configured backend's own diagnostics are ERRORs, and carrying
            # them forward would make this success read as a failed run to
            # every caller that checks `ok`. The warning below carries their
            # substance -- the reason and the fix command -- so nothing the
            # user needs is lost by leaving the errors behind.
            warning = _fallback_warning(config.parser, backend, first.diagnostics)
            return Result.success(
                _SubstitutedPipeline(attempt.unwrap(), warning),  # type: ignore[arg-type]
                warning,
                *attempt.diagnostics,
            )

    if not tried:
        return first
    return Result[Pipeline](
        None,
        (
            *first.diagnostics,
            Diagnostic.error(
                "PARSER_NO_BACKEND",
                f"no usable parser for {config.language!r}: {config.parser} could not load its model, "
                f"and {', '.join(tried)} could not either. Repair one of them with the fix above.",
                requested=config.parser,
                tried=tried,
            ),
        ),
    )
