"""Shared CLI plumbing — one place for the boring parts.

Every tool CLI is ~15 lines because the common arguments, corpus loading,
pipeline construction and envelope writing live here. Tools that need extra
flags add them on the returned parser and call the helpers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from core.config import NLPConfig
from core.io.reader import Corpus, read_corpus
from core.io.writer import OutputWriter
from core.pipelines.cache import PipelineCache
from core.pipelines.spacy_backend import build_spacy_pipeline
from core.result import Diagnostic, Result

__all__ = [
    "PARTIAL_PARSE_POLICY",
    "add_common_arguments",
    "build_common_parser",
    "exit_with_diagnostics",
    "get_pipeline",
    "handle_result_states",
    "load_corpus",
    "make_console_encoding_tolerant",
    "make_writer",
    "resolve_config",
    "write_or_die",
]

try:
    from core.pipelines.stanza_backend import build_stanza_pipeline

    _HAS_STANZA = True
except ImportError:
    _HAS_STANZA = False


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments every tool shares."""
    parser.add_argument("corpus", type=Path, help="corpus directory of .txt files")
    parser.add_argument("output", type=Path, help="output root directory")
    parser.add_argument(
        "--parser",
        choices=["spacy", "stanza"],
        default="stanza",
        help="parser backend (default matches NLPConfig)",
    )
    parser.add_argument("--language", default="en", help="language code (e.g. en, de)")
    parser.add_argument(
        "--strict-parser",
        dest="allow_parser_fallback",
        action="store_false",
        default=True,
        help=(
            "fail instead of parsing with a different backend when the chosen parser's model "
            "is missing or damaged (by default the run continues on another installed backend "
            "and says so)"
        ),
    )


def build_common_parser(description: str) -> argparse.ArgumentParser:
    """Create an ``ArgumentParser`` pre-populated with the common arguments."""
    parser = argparse.ArgumentParser(description=description)
    add_common_arguments(parser)
    return parser


def resolve_config(args: argparse.Namespace) -> NLPConfig:
    """Turn ``args.parser`` / ``args.language`` into a validated config."""
    return NLPConfig(parser=args.parser, language=args.language)


def load_corpus(args: argparse.Namespace) -> Corpus:
    """Load the corpus or exit with diagnostics."""
    result = read_corpus(args.corpus)
    if result.value is None:
        exit_with_diagnostics(result.diagnostics, code=1)
        raise SystemExit(1)
    if result.diagnostics:
        for diag in result.diagnostics:
            print(diag, file=sys.stderr)
    return result.unwrap()


def make_writer(
    args: argparse.Namespace,
    corpus: Corpus,
    tool: str,
    params: dict[str, Any],
) -> OutputWriter:
    """Create an :class:`OutputWriter` or exit."""
    try:
        return OutputWriter(args.output, tool=tool, params=params, corpus=corpus)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def exit_with_diagnostics(diagnostics: tuple[Diagnostic, ...] | list[Diagnostic], code: int = 1) -> None:
    for diag in diagnostics:
        print(diag, file=sys.stderr)
        # Missing-model failures carry the exact fix command; surfacing it on
        # its own line makes the failure actionable without reading prose.
        fix = diag.context.get("fix") if isinstance(diag.context, dict) else None
        if fix:
            print(f"fix: {fix}", file=sys.stderr)
    raise SystemExit(code)


PARTIAL_PARSE_POLICY = (
    "A parse that produced a value alongside ERROR diagnostics is a PARTIAL "
    "run: the CLI exits 1 after writing the artifacts and a failure envelope "
    "listing the failed document(s). It never silently analyzes a partial "
    "parse as success (exit 0), and never aborts without an envelope."
)


def handle_result_states(result: Result[Any], *, artifact: str | None = None) -> str:
    """Classify a Result per the documented four-state policy.

    Returns "ok" | "partial" | "failed" and prints all diagnostics.
    Callers treat "partial" as a failure exit (1) while still writing the
    envelope that names the failed documents — the value is never silently
    blessed as success, and a failure is never a bare traceback.
    """
    if result.value is None:
        exit_with_diagnostics(result.diagnostics, code=1)
        raise SystemExit(1)  # pragma: no cover — exit_with_diagnostics always raises
    if result.has_errors:
        for diag in result.diagnostics:
            print(diag, file=sys.stderr)
        print(f"partial run: {PARTIAL_PARSE_POLICY}", file=sys.stderr)
        return "partial"
    for diag in result.diagnostics:
        print(diag, file=sys.stderr)
    return "ok"


def get_pipeline(cache: PipelineCache, config: NLPConfig, args: argparse.Namespace | None = None):  # type: ignore[no-untyped-def]
    """Get a pipeline for *config*, falling back when its model is unusable.

    If the configured backend's model is missing or damaged but another
    installed backend has one, the run continues on that backend and the
    substitution is printed as a warning (see :mod:`core.pipelines.resolve`).
    Passing *args* lets this record the backend that actually did the work, so
    the run's provenance never names a parser that did not run.

    When no backend has a usable model the failure is still loud, with the fix
    command, and the doctor output is printed alongside so the whole
    environment state is visible at the point of failure.
    """
    from core.pipelines.resolve import resolve_pipeline

    result = resolve_pipeline(cache, config, allow_fallback=getattr(args, "allow_parser_fallback", True))
    if result.value is None:
        for diag in result.diagnostics:
            if diag.code.startswith("PIPELINE_MODEL_"):
                print("\nThe environment check below shows what else is missing:\n", file=sys.stderr)
                from tools.doctor import run_doctor

                run_doctor([config.language])
                break
        exit_with_diagnostics(result.diagnostics, code=1)
        raise SystemExit(1)  # pragma: no cover — exit_with_diagnostics always raises
    # Deliberately silent on success. A substitution is reported by the
    # pipeline itself, on every parse it performs, and every tool prints and
    # records its parse diagnostics -- printing here as well said it twice.
    # The printed diagnostic carries the fix command in its context.
    pipeline = result.unwrap()
    # Provenance (R8): the envelope's params come from ``vars(args)``, so the
    # requested parser has to be corrected to the one that ran, or the run
    # record would name a backend that never touched the text. The reason for
    # the substitution reaches the record by a different route -- the pipeline
    # attaches it to every parse it performs (core.pipelines.resolve).
    if args is not None and getattr(args, "parser", None) != pipeline.backend:
        args.parser = pipeline.backend
    return pipeline


def write_or_die(writer: OutputWriter, result: Result[Path]) -> None:
    """Write an artifact, or print the diagnostics and exit.

    Roughly twenty call sites used to discard the writer's Result, so a
    failed write left an envelope advertising artifacts that were never on
    disk. This is the one place where exiting is correct: a CLI without its
    artifact has nothing to report.
    """
    if result.value is None:
        writer.abandon()
        exit_with_diagnostics(result.diagnostics, code=1)
        raise SystemExit(1)  # pragma: no cover — exit_with_diagnostics always raises


def make_console_encoding_tolerant() -> None:
    """Never let the console's codepage turn a diagnostic into a traceback.

    On Windows a redirected or legacy-codepage stdout encodes with the ANSI
    codepage (cp1252, or cp437 on an unconfigured console). Printing any
    character outside it raises ``UnicodeEncodeError`` from inside ``print``,
    so the tool dies with a traceback instead of saying what was wrong --
    worst at exactly the moment it has bad news. Literal strings in this
    package are ASCII, but the text we print is not all ours: corpus paths,
    document names and third-party diagnostics all reach stdout.

    Switching the error handler to ``backslashreplace`` degrades an
    unprintable character to a visible escape and keeps the message. Called
    from the console entry points; a no-op on streams that do not support
    reconfiguration (a pytest capture buffer, a pipe wrapper).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="backslashreplace")
        except (OSError, ValueError):  # pragma: no cover -- exotic stream
            continue
