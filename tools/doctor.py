"""doctor — one command that answers "can this suite actually run?".

Checks, in order of how badly they break runs:

1. Python version and core dependencies (pandas, pyarrow).
2. Parser backends: is the package importable AND is a trained model
   loadable for the languages you will use? Importable-but-not-loadable is
   exactly the state that used to silently degrade to a blank pipeline.
3. Optional extras (plotly, streamlit) — charting and the viewer.

Exit codes: 0 = everything needed for the default path works;
1 = something required is missing (the report says exactly what to run).

Usage:
    python -m tools.doctor            # check en, the default language
    python -m tools.doctor en          # check a confirmed Stanza language
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from dataclasses import dataclass

from core.config import BACKEND_LANGUAGES, _normalize_language
from core.pipelines.spacy_backend import spacy_model_name

__all__ = ["Check", "run_doctor"]

DEFAULT_LANGUAGE = "en"


@dataclass(frozen=True)
class Check:
    """One environment check and its outcome.

    ``required`` is an explicit field (C6-14): the required set is never
    recovered from display strings like "python 3" (which never equals
    "python 3.12.10" — the exact bug this replaces).
    """

    name: str
    ok: bool
    detail: str
    fix: str = ""
    required: bool = False


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _check_core() -> list[Check]:
    checks: list[Check] = []
    version_ok = sys.version_info >= (3, 12)
    checks.append(
        Check(
            name=f"python {sys.version.split()[0]}",
            ok=version_ok,
            detail="requires >= 3.12",
            fix="install Python 3.12+ (see .python-version)",
            required=True,
        )
    )
    # Every declared core dependency (pyproject [project.dependencies]).
    for module in ("pandas", "pyarrow", "scipy", "scikit-learn"):
        import_name = "sklearn" if module == "scikit-learn" else module
        ok = _has_module(import_name)
        checks.append(
            Check(
                name=module,
                ok=ok,
                detail="core dependency" if ok else "missing",
                fix=f"pip install {module}",
                required=True,
            )
        )
    return checks


def _spacy_model_ok(language: str) -> tuple[bool, str]:
    try:
        import spacy

        spacy.load(spacy_model_name(language))
        return True, spacy_model_name(language)
    except ImportError:
        return False, "spacy package not installed"
    except Exception as exc:  # OSError for missing models, anything else is a broken install
        first_line = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        return False, f"{spacy_model_name(language)} not loadable ({first_line})"


def _stanza_model_ok(language: str) -> tuple[bool, str]:
    if not _has_module("stanza"):
        return False, "stanza package not installed"
    try:
        # Use the production builder so the probe exercises every canonical
        # processor (POS/lemma/dependency/NER), not just tokenizer resources.
        # A tokenizer-only probe can report success while the actual parse
        # path fails later when a downstream column is requested.
        from core.pipelines.stanza_backend import build_stanza_pipeline

        result = build_stanza_pipeline(language, frozenset())
        if result.value is None:
            first = result.diagnostics[0] if result.diagnostics else None
            if first is not None:
                return False, first.message
            return False, f"model for {language!r} is not loadable"
        return True, f"full model for {language!r} loadable"
    except Exception as exc:
        return False, f"stanza probe failed: {exc}"


def _check_parsers(languages: list[str]) -> list[Check]:
    checks: list[Check] = []
    checks.append(
        Check(
            name="stanza package",
            ok=_has_module("stanza"),
            detail="default parser backend (NLPConfig)",
            fix='pip install "nlp-suite-ng[stanza]"  (or: pip install stanza)',
            # Not required on its own: either backend can carry a run, and
            # "parser for <language>" below is the check that decides whether
            # analysis is possible at all.
            required=False,
        )
    )
    checks.append(
        Check(
            name="spacy package",
            ok=_has_module("spacy"),
            detail="alternate parser backend",
            fix='pip install "nlp-suite-ng[spacy]"  (or: pip install spacy)',
            required=False,
        )
    )
    for language in languages:
        stanza_ok, detail = _stanza_model_ok(language)
        checks.append(
            Check(
                name=f"stanza model ({language})",
                ok=stanza_ok,
                detail=detail,
                fix=f'python -c "import stanza; stanza.download({language!r})"',
                required=False,
            )
        )
        spacy_ok, detail = _spacy_model_ok(language)
        checks.append(
            Check(
                name=f"spacy model ({language})",
                ok=spacy_ok,
                detail=detail,
                fix=f"python -m spacy download {spacy_model_name(language)}",
                required=False,
            )
        )
        # What is required is a parser that works, not one specific parser: a
        # run whose configured backend has a damaged model continues on another
        # installed backend (see core.pipelines.resolve). Reporting the
        # default's absence as a blocker, on a machine where analysis runs
        # perfectly well, sends people to download gigabytes they do not need.
        usable = [name for name, ok in (("spacy", spacy_ok), ("stanza", stanza_ok)) if ok]
        checks.append(
            Check(
                name=f"parser for {language}",
                ok=bool(usable),
                detail=(
                    f"usable backend(s): {', '.join(usable)}"
                    if usable
                    else "no installed backend has a working model for this language"
                ),
                fix=f"python -m spacy download {spacy_model_name(language)}",
                required=True,
            )
        )
    return checks


def _check_extras() -> list[Check]:
    checks: list[Check] = []
    # Both forms, because the extra form resolves the distribution by name and
    # this project is not on PyPI yet: it works from a source checkout (where
    # pip sees the local project) but not from an arbitrary directory. The
    # direct package always works, so every fix line here is runnable as
    # printed. The stanza and spaCy checks above pair the forms the same way.
    for module, fix in (
        ("plotly", "pip install plotly"),
        ("streamlit", 'pip install "nlp-suite-ng[app]"  (or: pip install streamlit)'),
        ("vaderSentiment", 'pip install "nlp-suite-ng[sentiment]"  (or: pip install vaderSentiment)'),
        ("gensim", 'pip install "nlp-suite-ng[topics]"  (or: pip install gensim)'),
        ("nltk", 'pip install "nlp-suite-ng[wordnet]"  (or: pip install nltk)'),
        ("transformers", 'pip install "nlp-suite-ng[embeddings]"  (or: pip install transformers)'),
        ("wordcloud", 'pip install "nlp-suite-ng[wordcloud]"  (or: pip install wordcloud)'),
    ):
        ok = _has_module(module)
        checks.append(Check(name=module, ok=ok, detail="optional extra" if ok else "missing", fix=fix))
    for exe in ("java",):
        ok = shutil.which(exe) is not None
        checks.append(Check(name=exe, ok=ok, detail="optional (CoreNLP backend)" if ok else "not on PATH", fix=""))
    return checks


def run_doctor(languages: list[str] | None = None) -> tuple[int, list[Check]]:
    """Run all checks. Returns (exit_code, checks).

    Exit 1 only when something on the DEFAULT path is broken: python, core
    deps, the stanza package, or stanza models for the requested languages.
    spaCy is the alternate backend and plotly/streamlit/java are optional
    extras — they are reported as recommendations and never fail the run.
    """
    raw_languages = languages or [DEFAULT_LANGUAGE]
    try:
        languages = [_normalize_language(language) for language in raw_languages]
    except (AttributeError, TypeError):
        print("doctor: language names must be strings", file=sys.stderr)
        return 2, []
    for language in languages:
        if language not in BACKEND_LANGUAGES["stanza"]:
            print(
                f"doctor: {language!r} is not a language this suite's default Stanza path recognizes",
                file=sys.stderr,
            )
            return 2, []

    checks = _check_core() + _check_parsers(languages) + _check_extras()

    width = max(len(c.name) for c in checks) + 2
    print("NLP Suite doctor\n")
    for check in checks:
        marker = "OK     " if check.ok else ("MISSING" if not check.fix else "BROKEN")
        print(f"  [{marker}] {check.name:<{width}} {check.detail}")
        if not check.ok and check.fix:
            print(f"           fix: {check.fix}")

    # Required = the checks that actually stop analysis, via the explicit field.
    failed_required = [c for c in checks if not c.ok and c.required]
    if failed_required:
        print(f"\n{len(failed_required)} required item(s) missing - analysis WILL hard-fail.")
        return 1, checks

    # Everything runs, but say so precisely: a damaged default backend is a
    # real degradation even though it is no longer a blocker, and the person
    # reading this is the one who can repair it.
    degraded = [c for c in checks if not c.ok and c.name.startswith(("stanza model", "spacy model"))]
    print("\nDefault path is ready: parse -> analysis -> envelope all functional.")
    if degraded:
        names = ", ".join(c.name for c in degraded)
        print(
            f"\nDamaged or incomplete: {names}. Runs continue on another installed backend and say so, "
            "in their diagnostics and in the run record. Backends tokenize and tag differently, so repair "
            "with the fix above if you need a specific one."
        )
    print("Optional items above are recommendations, not blockers.")
    return 0, checks


def main(argv: list[str] | None = None) -> int:
    from tools._cli import make_console_encoding_tolerant

    make_console_encoding_tolerant()
    parser = argparse.ArgumentParser(description="Check the NLP Suite environment")
    parser.add_argument("languages", nargs="*", help="languages to check models for (default: en)")
    args = parser.parse_args(argv)
    code, _checks = run_doctor(args.languages or None)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
