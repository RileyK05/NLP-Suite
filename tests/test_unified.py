"""FR-9.1 — unified command contract tests."""

from __future__ import annotations

import contextlib

import pytest

from tools.unified import main


class TestUnified:
    def test_list_names_every_registry_tool(self, capsys: pytest.CaptureFixture[str]) -> None:
        from core.profiler.registry import tool_names

        assert main(["--list"]) == 0
        out = capsys.readouterr().out
        for name in tool_names():
            assert name in out

    def test_unknown_tool_fails_with_list_hint(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["no-such-tool"]) == 2
        err = capsys.readouterr().err
        assert "unknown tool" in err and "--list" in err

    def test_no_args_shows_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main([]) == 2
        assert "usage" in capsys.readouterr().err

    def test_routes_to_tool_main(self) -> None:
        # corpus_validation --help exercises real routing through argparse
        with pytest.raises(SystemExit) as exc:
            main(["corpus_validation", "--help"])
        assert exc.value.code == 0

    def test_every_registry_tool_has_routable_main(self) -> None:
        import importlib

        from core.profiler.registry import tool_names

        for name in tool_names():
            module = importlib.import_module(f"tools.{name}")
            assert callable(getattr(module, "main", None)), f"tools.{name} has no main()"


class TestExtras:
    def test_all_extra_covers_every_optional_extra(self) -> None:
        from pathlib import Path
        import tomllib

        pyproject = tomllib.loads(
            (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
        )
        optionals = pyproject["project"]["optional-dependencies"]
        assert "all" in optionals
        covered = " ".join(optionals["all"])
        for extra in optionals:
            if extra in ("all", "dev"):
                continue
            assert f"nlp-suite-ng[{extra}]" in covered, f"all extra misses [{extra}]"

    def test_doctor_reports_analysis_extras(self) -> None:
        from tools.doctor import run_doctor

        _, checks = run_doctor(["en"])
        names = {check.name: check for check in checks}
        for module, extra in (
            ("vaderSentiment", "sentiment"),
            ("gensim", "topics"),
            ("nltk", "wordnet"),
            ("transformers", "embeddings"),
        ):
            assert module in names, f"doctor never checks {module}"
            assert not names[module].required
            assert f"nlp-suite-ng[{extra}]" in names[module].fix


class TestRoutedProgName:
    """A routed tool must name the command the user actually typed.

    Tool parsers set no ``prog``, so argparse derives it from ``sys.argv[0]``.
    Unrouted, that is the router module, and ``nlp-suite kwic --help`` opens
    with ``usage: unified.py`` while every argparse error names a command the
    user never ran.
    """

    def test_help_names_the_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        from tools.unified import main

        with pytest.raises(SystemExit) as exc:
            main(["kwic", "--help"])
        assert exc.value.code == 0
        assert capsys.readouterr().out.startswith("usage: nlp-suite kwic")

    def test_usage_error_names_the_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        from tools.unified import main

        with pytest.raises(SystemExit) as exc:
            main(["keyness", "corpus", "output"])
        assert exc.value.code == 2
        assert "nlp-suite keyness: error:" in capsys.readouterr().err

    def test_argv_zero_is_restored_after_routing(self) -> None:
        import sys

        from tools.unified import main

        before = sys.argv[0]
        with pytest.raises(SystemExit):
            main(["kwic", "--help"])
        assert sys.argv[0] == before


class TestConsoleEncoding:
    """A CLI must not die on the console's codepage.

    On Windows a redirected or legacy-codepage stdout encodes with the ANSI
    codepage; printing anything outside it raises UnicodeEncodeError from
    inside ``print``. ``nlp-doctor`` did exactly this on the line announcing
    that the environment was broken -- the diagnostic died on its own bad
    news.
    """

    def test_no_printed_literal_needs_a_modern_codepage(self) -> None:
        import ast
        from pathlib import Path

        offenders: list[str] = []
        for path in sorted((Path(__file__).resolve().parent.parent / "tools").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"):
                    continue
                for sub in ast.walk(node):
                    if not (isinstance(sub, ast.Constant) and isinstance(sub.value, str)):
                        continue
                    try:
                        sub.value.encode("cp437")
                    except UnicodeEncodeError:
                        offenders.append(f"{path.name}:{sub.lineno}: {sub.value!r}")
        assert not offenders, "printed literals outside cp437 crash on a legacy console:\n" + "\n".join(offenders)

    def test_entry_points_make_the_console_tolerant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import tools._cli as cli
        import tools.doctor
        import tools.unified

        for module in (tools.unified, tools.doctor):
            called: list[bool] = []
            monkeypatch.setattr(
                cli,
                "make_console_encoding_tolerant",
                lambda sink=called: sink.append(True),
            )
            with contextlib.suppress(SystemExit):
                module.main(["--help"])
            assert called, f"{module.__name__}.main does not make the console tolerant"
