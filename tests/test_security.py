"""FR-9.6 — security/privacy static guards.

Parses every shipped Python file and forbids the dangerous shapes outright;
the small allowlist below names the modules whose job *is* the flagged
operation, with the control that makes each use safe. Anything new must
either avoid the shape or extend the allowlist with its control — silently
adding a bare ``shell=True`` fails here.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCOPES = ("core", "tools", "app", "scripts", "desktop_backend", "conftest.py")

# module stem -> (rule, control). Narrow: a whole-module pass, not a line excuse.
ALLOWLIST: dict[str, tuple[str, str]] = {
    "macos_payload": (
        "S603-subprocess",
        "fixed absolute Apple tools and argv; read-only DMG mount, fresh app copy, signature verification only",
    ),
    "build_desktop_backend": ("S603-subprocess", "fixed PyInstaller argv, current interpreter, no shell"),
    "verify_desktop_payload": (
        "S603-subprocess",
        "version-selected local build artifacts and fixed argv, no shell; silent installation restricted to explicit CI mode",
    ),
    "desktop_preview": ("S603-subprocess", "fixed local server argv, no shell"),
    "publish_snapshot": ("S603-subprocess", "fixed git ls-files argv on a local checkout, no shell"),
    "smoke_desktop": ("S603-subprocess S310-urlopen", "explicit test executable, local engine handshake, no shell"),
    "runner": ("S603-subprocess", "fixed local worker entrypoint and job UUID, no shell"),
    "server": ("socket", "binds only 127.0.0.1; bearer token, Origin and Host checks"),
    # Spawns the analyst's chosen tool CLI; argv list, shell=False.
    "jobs": ("S603-subprocess", "subprocess.Popen with a list argv and shell=False"),
    "srl_backend": ("S603-subprocess", "worker interpreter + fixed '-c' snippet, no shell"),
    "mallet": ("S603-subprocess", "configured binary + fixed train-topics flags, no shell"),
    # Talks to the analyst's own CoreNLP server; scheme pre-validated http(s).
    "corenlp_backend": ("S310-urlopen", "_checked_url rejects non-http(s) before any request"),
    # Google geocoding; host allowlisted to maps.googleapis.com.
    "online": ("S310-urlopen", "_default_request refuses non-Google hosts"),
    # Nominatim geocoding (OpenStreetMap); host allowlisted to nominatim.openstreetmap.org.
    "nominatim": ("S310-urlopen", "_default_request refuses non-Nominatim hosts; fixed User-Agent; no key"),
    # CoreNLP sentiment annotator over the analyst's own server; scheme
    # pre-validated http(s) exactly like the parse backend.
    "sentiment_neural": ("S310-urlopen", "_checked_url rejects non-http(s); analyst-chosen server URL only"),
    # DBpedia Spotlight; host allowlisted to api.dbpedia-spotlight.org.
    "dbpedia": ("S310-urlopen", "_default_request refuses non-Spotlight hosts"),
    # HW1 evidence scripts (one-off research, not shipped product code):
    # fetches fixed public .gov/.edu transcript URLs with urllib and hashes
    # responses into out/; no user input reaches the URL list.
    "hw1_duration_sources": ("S310-urlopen", "hard-coded public transcript URLs; responses hashed to out/"),
    # Runs the bundled Node VM over a saved local HTML file with four fixed
    # mock form fields; argv list, shell=False, no network.
    "run_gender_guesser": ("S603-subprocess", "fixed local node binary + script path, argv list, no shell"),
    # Boots the real desktop stack (local server over a throwaway workspace)
    # for the interactive frontend tests; argv lists, no shell, no network
    # beyond the engine's own loopback handshake.
    "interactive_stack": ("S603-subprocess", "fixed local server argv, taskkill on timeout, no shell"),
    # Model downloads from the project's own GitHub release; https only
    # (loopback http for tests), every file checked against its SHA-256.
    "download": ("S310-urlopen", "_checked refuses non-https URLs; files verified by SHA-256 before use"),
    # Maintainer tools, never shipped: fetch the Apache-2.0 text and upload
    # the models release to the project's own repository.
    "export_models": ("S310-urlopen", "fixed Hugging Face and apache.org license URLs"),
    "publish_models": (
        "S310-urlopen S603-subprocess",
        "fixed api.github.com / uploads.github.com URLs; git credential fill with a fixed argv, no shell",
    ),
    "run_interactive_frontend_test": (
        "S603-subprocess",
        "fixed local stack script + vitest argv, no shell; taskkill on timeout",
    ),
    # HW1 evidence generation (one-off research, not shipped product code):
    # runs the engine's own CLIs over the local added_corpus with argv lists,
    # capturing stdout/stderr into out/; no user input reaches any command.
    "generate_hw1_inaugural_outputs": (
        "S603-subprocess",
        "fixed tools.* module names + current interpreter, argv list, shell=False (default); outputs logged to out/",
    ),
}


# Personal scripts: tracked on dev but kept out of public by .publicignore,
# so public clones lack them and their existence is not asserted.
OWNER_LOCAL = frozenset({"hw1_duration_sources", "run_gender_guesser", "generate_hw1_inaugural_outputs"})


def _files() -> list[Path]:
    found: list[Path] = []
    for scope in SCOPES:
        target = ROOT / scope
        if target.is_file():
            found.append(target)
        elif target.is_dir():
            found.extend(sorted(target.rglob("*.py")))
    return [p for p in found if "__pycache__" not in p.parts]


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class TestSecurityShapes:
    def test_no_shell_execution(self) -> None:
        offenders: list[str] = []
        for path in _files():
            tree = _parse(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = ""
                    if isinstance(func, ast.Attribute):
                        name = func.attr
                    elif isinstance(func, ast.Name):
                        name = func.id
                    if name in ("system", "popen") and "os" in ast.dump(func):
                        offenders.append(f"{path}:{node.lineno}")
                    for keyword in node.keywords:
                        if keyword.arg == "shell" and not (
                            isinstance(keyword.value, ast.Constant) and keyword.value.value is False
                        ):
                            offenders.append(f"{path}:{node.lineno} shell=")
        assert not offenders, f"shell execution shapes: {offenders}"

    def test_no_dynamic_code_execution(self) -> None:
        offenders = []
        for path in _files():
            tree = _parse(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                    offenders.append(f"{path}:{node.lineno}")
        assert not offenders, f"eval/exec calls: {offenders}"

    def test_network_confined_to_allowlist(self) -> None:
        offenders = []
        for path in _files():
            tree = _parse(path)
            uses_network = any(
                isinstance(node, (ast.Import, ast.ImportFrom))
                and any((alias.name.split(".")[0] in ("urllib", "http", "socket", "requests")) for alias in node.names)
                for node in ast.walk(tree)
            )
            if uses_network:
                rule = ALLOWLIST.get(path.stem, ("", ""))[0]
                if "urlopen" not in rule and "socket" not in rule:
                    offenders.append(str(path))
        assert not offenders, f"network imports outside the allowlist: {offenders}"

    def test_subprocess_confined_to_allowlist(self) -> None:
        offenders = []
        for path in _files():
            tree = _parse(path)
            uses_subprocess = any(
                isinstance(node, (ast.Import, ast.ImportFrom))
                and any(alias.name.split(".")[0] == "subprocess" for alias in node.names)
                for node in ast.walk(tree)
            )
            if uses_subprocess and path.stem != "test_security":
                rule = ALLOWLIST.get(path.stem, ("", ""))[0]
                if "subprocess" not in rule:
                    offenders.append(str(path))
        assert not offenders, f"subprocess use outside the allowlist: {offenders}"

    def test_allowlist_entries_are_accurate(self) -> None:
        for stem, (rule, _control) in ALLOWLIST.items():
            matches = [p for p in _files() if p.stem == stem]
            if not matches and stem in OWNER_LOCAL:
                continue
            assert matches, f"allowlist names a module that does not exist: {stem}"
            if "subprocess" in rule:
                assert any("subprocess" in p.read_text(encoding="utf-8") for p in matches), stem
            if "urlopen" in rule:
                assert any("urlopen" in p.read_text(encoding="utf-8") for p in matches), stem
