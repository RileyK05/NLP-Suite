"""The preview launcher serves what is in the working tree.

Its docstring said "Build frontend first" and it never built anything -- it
checked that *a* bundle existed and served whatever was last compiled. Running
it after a change showed the previous interface, which is indistinguishable
from the change not having been made.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import types

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/desktop_preview.py"


def preview() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("desktop_preview", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def desktop(tmp_path: Path) -> Path:
    """A miniature desktop/ directory: some sources and an older bundle."""
    root = tmp_path / "desktop"
    (root / "src").mkdir(parents=True)
    (root / "src/App.tsx").write_text("export const App = () => null;", encoding="utf-8")
    (root / "index.html").write_text("<html></html>", encoding="utf-8")
    (root / "dist").mkdir()
    built = root / "dist/index.html"
    built.write_text("<html>built</html>", encoding="utf-8")
    newest = preview().newest_source(root)
    os.utime(built, (newest + 10, newest + 10))
    return root


class Recorder:
    """Stands in for npm so the tests never shell out to a real build."""

    def __init__(self, code: int = 0, *, writes: Path | None = None) -> None:
        self.code = code
        self.writes = writes
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **_: object) -> object:
        self.calls.append(command)
        if self.writes is not None:
            self.writes.parent.mkdir(parents=True, exist_ok=True)
            self.writes.write_text("<html>rebuilt</html>", encoding="utf-8")
        return types.SimpleNamespace(returncode=self.code)


def install(monkeypatch: pytest.MonkeyPatch, module: types.ModuleType, runner: Recorder) -> None:
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(module.subprocess, "run", runner)


def test_a_current_bundle_is_served_as_is(desktop: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = preview()
    runner = Recorder()
    install(monkeypatch, module, runner)
    module.build_if_stale(desktop, force=False)
    assert runner.calls == [], "nothing changed, so nothing should have been rebuilt"


def test_a_changed_source_triggers_a_rebuild(desktop: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = preview()
    built = desktop / "dist/index.html"
    stale = built.stat().st_mtime
    (desktop / "src/App.tsx").write_text("export const App = () => 'new';", encoding="utf-8")
    os.utime(desktop / "src/App.tsx", (stale + 60, stale + 60))
    runner = Recorder(writes=built)
    install(monkeypatch, module, runner)
    module.build_if_stale(desktop, force=False)
    assert runner.calls, "an edited source must be compiled before it is served"
    assert runner.calls[0][-2:] == ["run", "build"]


def test_a_new_source_file_counts_as_a_change(desktop: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The specific miss: a whole new page added to src/ and never compiled."""
    module = preview()
    built = desktop / "dist/index.html"
    stale = built.stat().st_mtime
    page = desktop / "src/Explore.tsx"
    page.write_text("export const Explore = () => null;", encoding="utf-8")
    os.utime(page, (stale + 60, stale + 60))
    runner = Recorder(writes=built)
    install(monkeypatch, module, runner)
    module.build_if_stale(desktop, force=False)
    assert runner.calls


def test_rebuild_is_forced_when_asked(desktop: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = preview()
    runner = Recorder(writes=desktop / "dist/index.html")
    install(monkeypatch, module, runner)
    module.build_if_stale(desktop, force=True)
    assert runner.calls


def test_a_failed_build_stops_rather_than_serving_the_old_one(desktop: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serving the previous bundle after a failed compile is the worst answer."""
    module = preview()
    (desktop / "src/App.tsx").write_text("syntax error", encoding="utf-8")
    runner = Recorder(code=1)
    install(monkeypatch, module, runner)
    with pytest.raises(SystemExit) as failure:
        module.build_if_stale(desktop, force=True)
    assert "build failed" in str(failure.value)


def test_a_build_that_produces_nothing_is_reported(desktop: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = preview()
    (desktop / "dist/index.html").unlink()
    runner = Recorder(code=0)  # claims success, writes nothing
    install(monkeypatch, module, runner)
    with pytest.raises(SystemExit) as failure:
        module.build_if_stale(desktop, force=False)
    assert "no desktop/dist/index.html" in str(failure.value)


def test_missing_npm_says_what_to_install(desktop: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = preview()
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as failure:
        module.build_if_stale(desktop, force=True)
    message = str(failure.value)
    assert "npm" in message and "npm run build" in message


def test_the_real_frontend_sources_are_the_ones_it_watches() -> None:
    """A source directory left out of SOURCES is a change that never rebuilds."""
    module = preview()
    root = SCRIPT.resolve().parents[1] / "desktop"
    for name in module.SOURCES:
        assert (root / name).exists(), f"{name} is watched but does not exist"
    assert "src" in module.SOURCES
    # Everything the bundle is compiled from lives under one of these.
    assert (root / "src/Explore.tsx").is_file()
