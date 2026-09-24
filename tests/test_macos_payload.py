"""Portable tests for Mac bundle metadata; native signing needs a Mac runner."""

import json
from pathlib import Path
import plistlib

import pytest

from scripts import macos_payload
from scripts.macos_payload import bundle_executable


def metadata(**overrides):
    return {
        "CFBundleExecutable": "nlp-suite-desktop",
        "CFBundleIdentifier": "org.nlpsuite.desktop",
        "CFBundleShortVersionString": "0.3.0",
        **overrides,
    }


def test_resolves_executable_in_app_with_spaces(tmp_path: Path) -> None:
    app = tmp_path / "copied application/NLP Suite.app"
    assert bundle_executable(app, metadata(), "0.3.0") == app / "Contents/MacOS/nlp-suite-desktop"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "x\\y", "C:drive", "", None])
def test_invalid_executable_names_are_rejected(tmp_path: Path, name) -> None:
    with pytest.raises(ValueError, match="executable"):
        bundle_executable(tmp_path, metadata(CFBundleExecutable=name), "0.3.0")


@pytest.mark.parametrize("overrides", [{"CFBundleIdentifier": "wrong.app"}, {"CFBundleShortVersionString": "0.2.2"}])
def test_mismatched_bundle_is_rejected(tmp_path: Path, overrides) -> None:
    with pytest.raises(ValueError, match="mismatch"):
        bundle_executable(tmp_path, metadata(**overrides), "0.3.0")


def test_dmg_detaches_when_copy_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(args):
        calls.append(args)
        if "attach" in args:
            (tmp_path / "mounted-dmg/NLP Suite.app").mkdir()
        if args[0].endswith("ditto"):
            raise RuntimeError("copy failed")
        return ""

    monkeypatch.setattr(macos_payload, "run", fake_run)
    with pytest.raises(RuntimeError, match="copy failed"):
        macos_payload.copy_and_verify(tmp_path / "test.dmg", tmp_path, "0.3.0")
    assert calls[-1][:2] == ["/usr/bin/hdiutil", "detach"]


@pytest.mark.parametrize("binary_arch", ["x86_64", "arm64"])
def test_copied_payload_architecture_is_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binary_arch: str
) -> None:
    calls = []

    def fake_run(args):
        calls.append(args)
        if "attach" in args:
            (tmp_path / "mounted-dmg/NLP Suite.app").mkdir()
        if args[0].endswith("ditto"):
            app = Path(args[-1])
            contents = app / "Contents"
            contents.mkdir(parents=True)
            (contents / "Info.plist").write_bytes(plistlib.dumps(metadata()))
            for relative in ("MacOS/nlp-suite-desktop", "Resources/nlp-runtime/nlp-runtime"):
                binary = contents / relative
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.write_bytes(b"\xcf\xfa\xed\xfe")
                binary.chmod(0o755)
        return binary_arch if args[0].endswith("lipo") else ""

    monkeypatch.setattr(macos_payload, "run", fake_run)
    monkeypatch.setattr(macos_payload.platform, "machine", lambda: "arm64")
    if binary_arch == "x86_64":
        with pytest.raises(ValueError, match="Wrong architecture"):
            macos_payload.copy_and_verify(tmp_path / "test.dmg", tmp_path, "0.3.0")
        assert not (tmp_path / "macos-validation.json").exists()
    else:
        executable, runtime = macos_payload.copy_and_verify(tmp_path / "test.dmg", tmp_path, "0.3.0")
        assert executable.is_file() and runtime.is_dir()
        report = json.loads((tmp_path / "macos-validation.json").read_text(encoding="utf-8"))
        assert len(report["binaries"]) == 2
        assert report["gatekeeper_acceptance_verified"] is False
        assert calls[-1][:4] == ["/usr/bin/codesign", "--verify", "--deep", "--strict"]
