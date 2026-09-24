"""Cross-platform build contracts; real native artifacts still need native CI."""

import json
from pathlib import Path
import re
import tomllib

import pytest

from scripts.build_desktop_backend import build_arguments
from scripts.collect_desktop_release import release_files


def test_desktop_release_versions_stay_in_sync() -> None:
    root = Path(__file__).resolve().parents[1] / "desktop"
    package = json.loads((root / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((root / "package-lock.json").read_text(encoding="utf-8"))
    tauri = json.loads((root / "src-tauri/tauri.conf.json").read_text(encoding="utf-8"))
    cargo = tomllib.loads((root / "src-tauri/Cargo.toml").read_text(encoding="utf-8"))
    assert package["version"] == lock["version"] == lock["packages"][""]["version"]
    assert package["version"] == tauri["version"] == cargo["package"]["version"]
    assert tauri["bundle"]["resources"] == {f"binaries/releases/{package['version']}/nlp-runtime/": "nlp-runtime/"}
    # The Python engine ships inside the same installer, so pyproject belongs in
    # this check. It silently drifted to 0.1.0 while the desktop reached 0.3.0
    # precisely because this assertion stopped at the desktop files.
    pyproject = tomllib.loads((root.parent / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == package["version"]


def test_macos_minimum_matches_the_documented_floor() -> None:
    """The DMG must not advertise an OS the build was never tested on.

    docs/MACOS.md: "The initial minimum is macOS 15 ... Do not promise macOS
    13/14 support merely because a Rust build flag can name it". The config
    said 13.0, so the installer claimed two untested OS versions.
    """
    root = Path(__file__).resolve().parents[1]
    tauri = json.loads((root / "desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8"))
    documented = re.search(r"minimum is \*\*macOS (\d+)\*\*", (root / "docs/MACOS.md").read_text(encoding="utf-8"))
    assert documented is not None, "docs/MACOS.md no longer states a minimum"
    assert tauri["bundle"]["macOS"]["minimumSystemVersion"].split(".")[0] == documented.group(1)


@pytest.mark.parametrize("system,separator", [("win32", ";"), ("darwin", ":"), ("linux", ":")])
def test_native_runtime_build_arguments(tmp_path: Path, system: str, separator: str) -> None:
    args = build_arguments(tmp_path, with_parser=True, system=system, python="python", version="0.3.0")
    assert "--onedir" in args and "--noupx" in args
    assert f"{tmp_path / 'out/desktop-package/resources'}{separator}desktop_resources" in args
    assert f"{tmp_path / 'desktop/.toolchain/nltk_data'}{separator}nltk_data" in args
    assert "en_core_web_sm" in args


def test_unknown_build_platform_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        build_arguments(tmp_path, with_parser=False, system="unknown", python="python", version="0.3.0")


def test_release_collection_does_not_include_old_installer(tmp_path: Path) -> None:
    directory = tmp_path / "nsis"
    directory.mkdir()
    old = directory / "NLP Suite_0.2.2_x64-setup.exe"
    new = directory / "NLP Suite_0.3.0_x64-setup.exe"
    old.touch()
    new.touch()
    assert release_files(tmp_path, "0.3.0", "win32") == [new]


def test_linux_requires_both_artifact_types(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Expected one"):
        release_files(tmp_path, "0.3.0", "linux")
