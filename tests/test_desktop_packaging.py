"""Cross-platform build contracts; real native artifacts still need native CI."""

import json
from pathlib import Path
import re
import tomllib

import pytest

from scripts.build_desktop_backend import build_arguments
from scripts.collect_desktop_release import release_files, updater_files


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


def test_frozen_engine_keeps_every_core_dependency(tmp_path: Path) -> None:
    """A core dependency excluded from the freeze fails only in the installed app.

    pyarrow was excluded to save space; the live bench later started caching
    parses as parquet, and every live load in the desktop app failed while
    every test and packaged job still passed.
    """
    root = Path(__file__).resolve().parents[1]
    core = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    import_names = {"scikit-learn": "sklearn"}
    wanted = {import_names.get(name, name) for name in (re.split(r"[<>=!~ ;\[]", dep)[0] for dep in core)}
    args = build_arguments(tmp_path, with_parser=True, system="linux", python="python", version="0.0.0")
    excluded = {args[i + 1] for i, arg in enumerate(args) if arg == "--exclude-module"}
    assert not wanted & excluded, f"core dependencies excluded from the desktop engine: {wanted & excluded}"
    hidden = {args[i + 1] for i, arg in enumerate(args) if arg == "--hidden-import"}
    assert "pyarrow.parquet" in hidden


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


def test_updater_files_follow_the_installer_names(tmp_path: Path) -> None:
    (tmp_path / "nsis").mkdir()
    (tmp_path / "macos").mkdir()
    (tmp_path / "nsis/NLP Suite_1.2.3_x64-setup.exe.sig").write_text("win")
    (tmp_path / "nsis/NLP Suite_1.2.2_x64-setup.exe.sig").write_text("stale")
    assert list(updater_files(tmp_path, "1.2.3", "win32", "AMD64").values()) == ["NLP Suite_1.2.3_x64-setup.exe.sig"]

    archive = tmp_path / "macos/NLP Suite.app.tar.gz"
    archive.write_bytes(b"app")
    assert updater_files(tmp_path, "1.2.3", "darwin", "arm64") == {}, "an unsigned archive is not an update"
    archive.with_name(archive.name + ".sig").write_text("mac")
    assert sorted(updater_files(tmp_path, "1.2.3", "darwin", "arm64").values()) == [
        "NLP Suite_1.2.3_aarch64.app.tar.gz",
        "NLP Suite_1.2.3_aarch64.app.tar.gz.sig",
    ]
    assert "NLP Suite_1.2.3_x64.app.tar.gz" in updater_files(tmp_path, "1.2.3", "darwin", "x86_64").values()
    assert updater_files(tmp_path, "1.2.3", "linux", "x86_64") == {}
