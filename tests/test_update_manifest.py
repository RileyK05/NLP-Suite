"""latest.json lists exactly the signed, updatable installers."""

import json
from pathlib import Path

from scripts.write_update_manifest import manifest

BASE = "https://github.com/RileyK05/NLP-Suite/releases/download/v1.2.3"


def _files(root: Path, names: list[str]) -> None:
    root.mkdir(exist_ok=True)
    for name in names:
        (root / name).write_text(name)


def test_every_signed_platform_is_listed_with_its_url(tmp_path: Path) -> None:
    release, sigs = tmp_path / "release", tmp_path / "sigs"
    _files(
        release,
        [
            "NLP-Suite_1.2.3_x64-setup.exe",
            "NLP-Suite_1.2.3_aarch64.app.tar.gz",
            "NLP-Suite_1.2.3_x64.app.tar.gz",
            "NLP-Suite_1.2.3_aarch64.dmg",
            "NLP-Suite_1.2.3_amd64.deb",
        ],
    )
    _files(
        sigs,
        [
            "NLP-Suite_1.2.3_x64-setup.exe.sig",
            "NLP-Suite_1.2.3_aarch64.app.tar.gz.sig",
            "NLP-Suite_1.2.3_x64.app.tar.gz.sig",
        ],
    )
    data = manifest(release, sigs, "1.2.3", BASE, "2026-01-01T00:00:00Z")
    assert data is not None
    assert data["version"] == "1.2.3"
    assert set(data["platforms"]) == {"windows-x86_64", "darwin-aarch64", "darwin-x86_64"}
    windows = data["platforms"]["windows-x86_64"]
    assert windows == {
        "signature": "NLP-Suite_1.2.3_x64-setup.exe.sig",
        "url": f"{BASE}/NLP-Suite_1.2.3_x64-setup.exe",
    }
    assert data["platforms"]["darwin-x86_64"]["url"].endswith("_x64.app.tar.gz")
    json.dumps(data)


def test_unsigned_builds_write_no_manifest(tmp_path: Path) -> None:
    release, sigs = tmp_path / "release", tmp_path / "sigs"
    _files(release, ["NLP-Suite_1.2.3_x64-setup.exe", "NLP-Suite_1.2.3_aarch64.dmg"])
    sigs.mkdir()
    assert manifest(release, sigs, "1.2.3", BASE, "2026-01-01T00:00:00Z") is None


def test_a_platform_without_its_signature_is_left_out(tmp_path: Path) -> None:
    release, sigs = tmp_path / "release", tmp_path / "sigs"
    _files(release, ["NLP-Suite_1.2.3_x64-setup.exe", "NLP-Suite_1.2.3_aarch64.app.tar.gz"])
    _files(sigs, ["NLP-Suite_1.2.3_x64-setup.exe.sig"])
    data = manifest(release, sigs, "1.2.3", BASE, "2026-01-01T00:00:00Z")
    assert data is not None
    assert list(data["platforms"]) == ["windows-x86_64"]
