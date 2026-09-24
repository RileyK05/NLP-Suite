"""scripts/bump_version.py moves every version site, and only those."""

from pathlib import Path
import shutil

import pytest

from scripts.bump_version import SITES, bump

ROOT = Path(__file__).resolve().parents[1]


def _copy_sites(destination: Path) -> None:
    for name in {site[0] for site in SITES}:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)


def test_bump_round_trips_every_site(tmp_path: Path) -> None:
    import tomllib

    _copy_sites(tmp_path)
    current = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    originals = {name: (tmp_path / name).read_bytes() for name in {site[0] for site in SITES}}

    changed = bump("9.8.7", tmp_path)
    assert sorted(changed) == sorted(originals)
    for name in originals:
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert "9.8.7" in text, name

    bump(current, tmp_path)
    for name, data in originals.items():
        assert (tmp_path / name).read_bytes() == data, f"{name} did not round-trip"


def test_bump_preserves_crlf(tmp_path: Path) -> None:
    _copy_sites(tmp_path)
    for name in {site[0] for site in SITES}:
        path = tmp_path / name
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    bump("9.8.7", tmp_path)
    lock = (tmp_path / "desktop/src-tauri/Cargo.lock").read_bytes()
    assert b'name = "nlp-suite-desktop"\r\nversion = "9.8.7"' in lock
    assert b"\n" not in lock.replace(b"\r\n", b"")


def test_bump_on_current_version_is_a_no_op() -> None:
    import tomllib

    current = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert bump(current) == []


def test_bump_rejects_a_malformed_version(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"1\.2\.3"):
        bump("v1.2", tmp_path)
