"""What the product ships, and what it must never ship.

This repository doubles as its owner's research workspace: `scripts/` holds
release tooling *and* one-off coursework builders, and a few personal notes sit
beside the product documentation. None of that is part of the software.

Setuptools' defaults already leave most of it out, but "happens to be excluded"
is not a guarantee -- adding one `package_data` line or a stray `__init__.py`
would change it silently. These tests make the boundary explicit.
"""

from __future__ import annotations

from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]

#: The only packages that belong to the product.
SHIPPED_PACKAGES = ("core", "tools", "app", "desktop_backend")

#: Directories that are development or personal material, never distributed.
NOT_SHIPPED = ("tests", "scripts", "docs", "desktop", "assets", ".github", ".opencode")

#: Markers for the owner's coursework. Product code must not mention them.
PERSONAL_MARKERS = ("homework", "hw1_", "hw2_", "soc 446", "soc446")


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_only_the_product_packages_are_distributed() -> None:
    include = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    assert include == [f"{name}*" for name in SHIPPED_PACKAGES], include


def test_manifest_excludes_development_and_personal_material() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    pruned = set(re.findall(r"^prune\s+(\S+)", manifest, re.MULTILINE))
    missing = [name for name in NOT_SHIPPED if name not in pruned]
    assert not missing, f"MANIFEST.in no longer prunes: {missing}"
    assert "exclude review_first.md" in manifest


def test_product_code_never_references_the_owners_coursework() -> None:
    """A homework path or dataset name inside core/tools/app is a leak."""
    offenders: list[str] = []
    for package in SHIPPED_PACKAGES:
        for source in (ROOT / package).rglob("*.py"):
            if "__pycache__" in source.parts:
                continue
            text = source.read_text(encoding="utf-8", errors="replace").lower()
            for marker in PERSONAL_MARKERS:
                if marker in text:
                    offenders.append(f"{source.relative_to(ROOT).as_posix()}: {marker!r}")
    assert not offenders, offenders


def test_personal_files_live_outside_the_shipped_packages() -> None:
    """Personal notes may stay in the repo, but never inside a shipped package."""
    for personal in (ROOT / "review_first.md", ROOT / "docs" / "HOMEWORK2_WORKING_PLAN.md"):
        if not personal.exists():
            continue  # removing them entirely is also a valid answer
        top = personal.relative_to(ROOT).parts[0]
        assert top not in SHIPPED_PACKAGES, personal


def test_coursework_scripts_are_not_importable_by_the_product() -> None:
    """`scripts/` is not a package, so nothing shipped can import from it."""
    assert not (ROOT / "scripts" / "__init__.py").exists()
    offenders: list[str] = []
    for package in SHIPPED_PACKAGES:
        for source in (ROOT / package).rglob("*.py"):
            if "__pycache__" in source.parts:
                continue
            text = source.read_text(encoding="utf-8", errors="replace")
            if re.search(r"^\s*(?:from|import)\s+scripts\b", text, re.MULTILINE):
                offenders.append(source.relative_to(ROOT).as_posix())
    assert not offenders, offenders
