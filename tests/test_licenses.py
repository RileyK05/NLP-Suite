"""FR-0.3 — no-vendoring guards (the license status quo, enforced).

Licensed lexicons must never land in the repo by accident: `assets/`
holds only placeholders, and no data-sized CSV/JSON lives outside
`tests/fixtures/` (hand-built minis) and `corpus/` (public-domain texts).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_INLINE_BYTES = 64 * 1024  # hand-built fixtures stay small


def source_files(suffix: str):
    """Prune generated dependencies, not shipped source, before inspecting data."""
    generated = {
        "desktop/.toolchain",
        "desktop/node_modules",
        "desktop/src-tauri/target",
        "desktop/src-tauri/gen",
        "desktop/src-tauri/binaries",
        # transient pip-install bundle used by scripts/build_hw1_draft.py;
        # gitignored, never shipped
        ".tmp_wordcloud",
    }
    for directory, folders, files in os.walk(ROOT):
        folders[:] = [
            name
            for name in folders
            if name not in (".git", "__pycache__", "out", ".pytest-tmp")
            and (Path(directory) / name).relative_to(ROOT).as_posix() not in generated
        ]
        for name in files:
            if name.endswith(suffix):
                yield Path(directory) / name


class TestNoVendoring:
    def test_assets_holds_no_lexicon_bytes(self) -> None:
        # Original fictional onboarding text, not third-party research data.
        examples = {ROOT / "assets/sample-corpus" / name for name in ("community.txt", "library.txt", "transport.txt")}
        offenders = [
            p
            for p in (ROOT / "assets").rglob("*")
            if p.is_file()
            and p.suffix.lower() in (".csv", ".json", ".txt", ".zip")
            and p.name != ".gitkeep"
            and p not in examples
        ]
        assert not offenders, f"licensed bytes vendored under assets/: {offenders}"

    def test_no_data_sized_csv_outside_fixtures(self) -> None:
        offenders = []
        for path in source_files(".csv"):
            if "__pycache__" in path.parts or ".git" in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith(("tests/fixtures/", "corpus/", "out/")):
                continue
            if path.stat().st_size > MAX_INLINE_BYTES:
                offenders.append(rel)
        assert not offenders, f"data-sized CSV outside fixtures/corpus: {offenders}"

    def test_no_lexicon_json_outside_fixtures(self) -> None:
        offenders = []
        for path in source_files(".json"):
            if "__pycache__" in path.parts or ".git" in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith(("tests/fixtures/", "out/")) or "package" in rel:
                continue
            if path.stat().st_size > MAX_INLINE_BYTES:
                offenders.append(rel)
        assert not offenders, f"data-sized JSON outside fixtures: {offenders}"
