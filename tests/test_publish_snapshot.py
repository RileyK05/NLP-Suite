"""Publishing dev to public leaves dev-only files out, and refuses leaks."""

from pathlib import Path

import pytest

from scripts.publish_snapshot import ignored, mirror, patterns, public_files, tracked_files

ROOT = Path(__file__).resolve().parents[1]


def test_the_repository_rules_keep_personal_files_out() -> None:
    rules = patterns(ROOT)
    for personal in (
        "scripts/build_hw1_draft.py",
        "scripts/run_legacy_wordclouds.py",
        "scripts/run_gender_guesser.py",
        "review_first.md",
        "docs/HOMEWORK2_WORKING_PLAN.md",
        "docs/Data visualization 446W Syllabus Fall 2026.docx",
        ".opencode/agents/x.md",
        ".publicignore",
    ):
        assert ignored(personal, rules), personal
    for product in ("core/io/writer.py", "scripts/bump_version.py", "README.md", "tests/test_security.py"):
        assert not ignored(product, rules), product


def test_a_personal_file_missing_from_the_rules_stops_the_publish() -> None:
    with pytest.raises(ValueError, match="hw3_essay"):
        public_files(["core/a.py", "scripts/hw3_essay.py"], [".publicignore"])
    with pytest.raises(ValueError, match="Syllabus"):
        public_files(["docs/Course Syllabus.pdf"], [])
    with pytest.raises(ValueError, match="hw3_data"):
        public_files(["hw3_data/a.txt"], [])


def test_every_file_dev_tracks_today_publishes_cleanly() -> None:
    public_files(tracked_files(ROOT), patterns(ROOT))


def test_mirror_replaces_the_public_tree_and_keeps_its_git(tmp_path: Path) -> None:
    dev, out = tmp_path / "dev", tmp_path / "out"
    (dev / "core").mkdir(parents=True)
    (dev / "core/a.py").write_text("new")
    (out / ".git").mkdir(parents=True)
    (out / ".git/HEAD").write_text("ref")
    (out / "core").mkdir()
    (out / "core/a.py").write_text("old")
    (out / "old/dir").mkdir(parents=True)
    (out / "old/dir/stale.py").write_text("stale")
    copied, removed = mirror(dev, out, ["core/a.py"])
    assert (copied, removed) == (1, 1)
    assert (out / "core/a.py").read_text() == "new"
    assert not (out / "old").exists()
    assert (out / ".git/HEAD").read_text() == "ref"
