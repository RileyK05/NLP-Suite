"""Mirror the dev repository into a public checkout, minus dev-only files.

Two tracks: dev (the private repository's main, where all work happens) and
public (RileyK05/NLP-Suite, what people download). Public is never edited by
hand: this script writes its tree from dev, leaving out every tracked path
matched by ``.publicignore`` (gitignore-style globs, one per line), and
refuses outright if a path that looks personal would still be published.

    python scripts/publish_snapshot.py --dev path/to/dev --out path/to/public

The caller commits and pushes; this only makes the public tree equal to dev.
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatch
from pathlib import Path
import re
import shutil
import subprocess
import sys

# Refused whatever .publicignore says: a new personal file that nobody added
# to the ignore list must stop the publish, not slip through it.
NEVER_PUBLIC = re.compile(r"(^|/)[^/]*(hw\d|homework|syllabus)[^/]*(/|$)|(^|/)review_first\.md$", re.IGNORECASE)
# Product files the guard would misread: the fixture is a list of tool names
# the tool registry is checked against, not course material.
PUBLIC_DESPITE_NAME = frozenset({"tests/fixtures/syllabus_tools.json"})


def patterns(dev: Path) -> list[str]:
    source = dev / ".publicignore"
    if not source.is_file():
        return [".publicignore"]
    lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines()]
    return [".publicignore", *(line for line in lines if line and not line.startswith("#"))]


def ignored(path: str, rules: list[str]) -> bool:
    for rule in rules:
        if rule.endswith("/") and (path.startswith(rule) or f"/{rule}" in f"/{path}"):
            return True
        if fnmatch(path, rule) or ("/" not in rule and fnmatch(Path(path).name, rule)):
            return True
    return False


def public_files(tracked: list[str], rules: list[str]) -> list[str]:
    """Dev's tracked files that belong in public; raises if a personal one leaks."""
    chosen = [path for path in tracked if not ignored(path, rules)]
    leaks = [path for path in chosen if NEVER_PUBLIC.search(path) and path not in PUBLIC_DESPITE_NAME]
    if leaks:
        raise ValueError(
            "Refusing to publish personal-looking files; add them to .publicignore:\n  " + "\n  ".join(leaks)
        )
    return chosen


def tracked_files(dev: Path) -> list[str]:
    git = shutil.which("git")
    if git is None:
        raise ValueError("git is not installed")
    done = subprocess.run(  # noqa: S603 - fixed git argv, no shell
        [git, "-C", str(dev), "ls-files", "-z"], check=True, capture_output=True
    )
    return [name for name in done.stdout.decode("utf-8").split("\0") if name]


def mirror(dev: Path, out: Path, files: list[str]) -> tuple[int, int]:
    """Make ``out`` hold exactly ``files`` from ``dev`` (its .git untouched)."""
    wanted = set(files)
    removed = 0
    for existing in sorted(out.rglob("*"), reverse=True):
        relative = existing.relative_to(out).as_posix()
        if relative == ".git" or relative.startswith(".git/"):
            continue
        if existing.is_file() or existing.is_symlink():
            if relative not in wanted:
                existing.unlink()
                removed += 1
        elif existing.is_dir() and not any(existing.iterdir()):
            existing.rmdir()
    for name in files:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dev / name, target)
    return len(files), removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev", type=Path, required=True, help="dev repository checkout")
    parser.add_argument("--out", type=Path, required=True, help="public repository checkout to overwrite")
    args = parser.parse_args()
    try:
        files = public_files(tracked_files(args.dev), patterns(args.dev))
    except ValueError as exc:
        sys.exit(str(exc))
    copied, removed = mirror(args.dev.resolve(), args.out.resolve(), files)
    print(f"Public tree: {copied} files from dev, {removed} stale files removed")


if __name__ == "__main__":
    main()
