"""Set the product version everywhere it is written, in one step.

The engine and the desktop app ship in one installer under one version, and
that version is spelled out in eight files. Editing them by hand is how
pyproject once drifted to 0.1.0. Usage, from the repository root:

    python scripts/bump_version.py 0.3.1
    git commit -am "Release 0.3.1" && git tag v0.3.1 && git push origin main v0.3.1

Pushing the tag runs the desktop workflow, which builds every platform and
drafts a GitHub Release with the installers attached.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.compile(r"\d+\.\d+\.\d+")
V = r"\d+\.\d+\.\d+"

# (file, pattern, replacement template, expected match count). Every pattern is
# anchored to its own context so no dependency that shares the number moves.
SITES: tuple[tuple[str, str, str, int], ...] = (
    ("pyproject.toml", rf'(?m)^version = "{V}"', 'version = "{v}"', 1),
    ("desktop/package.json", rf'(?m)^  "version": "{V}",', '  "version": "{v}",', 1),
    ("desktop/package-lock.json", rf'("name": "nlp-suite-desktop",\r?\n\s+"version": "){V}"', r'\g<1>{v}"', 2),
    ("desktop/src-tauri/tauri.conf.json", rf'(?m)^  "version": "{V}",', '  "version": "{v}",', 1),
    (
        "desktop/src-tauri/tauri.conf.json",
        rf"binaries/releases/{V}/nlp-runtime/",
        "binaries/releases/{v}/nlp-runtime/",
        1,
    ),
    ("desktop/src-tauri/Cargo.toml", rf'(?m)^version = "{V}"', 'version = "{v}"', 1),
    ("desktop/src-tauri/Cargo.lock", rf'(name = "nlp-suite-desktop"\r?\nversion = "){V}"', r'\g<1>{v}"', 1),
    ("desktop/src/App.tsx", rf"Desktop beta · {V}", "Desktop beta · {v}", 1),
    ("desktop_backend/server.py", rf'"ok": True, "version": "{V}"', '"ok": True, "version": "{v}"', 1),
)


def _read(path: Path) -> str:
    # newline="" keeps CRLF checkouts byte-for-byte outside the version itself.
    with path.open(encoding="utf-8", newline="") as handle:
        return handle.read()


def bump(version: str, root: Path = ROOT) -> list[str]:
    """Rewrite every site to ``version``; returns the files changed."""
    if not VERSION.fullmatch(version):
        raise ValueError(f"version must look like 1.2.3, got {version!r}")
    texts: dict[str, str] = {}
    for name, pattern, template, expected in SITES:
        text = texts.get(name) or _read(root / name)
        text, count = re.subn(pattern, template.format(v=version), text)
        if count != expected:
            raise ValueError(f"{name}: expected {expected} version site(s) for {pattern!r}, found {count}")
        texts[name] = text
    changed = []
    for name, text in texts.items():
        path = root / name
        if _read(path) != text:
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(text)
            changed.append(name)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", help="new version, e.g. 0.3.1")
    args = parser.parse_args()
    try:
        changed = bump(args.version)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"Version {args.version}: updated {len(changed)} file(s)")
    for name in changed:
        print(f"  {name}")
    print(
        f'Next: git commit -am "Release {args.version}" && git tag v{args.version} && git push origin main v{args.version}'
    )


if __name__ == "__main__":
    main()
