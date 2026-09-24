"""Collect only this version's platform installers, with checksums and handoff notes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys


def release_files(bundle: Path, version: str, system: str) -> list[Path]:
    patterns = {"win32": ["nsis/*-setup.exe"], "darwin": ["dmg/*.dmg"], "linux": ["deb/*.deb", "appimage/*.AppImage"]}
    if system not in patterns:
        raise ValueError("Unsupported release platform")
    # Linux ships a deb and may drop the AppImage: the sandboxed CI runners
    # cannot run linuxdeploy, and the START-HERE instructions lead with the
    # deb. Every other platform needs exactly one installer of its only form.
    optional = {"linux": ["appimage/*.AppImage"]}
    selected = []
    for pattern in patterns[system]:
        matches = [path for path in bundle.glob(pattern) if f"_{version}_" in path.name or f"-{version}-" in path.name]
        expected = 0 if pattern in optional.get(system, ()) else 1
        if len(matches) < expected or (expected == 0 and len(matches) > 1) or (expected == 1 and len(matches) != 1):
            raise ValueError(f"Expected one {version} installer matching {pattern}; found {len(matches)}")
        selected.extend(matches)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = json.loads((root / "desktop/package.json").read_text(encoding="utf-8"))["version"]
    files = release_files(root / "desktop/src-tauri/target/release/bundle", version, sys.platform)
    args.destination.mkdir(parents=True, exist_ok=True)
    if any(args.destination.iterdir()):
        raise SystemExit("Use an empty release directory so old installers cannot be mixed with this release.")
    checksums = []
    for source in files:
        target = args.destination / source.name
        shutil.copy2(source, target)
        with target.open("rb") as handle:
            checksums.append(f"{hashlib.file_digest(handle, 'sha256').hexdigest()}  {target.name}")
    (args.destination / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    (args.destination / "START-HERE.txt").write_text(
        f"NLP Suite {version} — {sys.platform}/{platform.machine()}\n\n"
        "Install the package for YOUR operating system and processor. A Windows EXE does not run on a Mac.\n"
        "The app includes Python, English spaCy, document converters, WordNet, Gensim and VADER.\n"
        "Create a project, add your corpus folder, and start with Readability. Open Learn for tool guidance.\n"
        "Original documents are copied, not modified. Use Environment & setup to back up projects.\n"
        "Windows: run the setup EXE. WebView2's offline installer is included.\n"
        "Mac: open the DMG and drag NLP Suite to Applications. Match Apple Silicon or Intel.\n"
        "Linux: install the DEB with your package manager, or make the AppImage executable and launch it.\n"
        "Linux package dependencies and a supported desktop/webview environment are still required.\n"
        "These are beta artifacts, not a signing/notarization certificate or full legacy-parity guarantee.\n"
        "Do not disable OS security protections to open a rejected package; request a signed release.\n"
        "Restricted lexicons and transformer models are not bundled. See Learn and Setup.\n",
        encoding="utf-8",
    )
    print(f"Collected {len(files)} installer(s) for {sys.platform}/{platform.machine()} in {args.destination}")


if __name__ == "__main__":
    main()
