"""Write latest.json, the file installed apps read to find an update.

Tauri's updater fetches releases/latest/download/latest.json, compares the
version with its own, and downloads the file for its platform, refusing it
unless the signature matches the public key built into the app. Linux debs
are not updatable by Tauri, so they are never listed.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

# Release asset suffix -> Tauri platform key.
PLATFORMS = {
    "_x64-setup.exe": "windows-x86_64",
    "_aarch64.app.tar.gz": "darwin-aarch64",
    "_x64.app.tar.gz": "darwin-x86_64",
}


def manifest(release: Path, signatures: Path, version: str, base_url: str, pub_date: str) -> dict[str, Any] | None:
    """The manifest for every signed asset, or None when nothing was signed."""
    platforms: dict[str, dict[str, str]] = {}
    for asset in sorted(release.iterdir()):
        for suffix, key in PLATFORMS.items():
            signature = signatures / f"{asset.name}.sig"
            if asset.name.endswith(suffix) and signature.is_file():
                platforms[key] = {
                    "signature": signature.read_text(encoding="utf-8").strip(),
                    "url": f"{base_url.rstrip('/')}/{asset.name}",
                }
    if not platforms:
        return None
    return {"version": version, "notes": f"NLP Suite {version}", "pub_date": pub_date, "platforms": platforms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True, help="staged release assets")
    parser.add_argument("--signatures", type=Path, required=True, help="updater .sig files, named after the assets")
    parser.add_argument("--version", required=True)
    parser.add_argument("--base-url", required=True, help="https://github.com/OWNER/REPO/releases/download/TAG")
    args = parser.parse_args()
    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    data = manifest(args.release, args.signatures, args.version, args.base_url, now)
    if data is None:
        print("No signed update files: latest.json not written (in-app updates need the signing secrets)")
        return
    (args.release / "latest.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"latest.json: {', '.join(sorted(data['platforms']))}")


if __name__ == "__main__":
    main()
