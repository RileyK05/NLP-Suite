"""Upload ``models/`` to the models release on the public repository.

Every published model's files go up as flat assets named ``<id>--<file>``
(``core.models.registry.release_url``) on the release tagged
``core.models._files.RELEASE`` (``models-1``), created if missing.

The release is a **prerelease that is never marked latest**: the app's
updater reads ``releases/latest/download/latest.json``, so a models release
that became "latest" would stop every installed app from finding updates.

An asset already there with the same SHA-256 is skipped, so an
interrupted upload is re-run, not restarted. Afterwards ``scripts/fetch_models.py``
into an empty folder proves the release serves what ``_files.py`` records.

Token: ``GITHUB_TOKEN``, else the one git's credential helper holds for
github.com.

    python scripts/publish_models.py            # every published model
    python scripts/publish_models.py --dry-run  # say what would be uploaded
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.models._files import RELEASE  # noqa: E402
from core.models.registry import MODELS  # noqa: E402

REPOSITORY = "RileyK05/NLP-Suite"
API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"


def token() -> str:
    found = os.environ.get("GITHUB_TOKEN", "").strip()
    if found:
        return found
    answer = subprocess.run(
        ["git", "credential", "fill"],  # noqa: S607 - the user's own git
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in answer.splitlines():
        if line.startswith("password="):
            return line.removeprefix("password=")
    raise SystemExit("no GitHub token: set GITHUB_TOKEN or sign in to github.com with git")


def call(
    method: str, url: str, secret: str, body: Any = None, *, data: Any = None, headers: dict[str, str] | None = None
) -> Any:
    sent = {"Authorization": f"Bearer {secret}", "Accept": "application/vnd.github+json", **(headers or {})}
    payload = data
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        sent["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=payload, method=method, headers=sent)  # noqa: S310 - fixed GitHub hosts
    with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310
        text = response.read().decode("utf-8")
    return json.loads(text) if text else None


def release(secret: str) -> dict[str, Any]:
    try:
        return call("GET", f"{API}/repos/{REPOSITORY}/releases/tags/{RELEASE}", secret)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    print(f"creating release {RELEASE} on {REPOSITORY} (prerelease, never latest)")
    return call(
        "POST",
        f"{API}/repos/{REPOSITORY}/releases",
        secret,
        {
            "tag_name": RELEASE,
            "target_commitish": "main",
            "name": "Models (set 1)",
            "body": (
                "Pretrained models the NLP Suite downloads or ships, converted to ONNX. "
                "The app fetches these itself; there is nothing here to install by hand.\n\n"
                "Each model's LICENSE and NOTICE (Apache-2.0, with its source and revision) is an asset beside it."
            ),
            "prerelease": True,
            "make_latest": "false",
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    planned: list[tuple[str, Path, int, str]] = []
    for spec in MODELS:
        for item in spec.files:
            path = args.models_dir / spec.id / item.name
            if not path.is_file() or path.stat().st_size != item.size:
                raise SystemExit(f"{path} is missing or not the recorded size; run scripts/export_models.py first")
            planned.append((f"{spec.id}--{item.name}", path, item.size, item.sha256))
    if args.dry_run:
        for name, _, size, _ in planned:
            print(f"would upload {name} ({size / 1e6:.1f} MB)")
        return 0
    secret = token()
    target = release(secret)
    if not target.get("prerelease"):
        raise SystemExit(
            f"release {RELEASE} exists but is not a prerelease; it could become 'latest' and break updates"
        )
    existing = {asset["name"]: asset for asset in target.get("assets", [])}
    for name, path, size, sha in planned:
        asset = existing.get(name)
        # GitHub reports an asset's SHA-256 as "digest"; a same-size file with
        # other bytes (a re-export) must be replaced, not kept.
        same = asset is not None and asset.get("state") == "uploaded" and asset.get("size") == size
        if same and asset is not None and asset.get("digest") in (f"sha256:{sha}", None):
            print(f"have   {name}")
            continue
        if asset is not None:
            call("DELETE", f"{API}/repos/{REPOSITORY}/releases/assets/{asset['id']}", secret)
        print(f"upload {name} ({size / 1e6:.1f} MB)", flush=True)
        with path.open("rb") as handle:
            call(
                "POST",
                f"{UPLOADS}/repos/{REPOSITORY}/releases/{target['id']}/assets?name={urllib.parse.quote(name)}",
                secret,
                data=handle,
                headers={"Content-Type": "application/octet-stream", "Content-Length": str(size)},
            )
    print(f"done: https://github.com/{REPOSITORY}/releases/tag/{RELEASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
