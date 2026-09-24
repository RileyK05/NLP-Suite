"""Verify native bridge + packaged runtime. Windows installer mode is CI-only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from collect_desktop_release import release_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-in-ci", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    release = root / "desktop/src-tauri/target/release"
    version = json.loads((root / "desktop/package.json").read_text(encoding="utf-8"))["version"]
    stage = root / "out/installed-payload-check"
    stage.mkdir(parents=True, exist_ok=False)
    executable = release / ("nlp-suite-desktop.exe" if sys.platform == "win32" else "nlp-suite-desktop")
    if sys.platform == "win32" and args.install_in_ci:
        if os.environ.get("GITHUB_ACTIONS") != "true":
            raise SystemExit(
                "Silent install validation is limited to an isolated GitHub runner to protect existing installations."
            )
        installer = release_files(release / "bundle", version, sys.platform)[0]
        destination = stage / "application"
        subprocess.run([str(installer), "/S", f"/D={destination}"], check=True, timeout=300)  # noqa: S603
        executable = destination / "nlp-suite-desktop.exe"
        runtime = destination / "nlp-runtime"
    elif sys.platform == "darwin":
        from macos_payload import copy_and_verify

        dmg = release_files(release / "bundle", version, sys.platform)[0]
        executable, runtime = copy_and_verify(dmg, stage, version)
    elif sys.platform == "linux":
        deb = next(path for path in release_files(release / "bundle", version, sys.platform) if path.suffix == ".deb")
        subprocess.run(["dpkg-deb", "--extract", str(deb), str(stage / "application")], check=True, timeout=60)  # noqa: S603,S607
        manifests = list((stage / "application").rglob("runtime.json"))
        if len(manifests) != 1:
            raise SystemExit("Expected exactly one installed runtime manifest")
        runtime = manifests[0].parents[2]
        executable = stage / "application/usr/bin/nlp-suite-desktop"
    else:
        runtime = root / "desktop/src-tauri/binaries/releases" / version / "nlp-runtime"
    subprocess.run(  # noqa: S603
        [str(executable), "--runtime-check", str(runtime), str(stage / "workspace")], check=True, timeout=120
    )
    runtime_exe = runtime / ("nlp-runtime.exe" if sys.platform == "win32" else "nlp-runtime")
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(root / "scripts/smoke_desktop.py"),
            str(runtime_exe),
            "--workspace",
            str(stage / "analysis-workspace"),
            "--with-parser",
            "--job-timeout",
            "300",
        ],
        check=True,
        timeout=1500,
    )
    print("Packaged runtime validation passed. Interactive OS/webview usability still requires acceptance.")


if __name__ == "__main__":
    main()
