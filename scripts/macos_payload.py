"""Native macOS DMG-copy validation; never bypass Gatekeeper or modify signing."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import plistlib
import subprocess
from typing import Any

MACHO_MAGIC = {
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
}


def bundle_executable(app: Path, info: dict[str, Any], version: str) -> Path:
    name = info.get("CFBundleExecutable")
    if not isinstance(name, str) or not name or Path(name).name != name or any(c in name for c in "/\\:"):
        raise ValueError("Invalid Mac bundle executable name")
    if info.get("CFBundleIdentifier") != "org.nlpsuite.desktop" or info.get("CFBundleShortVersionString") != version:
        raise ValueError("Mac bundle identity/version mismatch")
    return app / "Contents/MacOS" / name


def run(args: list[str]) -> str:
    result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=120)  # noqa: S603
    if result.returncode:
        raise RuntimeError(f"{Path(args[0]).name} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


def copy_and_verify(dmg: Path, stage: Path, version: str) -> tuple[Path, Path]:
    """Exercise the distributable DMG, not only the pre-packaging .app tree."""
    mount = stage / "mounted-dmg"
    mount.mkdir()
    app = stage / "copied application" / "NLP Suite.app"
    run(["/usr/bin/hdiutil", "verify", str(dmg)])
    run(["/usr/bin/hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint", str(mount), str(dmg)])
    try:
        source = mount / "NLP Suite.app"
        if not source.is_dir() or source.is_symlink():
            raise ValueError("DMG does not contain the expected application")
        app.parent.mkdir()
        run(["/usr/bin/ditto", str(source), str(app)])
    finally:
        run(["/usr/bin/hdiutil", "detach", str(mount)])
    with (app / "Contents/Info.plist").open("rb") as handle:
        executable = bundle_executable(app, plistlib.load(handle), version)
    runtime = app / "Contents/Resources/nlp-runtime"
    for entry in (executable, runtime / "nlp-runtime"):
        if not entry.is_file() or not os.access(entry, os.X_OK):
            raise ValueError(f"Missing/non-executable copied program: {entry}")
    architecture = {"aarch64": "arm64", "amd64": "x86_64"}.get(platform.machine().lower(), platform.machine().lower())
    binaries = []
    for entry in app.rglob("*"):
        if entry.is_symlink():
            if not entry.resolve(strict=True).is_relative_to(app.resolve()):
                raise ValueError(f"Bundle symlink escapes application: {entry}")
            continue
        if not entry.is_file():
            continue
        with entry.open("rb") as handle:
            magic = handle.read(4)
        if magic not in MACHO_MAGIC:
            continue
        slices = run(["/usr/bin/lipo", "-archs", str(entry)]).split()
        if architecture not in slices:
            raise ValueError(f"Wrong architecture in {entry}: {slices}")
        run(["/usr/bin/codesign", "--verify", "--strict", str(entry)])
        binaries.append({"path": entry.relative_to(app).as_posix(), "architectures": slices})
    if not binaries:
        raise ValueError("No native binaries found in Mac application")
    run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)])
    (stage / "macos-validation.json").write_text(
        json.dumps(
            {
                "version": version,
                "macos": platform.mac_ver()[0],
                "architecture": architecture,
                "dmg_verified": True,
                "copied_app_signatures_verified": True,
                "gatekeeper_acceptance_verified": False,
                "binaries": binaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return executable, runtime
