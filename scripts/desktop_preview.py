"""Build the frontend if it is out of date, then launch a browser preview of it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import webbrowser

# What the bundle is built from. If anything here is newer than the built
# index.html, the preview would otherwise show an older interface than the
# one in the working tree -- which looks exactly like a feature that was
# never written.
SOURCES = ("src", "index.html", "vite.config.ts", "tsconfig.json", "package.json")


def newest_source(desktop: Path) -> float:
    newest = 0.0
    for name in SOURCES:
        path = desktop / name
        if path.is_file():
            newest = max(newest, path.stat().st_mtime)
        elif path.is_dir():
            newest = max((f.stat().st_mtime for f in path.rglob("*") if f.is_file()), default=newest)
    return newest


def build_if_stale(desktop: Path, *, force: bool) -> None:
    """Rebuild when the sources have moved on, and say so while it happens."""
    built = desktop / "dist/index.html"
    if not force and built.is_file() and built.stat().st_mtime >= newest_source(desktop):
        return
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm is None:
        raise SystemExit(
            "The frontend is out of date and npm is not on PATH.\n"
            "Install Node.js, or build it yourself: cd desktop; npm install; npm run build"
        )
    reason = "Rebuilding" if built.is_file() else "Building"
    print(f"{reason} the interface (sources are newer than the last build)...", flush=True)
    result = subprocess.run([npm, "run", "build"], cwd=desktop, check=False)  # noqa: S603
    if result.returncode != 0:
        raise SystemExit(f"Frontend build failed (exit code {result.returncode}). The error above is the real cause.")
    if not built.is_file():
        raise SystemExit("The frontend build reported success but produced no desktop/dist/index.html.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("out/desktop-workspace"))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-build", action="store_true", help="Serve the existing bundle even if it is out of date.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the frontend even if it looks up to date.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    desktop = root / "desktop"
    if args.no_build:
        if not (desktop / "dist/index.html").is_file():
            raise SystemExit("Nothing built to serve. Drop --no-build, or: cd desktop; npm install; npm run build")
    else:
        build_if_stale(desktop, force=args.rebuild)
    command = [sys.executable, "-m", "desktop_backend.server", "--data-dir", str(args.data_dir.resolve()), "--desktop"]
    with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stdin=subprocess.PIPE, text=True) as proc:  # noqa: S603
        assert proc.stdout is not None  # noqa: S101
        first_line = proc.stdout.readline()
        if not first_line:
            # Server died before printing the handshake; let its traceback surface.
            code = proc.wait()
            raise SystemExit(
                f"NLP Suite engine failed to start (exit code {code}).\n"
                "The error above this message is the real cause -- most often the workspace "
                "is locked by another running instance."
            )
        connection = json.loads(first_line)
        url = f"{connection['baseUrl']}/#token={connection['token']}"
        print(f"Open this private local link (do not share):\n{url}\nPress Ctrl+C to stop after accepted jobs finish.")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            proc.wait()
        except KeyboardInterrupt:
            if proc.stdin:
                proc.stdin.close()
            proc.wait()


if __name__ == "__main__":
    main()
