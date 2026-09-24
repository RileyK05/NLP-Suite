"""Run the interactive frontend tests against the real desktop stack.

Boots :mod:`scripts.interactive_stack` -- the real server, the real corpus,
the shared cached parse, the built frontend -- hands its handshake to vitest
through the environment, and runs only the one test file that drives the
whole interface. Exits nonzero when the interface does not complete the
researcher's journey.

Every child gets a hard timeout and every stage prints as it happens, so a
hang is visible while it hangs rather than a morning surprise. Ctrl+C tears
the whole process tree down, not just this script.

Usage: python scripts/run_interactive_frontend_test.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

#: Longest wait for the server's handshake; the workspace build imports 87
#: documents and reads the cached parse, which is a minute on a slow disk.
HANDSHAKE_SECONDS = 300
#: Longest vitest run. The journey itself is capped inside the test file; this
#: is the outer fence that guarantees this script ends.
VITEST_SECONDS = 480
#: Grace period after stdin closes before a lingering stack process is killed.
SHUTDOWN_SECONDS = 20


def kill_tree(process: subprocess.Popen[Any]) -> None:
    """Terminate a process and its children, best effort."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            taskkill = shutil.which("taskkill") or "taskkill"
            subprocess.run(  # noqa: S603
                [taskkill, "/F", "/T", "/PID", str(process.pid)],
                check=False,
                capture_output=True,
            )
        else:
            process.send_signal(signal.SIGTERM)
    except OSError:
        pass
    process.kill()


def main() -> int:
    if shutil.which("node") is None:
        print("Node.js is required to drive the interface. Install Node and run again.")
        return 2
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm is None:
        print("npm is required to drive the interface. Install Node and run again.")
        return 2
    dist = ROOT / "desktop" / "dist" / "index.html"
    if not dist.is_file():
        print("Building the interface first (no desktop/dist/index.html).", flush=True)
        build = subprocess.run([npm, "run", "build"], cwd=ROOT / "desktop", check=False)  # noqa: S603
        if build.returncode != 0:
            print("The frontend build failed; the error above is the real cause.")
            return build.returncode

    with tempfile.TemporaryDirectory(prefix="nlp-interactive-env-") as env_dir:
        # The stack script reads the corpus location and the parse cache from
        # the same environment variables the pytest fixtures use.
        environment = dict(os.environ)
        environment["NLP_SUITE_INTERACTIVE_WORKSPACE_PARENT"] = env_dir
        environment["PYTHONUNBUFFERED"] = "1"
        stack = subprocess.Popen(  # noqa: S603
            [sys.executable, "-u", str(ROOT / "scripts" / "interactive_stack.py")],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
            env=environment,
        )
        assert stack.stdout is not None  # noqa: S101
        print("Booting the interactive stack (imports, cache read, server)...", flush=True)
        handshake = stack.stdout.readline()
        if not handshake:
            code = stack.wait(timeout=SHUTDOWN_SECONDS)
            print(f"The interactive stack failed to start (exit code {code}).")
            return 1
        connection = json.loads(handshake)
        print(
            f"Interactive stack up: {connection['baseUrl']} (workspace {connection['workspace']})",
            flush=True,
        )

        vitest_env = dict(os.environ)
        vitest_env["NLP_INTERACTIVE_BASE_URL"] = connection["baseUrl"]
        vitest_env["NLP_INTERACTIVE_TOKEN"] = connection["token"]
        print("Driving the interface (this is the slow part; live output follows)...", flush=True)
        try:
            vitest = subprocess.run(  # noqa: S603
                [npm, "exec", "--", "vitest", "run", "src/app.e2e.test.tsx"],
                cwd=ROOT / "desktop",
                env=vitest_env,
                check=False,
                timeout=VITEST_SECONDS,
            )
            return_code = vitest.returncode
        except subprocess.TimeoutExpired:
            print(f"vitest exceeded {VITEST_SECONDS}s; treating the journey as failed.")
            return_code = 1
        finally:
            # Closing stdin asks the stack to shut down; kill it if it lingers.
            if stack.stdin:
                stack.stdin.close()
            try:
                stack.wait(timeout=SHUTDOWN_SECONDS)
            except subprocess.TimeoutExpired:
                print("The stack did not exit on shutdown; killing its process tree.")
                kill_tree(stack)
        return return_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; the stack was torn down.")
        raise SystemExit(130) from None
