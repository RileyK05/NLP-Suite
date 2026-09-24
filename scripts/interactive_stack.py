"""Boot the real desktop stack for interactive frontend tests.

The workspace tests drive the app's own HTTP calls; the rendered tests mount
components against fakes. Neither establishes that the application a
researcher opens -- the built bundle, this server, the real corpus -- answers
when driven the way it is actually driven. This script boots exactly that
stack, in a throwaway workspace:

* the real ``desktop_backend.server`` on a free port, over a workspace
  holding the real documents;
* the shared cached parse, so warming costs a parquet read rather than a
  minute-long parse;
* the built frontend at ``desktop/dist``, served by the same server a
  browser would load it from.

It prints one JSON line -- the server's own handshake, extended with the
workspace location -- and waits on stdin, exactly as
:mod:`scripts.desktop_preview` does. ``scripts.run_interactive_frontend_test``
hands that line to vitest, which drives the interface by clicking and reports
its result through its exit code; this process is shut down from its stdin.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
# conftest.py lives at the repo root and is not an installed package; this
# script imports its prime_annotations helper directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def corpus_documents() -> list[Path]:
    configured = os.environ.get("NLP_SUITE_REAL_CORPUS")
    folder = Path(configured) if configured else Path.home() / "Downloads" / "POTUS State of the Union 1934-2024"
    documents = sorted(folder.glob("*.txt"))
    if not documents:
        raise SystemExit(f"No real corpus at {folder}; set NLP_SUITE_REAL_CORPUS to a folder of .txt documents.")
    return documents


def build_workspace(target: Path) -> None:
    """A workspace holding the real documents and the shared cached parse."""
    from conftest import prime_annotations
    from core.io.reader import (
        Corpus,
        Document,
        corpus_fingerprint,
        date_from_filename,
        hash_text,
        read_text,
    )
    from desktop_backend.live import Bench
    from desktop_backend.store import Workspace

    documents = corpus_documents()
    workspace = Workspace(target)
    project = workspace.create("State of the Union")
    for path in documents:
        workspace.import_document(project["id"], path.name, path.read_bytes())

    cache_root = Path(
        os.environ.get(
            "NLP_SUITE_REAL_CORPUS_CACHE",
            str(Path(tempfile.gettempdir()) / "nlp-suite-real-parses"),
        )
    )
    docs = []
    for index, path in enumerate(documents, 1):
        text = read_text(path).unwrap()
        docs.append(
            Document(
                index,
                path,
                text,
                date_from_filename(path),
                hash_text(text),
                f"doc-{index}",
                path.name,
            )
        )
    frozen = tuple(docs)
    corpus = Corpus(frozen, corpus_fingerprint(frozen))
    # Warming here obtains the content-addressed key the pytest fixtures use
    # and copies their parse into this workspace; a cold cache just costs the
    # parse itself, once, which is also what a real session would pay.
    snapshot = Bench(cache_root).warm(corpus, "spacy")
    prime_annotations(target, snapshot.key, cache_root)


def main() -> None:
    parent = Path(os.environ.get("NLP_SUITE_INTERACTIVE_WORKSPACE_PARENT", tempfile.gettempdir()))
    workspace_root = Path(tempfile.mkdtemp(prefix="nlp-interactive-", dir=parent))
    try:
        build_workspace(workspace_root)
    except Exception as exc:
        shutil.rmtree(workspace_root, ignore_errors=True)
        raise SystemExit(f"Could not prepare the interactive workspace: {exc}") from exc

    # The frontend test's jsdom origin is http://localhost:3000, which the
    # app's own connect() trusts; binding the server there means the built
    # connection flow needs no rewriting inside the test.
    command = [
        sys.executable,
        "-m",
        "desktop_backend.server",
        "--data-dir",
        str(workspace_root),
        "--port",
        "3000",
    ]
    with subprocess.Popen(  # noqa: S603
        command, cwd=ROOT, stdout=subprocess.PIPE, stdin=subprocess.PIPE, text=True
    ) as proc:
        assert proc.stdout is not None  # noqa: S101
        # A readline with no bound would wait forever on a server that died
        # without printing; the handshake either arrives or the engine is
        # dead, and waiting longer cannot change which.
        import queue as queue_module

        lines: queue_module.Queue[str] = queue_module.Queue()
        threading.Thread(
            target=lambda: lines.put(proc.stdout.readline() if proc.stdout else ""),
            daemon=True,
        ).start()
        try:
            handshake = lines.get(timeout=120)
        except queue_module.Empty:
            handshake = ""
        if not handshake:
            if os.name == "nt":
                taskkill = shutil.which("taskkill") or "taskkill"
                subprocess.run(  # noqa: S603
                    [taskkill, "/F", "/T", "/PID", str(proc.pid)],
                    check=False,
                    capture_output=True,
                )
            code = proc.wait()
            shutil.rmtree(workspace_root, ignore_errors=True)
            raise SystemExit(f"The desktop engine failed to start (exit code {code}).")
        connection = json.loads(handshake)
        connection["workspace"] = str(workspace_root)
        print(json.dumps(connection), flush=True)
        try:
            proc.wait()
        finally:
            shutil.rmtree(workspace_root, ignore_errors=True)


if __name__ == "__main__":
    main()
