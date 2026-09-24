"""R-C5: nothing a request or worker can reach is imported for the first time
on the request path.

This is the standard form of the fix for the topic-model hang: a tool adapter
imported ``gensim`` inside its function body, so the first import happened in
a request thread. On Windows that holds the loader lock while the main thread
needs it to grow the worker pool, and the server stops answering — no error,
no timeout, 0% CPU. Every shallow check passed, because they asked "is
``core.analysis.lda`` imported?" and it was; the cold thing was the *third
party package behind it*.

``tests/test_live_analysis.py`` holds the incident's own probe (live path
only). This file is the contract: it walks *every* adapter-reachable module —
the batch executor, the live bench, the table workflows and the job runner —
for third-party imports inside function bodies, and requires each one to be
named in ``desktop_backend.server.LAZY_BACKENDS`` and warm after
``preload_engine()``. A new lazy import anywhere in that chain fails here
rather than wedging a student's laptop.
"""

from __future__ import annotations

import subprocess
import sys

#: Modules whose function bodies must not hide an unpreloaded third-party
#: import. Everything the live bench, the batch runner or the table workflows
#: can reach from a request/worker thread belongs here.
ADAPTER_SOURCES = (
    "core/profiler/executor.py",
    "desktop_backend/live.py",
    "desktop_backend/tables.py",
    "desktop_backend/runner.py",
)

#: torch and transformers stay out on purpose: only ``core.analysis.contextual``
#: reaches them, no desktop tool does, and warming them costs seconds of
#: start-up and hundreds of megabytes. numpy/pandas/pyarrow are core
#: dependencies imported at module top across ``core/`` — warm from the first
#: line of any adapter, so a function-body mention of them is not a lazy load.
#: Kept here so each exclusion is a decision and not an oversight.
EXCLUDED = ("torch", "transformers", "numpy", "pandas", "pyarrow")

PROBE_TEMPLATE = """
import ast, importlib.util, sys
from pathlib import Path
from desktop_backend.server import preload_engine, LAZY_BACKENDS

SOURCES = %r
EXCLUDED = set(%r)

wanted = set()
for path_text in SOURCES:
    path = Path(path_text)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Module-level core imports: the wrappers a request reaches first.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("core."):
            wanted.add(node.module)
    # Function bodies: the level that actually matters. A wrapper being warm
    # says nothing about its backend -- core.analysis.lda imports gensim inside
    # fit_lda, so warming the wrapper left the DLL load in the request worker
    # while every shallow check passed.
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                name = node.module
            elif isinstance(node, ast.Import):
                name = node.names[0].name
            else:
                continue
            top = name.split(".")[0]
            if top in sys.stdlib_module_names or top in {"core", "tools", "desktop_backend"}:
                continue
            if top not in EXCLUDED:
                wanted.add(name)

# And one level further: the analysis modules those wrappers import, their
# function-body backends.
for module in sorted(wanted):
    path = Path(module.replace(".", "/") + ".py")
    if not path.is_file():
        continue
    for fn in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                name = node.module
            elif isinstance(node, ast.Import):
                name = node.names[0].name
            else:
                continue
            top = name.split(".")[0]
            if top in sys.stdlib_module_names or top in {"core", "tools", "desktop_backend"}:
                continue
            if top not in EXCLUDED:
                wanted.add(name)

backends = {w for w in wanted if w.split(".")[0] not in sys.stdlib_module_names
            and w.split(".")[0] not in {"core", "tools", "desktop_backend"}}

preload_engine()
# "Cold" means installed but left for a request thread: a missing optional
# extra cannot wedge anything -- its ImportError loads no native extension and
# the tool reports it cleanly. Only the installable-but-unimported are fatal.
def installed(name):
    try:
        return importlib.util.find_spec(name.split(".")[0]) is not None
    except (ImportError, ValueError):
        return False

print({
    "backends": sorted(backends),
    "cold": sorted(b for b in backends if installed(b) and b.split(".")[0] not in sys.modules),
    "unlisted": sorted(b for b in backends if b not in LAZY_BACKENDS),
})
"""
PROBE = PROBE_TEMPLATE % (ADAPTER_SOURCES, EXCLUDED)


def _probe() -> dict[str, list[str]]:
    """Run the AST walk + preload in a clean interpreter and read its report."""
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, f"probe crashed:\n{done.stderr}"
    line = done.stdout.strip().splitlines()[-1]
    return eval(line)  # noqa: S307 -- output of the probe above, not user input


def test_every_lazy_backend_is_preloaded_and_listed() -> None:
    """Cold, or unlisted, is the wedge waiting to happen.

    * cold: the first import lands in a request/worker thread (loader lock).
    * unlisted: leaving it cold was never a decision, so it will regress the
      moment someone adds another lazy import.
    """
    report = _probe()
    assert report["backends"], "the probe found no backends; it needs rewriting"
    assert report["cold"] == [], (
        f"preload_engine() leaves {report['cold']} for a request thread to import. "
        "Loading native extensions there holds the Windows loader lock while the "
        "main thread needs it, and the server stops answering with no error. "
        "Add them to LAZY_BACKENDS in desktop_backend/server.py."
    )
    assert report["unlisted"] == [], (
        f"{report['unlisted']} is imported inside an analysis function but is not "
        "named in LAZY_BACKENDS. Add it there (or to this test's EXCLUDED with the "
        "reason), so leaving it cold stays a decision rather than an oversight."
    )


def test_the_exclusions_are_named_and_defended() -> None:
    """The excluded heavyweights must not creep into a desktop-reachable path.

    They are excluded because no desktop tool reaches them. If that changes,
    the exclusion stops being a saving and becomes the bug this file exists for.
    """
    assert "torch" in EXCLUDED and "transformers" in EXCLUDED
    report = _probe()
    # Sanity: exclusions actually keep those names out of the required set.
    for name in report["backends"]:
        assert name.split(".")[0] not in EXCLUDED, (
            f"{name} is desktop-reachable but excluded from preloading; "
            "either warm it or remove it from the reachable chain"
        )
