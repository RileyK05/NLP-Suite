"""MALLET topic-model adapter (FR-5.8).

MALLET is a Java binary outside this suite: ``train_topics`` shells to
``mallet`` on PATH (``shell=False``, argv list) and maps every failure
to a diagnostic. No binary here means every local run fails loudly with
the install pointer — the adapter is code-complete but binary-gated, and
the ledger says so.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from core.result import Diagnostic, Result

__all__ = ["mallet_binary", "parse_doc_topics", "train_topics"]

_MALLET_TIMEOUT = 600.0


def mallet_binary() -> str | None:
    """Resolved ``mallet`` executable, or None when absent."""
    found = shutil.which("mallet")
    return found if found else None


def train_topics(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    n_topics: int = 10,
    seed: int | None = None,
    binary: str | None = None,
) -> Result[pd.DataFrame]:
    """Run MALLET ``train-topics`` over a directory of .txt files.

    Returns the topic-keys table (Topic, Weight, Words). ``binary``
    overrides PATH resolution (tests use it to force the failure path).
    ``seed`` is passed as ``--random-seed`` so a run can be repeated: the
    legacy GUI offered no seed at all, which made "run it twice" a coin toss.
    """
    if isinstance(n_topics, bool) or not isinstance(n_topics, int) or n_topics < 1:
        return Result.failure(
            Diagnostic.error("MALLET_BAD_TOPICS", f"n_topics must be a positive int, got {n_topics!r}")
        )
    resolved = binary or mallet_binary()
    if resolved is None:
        return Result.failure(
            Diagnostic.error(
                "MALLET_MISSING",
                "no 'mallet' binary on PATH; MALLET topics need the Java binary (https://mimno.github.io/Mallet)",
                fix="install MALLET 2.x and put its 'mallet' on PATH",
            )
        )
    target = Path(output_dir)
    # R3: analysis modules never touch the filesystem for writing — the
    # caller owns the directory (MALLET itself writes its outputs there).
    if not target.is_dir():
        return Result.failure(
            Diagnostic.error(
                "MALLET_NO_OUTPUT_DIR",
                f"output directory does not exist: {target}",
                fix="create the output directory before calling train_topics",
            )
        )
    try:
        # The analyst's configured binary is the feature (shell=False,
        # fixed argv); unsanitized input cannot escape the argv array.
        argv = [
            resolved,
            "train-topics",
            "--input",
            str(input_dir),
            "--num-topics",
            str(n_topics),
            "--output-topic-keys",
            str(target / "mallet_topic_keys.txt"),
            "--output-doc-topics",
            str(target / "mallet_doc_topics.txt"),
        ]
        if seed is not None:
            argv.extend(["--random-seed", str(seed)])
        completed = subprocess.run(  # noqa: S603
            argv,
            capture_output=True,
            text=True,
            timeout=_MALLET_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Result.failure(Diagnostic.error("MALLET_FAILED", f"MALLET would not run: {exc}"))
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        hint = tail[-1] if tail else f"exit {completed.returncode}"
        return Result.failure(Diagnostic.error("MALLET_FAILED", f"MALLET train-topics failed: {hint}"))
    keys = target / "mallet_topic_keys.txt"
    try:
        rows = []
        for line in keys.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            rows.append({"Topic": int(parts[0]), "Weight": float(parts[1]), "Words": " ".join(parts[2:])})
    except (OSError, ValueError, IndexError) as exc:
        return Result.failure(Diagnostic.error("MALLET_BAD_KEYS", f"cannot read MALLET topic keys: {exc}"))
    return Result.success(pd.DataFrame(rows, columns=["Topic", "Weight", "Words"]))


def parse_doc_topics(path: str | Path) -> Result[pd.DataFrame]:
    """Read MALLET ``--output-doc-topics`` into a dominant-topic table.

    MALLET writes one line per document: the document's source name, then the
    document's topic as ``topic:proportion`` pairs (``0:0.91 1:0.09``) or as
    alternating ``topic proportion`` columns depending on version. Both are
    read; anything else is a loud diagnostic rather than a silent empty table,
    because "no document leans on any topic" and "we could not read the file"
    are different findings.

    Returns ``Document, Dominant topic, Contribution, Topic proportions`` —
    the same shape as the Gensim tool's dominant table, so HW2's "compare
    MALLET with Gensim" is a table comparison and not a format puzzle.
    """
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return Result.failure(Diagnostic.error("MALLET_BAD_DOCS", f"cannot read MALLET doc topics: {exc}"))
    rows: list[dict[str, object]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        name = next((t for t in tokens if ("/" in t or "\\" in t or "." in t) and not t.replace(".", "").isdigit()), "")
        if not name:
            continue
        numbers = [t for t in tokens if t is not name]
        pairs: list[tuple[int, float]] = []
        for token in numbers:
            if ":" in token:
                topic, _, proportion = token.partition(":")
                try:
                    pairs.append((int(topic), float(proportion)))
                except ValueError:
                    continue
            else:
                # Alternating topic proportion columns; collect greedily and
                # pair below. A stray non-numeric token just breaks the walk.
                pass
        if not pairs:
            floats: list[float] = []
            for token in numbers:
                try:
                    floats.append(float(token))
                except ValueError:
                    floats = []
                    break
            # First value is MALLET's document index in some versions.
            if len(floats) >= 3 and floats[0].is_integer() and len(floats) % 2 == 1:
                floats = floats[1:]
            pairs = [(int(floats[i]), floats[i + 1]) for i in range(0, len(floats) - 1, 2)]
        if not pairs:
            continue
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        dominant, contribution = pairs[0]
        rows.append(
            {
                "Document": name,
                "Dominant topic": dominant,
                "Contribution": round(contribution, 4),
                "Topic proportions": ", ".join(f"{t}:{round(p, 4)}" for t, p in pairs),
            }
        )
    if not rows:
        return Result.failure(
            Diagnostic.error("MALLET_BAD_DOCS", f"no document topics could be read from {source.name}")
        )
    return Result.success(
        pd.DataFrame(rows, columns=["Document", "Dominant topic", "Contribution", "Topic proportions"])
    )
