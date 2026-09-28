"""MALLET topic-model adapter (FR-5.8).

MALLET is a Java binary outside this suite: ``train_topics`` shells to
``mallet`` on PATH (``shell=False``, argv list) and maps every failure
to a diagnostic. No binary here means every local run fails loudly with
the install pointer — the adapter is code-complete but binary-gated, and
the ledger says so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

from core.result import Diagnostic, Result

__all__ = ["mallet_binary", "parse_doc_topics", "train_topics"]

_MALLET_TIMEOUT = 600.0


def mallet_binary() -> str | None:
    """Resolve the MALLET launcher from PATH or a standard MALLET install.

    MALLET's Windows distribution is normally unpacked and exposed through
    ``MALLET_HOME`` rather than installed into a system-wide PATH.  The
    launcher is called ``mallet.bat`` there, while Unix installs generally
    expose ``bin/mallet``.
    """
    found = shutil.which("mallet")
    if found:
        return found
    mallet_home = os.environ.get("MALLET_HOME")
    if not mallet_home:
        return None
    for candidate in (Path(mallet_home) / "bin" / "mallet", Path(mallet_home) / "bin" / "mallet.bat"):
        if candidate.is_file():
            return str(candidate)
    return None


def _mallet_env(binary: str) -> dict[str, str]:
    """Return a subprocess environment that makes MALLET's launcher usable."""
    environment = dict(os.environ)
    if not environment.get("MALLET_HOME"):
        # A standard install is <MALLET_HOME>/bin/mallet[.bat].  Supplying the
        # variable is required by the Windows launcher and is harmless for the
        # Unix launcher.
        environment["MALLET_HOME"] = str(Path(binary).expanduser().resolve().parent.parent)
    return environment


def _run_mallet(argv: Sequence[str], *, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    """Run one fixed-argv MALLET command."""
    return subprocess.run(  # noqa: S603
        list(argv),
        capture_output=True,
        text=True,
        timeout=_MALLET_TIMEOUT,
        check=False,
        env=dict(env),
    )


def _failure_hint(completed: subprocess.CompletedProcess[str]) -> str:
    tail = (completed.stderr or completed.stdout or "").strip().splitlines()
    return tail[-1] if tail else f"exit {completed.returncode}"


def train_topics(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    n_topics: int = 10,
    seed: int | None = None,
    binary: str | None = None,
    optimize_interval: int = 20,
    num_iterations: int = 1000,
) -> Result[pd.DataFrame]:
    """Import and train MALLET LDA over a directory of .txt files.

    Returns the topic-keys table (Topic, Weight, Words). ``binary``
    overrides PATH resolution (tests use it to force the failure path).
    ``seed`` is passed as ``--random-seed`` so a run can be repeated: the
    legacy GUI offered no seed at all, which made "run it twice" a coin toss.
    """
    if isinstance(n_topics, bool) or not isinstance(n_topics, int) or n_topics < 1:
        return Result.failure(
            Diagnostic.error("MALLET_BAD_TOPICS", f"n_topics must be a positive int, got {n_topics!r}")
        )
    if isinstance(optimize_interval, bool) or not isinstance(optimize_interval, int) or optimize_interval < 1:
        return Result.failure(
            Diagnostic.error(
                "MALLET_BAD_OPTIMIZE_INTERVAL",
                f"optimize_interval must be a positive int, got {optimize_interval!r}",
            )
        )
    if isinstance(num_iterations, bool) or not isinstance(num_iterations, int) or num_iterations < 1:
        return Result.failure(
            Diagnostic.error(
                "MALLET_BAD_ITERATIONS",
                f"num_iterations must be a positive int, got {num_iterations!r}",
            )
        )
    resolved = binary or mallet_binary()
    if resolved is None:
        return Result.failure(
            Diagnostic.error(
                "MALLET_MISSING",
                "MALLET was not found on PATH or under MALLET_HOME/bin; MALLET topics need the Java binary "
                "(https://mimno.github.io/Mallet)",
                fix="install MALLET 2.x, set MALLET_HOME to its install directory, and put MALLET_HOME/bin on PATH",
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
    environment = _mallet_env(resolved)
    instances = target / "corpus.mallet"
    import_argv = [
        resolved,
        "import-dir",
        "--input",
        str(input_dir),
        "--output",
        str(instances),
        "--keep-sequence",
        "--remove-stopwords",
    ]
    argv = [
        resolved,
        "train-topics",
        "--input",
        str(instances),
        "--num-topics",
        str(n_topics),
        "--optimize-interval",
        str(optimize_interval),
        "--num-iterations",
        str(num_iterations),
        "--output-topic-keys",
        str(target / "mallet_topic_keys.txt"),
        "--output-doc-topics",
        str(target / "mallet_doc_topics.txt"),
    ]
    if seed is not None:
        argv.extend(["--random-seed", str(seed)])
    try:
        # MALLET train-topics consumes a serialized InstanceList, not the
        # source directory.  Import first, retaining sequence information so
        # train-topics can restore the instance list.
        imported = _run_mallet(import_argv, env=environment)
        if imported.returncode != 0:
            return Result.failure(
                Diagnostic.error("MALLET_FAILED", f"MALLET import-dir failed: {_failure_hint(imported)}")
            )
        completed = _run_mallet(argv, env=environment)
    except (OSError, subprocess.SubprocessError) as exc:
        return Result.failure(Diagnostic.error("MALLET_FAILED", f"MALLET would not run: {exc}"))
    if completed.returncode != 0:
        return Result.failure(
            Diagnostic.error("MALLET_FAILED", f"MALLET train-topics failed: {_failure_hint(completed)}")
        )
    keys = target / "mallet_topic_keys.txt"
    try:
        rows = []
        for line in keys.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split(maxsplit=2)
            rows.append({"Topic": int(parts[0]), "Weight": float(parts[1]), "Words": " ".join(parts[2:])})
    except (OSError, ValueError, IndexError) as exc:
        return Result.failure(Diagnostic.error("MALLET_BAD_KEYS", f"cannot read MALLET topic keys: {exc}"))
    return Result.success(pd.DataFrame(rows, columns=["Topic", "Weight", "Words"]))


def _document_name(raw_name: str, document_map: Mapping[int, str] | None) -> str:
    """Turn MALLET's file URL into the corpus document name."""
    parsed = urlparse(raw_name)
    is_file_url = parsed.scheme == "file"
    if is_file_url:
        decoded = unquote(parsed.path or parsed.netloc)
    else:
        decoded = unquote(raw_name)
    decoded = decoded.replace("\\", "/").rstrip("/")
    basename = decoded.rsplit("/", 1)[-1]
    position_match = re.match(r"^(\d{4})_", basename)
    position = int(position_match.group(1)) if position_match else None
    clean_basename = re.sub(r"^\d{4}_", "", basename)
    if document_map is not None and position is not None and position in document_map:
        return document_map[position]
    if is_file_url or "/" not in decoded:
        return clean_basename
    return f"{decoded.rsplit('/', 1)[0]}/{clean_basename}"


def _parse_topic_columns(
    columns: list[str],
    *,
    n_topics: int | None,
    tab_separated: bool,
) -> list[tuple[int, float]]:
    """Parse one document's topic columns, preferring the explicit arity."""
    topic_fields = " ".join(columns).split()
    if not topic_fields:
        return []

    colon_pairs: list[tuple[int, float]] = []
    for token in topic_fields:
        if ":" not in token:
            colon_pairs = []
            break
        topic_token, _, proportion_token = token.partition(":")
        try:
            colon_pairs.append((int(topic_token), float(proportion_token)))
        except ValueError:
            colon_pairs = []
            break
    if colon_pairs:
        return colon_pairs

    try:
        floats = [float(token) for token in topic_fields]
    except ValueError:
        return []

    if n_topics is not None:
        if len(floats) == n_topics:
            return list(enumerate(floats))
        if len(floats) == 2 * n_topics:
            pairs: list[tuple[int, float]] = []
            for index in range(0, len(floats), 2):
                topic_value = floats[index]
                if not topic_value.is_integer():
                    return []
                pairs.append((int(topic_value), floats[index + 1]))
            return pairs
        return []

    # The current MALLET format is tabular and has one float column per topic.
    # Without an explicit n_topics, retain that interpretation for tab-separated
    # rows.  Legacy whitespace rows use alternating topic/proportion columns.
    if tab_separated or len(floats) == 2:
        return list(enumerate(floats))
    if len(floats) % 2 == 0:
        pairs = []
        for index in range(0, len(floats), 2):
            topic_value = floats[index]
            if not topic_value.is_integer():
                return []
            pairs.append((int(topic_value), floats[index + 1]))
        return pairs
    return []


def parse_doc_topics(
    path: str | Path,
    n_topics: int | None = None,
    document_map: Mapping[int, str] | None = None,
) -> Result[pd.DataFrame]:
    """Read MALLET ``--output-doc-topics`` into a dominant-topic table.

    MALLET 2.0.8 writes tab-separated rows with a document index, a file URL,
    then one proportion per topic.  Older versions used ``topic:proportion``
    pairs or alternating ``topic proportion`` columns.  ``n_topics`` selects
    the numeric format by column count, preventing the document index from
    being mistaken for a topic.  ``document_map`` maps the staging position
    prefix back to the corpus's own document name.

    Returns ``Document, Dominant topic, Contribution, Topic proportions`` —
    the same shape as the Gensim tool's dominant table, so HW2's "compare
    MALLET with Gensim" is a table comparison and not a format puzzle.
    """
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return Result.failure(Diagnostic.error("MALLET_BAD_DOCS", f"cannot read MALLET doc topics: {exc}"))
    if n_topics is not None and (isinstance(n_topics, bool) or not isinstance(n_topics, int) or n_topics < 1):
        return Result.failure(Diagnostic.error("MALLET_BAD_DOCS", f"n_topics must be a positive int, got {n_topics!r}"))
    rows: list[dict[str, object]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tab_separated = "\t" in stripped
        columns = stripped.split("\t") if tab_separated else stripped.split()
        if len(columns) < 2:
            continue
        try:
            int(columns[0])
        except ValueError:
            # Legacy output can omit the document index and start with the
            # source name (including a tab-separated ``name<TAB>0:...`` row).
            name = _document_name(columns[0], document_map)
            topic_columns = columns[1:]
        else:
            if len(columns) < 3:
                continue
            name = _document_name(columns[1], document_map)
            topic_columns = columns[2:]
        pairs = _parse_topic_columns(topic_columns, n_topics=n_topics, tab_separated=tab_separated)
        if not pairs:
            continue
        dominant, contribution = max(pairs, key=lambda pair: pair[1])
        ordered_pairs = sorted(pairs, key=lambda pair: pair[0])
        rows.append(
            {
                "Document": name,
                "Dominant topic": dominant,
                "Contribution": round(contribution, 4),
                "Topic proportions": ", ".join(f"{t}:{round(p, 4)}" for t, p in ordered_pairs),
            }
        )
    if not rows:
        return Result.failure(
            Diagnostic.error("MALLET_BAD_DOCS", f"no document topics could be read from {source.name}")
        )
    return Result.success(
        pd.DataFrame(rows, columns=["Document", "Dominant topic", "Contribution", "Topic proportions"])
    )
