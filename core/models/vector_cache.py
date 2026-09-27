"""Sentence and document vectors, cached across runs (plan 5.4).

Comparisons, scripts and repeated runs embed the same sentences again and
again: a second ``doc_embeddings`` over the same corpus cost the same as the
first (backlog 4d-7). This is the shared store -- parquet files under
``<data>/vectors/``, one per key -- read through by the embedding analyses.

The key is (model, model sha, unit, document sha256, the items' own text
hash). The items' hash is there because the embedded texts come from a parse:
two parsers can split one document into different sentences, and the same
document under a different parse is a different embedding job. The document's
sha already covers the project's text cleaning, since cleaning happens at
read time (plan 5.2).

Writes follow ``core/models/download.py``'s shape: ``<file>.partial`` then an
atomic rename, so a reader never sees half a matrix. Nothing here touches a
run directory (R3); ``VectorCache`` is a file writer and is allowlisted for
it in ``tests/test_write_custody.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["VectorCache", "VectorKey"]

#: Parquet keeps the vectors beside the parse cache's own format and reads
#: back without Python pickling (no code is loaded from a cache file).
_SUFFIX = ".parquet"


@dataclass(frozen=True, slots=True)
class VectorKey:
    """What makes a matrix of vectors the same job twice.

    *unit* is what one row is ("sentence" or "document"); *document* is the
    sha256 of the document's (already cleaned) text; *items* is the sha256 of
    the item texts in order, so a different parse is a different key.
    """

    model: str
    model_sha: str
    unit: str
    document: str
    items: str

    def file_name(self) -> str:
        """The cache's file name for this key: its hash, plus the suffix."""
        joined = "|".join((self.model, self.model_sha, self.unit, self.document, self.items))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest() + _SUFFIX


class VectorCache:
    """The store under *root* (the workspace's data directory)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / "vectors"

    @staticmethod
    def key(*, model: str, model_sha: str, unit: str, document: str, items: str | Sequence[str]) -> VectorKey:
        """A key whose *items* hash covers the item texts in order."""
        if isinstance(items, str):
            payload = items.encode("utf-8")
        else:
            payload = "\n".join(str(text) for text in items).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return VectorKey(model=model, model_sha=model_sha, unit=unit, document=document, items=digest)

    def load(self, key: VectorKey) -> np.ndarray | None:
        """The cached matrix, or None on a miss or an unreadable file.

        A damaged cache entry is a miss, never a failure: the caller recomputes.
        """
        path = self.root / key.file_name()
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError):
            return None
        if frame.empty:
            return None
        return np.asarray(frame.to_numpy(dtype=float), dtype=float)

    def save(self, key: VectorKey, vectors: np.ndarray) -> None:
        """Store one document's vectors; safe against a reader mid-write."""
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / key.file_name()
        partial = target.with_name(target.name + ".partial")
        pd.DataFrame(np.asarray(vectors, dtype=float)).to_parquet(partial, index=False)
        os.replace(partial, target)

    def forget(self) -> int:
        """Clear every entry (Settings' "Clear"); returns the files removed."""
        if not self.root.is_dir():
            return 0
        removed = 0
        for path in self.root.glob(f"*{_SUFFIX}"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
        for path in self.root.glob(f"*{_SUFFIX}.partial"):
            try:
                path.unlink()
            except OSError:
                continue
        return removed

    def size_bytes(self) -> int:
        """What the cache occupies on disk, for Settings."""
        if not self.root.is_dir():
            return 0
        return sum(path.stat().st_size for path in self.root.glob("*") if path.is_file())
