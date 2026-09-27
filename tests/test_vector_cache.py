"""The embedding vector cache (plan 5.4): keyed tightly, damaged entries are misses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.models.vector_cache import VectorCache, VectorKey


def _key(**over: str) -> VectorKey:
    base = {"model": "m", "model_sha": "sha", "unit": "sentence", "document": "doc", "items": "items"}
    base.update(over)
    return VectorKey(**base)  # type: ignore[arg-type]


class TestKeys:
    def test_the_key_covers_model_unit_document_and_the_items(self) -> None:
        names = {
            _key().file_name(),
            _key(model="other").file_name(),
            _key(model_sha="other").file_name(),
            _key(unit="document").file_name(),
            _key(document="other").file_name(),
            _key(items="other").file_name(),
        }
        assert len(names) == 6

    def test_the_items_hash_covers_the_texts_in_order(self) -> None:
        first = VectorCache.key(model="m", model_sha="", unit="sentence", document="d", items=["a", "b"])
        second = VectorCache.key(model="m", model_sha="", unit="sentence", document="d", items=["b", "a"])
        assert first != second, "a different parse of one document is a different job"


class TestStore:
    def test_round_trip(self, tmp_path: Path) -> None:
        cache = VectorCache(tmp_path)
        vectors = np.array([[1.0, 2.0], [3.0, 4.0]])
        key = _key()
        assert cache.load(key) is None
        cache.save(key, vectors)
        got = cache.load(key)
        assert got is not None
        assert np.allclose(got, vectors)

    def test_a_damaged_entry_is_a_miss_not_a_failure(self, tmp_path: Path) -> None:
        cache = VectorCache(tmp_path)
        key = _key()
        cache.save(key, np.array([[1.0]]))
        (cache.root / key.file_name()).write_bytes(b"not a parquet")
        assert cache.load(key) is None

    def test_an_empty_entry_is_a_miss(self, tmp_path: Path) -> None:
        cache = VectorCache(tmp_path)
        key = _key()
        cache.root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_parquet(cache.root / key.file_name(), index=False)
        assert cache.load(key) is None

    def test_forget_clears_and_counts(self, tmp_path: Path) -> None:
        cache = VectorCache(tmp_path)
        cache.save(_key(), np.array([[1.0]]))
        cache.save(_key(document="other"), np.array([[2.0]]))
        assert cache.size_bytes() > 0
        assert cache.forget() == 2
        assert cache.size_bytes() == 0
        assert cache.forget() == 0

    def test_no_partial_is_left_behind(self, tmp_path: Path) -> None:
        cache = VectorCache(tmp_path)
        cache.save(_key(), np.array([[1.0, 2.0]]))
        assert list(cache.root.glob("*.partial")) == []
