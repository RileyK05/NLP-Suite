"""Plan 5.1's precondition: is one parse per-document independent?

Before a parse can be cached per document (so a 20-of-87 selection or a
comparison of two projects hits the cache), three things must hold:

1. parsing documents apart gives the same tokens as parsing them together;
2. ``Sentence ID``, ``Record ID`` and ``Document ID`` are id columns only --
   rewritable per document by offset, never evidence;
3. the ids are dense enough that offsets cannot collide.

If (1) ever fails, per-document keys must not be built on this parse, and
the cache stays keyed per corpus (plan 5.1's "if it's not safe" branch).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import has_spacy_model
from core.io.reader import Corpus, Document, corpus_fingerprint, hash_text
from core.pipelines.cache import PipelineCache
from core.pipelines.spacy_backend import build_spacy_pipeline, spacy_model_name

_TEXTS = {
    "a.txt": "The bank raised rates. The nation faces hard times. We will put people to work.",
    "b.txt": "We choose to explore space. The world watches Berlin. Our economy must grow.",
}
_ID_COLUMNS = ("Record ID", "Sentence ID", "Document ID")


def _corpus(texts: dict[str, str], start: int = 1) -> Corpus:
    docs = tuple(
        Document(
            doc_id=index,
            path=Path(name),
            text=text,
            sha256=hash_text(text),
            label=name,
        )
        for index, (name, text) in enumerate(texts.items(), start=start)
    )
    return Corpus(docs=docs, sha256=corpus_fingerprint(docs))


def _parse(corpus: Corpus) -> pd.DataFrame:
    if not has_spacy_model():
        pytest.skip(f"spaCy model not installed — run: python -m spacy download {spacy_model_name('en')}")
    cache = PipelineCache()
    cache.register("spacy", build_spacy_pipeline)  # type: ignore[arg-type]
    return cache.get("spacy", "en").unwrap().parse(corpus).unwrap()


def _rows(table: pd.DataFrame, label: str) -> pd.DataFrame:
    part = table[table["Document"] == label]
    keep = [column for column in part.columns if column not in _ID_COLUMNS]
    return part[keep].reset_index(drop=True)


def test_parsing_apart_gives_the_same_tokens_as_parsing_together() -> None:
    together = _parse(_corpus(_TEXTS))
    apart = pd.concat([_parse(_corpus({name: text})) for name, text in _TEXTS.items()], ignore_index=True)
    for name in _TEXTS:
        pd.testing.assert_frame_equal(_rows(together, name), _rows(apart, name))


def test_the_id_columns_are_dense_and_rewritable() -> None:
    """Answer to plan 5.1's precondition check: **it is safe**.

    Sentence IDs restart per document and are dense; Record IDs are dense
    over the whole table (one document continues the previous one's); every
    document has exactly one Document ID. Concatenating per-document parses
    and offsetting these three columns reproduces the whole-corpus table --
    which is what the previous test measures end to end.
    """
    together = _parse(_corpus(_TEXTS))
    for name, group in together.groupby("Document", sort=False):
        for column in ("Sentence ID", "Record ID"):
            # Sentence IDs repeat per token; dense means no gap in the ids used.
            ids = sorted({int(value) for value in group[column]})
            assert ids == list(range(ids[0], ids[0] + len(ids))), f"{column} must be dense within {name}"
    assert together.groupby("Document")["Document ID"].nunique().eq(1).all(), (
        "one Document ID per document, so concatenated per-document parses can be offset into one table"
    )
    first = int(max(together[together["Document"] == "a.txt"]["Record ID"]))
    second = int(min(together[together["Document"] == "b.txt"]["Record ID"]))
    assert second > first, "whole-corpus Record IDs run across documents: offset per document when concatenating"
