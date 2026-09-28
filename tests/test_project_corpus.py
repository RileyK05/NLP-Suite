"""One corpus reader and one parse cache for runs, the bench and scripts.

Before this module existed a published run parsed its documents every time:
two identical corpus_statistics runs over twenty speeches took 57.6 s and
49.4 s, the second as slow as the first, while the Interactive page kept the
same parse on disk. After it, the second run took 1.1 s. These tests pin the
behaviour that produced that: the second caller reads the parse back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.result import Result
from desktop_backend.project_corpus import load_corpus, parse_cached
from desktop_backend.runner import Runner, run_job
from desktop_backend.store import Workspace


class CountingPipeline:
    """The real spaCy pipeline, counting how often it is asked to parse."""

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.parses = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    def parse(self, corpus: object) -> object:
        self.parses += 1
        return self.inner.parse(corpus)  # type: ignore[attr-defined]


@pytest.fixture
def pipeline() -> CountingPipeline:
    pytest.importorskip("spacy")
    from desktop_backend.project_corpus import resolve_parser

    resolved = resolve_parser("spacy")
    if resolved.value is None:
        pytest.skip("spaCy's English model is not installed")
    return CountingPipeline(resolved.unwrap())


def _project(root: Path) -> tuple[Workspace, str]:
    workspace = Workspace(root)
    project = workspace.create("Cache")
    workspace.import_document(
        project["id"], "1990-01-01_first.txt", b"The first speech is short. It has two sentences."
    )
    workspace.import_document(project["id"], "1991-01-01_second.txt", b"A second speech follows. It is also short.")
    return workspace, project["id"]


def test_second_parse_of_the_same_corpus_is_read_back(tmp_path: Path, pipeline: CountingPipeline) -> None:
    workspace, project_id = _project(tmp_path)
    corpus, _ = load_corpus(workspace, project_id, workspace.documents(project_id))

    first = parse_cached(workspace.root, corpus, pipeline)
    second = parse_cached(workspace.root, corpus, pipeline)

    assert (first.source, second.source) == ("parsed", "disk")
    assert pipeline.parses == 1
    assert first.key == second.key
    assert second.table.equals(first.table)


def test_a_different_selection_is_a_different_parse(tmp_path: Path, pipeline: CountingPipeline) -> None:
    workspace, project_id = _project(tmp_path)
    documents = workspace.documents(project_id)
    both, _ = load_corpus(workspace, project_id, documents)
    one, _ = load_corpus(workspace, project_id, documents[:1])

    assert parse_cached(workspace.root, both, pipeline).key != parse_cached(workspace.root, one, pipeline).key
    # The keys differ, but the one document both corpora hold was parsed once:
    # per-document entries under the corpus key (plan 5.1).
    assert pipeline.parses == 1


def test_the_same_documents_in_another_order_are_another_parse(tmp_path: Path, pipeline: CountingPipeline) -> None:
    """Order matters in a parse: its ids are positions.

    The fingerprint is order-independent (a restored project is the same
    snapshot), but the *table* is not: Document/Sentence/Record IDs are
    positions. Keying the parse cache by the order-free hash meant a corpus
    read back the table of a different ordering -- the same documents numbered
    the other way -- so ids 1..N named the wrong documents, and
    ``nlp.keyness(a, b)`` compared b against a. The second ordering must not
    reuse the first ordering's table.
    """
    from dataclasses import replace

    from core.io.reader import Corpus, corpus_fingerprint

    workspace, project_id = _project(tmp_path)
    documents = workspace.documents(project_id)
    forward, _ = load_corpus(workspace, project_id, documents)
    reversed_docs = tuple(replace(doc, doc_id=i) for i, doc in enumerate(reversed(forward.docs), 1))
    backward = Corpus(reversed_docs, corpus_fingerprint(reversed_docs))
    assert forward.sha256 == backward.sha256, "the premise: the order-free hash is the same"

    first = parse_cached(workspace.root, forward, pipeline)
    second = parse_cached(workspace.root, backward, pipeline)
    # A different corpus key: the reordered table is not the forward table.
    assert first.key != second.key
    # The reordered parse describes the reordered corpus: its first document
    # id names the document that is first in that order.
    labels = second.table.drop_duplicates("Document ID").set_index("Document ID")["Document"]
    assert labels["1"] == backward.docs[0].name
    # Both orderings share the per-document entries (5.1), so nothing was
    # parsed twice -- the reused slices were just renumbered into the new order.
    assert pipeline.parses == 1


def test_a_comparison_only_parses_the_documents_it_has_not_seen(tmp_path: Path, pipeline: CountingPipeline) -> None:
    """The regression 5.1 exists for: a joined corpus reuses both sides' parses."""
    workspace, project_id = _project(tmp_path)
    documents = workspace.documents(project_id)
    speeches, _ = load_corpus(workspace, project_id, documents)
    parse_cached(workspace.root, speeches, pipeline)
    assert pipeline.parses == 1

    workspace.import_document(project_id, "1992-01-01_third.txt", b"A third speech arrives late.")
    grown, _ = load_corpus(workspace, project_id, workspace.documents(project_id))
    again = parse_cached(workspace.root, grown, pipeline)
    # Only the new document costs a parse; the first two come off disk, and
    # the assembled table is whole: dense Record IDs across the three.
    assert pipeline.parses == 2
    assert again.source == "parsed"
    assert sorted(again.table["Record ID"].astype(int)) == list(range(1, len(again.table) + 1))
    assert set(again.table["Document ID"].astype(str)) == {str(doc.doc_id) for doc in grown.docs}


def test_a_failed_cache_write_does_not_fail_the_parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache is optional even when its per-document write fails."""
    from types import SimpleNamespace

    import pandas as pd

    from core.io.reader import Corpus, Document, corpus_fingerprint, hash_text
    from desktop_backend.live import Annotations

    doc = Document(doc_id=1, path=Path("one.txt"), text="One word.", sha256=hash_text("One word."))
    corpus = Corpus((doc,), corpus_fingerprint((doc,)))
    table = pd.DataFrame({"Document ID": ["1"], "Document": ["one.txt"], "Record ID": [1], "Form": ["One"]})
    pipeline = SimpleNamespace(backend="fake", language="en", model="", parse=lambda _: Result.success(table))

    def fail_store(_cache: Annotations, _key: str, _frame: pd.DataFrame) -> None:
        raise OSError("cache is read-only")

    monkeypatch.setattr(Annotations, "store", fail_store)
    parsed = parse_cached(tmp_path, corpus, pipeline)
    assert parsed.source == "parsed"
    assert parsed.table["Form"].tolist() == ["One"]


def test_a_changed_document_is_refused_not_read(tmp_path: Path) -> None:
    workspace, project_id = _project(tmp_path)
    documents = workspace.documents(project_id)
    stored = workspace.project_dir(project_id) / "corpus" / documents[0]["stored_name"]
    stored.write_bytes(b"Edited behind the app's back.")
    with pytest.raises(ValueError, match="changed since import"):
        load_corpus(workspace, project_id, documents)


def test_published_runs_share_the_cache(
    tmp_path: Path, pipeline: CountingPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression itself: a second identical run must not parse again."""
    workspace, project_id = _project(tmp_path / "workspace")
    monkeypatch.setattr("desktop_backend.runner.resolve_parser", lambda parser: Result.success(pipeline))
    monkeypatch.setenv("NLP_SUITE_RUN_FIGURES", "0")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    try:
        jobs = [runner.submit(project_id, "corpus_statistics", {}, "spacy") for _ in range(2)]
        assert [run_job(workspace.root, job["id"]) for job in jobs] == [0, 0]
    finally:
        runner.close()
    assert pipeline.parses == 1
    assert {job["state"] for job in workspace.jobs(project_id)} == {"DONE"}


class TestCacheCap:
    """The parse cache holds a bounded amount of recomputable work (plan 5.1)."""

    @staticmethod
    def _annotations(tmp_path: Path) -> Any:
        from desktop_backend.live import Annotations

        return Annotations(tmp_path)

    @staticmethod
    def _frame() -> Any:
        import pandas as pd

        return pd.DataFrame({"Form": ["word"] * 400})

    def test_reading_a_parse_keeps_it_and_the_cap_drops_the_forgotten(self, tmp_path: Path) -> None:
        import os
        import time

        annotations = self._annotations(tmp_path)
        for name in ("older", "newer"):
            annotations.store(name, self._frame())
        older, newer = annotations.path("older"), annotations.path("newer")
        os.utime(older, (time.time() - 60, time.time() - 60))
        annotations.load("older")  # read is use: the old file is now the recent one
        assert older.stat().st_mtime >= newer.stat().st_mtime

        assert annotations.prune(limit=annotations.size_bytes() - 1) == 1
        assert {path.stem for path in annotations.root.glob("*.parquet")} == {"older"}

    def test_forget_clears_the_cache_and_size_bytes_sees_what_is_left(self, tmp_path: Path) -> None:
        annotations = self._annotations(tmp_path)
        annotations.store("one", self._frame())
        assert annotations.size_bytes() > 0
        assert annotations.forget() == 1
        assert annotations.size_bytes() == 0
