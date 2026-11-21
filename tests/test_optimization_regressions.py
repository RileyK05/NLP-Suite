"""Targeted regression tests for hardware, caching, I/O, viz, and store optimizations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from core.analysis import table_search
from core.analysis.word_embeddings import TrainedW2V
from core.analysis.word_vectors_bert import TrainedBert
from core.conll.schema import canonical_columns
from core.hardware import detect_hardware, held_corpora_limit, onnx_execution_providers
from core.io.reader import read_corpus
from core.result import Diagnostic, Result
from core.viz.panelspec import Annotation, Evidence, PanelEdge, PanelMark
from desktop_backend.store import Workspace
from tools._parse_cache import CachedPipeline, parse_with_cache


@dataclass(frozen=True)
class _FakePipeline:
    calls: list[str]
    backend: str = "spacy"
    language: str = "en"

    def parse(self, corpus: Any) -> Result[pd.DataFrame]:
        docs = corpus.docs
        self.calls.append(getattr(corpus, "sha256", ""))
        cols = canonical_columns()
        frame = pd.DataFrame(
            [
                {col: (1 if col in ("Document ID", "Sentence ID", "Token ID", "Head") else "hello") for col in cols}
                for _ in docs
            ]
        )
        return Result.success(frame, Diagnostic.info("TEST_DIAG", "cached diag"))


def test_parse_cache_reuses_parquet_and_diagnostics(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    # Exercise CRLF normalization under concurrent read_corpus too.
    (corpus_dir / "a.txt").write_bytes(b"Hello world.\r\nSecond line.")
    (corpus_dir / "b.txt").write_bytes(b"Third sentence.\r\n")

    loaded = read_corpus(corpus_dir)
    assert loaded.ok
    corpus = loaded.unwrap()
    assert "\r" not in corpus.docs[0].text
    assert "\n" in corpus.docs[0].text

    calls: list[str] = []
    fake = _FakePipeline(calls=calls)
    cache_dir = tmp_path / "cache"

    first = parse_with_cache(fake, corpus, cache_dir=cache_dir)  # type: ignore[arg-type]
    second = parse_with_cache(fake, corpus, cache_dir=cache_dir)  # type: ignore[arg-type]
    assert first.ok and second.ok
    assert len(calls) == 1
    assert [d.code for d in second.diagnostics] == ["TEST_DIAG"]
    pd.testing.assert_frame_equal(first.unwrap(), second.unwrap())

    uncached = CachedPipeline(fake, cache_dir=cache_dir)  # type: ignore[arg-type]
    assert uncached.backend == "spacy"
    third = uncached.parse(corpus)
    assert third.ok and len(calls) == 1

    bypassed = parse_with_cache(fake, corpus, cache_dir=cache_dir, use_cache=False)  # type: ignore[arg-type]
    assert bypassed.ok and len(calls) == 2


def test_parse_cache_skips_short_test_fingerprints(tmp_path: Path) -> None:
    calls: list[str] = []
    fake = _FakePipeline(calls=calls)
    dummy_corpus = SimpleNamespace(docs=(SimpleNamespace(doc_id=1),), sha256="short")
    res1 = parse_with_cache(fake, dummy_corpus, cache_dir=tmp_path)  # type: ignore[arg-type]
    res2 = parse_with_cache(fake, dummy_corpus, cache_dir=tmp_path)  # type: ignore[arg-type]
    assert res1.ok and res2.ok and len(calls) == 2


def test_workspace_targeted_queries_and_tables_and_digest_cache(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    with ws.connect() as db:
        version = int(db.execute("PRAGMA user_version").fetchone()[0])
    assert version >= 2

    proj = ws.create("Opt Project")
    pid = proj["id"]
    doc = ws.import_document(pid, "speech.txt", b"One two three four.")
    single = ws.document(pid, doc["id"])
    assert single is not None
    assert single["id"] == doc["id"]
    assert ws.document(pid, "missing") is None
    assert ws.has_active_jobs(pid) is False

    run_dir = ws.project_dir(pid) / "runs" / "r1"
    run_dir.mkdir(parents=True)
    csv_path = run_dir / "counts.csv"
    csv_path.write_text("Word,Count\nhello,3\n", encoding="utf-8")
    (run_dir / "result.json").write_text(
        '{"tool":"readability","created":"2026-01-01T00:00:00+00:00","inputs":[],"params":{},"artifacts":[{"path":"counts.csv","kind":"table","description":"Counts"}],"diagnostics":[]}',
        encoding="utf-8",
    )
    job_id = "job1"
    with ws.connect() as db:
        db.execute(
            "INSERT INTO jobs(id,project_id,tool,state,stage,created,run_dir,diagnostics,request) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (job_id, pid, "readability", "DONE", "Done", "2026-01-01T00:00:00+00:00", "runs/r1", "[]", "{}"),
        )

    fetched_job = ws.job(pid, job_id)
    assert fetched_job is not None and fetched_job["id"] == job_id
    tables = ws.tables(pid)
    assert len(tables) == 1 and tables[0]["path"] == "counts.csv"

    digest1 = ws.source_digest(pid, job_id, 0)
    digest2 = ws.source_digest(pid, job_id, 0)
    assert digest1 is not None and digest1 == digest2

    csv_path.write_text("Word,Count\nworld,9\n", encoding="utf-8")
    digest3 = ws.source_digest(pid, job_id, 0)
    assert digest3 is not None and digest3 != digest1


def test_render_bundle_multi_emits_png_and_svg() -> None:
    pytest.importorskip("seaborn")
    from core.viz.static.bundles import BUNDLES, render_bundle_multi

    fixture = Path(__file__).parent / "fixtures" / "figures" / "readability__readability.parquet"
    frame = pd.read_parquet(fixture)
    bundle = next(b for b in BUNDLES if b.name == "readability_measure_correlations")
    multi = render_bundle_multi(bundle, frame, ("png", "svg"), source="fixture.csv", dpi=60)
    assert multi.value is not None, [str(d) for d in multi.diagnostics]
    out = multi.unwrap()
    assert out["png"].startswith(b"\x89PNG")
    assert b"<svg" in out["svg"]


def _search_frame(forms: list[object], pos: list[str] | None = None) -> pd.DataFrame:
    """A small canonical CoNLL table with one row per form."""
    columns = list(canonical_columns())
    rows = []
    for index, form in enumerate(forms):
        row: dict[str, object] = {column: "" for column in columns}
        row.update(
            {
                "ID": index + 1,
                "Form": form,
                "Lemma": "" if form is None else str(form).casefold(),
                "POS": (pos[index] if pos is not None else "NOUN"),
                "Head": 0,
                "Record ID": index + 1,
                "Sentence ID": 1,
                "Document ID": 1,
                "Document": "a.txt",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def test_table_search_keeps_its_semantics_with_vectorized_matching() -> None:
    frame = _search_frame(
        ["Alpha", "ALPHA", "beta", "alphabet", None, float("nan")],
        ["NOUN", "NOUN", "NOUN", "ADJ", "NOUN", "VERB"],
    )

    def ids(filters: list[table_search.SearchFilter], logic: str = "AND") -> list[str]:
        result = table_search.search(frame, filters, logic=logic)  # type: ignore[arg-type]
        assert result.ok, [str(d) for d in result.diagnostics]
        return [str(value) for value in result.unwrap().frame["ID"].tolist()]

    assert ids([table_search.SearchFilter("Form", "eq", "alpha")]) == ["1", "2"]
    assert ids([table_search.SearchFilter("Form", "eq", "alpha", case_sensitive=True)]) == []
    assert ids([table_search.SearchFilter("Form", "eq", "alpha", negate=True)]) == ["3", "4", "5", "6"]
    assert ids([table_search.SearchFilter("Form", "contains", "ph")]) == ["1", "2", "4"]
    assert ids([table_search.SearchFilter("Form", "starts_with", "al")]) == ["1", "2", "4"]
    assert ids([table_search.SearchFilter("Form", "ends_with", "HA")]) == ["1", "2"]
    assert ids([table_search.SearchFilter("Form", "regex", "^a")]) == ["1", "2", "4"]
    assert ids([table_search.SearchFilter("Form", "regex", "^a", case_sensitive=True)]) == ["4"]
    assert ids([table_search.SearchFilter("Form", "regex", "^a", negate=True)]) == ["3", "5", "6"]
    assert ids(
        [
            table_search.SearchFilter("Form", "contains", "al"),
            table_search.SearchFilter("POS", "eq", "noun"),
        ]
    ) == ["1", "2"]
    assert ids(
        [
            table_search.SearchFilter("Form", "eq", "beta"),
            table_search.SearchFilter("POS", "eq", "adj"),
        ],
        logic="OR",
    ) == ["3", "4"]
    # A blank cell is blank: it is neither "None" nor "nan" to a search.
    assert ids([table_search.SearchFilter("Form", "eq", "none")]) == []
    assert ids([table_search.SearchFilter("Form", "eq", "nan")]) == []
    # A numeric column is matched by its printed form, as before.
    assert ids([table_search.SearchFilter("ID", "eq", "3")]) == ["3"]


def test_table_search_scales_to_a_large_table() -> None:
    """The old list-of-strings path built one Python object per row per filter."""
    size = 40_000
    frame = _search_frame(["alpha" if index % 2 else "beta" for index in range(size)])
    frame.loc[::10, "Form"] = None
    result = table_search.search(frame, [table_search.SearchFilter("Form", "eq", "alpha")])
    assert result.ok
    assert result.unwrap().matched_rows == size // 2
    result = table_search.search(frame, [table_search.SearchFilter("Form", "eq", "nan")])
    assert result.ok and result.unwrap().matched_rows == 0


def test_a_panel_mark_cannot_carry_a_non_finite_coordinate() -> None:
    evidence = Evidence(scope="rows", filters=(("Word", "a"),), count=1, describe="a stands for a row")

    def mark(**over: Any) -> PanelMark:
        fields: dict[str, Any] = {"key": "a", "label": "a", "x": 0.0, "y": 0.0, "evidence": evidence}
        fields.update(over)
        return PanelMark(**fields)

    assert mark().x == 0.0
    with pytest.raises(ValueError, match=r"PanelMark\.x must be finite"):
        mark(x=float("nan"))
    with pytest.raises(ValueError, match=r"PanelMark\.y must be finite"):
        mark(y=float("inf"))
    with pytest.raises(ValueError, match=r"PanelMark\.size must be finite"):
        mark(size=float("inf"))
    with pytest.raises(ValueError, match=r"PanelMark\.value must be finite"):
        mark(value=float("nan"))
    with pytest.raises(ValueError, match="edge weight must be finite"):
        PanelEdge(key="a-b", source="a", target="b", weight=float("nan"), evidence=evidence)
    with pytest.raises(ValueError, match=r"Annotation\.value must be finite"):
        Annotation(kind="hline", value=float("nan"), label="bad")


def test_vector_neighbour_queries_cache_the_matrix_and_vocabulary() -> None:
    vectors = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    words = ("alpha", "beta", "gamma")
    w2v = TrainedW2V(
        words=words,
        counts=(1, 1, 1),
        vectors=vectors,
        vector_size=2,
        window=2,
        min_count=1,
        sg=1,
        seed=1,
        epochs=1,
    )
    bert = TrainedBert(
        words=words,
        counts=(1, 1, 1),
        vectors=vectors,
        vector_size=2,
        model="test",
        field="lemma",
        min_count=1,
        layers=1,
    )
    for model in (w2v, bert):
        first = model.neighbors("alpha", 2)
        second = model.neighbors("alpha", 2)
        assert first == second
        assert [word for word, _ in first] == ["gamma", "beta"]
        assert model.similarity("alpha", "alpha") == pytest.approx(1.0)
        assert model.similarity("alpha", "beta") == pytest.approx(0.0, abs=1e-12)
    # The arrays and lookup are built once, on the first query.
    assert w2v._lookup is not None and w2v._matrix is not None
    assert bert._lookup is not None and bert._matrix is not None


def test_hardware_filters_azure_provider() -> None:
    eps = onnx_execution_providers(["AzureExecutionProvider", "CPUExecutionProvider"])
    assert eps == ("CPUExecutionProvider",)
    hw = detect_hardware()
    assert held_corpora_limit(hw) >= 1


def test_one_pass_sentence_depth_equals_the_per_token_walk() -> None:
    """The O(n) depth once gave the child of an already-resolved token its
    parent's depth instead of one more, and the shortfall compounded: on real
    speeches every sentence's Depth fell (12 -> 4). Random parses, including
    forests, cycles, self-heads and heads outside the sentence, must agree with
    the original depth_of walk."""
    import random

    from core.analysis.sentence_complexity import _sentence_depth_and_health, depth_of

    rng = random.Random(7)  # noqa: S311
    for _ in range(2000):
        n = rng.randint(1, 14)
        ids = list(range(1, n + 1))
        heads = {
            token: rng.choice([0, *ids, n + 3]) if rng.random() < 0.3 else rng.randint(0, token - 1) for token in ids
        }
        rng.shuffle(ids)
        expected = max(depth_of(heads, token) for token in ids)
        assert _sentence_depth_and_health(ids, heads)[1] == expected, (ids, heads)
    # A plain chain: 1 is the root, 5 hangs off 3, visited after 3 is resolved.
    assert _sentence_depth_and_health([1, 2, 3, 4, 5], {1: 0, 2: 1, 3: 2, 4: 3, 5: 3})[1] == 3
