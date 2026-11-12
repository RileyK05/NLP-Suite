"""Live analysis: the same answer, without the wait.

The bargain this makes is that the parse is expensive and everything after it
is not. On a twenty-speech corpus the parse takes thirty seconds and
readability over the result takes 170 milliseconds, so paying for the parse
once buys interactive parameters for the rest of the session.

That bargain is only safe if a live result is *the same result*. These tests
pin the equality first and the speed second: a fast preview that disagrees
with what publishing would write is worse than no preview, because someone
will act on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from desktop_backend.live import (
    PARSERS,
    SCHEMA_VERSION,
    Annotations,
    Bench,
    annotation_key,
    as_parser,
    parser_identity,
)

# Cold-start preload coverage lives in test_no_lazy_backend_imports.py.

TEXTS = {
    "1901-01-01_first.txt": (
        "The nation faces a question of freedom. Freedom is the work of every "
        "generation. We build the future together, and the future asks much of us."
    ),
    "1902-01-01_second.txt": (
        "Liberty and freedom are not the same word twice. The nation grows, the "
        "work continues, and the future belongs to those who build it."
    ),
}

IDENTITY = {
    "backend": "spacy",
    "language": "en",
    "model": "en_core_web_sm",
    "backend_version": "3.8.14",
    "model_version": "3.8.0",
    "schema": str(SCHEMA_VERSION),
}


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "corpus"
    directory.mkdir()
    for name, text in TEXTS.items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory


def a_corpus(directory: Path) -> Any:
    from core.io.reader import Corpus, Document, corpus_fingerprint, date_from_filename, hash_text, read_text

    documents = []
    for index, path in enumerate(sorted(directory.glob("*.txt")), 1):
        text = read_text(path).unwrap()
        documents.append(
            Document(
                doc_id=index,
                path=path,
                text=text,
                sha256=hash_text(text),
                date=date_from_filename(path),
            )
        )
    docs = tuple(documents)
    return Corpus(docs, corpus_fingerprint(docs))


# ------------------------------------------------------ naming a cached parse --


def test_the_key_covers_everything_that_changes_a_parse() -> None:
    """A cached parse reused after a model upgrade answers today's question
    with last month's annotations, and nothing about the file would say so."""
    base = annotation_key("fingerprint", IDENTITY)
    for field in IDENTITY:
        moved = annotation_key("fingerprint", {**IDENTITY, field: "changed"})
        assert moved != base, f"{field} does not reach the cache key"


def test_a_different_corpus_is_a_different_key() -> None:
    assert annotation_key("one", IDENTITY) != annotation_key("two", IDENTITY)


def test_the_same_corpus_parsed_the_same_way_is_the_same_key() -> None:
    assert annotation_key("one", IDENTITY) == annotation_key("one", dict(IDENTITY))


def test_the_key_is_a_usable_filename() -> None:
    key = annotation_key("one", IDENTITY)
    assert key.isalnum() and len(key) == 64


def test_the_identity_names_the_backend_that_ran_not_the_one_asked_for() -> None:
    """``resolve_pipeline`` falls back to another backend when a model is
    missing. A table the fallback produced must not be reused as though the
    configured parser had produced it."""

    class Fell:
        backend = "stanza"
        language = "en"
        model_name = ""

    assert parser_identity(Fell())["backend"] == "stanza"


def test_an_unknown_parser_is_refused_in_words() -> None:
    with pytest.raises(ValueError, match="spaCy or Stanza"):
        as_parser("regex")


def test_every_offered_parser_is_accepted() -> None:
    for name in PARSERS:
        assert as_parser(name) == name


# ---------------------------------------------------------------- the cache --


def test_a_stored_parse_comes_back_unchanged(tmp_path: Path) -> None:
    import pandas as pd

    frame = pd.DataFrame({"Form": ["the", "nation"], "Lemma": ["the", "nation"], "Document ID": [1, 1]})
    store = Annotations(tmp_path)
    store.store("abc", frame)
    assert store.load("abc").equals(frame)


def test_a_parse_that_was_never_stored_is_a_miss_not_a_failure(tmp_path: Path) -> None:
    assert Annotations(tmp_path).load("never") is None


def test_a_damaged_cache_file_is_dropped_rather_than_raised(tmp_path: Path) -> None:
    """A half-written parquet is a reason to parse again, not a reason to fail
    -- and leaving it there would make every later run retry it."""
    store = Annotations(tmp_path)
    store.root.mkdir(parents=True)
    store.path("broken").write_bytes(b"not a parquet file")
    assert store.load("broken") is None
    assert not store.path("broken").exists()


def test_the_cache_lives_beside_the_workspace_not_inside_a_project(tmp_path: Path) -> None:
    """It is a cache: deleting all of it costs time and nothing else. A run
    directory is a result and must never be confused with one."""
    store = Annotations(tmp_path)
    assert store.root == tmp_path / "annotations"
    assert "runs" not in store.path("k").parts


def test_forgetting_the_cache_removes_only_parses(tmp_path: Path) -> None:
    import pandas as pd

    store = Annotations(tmp_path)
    store.store("one", pd.DataFrame({"Form": ["a"]}))
    store.store("two", pd.DataFrame({"Form": ["b"]}))
    (store.root / "keep.txt").write_text("not a parse", encoding="utf-8")
    assert store.forget() == 2
    assert (store.root / "keep.txt").is_file()
    assert store.load("one") is None


def test_a_killed_write_leaves_no_half_parsed_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    store = Annotations(tmp_path)

    def die(self: Any, *args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", die)
    with pytest.raises(OSError):
        store.store("doomed", pd.DataFrame({"Form": ["a"]}))
    assert not store.path("doomed").exists()
    assert not list(store.root.glob("*.partial"))


# -------------------------------------------------- the same answer, faster --


@pytest.mark.model_integration
def test_a_live_result_is_what_publishing_would_write(corpus_dir: Path, tmp_path: Path) -> None:
    """The whole bargain. A preview that disagrees with the published run is
    worse than no preview, because somebody will act on it."""
    from core.profiler.executor import execute
    from core.profiler.plan import build_plan

    corpus = a_corpus(corpus_dir)
    bench = Bench(tmp_path / "workspace")
    warm = bench.warm(corpus, "spacy")

    live = bench.analyse(warm, "readability", {})
    assert live.ok, live.diagnostics

    # What run_job does, on the same corpus, with its own parse.
    plan = build_plan(["readability"], {}).unwrap()
    published = execute(plan, corpus=corpus, table=None, parse_diagnostics=()).outcomes[0]

    assert set(live.frames) == set(published.frames)
    for name, frame in published.frames.items():
        assert live.frames[name].equals(frame), name


@pytest.mark.model_integration
def test_the_second_warm_costs_nothing(corpus_dir: Path, tmp_path: Path) -> None:
    corpus = a_corpus(corpus_dir)
    bench = Bench(tmp_path / "workspace")
    first = bench.warm(corpus, "spacy")
    assert first.source == "parsed"

    again = bench.warm(corpus, "spacy")
    assert again.source == "memory", "a corpus already in hand was parsed again"
    assert again.key == first.key
    assert again.table is not None
    assert len(again.table) == len(first.table)


def test_a_memory_hit_carries_the_details_just_read(corpus_dir: Path, tmp_path: Path) -> None:
    """Details are outside the parse key, so an edit to one must not be lost to the held parse."""
    from dataclasses import replace

    corpus = a_corpus(corpus_dir)
    bench = Bench(tmp_path / "workspace")
    first = bench.warm(corpus, "spacy")
    edited = replace(corpus, docs=(replace(corpus.docs[0], fields=(("Party", "Federalist"),)), *corpus.docs[1:]))

    again = bench.warm(edited, "spacy")
    assert again.source == "memory"
    assert again.key == first.key
    assert again.corpus.docs[0].details == {"Party": "Federalist"}
    held = bench.held(again.key)
    assert held is not None
    assert held.corpus.docs[0].details == {"Party": "Federalist"}


@pytest.mark.model_integration
def test_a_new_process_recovers_the_parse_from_disk(corpus_dir: Path, tmp_path: Path) -> None:
    """Closing the app must not cost thirty seconds the next time it opens."""
    corpus = a_corpus(corpus_dir)
    root = tmp_path / "workspace"
    first = Bench(root).warm(corpus, "spacy")

    fresh = Bench(root)  # nothing held in memory
    recovered = fresh.warm(corpus, "spacy")
    assert recovered.source == "disk", "the parse was recomputed rather than read back"
    assert recovered.table.equals(first.table)


@pytest.mark.model_integration
def test_changing_a_parameter_does_not_reparse(corpus_dir: Path, tmp_path: Path) -> None:
    """The point of all of this: the second question is cheap."""
    corpus = a_corpus(corpus_dir)
    bench = Bench(tmp_path / "workspace")
    warm = bench.warm(corpus, "spacy")

    # Flat, exactly as /api/tools advertises them and the interface sends
    # them. Nesting these by tool name here would be testing the inside of
    # analyse against itself instead of testing what its callers pass.
    two = bench.analyse(warm, "ngrams", {"n": 2, "field": "form", "min-count": 1})
    three = bench.analyse(warm, "ngrams", {"n": 3, "field": "form", "min-count": 1})
    assert two.ok and three.ok
    assert bench.held(warm.key) is not None, "the warmed corpus was dropped between questions"

    first = next(iter(two.frames.values()))
    second = next(iter(three.frames.values()))
    assert not first.equals(second), "n=2 and n=3 produced the same table"


@pytest.mark.model_integration
def test_an_analysis_that_needs_no_parse_still_runs(corpus_dir: Path, tmp_path: Path) -> None:
    corpus = a_corpus(corpus_dir)
    bench = Bench(tmp_path / "workspace")
    warm = bench.warm(corpus, "spacy")
    assert bench.analyse(warm, "lexical_diversity", {}).ok


def test_a_parameter_reaches_the_tool_that_declares_it(tmp_path: Path) -> None:
    """Parameters arrive flat, the way ``/api/tools`` advertises them.

    ``build_plan`` wants ``{tool: {param: value}}``, so passing the interface's
    flat ``{"n": 3}`` straight through reads ``n`` as the name of a tool nobody
    selected. It fails quietly: analyses with no parameters worked, every
    analysis with one came back empty, and the message blamed the tool. Since
    changing a parameter and watching the answer change is the entire feature,
    that is worth a test that does not need a model.

    Getting as far as "needs parsed documents" is the proof: planning is over,
    and the parameter was understood.
    """
    from core.io.reader import Corpus

    bench = Bench(tmp_path / "workspace")
    warm = type("W", (), {"corpus": Corpus((), "x"), "table": None, "diagnostics": ()})()
    with pytest.raises(ValueError, match="needs parsed documents"):
        bench.analyse(warm, "ngrams", {"n": 3})


def test_live_and_published_runs_key_parameters_the_same_way(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The live path plans a run the way ``runner.py`` plans one.

    They diverged once and nothing failed, because each was tested against its
    own habit rather than against the other. Watching what the live path hands
    to ``build_plan`` is what pins them together: if the live preview and the
    published run plan differently, the preview is not previewing anything.
    """
    from core.io.reader import Corpus
    import core.profiler.plan as plan_module

    seen: dict[str, object] = {}
    real = plan_module.build_plan

    def spy(selected, params=None):  # type: ignore[no-untyped-def]
        seen["selected"], seen["params"] = list(selected), params
        return real(selected, params)

    monkeypatch.setattr(plan_module, "build_plan", spy)

    params = {"n": 3, "field": "form", "min-count": 1}
    bench = Bench(tmp_path / "workspace")
    warm = type("W", (), {"corpus": Corpus((), "x"), "table": None, "diagnostics": ()})()
    with pytest.raises(ValueError, match="needs parsed documents"):
        bench.analyse(warm, "ngrams", params)

    # runner.py builds exactly this from the parameters the run dialog sends.
    assert seen["selected"] == ["ngrams"]
    assert seen["params"] == {"ngrams": params}


def test_a_bad_parameter_is_a_diagnostic_not_an_exception(tmp_path: Path) -> None:
    """Live analysis runs on every keystroke's worth of change. A parameter the
    engine rejects has to come back as something the interface can show."""
    from core.io.reader import Corpus

    bench = Bench(tmp_path / "workspace")
    warm = type("W", (), {"corpus": Corpus((), "x"), "table": None, "diagnostics": ()})()
    result = bench.analyse(warm, "ngrams", {"n": -5})
    assert not result.ok
    assert result.diagnostics


# --------------------------------------------------------- through the API --


@pytest.fixture
def client(tmp_path: Path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from desktop_backend.server import create_app
    from desktop_backend.store import Workspace

    workspace = Workspace(tmp_path / "workspace")
    with TestClient(create_app(workspace, "tok")) as session:
        session.headers["Authorization"] = "Bearer tok"
        session.workspace = workspace  # type: ignore[attr-defined]
        yield session


def a_project(client: Any) -> str:
    project = client.workspace.create("Live")
    for name, text in TEXTS.items():
        client.workspace.import_document(project["id"], name, text.encode("utf-8"))
    return str(project["id"])


def test_a_project_starts_cold(client: Any) -> None:
    project = a_project(client)
    state = client.get(f"/api/projects/{project}/live").json()
    assert state["state"] == "cold"
    assert state["documents"] == 0


def test_asking_before_warming_says_what_to_do_first(client: Any) -> None:
    project = a_project(client)
    refused = client.post(
        f"/api/projects/{project}/live/analyse",
        json={"request_id": "test-req-0001", "tool": "readability", "params": {}},
    )
    assert refused.status_code == 400
    assert "documents" in refused.json()["detail"]


def test_an_abandoned_live_question_is_cancelled_not_answered(client: Any) -> None:
    """R-C8: the reader walked away; the engine stops pretending to answer.

    A question cancelled before it starts is never computed at all -- the
    machine should not warm up for an answer nobody wants. The cancellation is
    honoured once, so the same id is judged on its own merits afterwards (here:
    the ordinary "warm the corpus first" refusal) rather than staying dead.
    """
    project = a_project(client)
    ack = client.delete(f"/api/projects/{project}/live/analyse/test-req-0001")
    assert ack.status_code == 200
    assert ack.json() == {"ok": True, "request_id": "test-req-0001", "cancelled": True}

    answered = client.post(
        f"/api/projects/{project}/live/analyse",
        json={"request_id": "test-req-0001", "tool": "readability", "params": {}},
    )
    body = answered.json()
    assert answered.status_code == 200
    assert body["ok"] is False
    assert body["request_id"] == "test-req-0001"
    assert body["table"] is None
    codes = [d["code"] for d in body["diagnostics"]]
    assert codes == ["LIVE_CANCELLED"]

    again = client.post(
        f"/api/projects/{project}/live/analyse",
        json={"request_id": "test-req-0001", "tool": "readability", "params": {}},
    )
    assert again.status_code == 400
    assert "documents" in again.json()["detail"]


def test_cancelling_an_unknown_question_is_harmless(client: Any) -> None:
    """Abandoning a question that never existed must not invent a failure."""
    project = a_project(client)
    ack = client.delete(f"/api/projects/{project}/live/analyse/never-sent-000")
    assert ack.status_code == 200
    assert ack.json()["cancelled"] is True


def test_tracking_a_phrase_before_warming_says_what_to_do_first(client: Any) -> None:
    project = a_project(client)
    refused = client.post(
        f"/api/projects/{project}/live/phrase",
        json={"text": "public health"},
    )
    assert refused.status_code == 400
    assert "documents" in refused.json()["detail"]


def test_an_unknown_parser_is_refused_before_anything_is_read(client: Any) -> None:
    project = a_project(client)
    refused = client.post(f"/api/projects/{project}/live/warm", json={"parser": "regex"})
    assert refused.status_code == 400
    assert "spaCy or Stanza" in refused.json()["detail"]


def test_a_selection_matching_nothing_is_refused(client: Any) -> None:
    project = a_project(client)
    refused = client.post(
        f"/api/projects/{project}/live/warm",
        json={"selection": {"date_from": "2500-01-01", "date_to": "2500-12-31"}},
    )
    assert refused.status_code == 400


@pytest.mark.model_integration
def test_warming_then_asking_returns_rows_and_no_run(client: Any) -> None:
    """The point: an answer, with nothing published to get it."""
    project = a_project(client)
    warmed = client.post(f"/api/projects/{project}/live/warm", json={"parser": "spacy"})
    assert warmed.status_code == 200, warmed.text
    assert warmed.json()["state"] == "ready"
    assert warmed.json()["documents"] == 2

    answered = client.post(
        f"/api/projects/{project}/live/analyse",
        json={"request_id": "test-req-0001", "tool": "readability", "params": {}},
    )
    assert answered.status_code == 200, answered.text
    body = answered.json()
    assert body["ok"]
    assert body["table"]["columns"]
    assert body["table"]["total"] >= 1

    assert client.get(f"/api/projects/{project}/jobs").json() == [], "a live answer created a run"


@pytest.mark.model_integration
def test_the_second_question_is_much_faster_than_the_first(client: Any) -> None:
    project = a_project(client)
    client.post(f"/api/projects/{project}/live/warm", json={"parser": "spacy"})
    # Flat, as the interface posts them: the route hands body.params straight
    # to the bench, so nesting them here would test a shape nobody sends.
    grams = {"request_id": "test-req-0002", "tool": "ngrams", "params": {"n": 2, "field": "form", "min-count": 1}}
    first = client.post(f"/api/projects/{project}/live/analyse", json=grams).json()
    assert first["ok"], first["diagnostics"]
    # Not a benchmark -- a guard that the parse is not being redone. Reparsing
    # these two documents costs seconds; the analysis costs milliseconds.
    assert first["elapsed_ms"] < 5000, first["elapsed_ms"]


@pytest.mark.model_integration
def test_cooling_forgets_the_corpus_but_not_the_cached_parse(client: Any) -> None:
    project = a_project(client)
    client.post(f"/api/projects/{project}/live/warm", json={"parser": "spacy"})
    assert client.post(f"/api/projects/{project}/live/cool").json()["state"] == "cold"

    again = client.post(f"/api/projects/{project}/live/warm", json={"parser": "spacy"})
    assert again.json()["state"] == "ready"
    # From disk, specifically: cooling let go of the memory, so this proves the
    # parquet cache carried the parse rather than the bench still holding it.
    assert again.json()["source"] == "disk", again.json()["source"]


@pytest.mark.model_integration
def test_a_live_answer_matches_the_run_that_publishes_it(client: Any, tmp_path: Path) -> None:
    """The guarantee that makes a live preview safe to read.

    Both go through the same ``execute``; this checks it end to end, over HTTP,
    against a real published run rather than against another call in-process.
    """
    from desktop_backend.runner import Runner, run_job

    project = a_project(client)
    client.post(f"/api/projects/{project}/live/warm", json={"parser": "spacy"})
    live = client.post(
        f"/api/projects/{project}/live/analyse",
        json={"request_id": "test-req-0001", "tool": "readability", "params": {}},
    ).json()

    submitted = client.post(
        f"/api/projects/{project}/jobs",
        json={"tool": "readability", "params": {}, "parser": "spacy"},
    ).json()
    assert run_job(client.workspace.root, submitted["id"]) == 0, client.workspace.jobs(project)

    tables = [t for t in client.get(f"/api/projects/{project}/tables").json() if not t["manifest"]]
    published = client.get(f"/api/projects/{project}/jobs/{tables[0]['job']}/artifacts/{tables[0]['index']}").json()

    assert live["table"]["columns"] == published["columns"]
    assert live["table"]["total"] == published["total"]
    assert live["table"]["rows"] == published["rows"], "the live answer is not the published one"
    Runner(client.workspace).close()


def test_caches_report_their_size_and_clear_on_request(client: Any) -> None:
    """The Settings page's "clear" must drop only recomputable work."""
    import pandas as pd

    from desktop_backend.live import Annotations

    Annotations(client.workspace.root).store("some-parse", pd.DataFrame({"Form": ["word"] * 10}))
    sizes = client.get("/api/caches").json()
    assert sizes["parse_bytes"] > 0
    assert sizes["vector_bytes"] == 0

    cleared = client.post("/api/caches/clear").json()
    assert cleared == {"parses_removed": 1, "vectors_removed": 0}
    assert client.get("/api/caches").json() == {"parse_bytes": 0, "vector_bytes": 0}


def test_a_live_table_shows_missing_values_as_empty_cells() -> None:
    import pandas as pd

    from desktop_backend.live import as_table

    frame = pd.DataFrame({"nan": [float("nan")], "na": [pd.NA], "nat": [pd.NaT], "word": ["freedom"], "n": [0]})
    assert as_table(frame)["rows"] == [{"nan": "", "na": "", "nat": "", "word": "freedom", "n": "0"}]


def test_a_question_is_answered_from_the_snapshot_it_was_checked_against(
    corpus_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route resolves the snapshot, then analyses in a thread; a load in between must not swap the documents."""
    from desktop_backend.live import LiveResult, Session, Snapshot, Warm

    bench = Bench(tmp_path / "workspace")
    session = Session(bench)
    seen: list[Warm] = []

    def record(warm: Warm, tool: str, params: dict[str, Any], axis: dict[str, str] | None = None) -> LiveResult:
        seen.append(warm)
        return LiveResult(tool=tool, ok=True)

    monkeypatch.setattr(bench, "analyse", record)
    checked = Snapshot(
        warm=Warm(corpus=a_corpus(corpus_dir), key="checked", identity=IDENTITY, table=None),
        parser="spacy",
        selection=None,
        generation=1,
    )
    # Nothing is committed to the session, so re-resolving would raise.
    assert session.analyse("readability", {}, snapshot=checked).ok
    assert seen == [checked.warm]
