"""Real desktop contracts: import custody, authenticated API, runs and reopening."""

from __future__ import annotations

import io
import json
from pathlib import Path
import struct
import time
import zipfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from desktop_backend.runner import Runner, run_job
from desktop_backend.server import create_app, import_paths
from desktop_backend.store import Workspace, now


def test_catalog_covers_corpus_adapters(client: TestClient) -> None:
    from core.profiler.executor import ADAPTERS
    from desktop_backend.catalog import CORPUS_TOOLS, INTERNAL_TOOLS
    from desktop_backend.tables import TABLE_TOOLS

    names = {tool["name"] for tool in client.get("/api/tools").json()}
    expected = set(CORPUS_TOOLS)
    # Job-only tools run through desktop_backend.runner's own path (the MALLET
    # staging workflow), never through an executor adapter.
    job_only = {"lda_mallet"}
    assert names == expected | set(TABLE_TOOLS)
    assert names.isdisjoint(INTERNAL_TOOLS)
    assert expected - job_only <= set(ADAPTERS)


def test_catalog_names_every_tool_and_setting(client: TestClient) -> None:
    """The desktop prints what this payload carries, and never invents a name.

    It used to keep its own table of tool names and derive setting labels from
    the flag, which is how three tools came to be shown as bare identifiers and
    every setting was labelled "sg", "op", "col x" and the like.
    """
    catalog = client.get("/api/tools").json()
    assert catalog
    for tool in catalog:
        assert tool["label"], tool["name"]
        assert tool["label"] != tool["name"] or "_" not in tool["name"]
        assert tool["description"].endswith("."), tool["name"]
        for param in tool["params"]:
            assert param["label"], f"{tool['name']}.{param['name']}"
            assert param["label"] != param["name"], f"{tool['name']}.{param['name']}"


def test_catalog_offers_every_chart_the_engine_can_draw(client: TestClient) -> None:
    """The desktop's chart form reads these choices; a short list hides a kind."""
    from core.viz.chartspec import CHART_KINDS

    catalog = {tool["name"]: tool for tool in client.get("/api/tools").json()}
    kind = next(p for p in catalog["table_charts"]["params"] if p["name"] == "kind")
    assert tuple(kind["choices"]) == tuple(CHART_KINDS)


def test_catalog_separates_visualization_from_analysis(client: TestClient) -> None:
    from desktop_backend.runner import VISUALIZATION_TOOLS

    catalog = {tool["name"]: tool for tool in client.get("/api/tools").json()}
    visual = {name for name, tool in catalog.items() if tool["category"] == "visualization"}
    assert visual == VISUALIZATION_TOOLS & set(catalog)
    assert {"table_charts", "table_wordcloud_gephi", "narrative_arc", "shapes"} <= visual
    assert visual & set(catalog) == visual
    assert catalog["table_charts"]["category"] != catalog["table_chi2"]["category"]


@pytest.mark.parametrize(
    "tool,params",
    [
        ("table_chi2", {"col1": "group", "col2": "vote"}),
        ("table_crosstab", {"col1": "group", "col2": "vote"}),
        ("table_keyness", {"word-col": "word", "freq1": "x", "freq2": "y"}),
        ("table_mw", {"value-col": "x", "group-col": "group"}),
        ("table_kw", {"value-col": "x", "group-col": "group"}),
        ("table_trend", {"date-col": "date", "value-col": "x"}),
        ("table_rankcorr", {"col-x": "x", "col-y": "y"}),
    ],
)
def test_table_workflow_without_corpus(
    workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str, params: dict
) -> None:
    source = tmp_path / "dataset.csv"
    source.write_text(
        "group,vote,word,x,y,date\nA,yes,one,1,2,2020-01-01\nA,no,two,2,4,2020-01-02\nB,yes,three,3,5,2020-01-03\nB,no,four,4,8,2020-01-04\nA,yes,five,5,9,2020-01-05\nB,no,six,6,11,2020-01-06\nA,yes,seven,7,13,2020-01-07\nB,no,eight,8,15,2020-01-08\n",
        encoding="utf-8",
    )
    project = workspace.create("Table research")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    job = runner.submit(project["id"], tool, {**params, "input": str(source)}, "spacy")
    source.write_text("changed original", encoding="utf-8")
    assert run_job(workspace.root, job["id"]) == 0, workspace.jobs(project["id"])
    directory, envelope = workspace.artifacts(project["id"], job["id"])
    assert len(envelope.inputs) == 1
    assert envelope.artifacts and directory.is_dir()
    runner.close()


def test_visualize_endpoint_reuses_artifact_without_exposing_paths(
    workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dataset.csv"
    source.write_text("word,freq1,freq2\nfreedom,120,10\nliberty,95,40\nnation,80,30\nfuture,60,55\n", encoding="utf-8")
    project = workspace.create("Visualize research")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    job = runner.submit(
        project["id"],
        "table_keyness",
        {"input": str(source), "word-col": "word", "freq1": "freq1", "freq2": "freq2"},
        "spacy",
    )
    assert run_job(workspace.root, job["id"]) == 0, workspace.jobs(project["id"])
    runner.close()
    _, envelope = workspace.artifacts(project["id"], job["id"])
    csvs = [i for i, artifact in enumerate(envelope.artifacts) if artifact.path.endswith(".csv")]
    with TestClient(create_app(workspace, "test-local-token")) as session:
        # This test executes the submitted job synchronously below. Do not
        # also dispatch it in a worker: two runs race to publish the same
        # artifacts and shutdown waits on a redundant cold worker startup.
        monkeypatch.setattr(session.app.state.runner.pool, "submit", lambda *args: None)
        session.headers["Authorization"] = "Bearer test-local-token"
        response = session.post(
            f"/api/projects/{project['id']}/jobs/{job['id']}/visualize",
            json={
                "tool": "table_charts",
                "index": csvs[0],
                "params": {"kind": "bar", "x": "Word", "y": "G2 (log-likelihood)", "agg": "sum"},
            },
        )
        assert response.status_code == 200, response.text
        submitted = response.json()
        assert submitted["tool"] == "table_charts"
        assert submitted["state"] in ("QUEUED", "RUNNING")
        assert run_job(workspace.root, submitted["id"]) == 0, workspace.jobs(project["id"])
        _, viz_envelope = workspace.artifacts(project["id"], submitted["id"])
        assert any(artifact.path.startswith("chart.") for artifact in viz_envelope.artifacts)
        bad = session.post(
            f"/api/projects/{project['id']}/jobs/{job['id']}/visualize",
            json={"tool": "table_charts", "index": 999, "params": {}},
        )
        assert bad.status_code in (400, 404, 500)
    # the run result.json must never advertise the original absolute path
    directory, _ = workspace.artifacts(project["id"], submitted["id"])
    recorded = (directory / "result.json").read_text(encoding="utf-8")
    assert "dataset" not in recorded


def test_inline_artifact_preview_serves_viewable_types(
    workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dataset.csv"
    source.write_text("word,count\nfreedom,120\nliberty,95\n", encoding="utf-8")
    project = workspace.create("Preview research")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    job = runner.submit(
        project["id"],
        "table_wordcloud_gephi",
        {
            "input": str(source),
            "mode": "wordcloud",
            "word-col": "word",
            "weight-col": "count",
            "image": True,
        },
        "spacy",
    )
    assert run_job(workspace.root, job["id"]) == 0, workspace.jobs(project["id"])
    runner.close()
    # The PNG is rendered through a scratch file that must not be left in runs/.
    assert not (workspace.project_dir(project["id"]) / "runs" / ".staging_wc.png").exists()
    _, envelope = workspace.artifacts(project["id"], job["id"])
    with TestClient(create_app(workspace, "test-local-token")) as session:
        session.headers["Authorization"] = "Bearer test-local-token"
        for index, artifact in enumerate(envelope.artifacts):
            response = session.get(f"/api/projects/{project['id']}/jobs/{job['id']}/artifacts/{index}")
            if artifact.path.endswith(".csv"):
                assert response.headers["content-type"].startswith("application/json")
            elif artifact.path.endswith(".html"):
                assert response.headers["content-type"].startswith("text/html")
            elif artifact.path.endswith(".png"):
                assert response.headers["content-type"].startswith("image/png")
        inline = session.get(f"/api/projects/{project['id']}/jobs/{job['id']}/artifacts/0")
        assert inline.status_code == 200
        download = session.get(f"/api/projects/{project['id']}/jobs/{job['id']}/artifacts/0?download=true")
        assert download.headers["content-type"] == "application/octet-stream"


def test_uploaded_auxiliary_resource_is_scoped(client: TestClient, workspace: Workspace) -> None:
    response = client.post("/api/resources?name=../../words.txt", content=b"apple\nbanana")
    assert response.status_code == 200
    path = Path(response.json()["path"])
    assert path.is_relative_to(workspace.root / "resources")
    assert path.read_bytes() == b"apple\nbanana"
    assert client.post("/api/resources?name=run.exe", content=b"bad").status_code == 400


def test_converted_import_retains_source_and_distinct_identity(workspace: Workspace) -> None:
    project = workspace.create("Converted")
    first = b"<p>Same text</p>"
    second = b"<div>Same text</div>"
    a = workspace.import_document(project["id"], "page.html", first)
    b = workspace.import_document(project["id"], "page.html", second)
    assert a["sha256"] == b["sha256"]
    assert a["source_sha256"] != b["source_sha256"]
    assert a["stored_name"] != b["stored_name"]
    assert (workspace.project_dir(project["id"]) / "originals" / a["source_name"]).read_bytes() == first
    assert workspace.import_document(project["id"], "page.html", first)["duplicate"]
    assert workspace.preview(project["id"], a["id"])["text"] == "Same text"


def test_cancel_queued_job_is_terminal(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    project = workspace.create("Cancel")
    workspace.import_document(project["id"], "input.txt", b"A test document.")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    job = runner.submit(project["id"], "readability", {}, "spacy")
    assert runner.cancel(project["id"], job["id"])["state"] == "CANCELLED"
    runner._dispatch(job["id"])
    workspace.update_job(job["id"], state="DONE", stage="Late update")
    assert workspace.jobs(project["id"])[0]["state"] == "CANCELLED"
    runner.close()


def test_cancel_running_worker_promptly(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation must return while a worker's completion read is blocked.

    Regression for the review finding: _dispatch used to hold the state
    lock across the blocking pipe read, so cancel() blocked until the
    analysis finished. The pipe sever must interrupt the dispatch instead.
    """
    import sys
    import threading
    import time as _time

    project = workspace.create("Cancel running")
    workspace.import_document(project["id"], "input.txt", b"A test document.")
    received = workspace.root / "worker-received"
    # Real pipes and a real process, with no unrelated NLP imports. The marker
    # proves a job reached the worker before cancellation, unlike RUNNING,
    # which is set before the worker has even started.
    code = (
        "import sys, threading\nfrom pathlib import Path\n"
        'print(\'{"event": "ready"}\', flush=True)\n'
        "job = sys.stdin.readline()\n"
        f"Path({str(received)!r}).write_text(job)\n"
        "threading.Event().wait()\n"
    )
    monkeypatch.setattr("desktop_backend.runner.worker_command", lambda root: [sys.executable, "-u", "-c", code])
    runner = Runner(workspace)
    try:
        job = runner.submit(project["id"], "readability", {}, "spacy")
        # Let the dispatch thread reach its blocking readline.
        deadline = _time.monotonic() + 10
        while _time.monotonic() < deadline:
            if received.exists():
                break
            _time.sleep(0.05)
        else:
            raise AssertionError("worker never received the job")

        def do_cancel() -> None:
            runner.cancel(project["id"], job["id"])

        worker = runner._worker
        assert worker is not None
        canceller = threading.Thread(target=do_cancel, daemon=True)
        started = _time.monotonic()
        canceller.start()
        canceller.join(timeout=5)
        assert not canceller.is_alive(), "cancel() blocked behind the running analysis"
        assert _time.monotonic() - started < 5
        assert workspace.jobs(project["id"])[0]["state"] == "CANCELLED"
        assert worker.poll() is not None, "cancelled worker was not reaped"
    finally:
        # Even a locking regression must not strand the pool during teardown.
        if runner._worker is not None and runner._worker.poll() is None:
            runner._worker.kill()
        runner.close()


def test_project_archive_roundtrip(client: TestClient, workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    project = workspace.create("Portable")
    workspace.import_document(project["id"], "speech.html", b"<p>Our words deserve a careful reading.</p>")
    runner = client.app.state.runner
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    job = runner.submit(project["id"], "readability", {}, "spacy")
    assert run_job(workspace.root, job["id"]) == 0
    finished = workspace.jobs(project["id"])[0]
    assert finished["state"] == "DONE", finished
    archive = client.get(f"/api/projects/{project['id']}/backup")
    assert archive.status_code == 200
    restored = client.post("/api/projects/restore", content=archive.content)
    assert restored.status_code == 200, restored.text
    copy = restored.json()
    assert copy["id"] != project["id"] and copy["documents"] == 1
    copied_job = workspace.jobs(copy["id"])[0]
    assert workspace.artifact(copy["id"], copied_job["id"], 0).is_file()
    assert workspace.artifacts(copy["id"], copied_job["id"])[1].tool == job["tool"]
    assert len(workspace.projects()) == 2


def test_readout_markdown_is_viewable_inline(
    client: TestClient, workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plain-language reading ships with every run; the app must be able to show it."""
    project = workspace.create("Readout view")
    workspace.import_document(project["id"], "speech.txt", b"Our words deserve a careful reading.")
    runner = client.app.state.runner
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    job = runner.submit(project["id"], "readability", {}, "spacy")
    assert run_job(workspace.root, job["id"]) == 0
    _, envelope = workspace.artifacts(project["id"], job["id"])
    index = next(i for i, a in enumerate(envelope.artifacts) if a.path == "readout.md")
    response = client.get(f"/api/projects/{project['id']}/jobs/{job['id']}/artifacts/{index}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "readability" in response.text


def test_partial_run_survives_backup_roundtrip(workspace: Workspace) -> None:
    """A PARTIAL job (artifacts + diagnostics) is terminal: backup and restore keep it."""
    from desktop_backend.archives import backup, restore

    project = workspace.create("Partial portable")
    job_id = "partial" + "0" * 26
    with workspace.connect() as db:
        db.execute(
            "INSERT INTO jobs (id, project_id, tool, state, stage, created, finished, run_dir, diagnostics, request) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                job_id,
                project["id"],
                "readability",
                "PARTIAL",
                "Completed with warnings",
                now(),
                now(),
                "",
                "[]",
                "{}",
            ),
        )
    archive_path = workspace.root / "partial-backup.nlpsuite"
    backup(workspace, project["id"], archive_path)
    restored = restore(workspace, archive_path)
    assert restored["id"] != project["id"]
    assert workspace.jobs(restored["id"])[0]["state"] == "PARTIAL"


@pytest.mark.parametrize("member", ["../escape.txt", "files/../../escape.txt", "C:/escape.txt", "files\\escape.txt"])
def test_restore_rejects_unsafe_paths(client: TestClient, workspace: Workspace, member: str) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member, "payload")
    response = client.post("/api/projects/restore", content=buffer.getvalue())
    assert response.status_code == 400
    assert workspace.projects() == []


def test_restore_rejects_changed_payload(client: TestClient, workspace: Workspace) -> None:
    project = workspace.create("Checksum")
    workspace.import_document(project["id"], "text.txt", b"Text worth preserving.")
    original = client.get(f"/api/projects/{project['id']}/backup").content
    corrupted = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as incoming, zipfile.ZipFile(corrupted, "w") as outgoing:
        for member in incoming.namelist():
            outgoing.writestr(member, incoming.read(member) if member == "project.json" else b"changed")
    response = client.post("/api/projects/restore", content=corrupted.getvalue())
    assert response.status_code == 400
    assert "checksum" in response.json()["detail"]
    assert len(workspace.projects()) == 1


@pytest.mark.parametrize("damage", ["corrupt-deflate", "encrypted-flag", "created"])
def test_restore_rejects_malformed_archives(client: TestClient, workspace: Workspace, damage: str) -> None:
    project = workspace.create("Malformed")
    workspace.import_document(project["id"], "text.txt", b"Text worth preserving.")
    original = client.get(f"/api/projects/{project['id']}/backup").content
    if damage == "created":
        with zipfile.ZipFile(io.BytesIO(original)) as incoming:
            members = {name: incoming.read(name) for name in incoming.namelist()}
        manifest = json.loads(members["project.json"])
        manifest["project"] = dict(manifest["project"], created={})
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as outgoing:
            for name, payload in members.items():
                outgoing.writestr(name, json.dumps(manifest) if name == "project.json" else payload)
        content = buffer.getvalue()
    elif damage == "corrupt-deflate":
        buffer = io.BytesIO()
        with (
            zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as outgoing,
            zipfile.ZipFile(io.BytesIO(original)) as incoming,
        ):
            for name in incoming.namelist():
                outgoing.writestr(name, incoming.read(name))
        data = bytearray(buffer.getvalue())
        with zipfile.ZipFile(io.BytesIO(bytes(data))) as archive:
            info = archive.getinfo("project.json")
            offset = info.header_offset + 30 + len(info.filename) + len(info.extra)
        for index in range(offset + 5, offset + 40):
            data[index] ^= 0xFF
        content = bytes(data)
    else:
        data = bytearray(original)
        with zipfile.ZipFile(io.BytesIO(original)) as archive:
            infos = archive.infolist()
        for info in infos:
            flags = struct.unpack_from("<H", data, info.header_offset + 6)[0]
            struct.pack_into("<H", data, info.header_offset + 6, flags | 0x1)
        central = data.find(b"PK\x01\x02")
        while central != -1:
            flags = struct.unpack_from("<H", data, central + 8)[0]
            struct.pack_into("<H", data, central + 8, flags | 0x1)
            central = data.find(b"PK\x01\x02", central + 4)
        content = bytes(data)
    before = workspace.projects()
    response = client.post("/api/projects/restore", content=content)
    assert response.status_code == 400, response.text
    assert workspace.projects() == before


def _question_payload(document_id: str, matching: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "Public health",
        "snapshot_id": "a" * 64,
        "parser": "spacy",
        "document_ids": [document_id],
        "specification": {
            "schema_version": 1,
            "kind": "phrase_distribution",
            "text": "public health",
            "comparison_text": "private health",
            "case_sensitive": False,
            "position_bins": 10,
            "evidence_year": 2020,
            "evidence_document_id": document_id,
        },
    }
    if matching is not None:
        payload["matching"] = matching
    return payload


def test_restore_keeps_question_matching_rules(client: TestClient, workspace: Workspace) -> None:
    from desktop_backend.archives import backup, restore

    project = workspace.create("Questions")
    document = workspace.import_document(project["id"], "text.txt", b"public health")
    matching = {
        "version": 1,
        "tokenizer": "spacy",
        "source": "saved",
        "tokens": ["public", "health"],
        "comparison_tokens": ["private", "health"],
    }
    saved = client.post(
        f"/api/projects/{project['id']}/questions", json=_question_payload(str(document["id"]), matching)
    )
    assert saved.status_code == 200, saved.text
    archive = workspace.root / "question-matching.nlpsuite"
    backup(workspace, project["id"], archive)
    restored = restore(workspace, archive)
    assert workspace.questions(restored["id"])[0]["matching"] == matching


def test_restore_defaults_question_matching_when_absent(client: TestClient, workspace: Workspace) -> None:
    from desktop_backend.archives import restore
    from desktop_backend.questions import MatchingProfile

    project = workspace.create("Old questions")
    document = workspace.import_document(project["id"], "text.txt", b"public health")
    saved = client.post(f"/api/projects/{project['id']}/questions", json=_question_payload(str(document["id"])))
    assert saved.status_code == 200, saved.text
    original = client.get(f"/api/projects/{project['id']}/backup").content
    with zipfile.ZipFile(io.BytesIO(original)) as incoming:
        members = {name: incoming.read(name) for name in incoming.namelist()}
    manifest = json.loads(members["project.json"])
    for question in manifest["questions"]:
        question.pop("matching", None)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as outgoing:
        for name, payload in members.items():
            outgoing.writestr(name, json.dumps(manifest) if name == "project.json" else payload)
    archive = workspace.root / "question-default-matching.nlpsuite"
    archive.write_bytes(buffer.getvalue())
    restored = restore(workspace, archive)
    assert workspace.questions(restored["id"])[0]["matching"] == MatchingProfile().model_dump()


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "workspace")


@pytest.fixture
def client(workspace: Workspace):
    with TestClient(create_app(workspace, "test-local-token")) as session:
        session.headers["Authorization"] = "Bearer test-local-token"
        yield session


def test_requires_authentication(client: TestClient) -> None:
    assert client.get("/api/projects", headers={"Authorization": ""}).status_code == 401
    assert client.get("/api/projects", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/projects", headers={"Host": "evil.example"}).status_code == 403
    assert client.get("/api/projects").status_code == 200


def test_a_token_with_non_ascii_bytes_is_refused_not_a_server_error(client: TestClient) -> None:
    """``compare_digest`` rejects non-ASCII text outright; that must read as a wrong token (401), not a 500."""
    refused = client.get("/api/projects", headers={"Authorization": "Bearer café".encode("latin-1")})
    assert refused.status_code == 401


def test_project_persists_after_reopening(workspace: Workspace) -> None:
    project = workspace.create("A research project")
    assert Workspace(workspace.root).projects()[0]["id"] == project["id"]


def test_archive_and_rename_preserve_documents(client: TestClient, workspace: Workspace) -> None:
    project = workspace.create("Old name")
    document = workspace.import_document(project["id"], "text.txt", b"Preserved words.")
    base = f"/api/projects/{project['id']}"
    assert client.post(base + "/rename", json={"name": "New name"}).status_code == 200
    assert client.post(base + "/archive", json={"archived": True}).status_code == 200
    assert client.get("/api/projects").json() == []
    assert client.get("/api/projects?archived=true").json()[0]["name"] == "New name"
    assert workspace.preview(project["id"], document["id"])["text"] == "Preserved words."
    assert client.post(base + "/archive", json={"archived": False}).status_code == 200
    assert client.get("/api/projects").json()[0]["documents"] == 1


def test_import_preserves_original_and_handles_duplicates(workspace: Workspace, tmp_path: Path) -> None:
    source = tmp_path / "speech.txt"
    source.write_bytes(b"Words are worth reading.")
    project = workspace.create("Speeches")
    first = import_paths(workspace, project["id"], [source])
    second = import_paths(workspace, project["id"], [source])
    assert first["imported"] == 1 and second["duplicates"] == 1
    assert source.read_bytes() == b"Words are worth reading."
    assert workspace.project(project["id"])["words"] == 4
    doc = workspace.documents(project["id"])[0]
    assert workspace.preview(project["id"], doc["id"])["text"] == "Words are worth reading."


def test_same_name_different_content_is_not_overwritten(workspace: Workspace) -> None:
    project_id = workspace.create("Speeches")["id"]
    first = workspace.import_document(project_id, "speech.txt", b"First speech")
    second = workspace.import_document(project_id, "speech.txt", b"Second speech")
    assert first["stored_name"] != second["stored_name"]
    assert workspace.preview(project_id, first["id"])["text"] == "First speech"
    assert workspace.preview(project_id, second["id"])["text"] == "Second speech"


@pytest.mark.parametrize("name,data", [("a.exe", b"text"), ("a.txt", b""), ("a.txt", b"\x00binary"), ("a.txt", b"  ")])
def test_bad_imports_are_rejected_without_artifacts(workspace: Workspace, name: str, data: bytes) -> None:
    project_id = workspace.create("Test")["id"]
    with pytest.raises(ValueError):
        workspace.import_document(project_id, name, data)
    assert workspace.documents(project_id) == []
    assert list((workspace.project_dir(project_id) / "corpus").iterdir()) == []


def test_upload_does_not_trust_filename_paths(client: TestClient, workspace: Workspace) -> None:
    project = client.post("/api/projects", json={"name": "My corpus"}).json()
    response = client.post(f"/api/projects/{project['id']}/documents?name=../../outside.txt", content=b"A little text")
    assert response.status_code == 200
    assert response.json()["name"] == "outside.txt"
    assert not (workspace.root / "outside.txt").exists()


def test_unknown_project_and_empty_run_are_actionable(client: TestClient) -> None:
    assert client.get("/api/projects/invalid/documents").status_code == 404
    project = client.post("/api/projects", json={"name": "Empty"}).json()
    response = client.post(f"/api/projects/{project['id']}/jobs", json={"tool": "readability"})
    assert response.status_code == 400
    assert "Import at least" in response.json()["detail"]


def test_warm_worker_reuses_one_process_for_several_jobs(workspace: Workspace) -> None:
    """Second analysis must not pay a second import/model-load cold start."""
    project_id = workspace.create("Warm")["id"]
    workspace.import_document(project_id, "text.txt", b"Simple words for a simple read.")
    runner = Runner(workspace)
    try:
        runner.submit(project_id, "readability", {}, "spacy")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if workspace.jobs(project_id)[0]["state"] not in ("QUEUED", "RUNNING"):
                break
            time.sleep(0.1)
        assert workspace.jobs(project_id)[0]["state"] == "DONE", workspace.jobs(project_id)[0]
        pid_first = runner._worker.pid if runner._worker else None
        assert pid_first is not None
        second = runner.submit(project_id, "lexical_diversity", {}, "spacy")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            jobs = {j["id"]: j for j in workspace.jobs(project_id)}
            if jobs[second["id"]]["state"] not in ("QUEUED", "RUNNING"):
                break
            time.sleep(0.1)
        assert jobs[second["id"]]["state"] == "DONE", jobs[second["id"]]
        assert runner._worker is not None and runner._worker.pid == pid_first
    finally:
        runner.close()


def test_real_background_analysis_and_export(client: TestClient, workspace: Workspace) -> None:
    project_id = client.post("/api/projects", json={"name": "End to end"}).json()["id"]
    client.post(
        f"/api/projects/{project_id}/documents?name=speech.txt",
        content=b"This is a simple sentence. This is another clear sentence. We can read them.",
    )
    response = client.post(f"/api/projects/{project_id}/jobs", json={"tool": "readability"})
    assert response.status_code == 200
    job_id = response.json()["id"]
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        job = client.get(f"/api/projects/{project_id}/jobs").json()[0]
        if job["state"] not in ("RUNNING", "QUEUED"):
            break
        time.sleep(0.1)
    assert job["state"] == "DONE", job
    result = client.get(f"/api/projects/{project_id}/jobs/{job_id}/results").json()
    assert result["tool"] == "readability" and len(result["inputs"]) == 1
    table = client.get(f"/api/projects/{project_id}/jobs/{job_id}/artifacts/0").json()
    assert table["total"] == 1
    assert client.get(f"/api/projects/{project_id}/jobs/{job_id}/artifacts/-1").status_code == 404
    exported = client.get(f"/api/projects/{project_id}/jobs/{job_id}/export")
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        assert "readability.csv" in archive.namelist() and "result.json" in archive.namelist()
    reopened = Workspace(workspace.root)
    assert reopened.jobs(project_id)[0]["state"] == "DONE"
    assert reopened.artifacts(project_id, job_id)[1].tool == "readability"


def test_run_captures_submission_document_selection(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    project_id = workspace.create("Snapshot")["id"]
    workspace.import_document(project_id, "first.txt", b"A first short document.")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    job = runner.submit(project_id, "readability", {}, "spacy")
    workspace.import_document(project_id, "second.txt", b"An unrelated later document.")
    assert run_job(workspace.root, job["id"]) == 0
    assert len(workspace.artifacts(project_id, job["id"])[1].inputs) == 1
    runner.close()


def test_interrupted_jobs_are_not_left_running(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    project_id = workspace.create("Interrupted")["id"]
    workspace.import_document(project_id, "text.txt", b"Some readable words.")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    runner.submit(project_id, "readability", {}, "spacy")
    runner.close()
    restarted = Runner(workspace)
    assert workspace.jobs(project_id)[0]["state"] == "INTERRUPTED"
    restarted.close()


def test_changed_import_is_not_silently_analyzed(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    project_id = workspace.create("Integrity")["id"]
    document = workspace.import_document(project_id, "speech.txt", b"Original research text.")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    job = runner.submit(project_id, "readability", {}, "spacy")
    (workspace.project_dir(project_id) / "corpus" / document["stored_name"]).write_bytes(b"Changed text.")
    assert run_job(workspace.root, job["id"]) == 1
    failed = workspace.jobs(project_id)[0]
    assert failed["state"] == "FAILED"
    assert "changed since import" in failed["diagnostics"][0]["message"]
    runner.close()


def test_declared_artifacts_cannot_escape(workspace: Workspace) -> None:
    from core.artifacts.envelope import Artifact, Envelope

    project_id = workspace.create("Scope")["id"]
    directory = workspace.project_dir(project_id) / "runs" / "test"
    directory.mkdir(parents=True)
    assert Envelope(tool="test", artifacts=(Artifact(kind="table", path="../../outside.csv"),)).write(directory).ok
    with workspace.connect() as db:
        db.execute(
            "INSERT INTO jobs(id,project_id,tool,state,stage,created,run_dir,request) VALUES(?,?,?,?,?,?,?,?)",
            ("job", project_id, "test", "DONE", "Done", "today", "runs/test", json.dumps({})),
        )
    with pytest.raises(ValueError, match="outside"):
        workspace.artifact(project_id, "job", 0)


def test_result_search_reaches_rows_beyond_first_page(client: TestClient, workspace: Workspace) -> None:
    import pandas as pd

    from core.io.writer import OutputWriter

    project = workspace.create("Full table")
    writer = OutputWriter(workspace.project_dir(project["id"]) / "runs", tool="test_table", params={})
    assert writer.write_table(pd.DataFrame({"label": [f"row-{i}" for i in range(800)]}), "rows.csv", kind="table").ok
    assert writer.finalize().ok
    with workspace.connect() as db:
        db.execute(
            "INSERT INTO jobs(id,project_id,tool,state,stage,created,run_dir,request) VALUES(?,?,?,?,?,?,?,?)",
            (
                "table",
                project["id"],
                "test_table",
                "DONE",
                "Done",
                "now",
                writer.run_dir.relative_to(workspace.project_dir(project["id"])).as_posix(),
                "{}",
            ),
        )
    base = f"/api/projects/{project['id']}/jobs/table/artifacts/0"
    first = client.get(base).json()
    assert len(first["rows"]) == 500 and first["total"] == 800
    second = client.get(base + "?offset=500").json()
    assert len(second["rows"]) == 300 and second["rows"][0]["label"] == "row-500"
    result = client.get(base + "?q=row-799").json()
    assert result["filtered_total"] == 1 and result["rows"][0]["label"] == "row-799"
    assert client.get(base + "?offset=-1").status_code == 422


def test_desynced_worker_is_reaped_not_leaked(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review finding S7: a worker emitting a malformed line must be stopped.

    The old code dropped the handle (self._worker = None) without
    stdin.close()/kill()/wait(), leaving a live second worker alongside the
    first against one SQLite workspace. The dispatch now routes the desync
    through _stop_worker, so nothing survives.
    """

    import subprocess
    import sys

    code = (
        "import json, sys\nfrom pathlib import Path\n"
        "from desktop_backend.store import Workspace\n"
        "workspace = Workspace(Path(sys.argv[1]))\n"
        "print(json.dumps({'event': 'ready'}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    job = line.strip()\n"
        "    if sys.argv[2] == 'bad':\n"
        "        job = 'not-this-job'\n"
        "    else:\n"
        "        workspace.update_job(job, state='DONE', stage='Protocol test')\n"
        "    print(json.dumps({'event': 'done', 'job': job, 'returncode': 0}), flush=True)\n"
    )
    broken = True
    monkeypatch.setattr(
        "desktop_backend.runner.worker_command",
        lambda root: [sys.executable, "-u", "-c", code, str(root), "bad" if broken else "good"],
    )
    project_id = workspace.create("Desync")["id"]
    workspace.import_document(project_id, "text.txt", b"Simple words for a simple read.")
    runner = Runner(workspace)
    spawned: list[subprocess.Popen[str]] = []
    original_spawn = runner._spawn_worker

    def track_spawn() -> subprocess.Popen[str] | None:
        worker = original_spawn()
        if worker is not None:
            spawned.append(worker)
        return worker

    monkeypatch.setattr(runner, "_spawn_worker", track_spawn)
    try:
        job = runner.submit(project_id, "readability", {}, "spacy")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if workspace.jobs(project_id)[0]["state"] not in ("QUEUED", "RUNNING"):
                break
            time.sleep(0.1)
        assert workspace.jobs(project_id)[0]["state"] == "FAILED"
        assert len(spawned) == 1
        assert spawned[0].poll() is not None, "desynchronized worker was not reaped"
        assert runner._worker is None
        # Recovery must happen on this same runner, not a fresh instance whose
        # state says nothing about the failed dispatch.
        broken = False
        again = runner.submit(project_id, "readability", {}, "spacy")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            jobs = {item["id"]: item for item in workspace.jobs(project_id)}
            if jobs[again["id"]]["state"] not in ("QUEUED", "RUNNING"):
                break
            time.sleep(0.1)
        assert jobs[again["id"]]["state"] == "DONE"
        assert jobs[job["id"]]["state"] == "FAILED"
        assert len(spawned) == 2 and spawned[0].pid != spawned[1].pid
    finally:
        for worker in spawned:
            if worker.poll() is None:
                worker.kill()
        runner.close()


def test_insight_endpoint_reads_a_finished_run(
    workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The desktop reads the published CSV, not a recomputation of it."""
    source = tmp_path / "dataset.csv"
    source.write_text(
        "word,freq1,freq2\nfreedom,120,10\nliberty,95,40\nnation,80,30\nfuture,60,55\n",
        encoding="utf-8",
    )
    project = workspace.create("Insight research")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    job = runner.submit(
        project["id"],
        "table_keyness",
        {"input": str(source), "word-col": "word", "freq1": "freq1", "freq2": "freq2"},
        "spacy",
    )
    assert run_job(workspace.root, job["id"]) == 0, workspace.jobs(project["id"])
    runner.close()
    with TestClient(create_app(workspace, "test-local-token")) as session:
        session.headers["Authorization"] = "Bearer test-local-token"
        payload = session.get(f"/api/projects/{project['id']}/jobs/{job['id']}/insight").json()
    assert payload["available"] is True
    assert payload["table"].endswith(".csv")
    assert payload["headline"]
    assert isinstance(payload["observations"], list)
    for chart in payload["recommended_charts"]:
        assert {"kind", "x", "y", "question", "why"} <= set(chart)


def test_visualizations_endpoint_marks_legacy_provenance(client: TestClient) -> None:
    """The desktop can badge a chart as carried over from 1.6.38."""
    payload = client.get("/api/visualizations").json()
    assert payload
    by_name = {item["name"]: item for item in payload}
    assert by_name["excel_charts"]["is_legacy"] is True
    assert by_name["excel_charts"]["legacy_module"] == "charts_Excel_util.py"
    assert by_name["dispersion_plot"]["is_legacy"] is False
    for item in payload:
        assert item["origin"] in ("legacy", "legacy-extended", "new")
        assert item["produces"]


def test_setup_names_every_component_it_reports(client: TestClient) -> None:
    """Settings shows this list; raw import names are not something to read."""
    setup = client.get("/api/setup").json()
    assert setup["components"], "the setup payload carries no component inventory"
    for component in setup["components"]:
        assert component["label"] and component["purpose"].endswith("."), component["name"]
        assert isinstance(component["present"], bool)
    assert {"kaleido", "openpyxl"} <= {c["name"] for c in setup["components"]}, (
        "the formats the Visualize dialog offers depend on these two; "
        "without them in the inventory it cannot say which exports will work"
    )


def test_a_run_may_choose_either_parser(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The desktop sent parser="spacy" with every job and offered no choice.

    Both backends have always been accepted here, so a machine with a working
    Stanza install could not use it.
    """
    monkeypatch.setattr(client.app.state.runner.pool, "submit", lambda *args: None)
    project = client.post("/api/projects", json={"name": "Parsers"}).json()
    client.post(
        f"/api/projects/{project['id']}/documents?name=a.txt",
        content=b"The cat sat down. The dog ran off. A bird flew away today.",
    )
    for parser in ("spacy", "stanza"):
        response = client.post(
            f"/api/projects/{project['id']}/jobs",
            json={"tool": "readability", "params": {}, "parser": parser},
        )
        assert response.status_code == 200, (parser, response.json())
    refused = client.post(
        f"/api/projects/{project['id']}/jobs",
        json={"tool": "readability", "params": {}, "parser": "elsewhere"},
    )
    assert refused.status_code == 400


def test_two_runs_can_be_compared_in_one_click(
    client: TestClient, workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The graded "which backend is better" questions, minus the spreadsheet.

    HW2 and HW5 ask which of two backends is better. Answering used to mean
    two exports and a reader with a diff tool; this is the same question as
    one click over ``core.compare``. Two runs of one tool over one document
    are the control: they must agree, and the comparison must say so.
    """
    monkeypatch.setattr(client.app.state.runner.pool, "submit", lambda *args: None)
    project_id = client.post("/api/projects", json={"name": "Compare"}).json()["id"]
    client.post(
        f"/api/projects/{project_id}/documents?name=text.txt",
        content=b"Simple words for a simple read.",
    )
    ids = []
    for _ in range(2):
        response = client.post(f"/api/projects/{project_id}/jobs", json={"tool": "readability"})
        assert response.status_code == 200
        ids.append(response.json()["id"])
        assert run_job(workspace.root, ids[-1]) == 0
    jobs = {j["id"]: j for j in client.get(f"/api/projects/{project_id}/jobs").json()}
    assert [jobs[i]["state"] for i in ids] == ["DONE", "DONE"]

    compared = client.post(
        f"/api/projects/{project_id}/compare",
        json={"expected": ids[0], "actual": ids[1]},
    )
    body = compared.json()
    assert compared.status_code == 200
    assert body["ok"] is True
    assert body["passed"] is True
    assert body["diff"] is None or body["diff"]["total"] == 0


def test_comparing_a_run_that_does_not_exist_says_why(client: TestClient) -> None:
    project_id = client.post("/api/projects", json={"name": "Compare"}).json()["id"]
    body = client.post(
        f"/api/projects/{project_id}/compare",
        json={"expected": "missing-a", "actual": "missing-b"},
    ).json()
    assert body["ok"] is False
    assert body["diagnostics"][0]["code"] == "COMPARE_NO_RUN"


def test_wordcloud_write_failure_reports_the_writer_diagnostic(
    workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wordcloud whose HTML cannot be written fails with the writer's reason,
    not with the (empty) diagnostics of the build that succeeded."""
    from core.io.writer import OutputWriter
    from core.result import Diagnostic, Result

    source = tmp_path / "dataset.csv"
    source.write_text("word,count\nfreedom,120\nliberty,95\n", encoding="utf-8")
    project = workspace.create("Wordcloud write failure")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    job = runner.submit(
        project["id"],
        "table_wordcloud_gephi",
        {"input": str(source), "mode": "wordcloud", "word-col": "word", "weight-col": "count", "image": False},
        "spacy",
    )
    runner.close()

    def refuse(self: OutputWriter, *args: object, **kwargs: object) -> Result[Path]:
        return Result.failure(Diagnostic.error("WRITER_TEST_REFUSED", "disk full"))

    monkeypatch.setattr(OutputWriter, "write_html", refuse)
    assert run_job(workspace.root, job["id"]) == 1
    failed = next(row for row in workspace.jobs(project["id"]) if row["id"] == job["id"])
    assert failed["state"] == "FAILED"
    assert any(d["code"] == "WRITER_TEST_REFUSED" for d in failed["diagnostics"])


def test_a_trashed_project_takes_no_new_runs(workspace: Workspace) -> None:
    """A run queued in Trash would recreate the project's folder and make it unrestorable."""
    project = workspace.create("Trashed")
    workspace.import_document(project["id"], "a.txt", b"Hello world. This is text.")
    workspace.trash_project(project["id"], True)
    runner = Runner(workspace)
    try:
        with pytest.raises(ValueError, match="from Trash before starting an analysis"):
            runner.submit(project["id"], "text_statistics", {}, "spacy")
    finally:
        runner.close()
    assert workspace.jobs(project["id"], include_trashed=True) == []
    workspace.trash_project(project["id"], False)


def test_a_changed_corpus_file_is_a_400_on_the_scripts_page(client: TestClient, workspace: Workspace) -> None:
    """The notebook library refuses with a SuiteError; the page must get its words, not a 500."""
    project = workspace.create("Scripts")
    document = workspace.import_document(project["id"], "a.txt", b"Hello world. This is text.")
    stored = workspace.project_dir(project["id"]) / "corpus" / str(document["stored_name"])
    stored.write_bytes(b"Edited behind the app's back.")
    response = client.get(f"/api/projects/{project['id']}/script-corpus")
    assert response.status_code == 400, response.text
    assert response.json()["detail"]


def test_a_cleared_resource_field_submits_as_no_file(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Clearing a resource sends "", which used to be opened as the current directory and refused."""
    project = workspace.create("Cleared")
    workspace.import_document(project["id"], "a.txt", b"A test document. Another one.")
    workspace.import_document(project["id"], "b.txt", b"A second document here.")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    try:
        job = runner.submit(project["id"], "doc_similarity", {"stopwords": ""}, "spacy")
    finally:
        runner.close()
    assert job["state"] == "QUEUED"
    [stored] = workspace.jobs(project["id"])
    request = json.loads(stored["request"]) if isinstance(stored["request"], str) else stored["request"]
    assert request["params"]["doc_similarity"]["stopwords"] is None
    assert request["resources"] == []
