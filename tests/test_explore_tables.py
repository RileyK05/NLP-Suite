"""The list of tables the interaction lab can open.

Explore asks one question -- which table? -- so the answer has to be the
whole answer: every CSV any finished run in the project published, named
well enough to tell two of them apart, and never a hard failure because one
run directory has gone missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from desktop_backend.runner import INPUT_MANIFEST, Runner, run_job
from desktop_backend.server import create_app
from desktop_backend.store import Workspace

DATASET = "word,freq1,freq2\nfreedom,120,10\nliberty,95,40\nnation,80,30\nfuture,60,55\n"


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "workspace")


@pytest.fixture
def client(workspace: Workspace):
    with TestClient(create_app(workspace, "test-local-token")) as session:
        session.headers["Authorization"] = "Bearer test-local-token"
        yield session


def finished_run(workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> dict:
    """A project with one real completed run in it."""
    source = tmp_path / f"{name}.csv"
    source.write_text(DATASET, encoding="utf-8")
    project = workspace.create(name)
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
    return project


def test_an_empty_project_offers_nothing_rather_than_failing(client: TestClient, workspace: Workspace) -> None:
    project = workspace.create("Nothing run yet")
    response = client.get(f"/api/projects/{project['id']}/tables")
    assert response.status_code == 200
    assert response.json() == []


def test_every_published_csv_is_offered(
    client: TestClient, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = finished_run(workspace, tmp_path, monkeypatch, "keyness")
    _, envelope = workspace.artifacts(project["id"], workspace.jobs(project["id"])[0]["id"])
    published = {a.path for a in envelope.artifacts if a.path.endswith(".csv")}
    offered = {row["path"] for row in client.get(f"/api/projects/{project['id']}/tables").json()}
    assert offered == published
    assert published, "the fixture run published no CSV, so this test proves nothing"


def test_each_table_carries_what_it_takes_to_open_and_name_it(
    client: TestClient, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = finished_run(workspace, tmp_path, monkeypatch, "keyness")
    rows = client.get(f"/api/projects/{project['id']}/tables").json()
    for row in rows:
        assert row["label"] and row["label"] != row["tool"], row
        assert row["created"]
        assert isinstance(row["manifest"], bool)
        # The index must address the artifact it claims to.
        artifact = client.get(f"/api/projects/{project['id']}/jobs/{row['job']}/artifacts/{row['index']}")
        assert artifact.status_code == 200, artifact.text
        assert artifact.json()["columns"], row["path"]


def test_the_record_of_what_was_read_is_flagged_not_hidden(
    client: TestClient, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explore opens on a result; the input manifest stays available."""
    project = workspace.create("Corpus run")
    workspace.import_document(project["id"], "a.txt", b"One sentence here. And a second one.")
    runner = client.app.state.runner
    runner.submit(project["id"], "readability", {}, "spacy")
    runner.close()
    assert workspace.jobs(project["id"])[0]["state"] == "DONE", workspace.jobs(project["id"])
    rows = client.get(f"/api/projects/{project['id']}/tables").json()
    by_path = {row["path"]: row for row in rows}
    assert INPUT_MANIFEST in by_path, "the manifest should still be offered"
    assert by_path[INPUT_MANIFEST]["manifest"] is True
    assert [row["path"] for row in rows if not row["manifest"]], "no result table was offered"


def test_a_missing_run_directory_costs_its_own_row_only(
    client: TestClient, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One broken run must not make the whole project unexplorable."""
    project = finished_run(workspace, tmp_path, monkeypatch, "first")
    before = client.get(f"/api/projects/{project['id']}/tables").json()
    assert before

    source = tmp_path / "second.csv"
    source.write_text(DATASET, encoding="utf-8")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    second = runner.submit(
        project["id"],
        "table_keyness",
        {"input": str(source), "word-col": "word", "freq1": "freq1", "freq2": "freq2"},
        "spacy",
    )
    assert run_job(workspace.root, second["id"]) == 0
    runner.close()
    directory, _ = workspace.artifacts(project["id"], second["id"])
    (directory / "result.json").unlink()

    response = client.get(f"/api/projects/{project['id']}/tables")
    assert response.status_code == 200, response.text
    assert {row["job"] for row in response.json()} == {row["job"] for row in before}


def test_a_failed_run_is_not_offered(
    client: TestClient, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "wrong-shape.csv"
    source.write_text(DATASET, encoding="utf-8")
    project = workspace.create("Broken run")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    job = runner.submit(
        project["id"],
        "table_keyness",
        {"input": str(source), "word-col": "absent", "freq1": "freq1", "freq2": "freq2"},
        "spacy",
    )
    run_job(workspace.root, job["id"])
    runner.close()
    assert workspace.jobs(project["id"])[0]["state"] != "DONE"
    assert client.get(f"/api/projects/{project['id']}/tables").json() == []


def test_tables_are_newest_first(
    client: TestClient, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = finished_run(workspace, tmp_path, monkeypatch, "older")
    source = tmp_path / "newer.csv"
    source.write_text(DATASET, encoding="utf-8")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    newer = runner.submit(
        project["id"],
        "table_keyness",
        {"input": str(source), "word-col": "word", "freq1": "freq1", "freq2": "freq2"},
        "spacy",
    )
    assert run_job(workspace.root, newer["id"]) == 0
    runner.close()
    rows = client.get(f"/api/projects/{project['id']}/tables").json()
    assert rows[0]["job"] == newer["id"], "the run just finished should be at the top"


def test_listing_tables_reads_and_writes_nothing(
    client: TestClient, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3: a read must leave the run directory byte-identical."""
    project = finished_run(workspace, tmp_path, monkeypatch, "keyness")
    directory, _ = workspace.artifacts(project["id"], workspace.jobs(project["id"])[0]["id"])
    before = {p.name: p.stat().st_mtime_ns for p in sorted(directory.rglob("*")) if p.is_file()}
    client.get(f"/api/projects/{project['id']}/tables")
    after = {p.name: p.stat().st_mtime_ns for p in sorted(directory.rglob("*")) if p.is_file()}
    assert before == after


def test_a_table_whose_file_is_gone_is_not_offered(
    client: TestClient, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The envelope records what a run published, not what is on disk now.

    Found by pointing the running app at a workspace whose run directories had
    been copied incompletely: every listed table was offered and every one of
    them answered 400 when opened.
    """
    project = finished_run(workspace, tmp_path, monkeypatch, "keyness")
    job_id = workspace.jobs(project["id"])[0]["id"]
    directory, envelope = workspace.artifacts(project["id"], job_id)
    gone = next(a.path for a in envelope.artifacts if a.path.endswith(".csv"))
    (directory / gone).unlink()

    rows = client.get(f"/api/projects/{project['id']}/tables").json()
    assert gone not in {row["path"] for row in rows}
    # Whatever is still offered must still open.
    for row in rows:
        response = client.get(f"/api/projects/{project['id']}/jobs/{row['job']}/artifacts/{row['index']}")
        assert response.status_code == 200, f"{row['path']}: {response.text}"


def test_every_offered_table_can_actually_be_opened(
    client: TestClient, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The listing is a promise; this is the promise being kept."""
    project = finished_run(workspace, tmp_path, monkeypatch, "keyness")
    rows = client.get(f"/api/projects/{project['id']}/tables").json()
    assert rows
    for row in rows:
        response = client.get(f"/api/projects/{project['id']}/jobs/{row['job']}/artifacts/{row['index']}")
        assert response.status_code == 200, f"{row['path']}: {response.text}"
        assert response.json()["columns"]
