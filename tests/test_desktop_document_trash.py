from __future__ import annotations

import json
from pathlib import Path

import pytest

from desktop_backend.archives import backup, restore
from desktop_backend.server import create_app
from desktop_backend.store import Workspace, now


def test_document_trash_restore_and_purge_cover_both_owned_files(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create("Corpus")
    doc = workspace.import_document(project["id"], "paper.txt", b"A small test corpus.")
    assert workspace.trash_document(project["id"], doc["id"], True)["trashed"] == 1
    assert workspace.documents(project["id"]) == []
    assert workspace.documents(project["id"], trashed=True)[0]["id"] == doc["id"]
    workspace.trash_document(project["id"], doc["id"], False)
    assert workspace.documents(project["id"])[0]["id"] == doc["id"]
    workspace.trash_document(project["id"], doc["id"], True)
    corpus_file = workspace.project_dir(project["id"]) / "corpus" / doc["stored_name"]
    original_file = workspace.project_dir(project["id"]) / "originals" / "source.docx"
    original_file.parent.mkdir(parents=True)
    original_file.write_bytes(b"source bytes")
    with workspace.connect() as db:
        db.execute("UPDATE documents SET source_name='source.docx', source_sha256=? WHERE id=?", ("0" * 64, doc["id"]))
    workspace.purge_document(project["id"], doc["id"])
    assert not corpus_file.exists()
    assert not original_file.exists()
    assert workspace.documents(project["id"], include_trashed=True) == []


def test_trash_survives_backup_restore_and_saved_question_reference_blocks_purge(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create("Corpus")
    doc = workspace.import_document(project["id"], "paper.txt", b"A small test corpus.")
    with workspace.connect() as db:
        db.execute(
            "INSERT INTO questions(id,project_id,name,snapshot_id,parser,document_ids,specification,matching,created,updated) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                "q1",
                project["id"],
                "Question",
                "0" * 64,
                "spacy",
                json.dumps([doc["id"]]),
                json.dumps({"text": "paper"}),
                "{}",
                now(),
                now(),
            ),
        )
    workspace.trash_document(project["id"], doc["id"], True)
    try:
        workspace.purge_document(project["id"], doc["id"])
    except ValueError as exc:
        assert "Question" in str(exc)
    else:
        raise AssertionError("purge must preserve saved question document references")
    archive = tmp_path / "backup.nlpsuite"
    backup(workspace, project["id"], archive)
    restored = restore(workspace, archive)
    active = workspace.documents(restored["id"])
    trashed = workspace.documents(restored["id"], trashed=True)
    assert active == []
    assert len(trashed) == 1 and trashed[0]["trashed"] == 1
    assert len(workspace.questions(restored["id"])) == 1


def test_document_api_hides_storage_names_and_refuses_changes_during_active_jobs(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create("Corpus")
    doc = workspace.import_document(project["id"], "paper.txt", b"A small test corpus.")
    with TestClient(create_app(workspace, "token")) as client:
        client.headers["Authorization"] = "Bearer token"
        response = client.get(f"/api/projects/{project['id']}/documents")
        assert response.status_code == 200
        assert "stored_name" not in response.json()[0]
        assert "source_name" not in response.json()[0]
        with workspace.connect() as db:
            db.execute(
                "INSERT INTO jobs(id,project_id,tool,state,stage,created,request) VALUES(?,?,?,?,?,?,?)",
                ("active", project["id"], "search", "RUNNING", "Working", now(), "{}"),
            )
        refused = client.post(f"/api/projects/{project['id']}/documents/{doc['id']}/trash", json={})
        assert refused.status_code == 400
        with workspace.connect() as db:
            db.execute("UPDATE jobs SET state='DONE' WHERE id='active'")
        moved = client.post(f"/api/projects/{project['id']}/documents/{doc['id']}/trash", json={})
        assert moved.status_code == 200
        assert moved.json()["id"] == doc["id"]
        assert client.get(f"/api/projects/{project['id']}/documents").json() == []


def test_run_trash_restore_and_purge_keeps_saved_view_explicitly_dangling(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create("Corpus")
    run_id = "run1"
    run_dir = workspace.project_dir(project["id"]) / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text("{}", encoding="utf-8")
    input_dir = workspace.project_dir(project["id"]) / "job-inputs" / run_id
    input_dir.mkdir(parents=True)
    (input_dir / "frozen.csv").write_text("data", encoding="utf-8")
    (workspace.root / "logs").mkdir()
    (workspace.root / "logs" / f"{run_id}.log").write_text("failure", encoding="utf-8")
    with workspace.connect() as db:
        db.execute(
            "INSERT INTO jobs(id,project_id,tool,state,stage,created,run_dir,request) VALUES(?,?,?,?,?,?,?,?)",
            (run_id, project["id"], "search", "FAILED", "Failed", now(), f"runs/{run_id}", "{}"),
        )
        db.execute(
            "INSERT INTO views(id,project_id,name,job_id,artifact_index,artifact_path,source_sha256,settings,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("view1", project["id"], "Kept settings", run_id, 0, "table.csv", "0" * 64, "{}", now(), now()),
        )
    workspace.trash_run(project["id"], run_id, True)
    assert workspace.jobs(project["id"]) == []
    assert workspace.jobs(project["id"], trashed=True)[0]["id"] == run_id
    workspace.trash_run(project["id"], run_id, False)
    assert workspace.jobs(project["id"])[0]["id"] == run_id
    workspace.trash_run(project["id"], run_id, True)
    workspace.purge_run(project["id"], run_id)
    assert not run_dir.exists() and not input_dir.exists()
    assert not (workspace.root / "logs" / f"{run_id}.log").exists()
    assert workspace.views(project["id"])[0]["job"] == run_id


def test_archived_project_trash_restore_and_purge_removes_dependent_rows(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create("Archived")
    doc = workspace.import_document(project["id"], "paper.txt", b"Content")
    other = workspace.create("Keep me")
    other_doc = workspace.import_document(other["id"], "safe.txt", b"Keep this content")
    (workspace.root / "logs").mkdir(exist_ok=True)
    (workspace.root / "logs" / "project-run.log").write_text("run log", encoding="utf-8")
    with workspace.connect() as db:
        db.execute(
            "INSERT INTO jobs(id,project_id,tool,state,stage,created,request) VALUES(?,?,?,?,?,?,?)",
            ("project-run", project["id"], "search", "FAILED", "Failed", now(), "{}"),
        )
    folder = workspace.project_dir(project["id"])
    workspace.archive(project["id"], True)
    workspace.trash_project(project["id"], True)
    assert project["id"] not in {item["id"] for item in workspace.projects()}
    assert project["id"] in {item["id"] for item in workspace.projects(trashed=True)}
    held = workspace.root / "trash" / f"project-{project['id']}"
    assert held.is_dir() and not folder.exists()
    workspace.trash_project(project["id"], False)
    assert workspace.project_dir(project["id"]).is_dir()
    workspace.trash_project(project["id"], True)
    workspace.purge_project(project["id"])
    assert not held.exists()
    assert not (workspace.root / "logs" / "project-run.log").exists()
    assert workspace.project(other["id"])["id"] == other["id"]
    assert workspace.preview(other["id"], other_doc["id"])["text"] == "Keep this content"
    with workspace.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM projects WHERE id=?", (project["id"],)).fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM documents WHERE id=?", (doc["id"],)).fetchone()[0] == 0


def test_backup_preserves_trashed_run_state_on_restore(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create("Runs")
    with workspace.connect() as db:
        db.execute(
            "INSERT INTO jobs(id,project_id,tool,state,stage,created,request,trashed) VALUES(?,?,?,?,?,?,?,1)",
            ("old-run", project["id"], "search", "FAILED", "Failed", now(), "{}"),
        )
    archive = tmp_path / "trashed-run.nlpsuite"
    backup(workspace, project["id"], archive)
    restored = restore(workspace, archive)
    assert workspace.jobs(restored["id"]) == []
    assert len(workspace.jobs(restored["id"], trashed=True)) == 1


def test_project_without_run_logs_can_be_purged(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create("Empty")
    workspace.trash_project(project["id"], True)
    workspace.purge_project(project["id"])
    with pytest.raises(KeyError):
        workspace.project(project["id"])


def test_run_purge_refuses_symlinked_artifact_directory(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create("Safe")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "important.txt").write_text("keep", encoding="utf-8")
    link = workspace.project_dir(project["id"]) / "runs" / "escaped"
    link.parent.mkdir()
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this Windows host")
    with workspace.connect() as db:
        db.execute(
            "INSERT INTO jobs(id,project_id,tool,state,stage,created,run_dir,request,trashed) VALUES(?,?,?,?,?,?,?,?,1)",
            ("unsafe", project["id"], "search", "FAILED", "Failed", now(), "runs/escaped", "{}"),
        )
    with pytest.raises(ValueError, match="unsafe"):
        workspace.purge_run(project["id"], "unsafe")
    assert (outside / "important.txt").read_text(encoding="utf-8") == "keep"
