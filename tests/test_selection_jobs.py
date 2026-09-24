"""Integration contracts for interactive corpus selection and provenance."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from desktop_backend.archives import backup, restore
from desktop_backend.runner import Runner, run_job
from desktop_backend.selection import CorpusSelection
from desktop_backend.server import create_app
from desktop_backend.store import Workspace


def _app(workspace: Workspace) -> TestClient:
    client = TestClient(create_app(workspace, "selection-test-token"))
    client.headers["Authorization"] = "Bearer selection-test-token"
    return client


def test_subset_is_frozen_and_published_with_scope_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = Workspace(tmp_path / "workspace")
    project_id = workspace.create("Selection")["id"]
    workspace.import_document(project_id, "speech_2020-01-15.txt", b"First document with enough words.")
    second = workspace.import_document(project_id, "speech_2021-02-28.txt", b"Second document should not run.")
    runner = Runner(workspace)
    try:
        monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
        selection = CorpusSelection(document_ids=[second["id"]], date_from="2021-01-01", date_to="2021-12-31")
        job = runner.submit(project_id, "readability", {}, "spacy", selection=selection)
        workspace.import_document(project_id, "speech_2022-03-01.txt", b"A later import must not enter this run.")

        request = json.loads(workspace.jobs(project_id)[0]["request"])
        assert request["selection"] == selection.model_dump()
        assert [document["id"] for document in request["documents"]] == [second["id"]]
        assert request["documents"][0]["document_date"] == "2021-02-28"
        assert run_job(workspace.root, job["id"]) == 0
        directory, envelope = workspace.artifacts(project_id, job["id"])
        assert len(envelope.inputs) == 1
        with (directory / "desktop_inputs.csv").open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["id"] == second["id"]
        assert rows[0]["selection_mode"] == "explicit"
        assert rows[0]["date_from"] == "2021-01-01"
        assert rows[0]["date_to"] == "2021-12-31"
        assert rows[0]["include_undated"] == "False"
    finally:
        runner.close()


def test_get_documents_exposes_filename_date_provenance(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    project_id = workspace.create("Metadata")["id"]
    workspace.import_document(project_id, "report_2024-01-02.txt", b"A short report.")
    workspace.import_document(project_id, "notes.txt", b"No date in this name.")
    with _app(workspace) as client:
        response = client.get(f"/api/projects/{project_id}/documents")
    assert response.status_code == 200
    rows = response.json()
    dated = next(row for row in rows if row["name"].startswith("report"))
    undated = next(row for row in rows if row["name"] == "notes.txt")
    assert dated["document_date"] == "2024-01-02"
    assert dated["date_source"] == "filename"
    assert undated["document_date"] is None
    assert undated["date_source"] is None


def test_omitted_selection_freezes_the_whole_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = Workspace(tmp_path / "workspace")
    project_id = workspace.create("All documents")["id"]
    first = workspace.import_document(project_id, "one.txt", b"First document.")
    second = workspace.import_document(project_id, "two.txt", b"Second document.")
    runner = Runner(workspace)
    try:
        monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
        runner.submit(project_id, "readability", {}, "spacy")
        workspace.import_document(project_id, "three.txt", b"Later document.")
        request = json.loads(workspace.jobs(project_id)[0]["request"])
        assert request["selection"] is None
        assert [doc["id"] for doc in request["documents"]] == [first["id"], second["id"]]
    finally:
        runner.close()


def test_selection_criteria_survive_project_backup_and_restore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = Workspace(tmp_path / "workspace")
    project_id = workspace.create("Backup selection")["id"]
    document = workspace.import_document(project_id, "speech_2020-01-15.txt", b"Backed up selection.")
    runner = Runner(workspace)
    try:
        monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
        selection = CorpusSelection(document_ids=[document["id"]], date_from="2020-01-01", date_to="2020-12-31")
        job = runner.submit(project_id, "readability", {}, "spacy", selection=selection)
        assert run_job(workspace.root, job["id"]) == 0
    finally:
        runner.close()
    archive = workspace.root / "selection.nlpsuite"
    backup(workspace, project_id, archive)
    restored = restore(workspace, archive)
    restored_request = json.loads(workspace.jobs(restored["id"])[0]["request"])
    assert restored_request["selection"] == selection.model_dump()
    assert restored_request["documents"][0]["document_date"] == "2020-01-15"


def test_api_rejects_invalid_or_foreign_selection_before_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = Workspace(tmp_path / "workspace")
    project_id = workspace.create("API selection")["id"]
    workspace.import_document(project_id, "speech_2020-01-01.txt", b"A simple document.")
    client = _app(workspace)
    runner = client.app.state.runner
    monkeypatch.setattr(runner.pool, "submit", lambda *args: (_ for _ in ()).throw(AssertionError("queued")))
    try:
        foreign = client.post(
            f"/api/projects/{project_id}/jobs",
            json={"tool": "readability", "selection": {"document_ids": ["missing"]}},
        )
        assert foreign.status_code in (400, 422)
        empty = client.post(
            f"/api/projects/{project_id}/jobs",
            json={"tool": "readability", "selection": {"document_ids": []}},
        )
        assert empty.status_code in (400, 422)
        bad_date = client.post(
            f"/api/projects/{project_id}/jobs",
            json={"tool": "readability", "selection": {"date_from": "2020-02-30"}},
        )
        assert bad_date.status_code == 422
        table = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "tool": "table_chi2",
                "selection": {"document_ids": ["anything"]},
                "params": {"input": str(tmp_path / "missing.csv"), "col1": "a", "col2": "b"},
            },
        )
        assert table.status_code in (400, 422)
        assert workspace.jobs(project_id) == []
    finally:
        client.close()


def test_public_job_scope_contains_only_safe_selection_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = Workspace(tmp_path / "workspace")
    project_id = workspace.create("Public scope")["id"]
    document = workspace.import_document(project_id, "speech_2020-01-01.txt", b"A public scope test.")
    client = _app(workspace)
    runner = client.app.state.runner
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    try:
        response = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "tool": "readability",
                "selection": {
                    "document_ids": [document["id"]],
                    "date_from": "2020-01-01",
                    "date_to": "2020-12-31",
                    "include_undated": True,
                },
            },
        )
        assert response.status_code == 200
        assert set(response.json()["scope"]) == {
            "document_count",
            "date_from",
            "date_to",
            "explicit_documents",
            "include_undated",
        }
        assert response.json()["scope"] == {
            "document_count": 1,
            "date_from": "2020-01-01",
            "date_to": "2020-12-31",
            "explicit_documents": True,
            "include_undated": True,
        }
    finally:
        client.close()
