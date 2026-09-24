"""Saved research questions survive navigation without losing their scope."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from desktop_backend.server import create_app
from desktop_backend.store import Workspace


@pytest.fixture
def client(tmp_path: Path):
    workspace = Workspace(tmp_path / "workspace")
    with TestClient(create_app(workspace, "tok")) as session:
        session.headers["Authorization"] = "Bearer tok"
        session.workspace = workspace  # type: ignore[attr-defined]
        yield session


def _project(client: Any) -> tuple[str, list[str]]:
    project = client.workspace.create("Questions")
    first = client.workspace.import_document(project["id"], "2020-first.txt", b"public health")
    second = client.workspace.import_document(project["id"], "2021-second.txt", b"other words")
    return str(project["id"]), [str(first["id"]), str(second["id"])]


def _body(ids: list[str], name: str = "Public health over time") -> dict[str, Any]:
    return {
        "name": name,
        "snapshot_id": "a" * 64,
        "parser": "spacy",
        "document_ids": ids,
        "specification": {
            "schema_version": 1,
            "kind": "phrase_distribution",
            "text": "public health",
            "comparison_text": "private health",
            "case_sensitive": False,
            "position_bins": 10,
            "evidence_year": 2020,
            "evidence_document_id": ids[0],
        },
    }


def test_question_crud_preserves_scope_and_revision(client: Any) -> None:
    project, ids = _project(client)
    saved = client.post(f"/api/projects/{project}/questions", json=_body(ids))
    assert saved.status_code == 200, saved.text
    question = saved.json()
    assert question["document_ids"] == ids
    assert question["specification"]["text"] == "public health"
    assert question["specification"]["comparison_text"] == "private health"
    assert question["revision"] == 1

    changed = _body(ids, "Public health by year")
    changed["specification"]["position_bins"] = 20
    updated = client.post(f"/api/projects/{project}/questions/{question['id']}", json=changed).json()
    assert updated["revision"] == 2
    assert updated["specification"]["position_bins"] == 20

    copy = client.post(f"/api/projects/{project}/questions/{question['id']}/duplicate").json()
    assert copy["id"] != question["id"]
    assert copy["name"] == "Public health by year (copy)"
    assert copy["revision"] == 1

    assert client.post(f"/api/projects/{project}/questions/{copy['id']}/delete").json() == {"deleted": True}
    assert [item["id"] for item in client.get(f"/api/projects/{project}/questions").json()] == [question["id"]]


def test_question_rejects_documents_outside_the_project(client: Any) -> None:
    project, ids = _project(client)
    refused = client.post(f"/api/projects/{project}/questions", json=_body([*ids, "0" * 32]))
    assert refused.status_code == 400
    assert "no longer" in refused.json()["detail"]


def test_question_rejects_an_evidence_filter_outside_its_scope(client: Any) -> None:
    project, ids = _project(client)
    body = _body([ids[0]])
    body["specification"]["evidence_document_id"] = ids[1]
    refused = client.post(f"/api/projects/{project}/questions", json=body)
    assert refused.status_code == 400
    assert "outside" in refused.json()["detail"]


def test_question_payload_is_strict(client: Any) -> None:
    project, ids = _project(client)
    refused = client.post(f"/api/projects/{project}/questions", json={**_body(ids), "surprise": True})
    assert refused.status_code == 422


def test_new_project_has_an_empty_question_list(client: Any) -> None:
    project = client.workspace.create("Empty")
    assert client.get(f"/api/projects/{project['id']}/questions").json() == []


def test_saved_question_publishes_an_immutable_job(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    project, ids = _project(client)
    question = client.post(f"/api/projects/{project}/questions", json=_body(ids)).json()
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(client.app.state.runner.pool, "submit", lambda *_args, **_kwargs: None)

    response = client.post(f"/api/projects/{project}/questions/{question['id']}/publish", json={})
    assert response.status_code == 200, response.text
    job = response.json()
    assert job["tool"] == "phrase_distribution"
    assert job["scope"]["document_count"] == 2
    assert job["scope"]["explicit_documents"] is True
    assert "phrase_distribution" not in {tool["name"] for tool in client.get("/api/tools").json()}

    with client.workspace.connect() as db:
        request = json.loads(db.execute("SELECT request FROM jobs WHERE id=?", (job["id"],)).fetchone()[0])
    params = request["params"]["phrase_distribution"]
    assert [item["id"] for item in request["documents"]] == ids
    assert params["phrase"] == "public health"
    assert params["comparison"] == "private health"
    assert params["question-revision"] == 1
    assert params["snapshot-id"] == "a" * 64


def test_questions_survive_backup_with_document_filters_remapped(client: Any) -> None:
    from desktop_backend.archives import backup, restore

    project, ids = _project(client)
    saved = client.post(f"/api/projects/{project}/questions", json=_body(ids)).json()
    archive = client.workspace.root / "questions.nlpsuite"
    backup(client.workspace, project, archive)
    restored = restore(client.workspace, archive)

    question = client.workspace.questions(restored["id"])[0]
    restored_ids = [str(doc["id"]) for doc in client.workspace.documents(restored["id"])]
    assert question["name"] == saved["name"]
    assert question["snapshot_id"] == saved["snapshot_id"]
    assert question["document_ids"] != ids
    assert set(question["document_ids"]) == set(restored_ids)
    assert question["specification"]["evidence_document_id"] in restored_ids
