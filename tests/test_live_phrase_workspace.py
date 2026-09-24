"""The live phrase endpoints: position filters, and the linked source reader.

The workspace answer is only as followable as its links. A count that cannot
open the passage it counted, and a filter that silently redefines the
aggregates it sits beside, are the two failures these endpoints exist to
prevent -- so both are pinned at the HTTP boundary, not only in the core
module.
"""

from __future__ import annotations

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


def _warm(client: Any, project: str) -> str:
    state = client.post(
        f"/api/projects/{project}/live/warm",
        json={"parser": "spacy"},
    )
    assert state.status_code == 200, state.text
    snapshot = state.json()["snapshot_id"]
    assert snapshot
    return snapshot


def _project_with_corpus(client: Any) -> str:
    project = client.workspace.create("Reader")
    client.workspace.import_document(project["id"], "first.txt", b"public health matters here")
    return str(project["id"])


def test_source_passage_serves_the_loaded_documents(client: Any) -> None:
    project = _project_with_corpus(client)
    _warm(client, project)
    documents = client.get(f"/api/projects/{project}/documents").json()
    document_id = documents[0]["id"]
    read = client.get(
        f"/api/projects/{project}/live/source",
        params={"document_id": document_id, "character_start": 7, "character_end": 13},
    )
    assert read.status_code == 200, read.text
    passage = read.json()
    assert passage["document_id"] == document_id
    assert passage["text"][7 - passage["character_start"] : 13 - passage["character_start"]] == "health"
    assert passage["character_offset_unit"] == "unicode_code_point"


def test_source_passage_is_refused_before_anything_is_loaded(client: Any) -> None:
    project = str(client.workspace.create("Cold")["id"])
    read = client.get(
        f"/api/projects/{project}/live/source",
        params={"document_id": "anything", "character_start": 0},
    )
    assert read.status_code == 400
    assert "documents" in read.json()["detail"]


def test_source_passage_refuses_an_offset_outside_the_document(client: Any) -> None:
    project = _project_with_corpus(client)
    _warm(client, project)
    documents = client.get(f"/api/projects/{project}/documents").json()
    read = client.get(
        f"/api/projects/{project}/live/source",
        params={"document_id": documents[0]["id"], "character_start": 100_000},
    )
    assert read.status_code == 400
    assert "not part" in read.json()["detail"]


def test_phrase_position_filter_reaches_the_core(client: Any) -> None:
    project = _project_with_corpus(client)
    snapshot = _warm(client, project)
    answered = client.post(
        f"/api/projects/{project}/live/phrase",
        json={
            "text": "health",
            "snapshot_id": snapshot,
            "evidence_position_start": 0.5,
            "evidence_position_end": 1.0,
        },
    )
    assert answered.status_code == 200, answered.text
    body = answered.json()
    assert body["evidence"]["filter"]["position_start"] == 0.5
    # "health" sits at 7/27 of this document: before the midpoint, so the
    # filter keeps it out of the evidence while the summary stays global.
    assert body["evidence"]["filtered_total"] == 0
    assert body["summary"]["occurrences"] == 1


def test_phrase_rejects_an_inverted_position_range(client: Any) -> None:
    project = _project_with_corpus(client)
    snapshot = _warm(client, project)
    answered = client.post(
        f"/api/projects/{project}/live/phrase",
        json={
            "text": "health",
            "snapshot_id": snapshot,
            "evidence_position_start": 0.9,
            "evidence_position_end": 0.2,
        },
    )
    assert answered.status_code == 400
    assert "end after its start" in answered.json()["detail"]
