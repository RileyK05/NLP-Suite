"""A comparison's source context must reach beyond the ordinary preview."""

from pathlib import Path

from fastapi.testclient import TestClient

from desktop_backend.server import create_app
from desktop_backend.store import Workspace


def test_passage_context_uses_full_source_and_parser_spacing(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create("Speeches")
    text = "Opening. " * 4_000 + "To our sister republics -- we offer a pledge. Closing."
    document = workspace.import_document(project["id"], "speech.txt", text.encode())
    assert workspace.preview(project["id"], document["id"])["truncated"]

    context = workspace.passage_context(project["id"], document["id"], "To our sister republics -- we offer a pledge.")
    assert context["found"]
    assert "To our sister republics -- we offer a pledge." in context["excerpt"]
    assert "Closing." in context["excerpt"]

    spaced = workspace.passage_context(project["id"], document["id"], "To our sister republics  --  we offer a pledge.")
    assert spaced["found"]
    assert "To our sister republics -- we offer a pledge." in spaced["excerpt"]
    assert workspace.passage_context(project["id"], document["id"], "unrelated words")["found"] is False

    with TestClient(create_app(workspace, "test-token")) as client:
        client.headers["Authorization"] = "Bearer test-token"
        response = client.post(
            f"/api/projects/{project['id']}/documents/{document['id']}/context",
            json={"passage": "To our sister republics  --  we offer a pledge."},
        )
        assert response.status_code == 200
        assert response.json()["found"]
        assert "Closing." in response.json()["excerpt"]
