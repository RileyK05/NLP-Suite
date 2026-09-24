"""Frame responses have independent policies without exposing workspace auth."""

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from desktop_backend.previews import PreviewTickets
from desktop_backend.server import create_app
from desktop_backend.store import Workspace


def test_browser_preview_is_scoped_and_has_independent_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = Workspace(tmp_path / "workspace")
    chart = tmp_path / "chart.html"
    chart.write_text("<script>document.body.textContent='rendered';</script>", encoding="utf-8")
    monkeypatch.setattr(workspace, "artifact", lambda *args: chart)
    endpoint = "/api/projects/" + "a" * 32 + "/jobs/" + "b" * 32 + "/artifacts/0"
    with TestClient(create_app(workspace, "workspace-secret")) as client:
        assert client.post(endpoint + "/preview").status_code == 401
        client.headers["Authorization"] = "Bearer workspace-secret"
        prepared = client.post(endpoint + "/preview")
        assert prepared.status_code == 200
        preview = prepared.json()["path"]
        assert "workspace-secret" not in preview
        main_policy = client.get("/api/health").headers["Content-Security-Policy"]
        assert "script-src 'self'" in main_policy
        assert "frame-src 'self'" in main_policy
        download = client.get(endpoint + "?download=true")
        assert download.headers["content-type"] == "application/octet-stream"
        assert "attachment" in download.headers["content-disposition"]
        client.headers.clear()
        response = client.get(preview)
        assert response.status_code == 200 and "rendered" in response.text
        policy = response.headers["Content-Security-Policy"]
        assert "script-src 'unsafe-inline'" in policy
        assert "connect-src 'none'" in policy and "sandbox allow-scripts" in policy
        assert "allow-same-origin" not in policy
        assert response.headers["Cache-Control"] == "no-store"
        assert client.get(endpoint).status_code == 401
        assert client.get(preview + "invalid").status_code == 404


def test_tickets_expire_and_are_bounded() -> None:
    tickets = PreviewTickets(capacity=1)
    first = tickets.issue("project", "first", 0)
    second = tickets.issue("project", "second", 1)
    with pytest.raises(KeyError):
        tickets.resolve(first)
    assert tickets.resolve(second) == ("project", "second", 1)
    expired = PreviewTickets(ttl=0)
    ticket = expired.issue("project", "job", 0)
    with pytest.raises(KeyError):
        expired.resolve(ticket)
