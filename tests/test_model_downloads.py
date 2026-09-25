"""Model downloads: resume, verification, cancel, disk space, offline, and the API.

A local HTTP server stands in for the GitHub release. It honours ``Range``
(as GitHub's asset CDN does), can drop a connection part-way through a
file, and can serve damaged bytes, so every path of
:mod:`core.models.download` runs against real sockets.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import Any

import pytest

from core.models import _files, download, locate, registry

TINY = Path(__file__).parent / "fixtures" / "models" / "tiny-classifier"
MODEL = "distilbert-sst2"


@dataclass
class Release:
    """What the fake release serves, and how it misbehaves."""

    assets: dict[str, bytes] = field(default_factory=dict)
    #: Close the connection after this many bytes of the next response.
    drop_after: int | None = None
    #: Serve these bytes instead of the real ones.
    damaged: set[str] = field(default_factory=set)
    #: Answer Range requests with the whole file (a server that ignores Range).
    ignore_range: bool = False
    ranges: list[str] = field(default_factory=list)


def _handler(release: Release) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            name = self.path.rsplit("/", 1)[-1]
            body = release.assets.get(name)
            if body is None:
                self.send_error(404)
                return
            if name in release.damaged:
                body = bytes(len(body))
            start = 0
            header = self.headers.get("Range")
            if header:
                release.ranges.append(header)
            if header and not release.ignore_range:
                start = int(header.removeprefix("bytes=").split("-")[0])
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
            else:
                self.send_response(200)
            payload = body[start:]
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if release.drop_after is not None:
                cut, release.drop_after = release.drop_after, None
                self.wfile.write(payload[:cut])
                self.wfile.flush()
                self.connection.shutdown(2)
                return
            self.wfile.write(payload)

    return Handler


@pytest.fixture
def release(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Release]:
    """The tiny classifier published on a local server as distilbert-sst2."""
    served = Release()
    files: dict[str, tuple[int, str]] = {}
    for path in sorted(TINY.iterdir()):
        data = path.read_bytes()
        served.assets[f"{MODEL}--{path.name}"] = data
        files[path.name] = (len(data), hashlib.sha256(data).hexdigest())
    monkeypatch.setitem(_files.FILES, MODEL, files)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(served))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(registry, "_RELEASES", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("NLP_SUITE_MODELS_DIR", str(tmp_path / "user-models"))
    monkeypatch.delenv("NLP_SUITE_MODELS", raising=False)
    monkeypatch.setattr(locate, "bundled_models_dir", lambda: tmp_path / "bundled")
    yield served
    server.shutdown()
    server.server_close()


def spec() -> registry.ModelSpec:
    found = registry.get_model(MODEL)
    assert found is not None
    return found


class TestDownload:
    def test_fetches_verifies_and_is_then_ready(self, release: Release) -> None:
        seen: list[tuple[int, int]] = []
        result = download.download(spec(), progress=lambda done, total: seen.append((done, total)))
        assert result.ok, result.diagnostics
        assert locate.status(spec()) == "ready"
        assert seen[-1][0] == seen[-1][1] == spec().size_bytes
        assert not list(result.unwrap().glob("*.partial"))

    def test_second_run_downloads_nothing(self, release: Release) -> None:
        assert download.download(spec()).ok
        release.assets.clear()  # anything fetched now would 404
        assert download.download(spec()).ok

    def test_a_dropped_connection_resumes_where_it_stopped(self, release: Release) -> None:
        release.drop_after = 50_000
        first = download.download(spec())
        assert first.diagnostics[0].code == "MODEL_DOWNLOAD_OFFLINE"
        folder = locate.user_models_dir() / MODEL
        partials = list(folder.glob("*.partial"))
        assert len(partials) == 1 and partials[0].stat().st_size == 50_000
        second = download.download(spec())
        assert second.ok, second.diagnostics
        assert locate.status(spec()) == "ready"
        # The second attempt asked only for the rest of the file.
        assert release.ranges == ["bytes=50000-"]

    def test_a_server_that_ignores_range_restarts_the_file(self, release: Release) -> None:
        release.drop_after = 50_000
        download.download(spec())
        release.ignore_range = True
        result = download.download(spec())
        assert result.ok, result.diagnostics
        assert locate.status(spec()) == "ready"

    def test_damaged_bytes_are_deleted_not_installed(self, release: Release) -> None:
        release.damaged.add(f"{MODEL}--model.onnx")
        result = download.download(spec())
        assert result.diagnostics[0].code == "MODEL_DOWNLOAD_CORRUPT"
        folder = locate.user_models_dir() / MODEL
        assert not (folder / "model.onnx").exists()
        assert not (folder / "model.onnx.partial").exists()
        release.damaged.clear()
        assert download.download(spec()).ok

    def test_cancel_keeps_the_partial_for_later(self, release: Release) -> None:
        cancel = threading.Event()
        cancel.set()
        result = download.download(spec(), cancel=cancel)
        assert result.diagnostics[0].code == "MODEL_DOWNLOAD_CANCELLED"
        assert "picks up where it stopped" in result.diagnostics[0].message
        assert locate.status(spec()) != "ready"

    def test_not_enough_disk_is_one_sentence(self, release: Release, monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil

        real = shutil.disk_usage
        monkeypatch.setattr(
            download.shutil,
            "disk_usage",
            lambda path: real(path)._replace(free=10),  # type: ignore[attr-defined]
        )
        result = download.download(spec())
        assert result.diagnostics[0].code == "MODEL_DOWNLOAD_DISK"
        assert "GB free" in result.diagnostics[0].message

    def test_offline_is_one_sentence(self, release: Release, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(registry, "_RELEASES", "http://127.0.0.1:9")
        result = download.download(spec())
        diag = result.diagnostics[0]
        assert diag.code == "MODEL_DOWNLOAD_OFFLINE"
        assert "internet connection" in diag.message and "Traceback" not in diag.message

    def test_a_missing_asset_says_so(self, release: Release) -> None:
        release.assets.pop(f"{MODEL}--tokenizer.json")
        assert download.download(spec()).diagnostics[0].code == "MODEL_DOWNLOAD_MISSING"

    def test_delete_frees_the_folder(self, release: Release) -> None:
        assert download.download(spec()).ok
        freed = download.delete(spec())
        assert freed.ok and freed.unwrap() == spec().size_bytes
        assert locate.status(spec()) == "not_downloaded"
        assert download.delete(spec()).unwrap() == 0

    def test_unpublished_model_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(_files.FILES, "qwen3-embedding-0.6b", raising=False)
        qwen = registry.get_model("qwen3-embedding-0.6b")
        assert qwen is not None
        assert download.download(qwen).diagnostics[0].code == "MODEL_UNPUBLISHED"


class TestModelsApi:
    @pytest.fixture
    def client(self, release: Release, tmp_path: Path) -> Iterator[Any]:
        from fastapi.testclient import TestClient

        from desktop_backend.server import create_app
        from desktop_backend.store import Workspace

        with TestClient(create_app(Workspace(tmp_path / "workspace"), "test-local-token")) as session:
            session.headers["Authorization"] = "Bearer test-local-token"
            yield session

    def listed(self, client: Any) -> dict[str, Any]:
        response = client.get("/api/models")
        assert response.status_code == 200
        return {row["id"]: row for row in response.json()["models"]}

    def test_listing_names_every_model_and_its_state(self, client: Any) -> None:
        rows = self.listed(client)
        assert set(rows) == {spec.id for spec in registry.MODELS}
        row = rows[MODEL]
        assert row["status"] == "not_downloaded"
        assert row["kindLabel"] == "Sentiment"
        assert row["license"] == "Apache-2.0"
        assert "sentiment_neural_bert" in row["usedBy"]
        assert row["removable"] is False

    def test_download_then_delete(self, client: Any) -> None:
        started = client.post(f"/api/models/{MODEL}/download")
        assert started.status_code == 200
        client.app.state.downloads.wait(MODEL, timeout=30)
        row = self.listed(client)[MODEL]
        assert row["status"] == "ready" and row["removable"] is True
        # Tools that run it are now available rather than waiting on a model.
        tools = {tool["name"]: tool for tool in client.get("/api/tools").json()}
        if "sentiment_neural_bert" in tools:
            assert tools["sentiment_neural_bert"]["availability"]["state"] != "needs_model"
        removed = client.delete(f"/api/models/{MODEL}")
        assert removed.status_code == 200
        assert removed.json()["status"] == "not_downloaded"

    def test_failed_download_reports_its_reason(self, client: Any, release: Release) -> None:
        release.assets.clear()
        client.post(f"/api/models/{MODEL}/download")
        client.app.state.downloads.wait(MODEL, timeout=30)
        row = self.listed(client)[MODEL]
        assert row["download"]["state"] == "failed"
        assert "not on the download server" in row["download"]["message"]

    def test_model_choices_are_named_by_what_they_do(self, client: Any) -> None:
        tools = {tool["name"]: tool for tool in client.get("/api/tools").json()}
        model = next(p for p in tools["bert_topics"]["params"] if p["name"] == "model")
        labels = model["choice_labels"]
        assert labels["bert-base-uncased"].startswith("BERT base (uncased): word vectors")
        assert (
            labels["granite-embedding-english-r2"].endswith("sentence and document vectors")
            or "not installed" in (labels["granite-embedding-english-r2"])
        )
        sentiment = next(p for p in tools["sentiment_neural_bert"]["params"] if p["name"] == "model")
        assert "sentiment" in sentiment["choice_labels"]["distilbert-sst2"]

    def test_unknown_model_is_404(self, client: Any) -> None:
        assert client.post("/api/models/no-such-model/download").status_code == 404
