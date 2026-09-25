"""The live specialist view reads the exact frame returned by its engine."""

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pandas as pd

from core.analysis.ngram_viewer import ngram_series
from desktop_backend import server
from desktop_backend.live import LiveResult
from desktop_backend.live_panels import draw_live_panel, live_panel, live_panels_offered
from desktop_backend.server import create_app
from desktop_backend.store import Workspace


def test_live_culturomics_opens_annual_lines_with_unsaved_provenance() -> None:
    series = ngram_series(
        {"a": ["war"], "b": ["peace"], "c": ["war"]},
        {"a": 1900, "b": 1901, "c": 1903},
        ["war"],
    ).unwrap()
    result = LiveResult(tool="ngram_viewer", ok=True, frames={"ngram_series.csv": series})
    answer = live_panel(result, {"queries": "war", "smooth": 1})
    assert answer is not None and answer["ok"]
    assert answer["shape"] == "line_series"
    assert [mark["x"] for mark in answer["marks"]] == [1900.0, 1901.0, 1903.0]
    assert [mark["y"] for mark in answer["marks"]] == [1000000.0, 0.0, 1000000.0]
    assert "unsaved live ngram_series.csv" in answer["caption"]


def test_live_panel_absent_for_unmatched_or_failed_answer() -> None:
    unmatched = LiveResult(tool="ngram_viewer", ok=True, frames={"old.csv": pd.DataFrame({"A": [1]})})
    assert live_panel(unmatched, {}) is None
    failed = LiveResult(tool="ngram_viewer", ok=False, frames={"ngram_series.csv": pd.DataFrame()})
    assert live_panel(failed, {}) is None


def test_live_embedding_result_exposes_saved_vector_query_without_retraining() -> None:
    vectors = pd.DataFrame(
        [
            {"Word": "government", "Count": 12, "Vector": "1,0"},
            {"Word": "state", "Count": 10, "Vector": "0.9,0.1"},
            {"Word": "nation", "Count": 7, "Vector": "0,1"},
        ]
    )
    result = LiveResult(tool="word2vec_gensim", ok=True, frames={"vectors.csv": vectors})
    settings = {"field": "lemma", "vector-size": 2, "seed": 42}

    offered = live_panels_offered(result)
    assert "word2vec_gensim_saved_vectors" in [panel["name"] for panel in offered]
    # Three words are too few for meaning groups (the first figure) or a
    # bad-good axis; the first figure that can read them is drawn instead.
    default = live_panel(result, settings)
    assert default is not None and default["ok"]
    assert default["title"] == "The neighbourhood of government"
    queried = draw_live_panel(result, settings, "word2vec_gensim_saved_vectors", {"query": "state"})
    assert queried["ok"]
    assert next(mark["label"] for mark in queried["marks"]) == "government"


def test_live_analyse_route_returns_the_prepared_panel(tmp_path: Path, monkeypatch) -> None:
    series = ngram_series({"a": ["war"], "b": ["peace"]}, {"a": 1900, "b": 1901}, ["war"]).unwrap()
    result = LiveResult(tool="ngram_viewer", ok=True, frames={"ngram_series.csv": series})

    class ReadySession:
        def analysis_cancelled(self, request_id: str) -> bool:
            return False

        def resolved(self, snapshot_id: str | None = None):
            return SimpleNamespace(id="snapshot-1")

        def analyse(self, tool: str, params: dict[str, object]) -> LiveResult:
            return result

        def remember(self, result: LiveResult, settings: dict[str, object], snapshot_id: str) -> str:
            return "answer-1"

        def answer(self, answer_id: str):
            return result, {}

    monkeypatch.setattr(server, "Session", lambda bench: ReadySession())
    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create("Live")
    with TestClient(create_app(workspace, "token")) as client:
        response = client.post(
            f"/api/projects/{project['id']}/live/analyse",
            headers={"Authorization": "Bearer token"},
            json={"request_id": "test-request", "tool": "ngram_viewer", "params": {"queries": "war"}},
        )
    assert response.status_code == 200, response.text
    panel = response.json()["panel"]
    assert panel["ok"] and panel["shape"] == "line_series"
    assert [mark["x"] for mark in panel["marks"]] == [1900.0, 1901.0]
