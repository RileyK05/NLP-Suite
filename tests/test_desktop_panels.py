"""Panels reaching the desktop: offered where they fit, drawn from the run.

Two properties this file exists to hold.

**A panel is offered by the tool that produced the run**, not from a
catalogue the reader has to pick through. A keyness run offers the volcano;
a readability run offers nothing, and says nothing rather than listing
figures that would fail when clicked.

**The caption's facts come from the envelope, not the request.** The source
name, its hash and the analysis settings are read from the run being drawn
over, so a caller cannot talk a figure into claiming provenance it does not
have.

Failures travel as ``ok: False`` plus diagnostics (R-C7) rather than HTTP
errors, so the interface has one shape to render.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from desktop_backend.server import create_app
from desktop_backend.store import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "workspace")


@pytest.fixture
def client(workspace: Workspace) -> Any:
    with TestClient(create_app(workspace, "test-local-token")) as session:
        session.headers["Authorization"] = "Bearer test-local-token"
        yield session


_KEYNESS_ROWS = "\n".join(
    [
        "Word,Freq Group A (pattern docs),Freq Group B (other docs),G2 (log-likelihood),p-value,Log Ratio,Overrepresented in",
        "freedom,120,12,88.4,0.0,2.31,Group A",
        "the,9000,8700,41.2,0.0,0.04,Group A",
        "tariff,1,19,2.1,0.14,-3.20,Group B",
    ]
)


def finished_run(
    workspace: Workspace, tool: str, tables: dict[str, str], params: dict[str, Any] | None = None
) -> tuple[str, str, Path]:
    """A DONE job for ``tool`` with a real run directory and envelope on disk.

    Tables are written in the order given, because order is exactly what the
    first version of this layer got wrong: it took the first CSV of a run, and
    the first CSV an LDA run writes is the one table no LDA panel draws.

    Built directly rather than by running the tool: this file is about the
    route from a finished run to a figure, and parsing a corpus to get there
    would test spaCy instead.
    """
    from core.io.writer import OutputWriter
    from desktop_backend.store import now

    project = workspace.create(f"Panels {tool}")
    # The run has to live inside the project: ``Workspace.artifacts`` resolves
    # a job's run_dir against the project directory and refuses anything that
    # escapes it, which is also why the row below stores a relative name.
    root = workspace.project_dir(project["id"]).resolve()
    writer = OutputWriter(root / "runs", tool=tool, params=params or {}, inputs=[])
    for name, text in tables.items():
        writer.write_text(text, name, kind="table", description=name)
    envelope = writer.finalize()
    assert envelope.ok, [str(d) for d in envelope.diagnostics]
    run_dir = writer.run_dir.resolve()
    relative = run_dir.relative_to(root).as_posix()

    job_id = (tool.replace("_", "") + "0" * 33)[:33]
    with workspace.connect() as db:
        db.execute(
            "INSERT INTO jobs (id, project_id, tool, state, stage, created, finished, run_dir, diagnostics, request) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (job_id, project["id"], tool, "DONE", "Completed", now(), now(), relative, "[]", "{}"),
        )
    return project["id"], job_id, run_dir


def finished_keyness_run(workspace: Workspace) -> tuple[str, str, Path]:
    return finished_run(
        workspace, "keyness", {"keyness.csv": _KEYNESS_ROWS}, {"group-pattern": "^19", "field": "Lemma"}
    )


# The four tables `_adapt_lda_gensim` writes, in the order it writes them
# (core/profiler/executor.py). topics.csv comes first and no panel reads it.
_LDA_TABLES: dict[str, str] = {
    "topics.csv": "\n".join(["Topic,Word,Weight", "0,war,0.05", "0,peace,0.04", "1,tax,0.06", "1,job,0.05"]),
    "topics_dominant.csv": "\n".join(
        [
            "Document ID,Document,Dominant topic,Contribution,Topic keywords",
            'd1,1934-01-03_franklin d roosevelt_sotu.txt,0,0.8,"war, peace"',
            'd2,1936-01-03_franklin d roosevelt_sotu.txt,1,0.6,"tax, job"',
            'd3,1944-01-11_franklin d roosevelt_sotu.txt,0,0.7,"war, peace"',
            'd4,1946-01-21_harry s truman_sotu.txt,1,0.5,"tax, job"',
        ]
    ),
    "terms_by_relevance.csv": "\n".join(
        [
            "Topic,Word,Relevance,Saliency,Corpus frequency,Topic frequency",
            "0,war,-4.5,-0.05,40,36.5",
            "0,peace,-4.7,-0.04,25,21.0",
            "1,tax,-4.4,-0.06,30,27.2",
            "1,job,-4.6,-0.03,22,18.9",
        ]
    ),
    "intertopic_distances.csv": "\n".join(["Topic,X,Y,Prevalence", "0,-0.2,0.1,1.0", "1,0.2,-0.1,0.6"]),
}


class TestCatalogue:
    def test_the_app_is_told_what_panels_exist_and_what_they_take(self, client: TestClient) -> None:
        """Shipped rather than restated in TypeScript: a second copy across a
        language boundary drifts without anything failing."""
        response = client.get("/api/panels")
        assert response.status_code == 200
        panels = {panel["name"]: panel for panel in response.json()}
        assert "keyness_volcano" in panels
        volcano = panels["keyness_volcano"]
        assert volcano["tool"] == "keyness"
        assert volcano["notes"], "a panel must ship what it cannot show"
        assert {param["name"] for param in volcano["params"]} >= {"label-top", "significance"}
        for panel in panels.values():
            for param in panel["params"]:
                assert param["help"], f"{panel['name']}.{param['name']} has no help"

    def test_a_run_is_offered_the_panels_that_fit_it(self, client: TestClient, workspace: Workspace) -> None:
        project_id, job_id, _ = finished_keyness_run(workspace)
        response = client.get(f"/api/projects/{project_id}/jobs/{job_id}/panels")
        assert response.status_code == 200
        assert [panel["name"] for panel in response.json()] == ["keyness_volcano"]

    def test_saved_embedding_vectors_offer_the_query_panel(self, client: TestClient, workspace: Workspace) -> None:
        project_id, job_id, _ = finished_run(
            workspace,
            "word2vec_gensim",
            {"vectors.csv": 'Word,Count,Vector\ngovernment,4,"1,0"\nstate,3,"0.9,0.1"\n'},
        )
        response = client.get(f"/api/projects/{project_id}/jobs/{job_id}/panels")
        assert response.status_code == 200
        names = [panel["name"] for panel in response.json()]
        assert "word2vec_gensim_saved_vectors" in names
        assert "word2vec_gensim_neighbours" not in names, "precomputed neighbours need their own optional artifact"


class TestDrawing:
    def test_a_panel_comes_back_as_marks_the_app_can_draw_and_click(
        self, client: TestClient, workspace: Workspace
    ) -> None:
        project_id, job_id, _ = finished_keyness_run(workspace)
        response = client.post(
            f"/api/projects/{project_id}/jobs/{job_id}/panels",
            json={"panel": "keyness_volcano", "params": {"label-top": 2}},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"], body
        assert body["shape"] == "scatter_labelled"
        assert {mark["label"] for mark in body["marks"]} == {"freedom", "the", "tariff"}
        assert body["groups"] == ["Group A", "Group B"]
        assert body["notes"], "the limits travel with the figure"
        assert [a["kind"] for a in body["annotations"]] == ["vline", "hline"]

    def test_every_mark_carries_evidence_a_click_can_answer(self, client: TestClient, workspace: Workspace) -> None:
        """Answerable without a second round trip -- the evidence crosses with
        the mark, so pointing at a point can say what it stands for."""
        project_id, job_id, _ = finished_keyness_run(workspace)
        body = client.post(f"/api/projects/{project_id}/jobs/{job_id}/panels", json={"panel": "keyness_volcano"}).json()
        for mark in body["marks"]:
            evidence = mark["evidence"]
            assert evidence["describe"]
            assert evidence["filters"], "a mark with no filters cannot be resolved"
            assert evidence["filters"][0][0] == "Word"

    def test_the_caption_is_built_from_the_run_not_the_request(self, client: TestClient, workspace: Workspace) -> None:
        project_id, job_id, run_dir = finished_keyness_run(workspace)
        body = client.post(f"/api/projects/{project_id}/jobs/{job_id}/panels", json={"panel": "keyness_volcano"}).json()
        digest = hashlib.sha256((run_dir / "keyness.csv").read_bytes()).hexdigest()
        assert "keyness.csv" in body["caption"]
        assert digest[:12] in body["caption"], "the caption must carry the run's own hash"
        assert "group-pattern=^19" in body["caption"], "the settings come from the envelope"

    def test_the_table_beside_the_figure_travels_with_it(self, client: TestClient, workspace: Workspace) -> None:
        project_id, job_id, _ = finished_keyness_run(workspace)
        body = client.post(f"/api/projects/{project_id}/jobs/{job_id}/panels", json={"panel": "keyness_volcano"}).json()
        assert body["table"]["total"] == 3
        assert not body["table"]["truncated"]
        assert "Word" in body["table"]["columns"]

    def test_saved_vectors_draw_neighbours_from_the_selected_query(
        self, client: TestClient, workspace: Workspace
    ) -> None:
        project_id, job_id, _ = finished_run(
            workspace,
            "word2vec_gensim",
            {"vectors.csv": ('Word,Count,Vector\ngovernment,4,"1,0"\nstate,3,"0.9,0.1"\nnation,2,"0,1"\n')},
        )
        body = client.post(
            f"/api/projects/{project_id}/jobs/{job_id}/panels",
            json={"panel": "word2vec_gensim_saved_vectors", "params": {"query": "government"}},
        ).json()

        assert body["ok"], body
        assert [mark["label"] for mark in body["marks"]] == ["state", "nation"]
        assert body["table"]["columns"] == ["Word", "Neighbor", "Cosine"]
        assert body["marks"][0]["evidence"]["phrase"] == "state"
        assert body["marks"][0]["evidence"]["filters"] == [["Word", "government"], ["Neighbor", "state"]]

    def test_mallet_topic_terms_are_offered_and_preserve_rank(self, client: TestClient, workspace: Workspace) -> None:
        project_id, job_id, _ = finished_run(
            workspace,
            "lda_mallet",
            {
                "topics.csv": "Topic,Weight,Words\n0,44,war peace defense\n1,18,jobs wages labor\n",
                "topics_dominant.csv": (
                    'Document,Dominant topic,Contribution,Topic proportions\naddress.txt,0,0.8,"0:0.8, 1:0.2"\n'
                ),
            },
        )
        offered = client.get(f"/api/projects/{project_id}/jobs/{job_id}/panels").json()
        assert "mallet_topic_terms" in [panel["name"] for panel in offered]
        body = client.post(
            f"/api/projects/{project_id}/jobs/{job_id}/panels",
            json={"panel": "mallet_topic_terms", "params": {"topic": 1, "top-n": 2}},
        ).json()
        assert body["ok"], body
        assert body["shape"] == "positions"
        assert [mark["label"] for mark in body["marks"]] == ["jobs", "wages"]
        assert body["table"]["columns"] == ["Topic", "Rank", "Word", "Words"]


class TestRefusals:
    def test_an_unknown_panel_is_a_diagnostic_not_an_http_error(self, client: TestClient, workspace: Workspace) -> None:
        project_id, job_id, _ = finished_keyness_run(workspace)
        response = client.post(f"/api/projects/{project_id}/jobs/{job_id}/panels", json={"panel": "volcano"})
        assert response.status_code == 200, "the interface has one shape to render (R-C7)"
        body = response.json()
        assert not body["ok"]
        assert body["diagnostics"][0]["code"] == "PANEL_UNKNOWN"

    def test_a_panel_for_another_tool_is_refused_by_name(self, client: TestClient, workspace: Workspace) -> None:
        """Offering every panel over every run is how a figure that cannot
        work becomes a button that does nothing."""
        project_id, job_id, _ = finished_keyness_run(workspace)
        body = client.post(f"/api/projects/{project_id}/jobs/{job_id}/panels", json={"panel": "lda_relevance"}).json()
        assert not body["ok"]
        assert body["diagnostics"][0]["code"] == "PANEL_WRONG_TOOL"
        assert "lda_gensim" in body["diagnostics"][0]["message"]

    def test_a_bad_parameter_is_refused_with_the_panels_own_vocabulary(
        self, client: TestClient, workspace: Workspace
    ) -> None:
        project_id, job_id, _ = finished_keyness_run(workspace)
        body = client.post(
            f"/api/projects/{project_id}/jobs/{job_id}/panels",
            json={"panel": "keyness_volcano", "params": {"significance": "0.5"}},
        ).json()
        assert not body["ok"]
        assert body["diagnostics"][0]["code"] == "PANEL_BAD_PARAM"

    def test_an_unknown_field_in_the_request_is_a_caller_bug(self, client: TestClient, workspace: Workspace) -> None:
        project_id, job_id, _ = finished_keyness_run(workspace)
        response = client.post(
            f"/api/projects/{project_id}/jobs/{job_id}/panels",
            json={"panel": "keyness_volcano", "parms": {}},
        )
        assert response.status_code == 422, "extra='forbid' keeps the contract honest"


class TestMultiTableRuns:
    """A run that writes several tables: each panel must find its own."""

    def test_an_lda_run_is_offered_every_panel_it_can_draw(self, client: TestClient, workspace: Workspace) -> None:
        project_id, job_id, _ = finished_run(workspace, "lda_gensim", _LDA_TABLES)
        offered = [p["name"] for p in client.get(f"/api/projects/{project_id}/jobs/{job_id}/panels").json()]
        assert sorted(offered) == ["lda_intertopic", "lda_prevalence", "lda_relevance"]

    @pytest.mark.parametrize(
        ("panel", "table"),
        [
            ("lda_relevance", "terms_by_relevance.csv"),
            ("lda_intertopic", "intertopic_distances.csv"),
            ("lda_prevalence", "topics_dominant.csv"),
        ],
    )
    def test_each_panel_draws_from_the_table_that_carries_its_columns(
        self, client: TestClient, workspace: Workspace, panel: str, table: str
    ) -> None:
        """Not the first CSV: the first one an LDA run writes is topics.csv,
        which none of these panels reads."""
        project_id, job_id, _ = finished_run(workspace, "lda_gensim", _LDA_TABLES)
        body = client.post(f"/api/projects/{project_id}/jobs/{job_id}/panels", json={"panel": panel}).json()
        assert body["ok"], body
        assert table in body["caption"], f"{panel} drew from the wrong table: {body['caption']}"
        assert body["marks"]

    def test_a_panel_whose_table_is_missing_is_not_offered(self, client: TestClient, workspace: Workspace) -> None:
        """A partial run, or one from an older tool version, must not offer a
        figure that fails with a missing column the moment it is clicked."""
        partial = {name: text for name, text in _LDA_TABLES.items() if name != "intertopic_distances.csv"}
        project_id, job_id, _ = finished_run(workspace, "lda_gensim", partial)
        offered = [p["name"] for p in client.get(f"/api/projects/{project_id}/jobs/{job_id}/panels").json()]
        assert "lda_intertopic" not in offered
        assert "lda_relevance" in offered

    def test_drawing_a_panel_with_no_table_names_the_columns_it_needed(
        self, client: TestClient, workspace: Workspace
    ) -> None:
        partial = {name: text for name, text in _LDA_TABLES.items() if name != "intertopic_distances.csv"}
        project_id, job_id, _ = finished_run(workspace, "lda_gensim", partial)
        body = client.post(f"/api/projects/{project_id}/jobs/{job_id}/panels", json={"panel": "lda_intertopic"}).json()
        assert not body["ok"]
        assert body["diagnostics"][0]["code"] == "PANEL_NO_TABLE"
        assert "Prevalence" in body["diagnostics"][0]["message"]


class TestPublicationFigure:
    """The same panel, drawn by matplotlib and seaborn, served as an image."""

    @pytest.mark.parametrize(
        ("fmt", "media", "magic"),
        [("png", "image/png", b"\x89PNG"), ("svg", "image/svg+xml", b"<?xml"), ("pdf", "application/pdf", b"%PDF")],
    )
    def test_a_run_panel_comes_back_as_an_image(
        self, client: TestClient, workspace: Workspace, fmt: str, media: str, magic: bytes
    ) -> None:
        pytest.importorskip("seaborn")
        project_id, job_id, _ = finished_keyness_run(workspace)
        response = client.post(
            f"/api/projects/{project_id}/jobs/{job_id}/panels/static",
            json={"panel": "keyness_volcano", "params": {"label-top": 2}, "format": fmt, "dpi": 72},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(media)
        assert response.content[: len(magic)] == magic
        assert response.headers["x-figure-warnings"] == "0"

    def test_a_refusal_is_the_panel_refusal_shape_not_an_image(self, client: TestClient, workspace: Workspace) -> None:
        project_id, job_id, _ = finished_keyness_run(workspace)
        response = client.post(
            f"/api/projects/{project_id}/jobs/{job_id}/panels/static",
            json={"panel": "not_a_panel", "params": {}, "format": "png"},
        )
        assert response.status_code == 200
        assert response.json()["ok"] is False
        assert response.json()["diagnostics"][0]["code"] == "PANEL_UNKNOWN"

    def test_an_unknown_format_is_a_caller_bug(self, client: TestClient, workspace: Workspace) -> None:
        project_id, job_id, _ = finished_keyness_run(workspace)
        response = client.post(
            f"/api/projects/{project_id}/jobs/{job_id}/panels/static",
            json={"panel": "keyness_volcano", "params": {}, "format": "gif"},
        )
        assert response.status_code == 422


def test_a_panel_takes_the_table_written_for_it_not_the_first_that_fits(workspace: Workspace) -> None:
    """Regression: LDA's paragraph-level topic flow carries the same Document
    and Dominant topic columns as topics_dominant.csv. Taken first, it made
    the prevalence figure count passages as speeches."""
    from core.viz.panels import get_panel
    from desktop_backend.panels import _table_carrying

    flow = [
        "Document ID",
        "Document",
        "Segment",
        "Segments",
        "Tokens",
        "Dominant topic",
        "Contribution",
        "Topic keywords",
        "Start",
        "End",
    ]
    dominant = ["Document ID", "Document", "Dominant topic", "Contribution", "Topic keywords"]
    chosen = _table_carrying([("topic_flow.csv", flow), ("topics_dominant.csv", dominant)], get_panel("lda_prevalence"))
    assert chosen == "topics_dominant.csv"


class TestBundles:
    """Publication-only figures over a finished run, listed and drawn."""

    def run(self, workspace: Workspace) -> tuple[str, str]:
        table = pd.read_parquet(Path(__file__).parent / "fixtures" / "figures" / "readability__readability.parquet")
        project_id, job_id, _ = finished_run(workspace, "readability", {"readability.csv": table.to_csv(index=False)})
        return project_id, job_id

    def test_a_run_lists_the_bundles_its_tables_can_feed(self, client: TestClient, workspace: Workspace) -> None:
        pytest.importorskip("seaborn")
        project_id, job_id = self.run(workspace)
        names = [b["name"] for b in client.get(f"/api/projects/{project_id}/jobs/{job_id}/bundles").json()]
        assert names == ["readability_measure_correlations", "readability_against_length"]

    def test_a_bundle_comes_back_as_an_image(self, client: TestClient, workspace: Workspace) -> None:
        pytest.importorskip("seaborn")
        project_id, job_id = self.run(workspace)
        response = client.post(
            f"/api/projects/{project_id}/jobs/{job_id}/bundles",
            json={"bundle": "readability_measure_correlations", "format": "svg"},
        )
        assert response.headers["content-type"].startswith("image/svg+xml")
        assert b"readability.csv" in response.content, "captioned from the run's own table"

    def test_another_tools_bundle_is_refused(self, client: TestClient, workspace: Workspace) -> None:
        project_id, job_id = self.run(workspace)
        response = client.post(
            f"/api/projects/{project_id}/jobs/{job_id}/bundles", json={"bundle": "doc_similarity_map"}
        )
        assert response.json()["ok"] is False
        assert response.json()["diagnostics"][0]["code"] == "BUNDLE_UNKNOWN"
