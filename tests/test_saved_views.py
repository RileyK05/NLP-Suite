"""Saved views: a way of looking at a result that survives closing the app.

The workbench draws instantly and forgets everything. These tests pin the
part that does not forget -- and, just as much, the part that refuses to
pretend: a view whose source has been replaced says so, a view whose run has
gone says so, and publishing never silently produces a chart arranged
differently from the one on screen without naming the difference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from desktop_backend.runner import Runner, run_job
from desktop_backend.server import create_app
from desktop_backend.store import Workspace, free_name
from desktop_backend.views import MAX_VIEWS_PER_PROJECT, ViewSettings, chart_contract, publication_gaps

DATASET = "word,freq1,freq2\nfreedom,120,10\nliberty,95,40\nnation,80,30\nfuture,60,55\n"

SETTINGS: dict[str, Any] = {
    "kind": "bar",
    "x": "word",
    "y": "freq1",
    "group": "",
    "agg": "sum",
    "top_n": 25,
    "sort": "high",
    "bins": 12,
}


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "workspace")


@pytest.fixture
def client(workspace: Workspace):
    with TestClient(create_app(workspace, "test-local-token")) as session:
        session.headers["Authorization"] = "Bearer test-local-token"
        yield session


def finished_run(workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> dict[str, Any]:
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


def a_table(client: TestClient, project_id: str) -> dict[str, Any]:
    """The first real result table in a project -- not its input manifest."""
    tables = client.get(f"/api/projects/{project_id}/tables").json()
    return next(table for table in tables if not table["manifest"])


def body(table: dict[str, Any], name: str = "Freedom by word", **over: Any) -> dict[str, Any]:
    return {
        "name": name,
        "job": table["job"],
        "index": table["index"],
        "settings": {**SETTINGS, **over.pop("settings", {})},
        "filters": over.pop("filters", []),
        **over,
    }


@pytest.fixture
def project(
    client: TestClient, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    return finished_run(workspace, tmp_path, monkeypatch, "keyness")


# ------------------------------------------------------------ saving them --


def test_a_new_project_has_no_views_rather_than_an_error(client: TestClient, workspace: Workspace) -> None:
    empty = workspace.create("Nothing saved yet")
    response = client.get(f"/api/projects/{empty['id']}/views")
    assert response.status_code == 200
    assert response.json() == []


def test_a_saved_view_comes_back_with_its_settings_intact(client: TestClient, project: dict[str, Any]) -> None:
    table = a_table(client, project["id"])
    saved = client.post(f"/api/projects/{project['id']}/views", json=body(table))
    assert saved.status_code == 200, saved.text
    view = saved.json()
    assert view["settings"] == SETTINGS
    assert view["name"] == "Freedom by word"
    assert view["revision"] == 1
    assert view["source"]["state"] == "ok"

    listed = client.get(f"/api/projects/{project['id']}/views").json()
    assert [v["id"] for v in listed] == [view["id"]]
    assert listed[0]["settings"] == SETTINGS


def test_saving_a_view_runs_nothing(client: TestClient, project: dict[str, Any]) -> None:
    """A view is a draft. If saving one could start a job, editing would cost
    a run and nobody would edit."""
    before = client.get(f"/api/projects/{project['id']}/jobs").json()
    table = a_table(client, project["id"])
    assert client.post(f"/api/projects/{project['id']}/views", json=body(table)).status_code == 200
    after = client.get(f"/api/projects/{project['id']}/jobs").json()
    assert [j["id"] for j in after] == [j["id"] for j in before]


def test_a_view_records_the_file_it_was_saved_over(client: TestClient, project: dict[str, Any]) -> None:
    table = a_table(client, project["id"])
    view = client.post(f"/api/projects/{project['id']}/views", json=body(table)).json()
    assert len(view["source_sha256"]) == 64
    assert view["artifact_path"] == table["path"]


def test_two_views_of_the_same_table_are_kept_apart(client: TestClient, project: dict[str, Any]) -> None:
    """The acceptance case: two independently edited views over one artifact.

    Identical columns is exactly when a shared draft would be invisible --
    both would open, both would look plausible, and one would be wrong.
    """
    table = a_table(client, project["id"])
    first = client.post(
        f"/api/projects/{project['id']}/views",
        json=body(table, "By freq1", settings={"y": "freq1", "sort": "high"}),
    ).json()
    second = client.post(
        f"/api/projects/{project['id']}/views",
        json=body(table, "By freq2", settings={"y": "freq2", "sort": "low"}),
    ).json()

    reopened = {v["name"]: v for v in client.get(f"/api/projects/{project['id']}/views").json()}
    assert reopened["By freq1"]["settings"]["y"] == "freq1"
    assert reopened["By freq1"]["settings"]["sort"] == "high"
    assert reopened["By freq2"]["settings"]["y"] == "freq2"
    assert reopened["By freq2"]["settings"]["sort"] == "low"
    assert first["id"] != second["id"]
    assert first["job"] == second["job"] and first["index"] == second["index"]


def test_editing_one_view_leaves_the_other_alone(client: TestClient, project: dict[str, Any]) -> None:
    table = a_table(client, project["id"])
    first = client.post(f"/api/projects/{project['id']}/views", json=body(table, "First")).json()
    second = client.post(f"/api/projects/{project['id']}/views", json=body(table, "Second")).json()

    client.post(
        f"/api/projects/{project['id']}/views/{first['id']}",
        json=body(table, "First", settings={"kind": "line", "y": "freq2"}),
    )
    assert client.get(f"/api/projects/{project['id']}/views/{second['id']}").json()["settings"] == SETTINGS


def test_editing_counts_a_revision(client: TestClient, project: dict[str, Any]) -> None:
    """A published chart's provenance names the revision, so it has to move."""
    table = a_table(client, project["id"])
    view = client.post(f"/api/projects/{project['id']}/views", json=body(table)).json()
    assert view["revision"] == 1
    edited = client.post(
        f"/api/projects/{project['id']}/views/{view['id']}",
        json=body(table, settings={"kind": "line"}),
    ).json()
    assert edited["revision"] == 2
    assert edited["id"] == view["id"]


def test_a_drill_down_is_saved_with_the_view(client: TestClient, project: dict[str, Any]) -> None:
    table = a_table(client, project["id"])
    view = client.post(
        f"/api/projects/{project['id']}/views",
        json=body(table, filters=[{"column": "word", "equals": "freedom"}]),
    ).json()
    assert client.get(f"/api/projects/{project['id']}/views/{view['id']}").json()["filters"] == [
        {"column": "word", "equals": "freedom"}
    ]


# ------------------------------------------------------- refusing to guess --


def test_two_views_cannot_share_a_name(client: TestClient, project: dict[str, Any]) -> None:
    table = a_table(client, project["id"])
    client.post(f"/api/projects/{project['id']}/views", json=body(table, "Trend"))
    clash = client.post(f"/api/projects/{project['id']}/views", json=body(table, "Trend"))
    assert clash.status_code == 400
    assert "Trend" in clash.json()["detail"]


def test_an_unnamed_view_is_refused(client: TestClient, project: dict[str, Any]) -> None:
    table = a_table(client, project["id"])
    assert client.post(f"/api/projects/{project['id']}/views", json=body(table, "   ")).status_code == 422


def test_a_view_of_a_run_that_does_not_exist_is_refused(client: TestClient, project: dict[str, Any]) -> None:
    table = a_table(client, project["id"])
    missing = client.post(f"/api/projects/{project['id']}/views", json={**body(table), "job": "0" * 32})
    assert missing.status_code == 404


def test_a_view_of_a_non_csv_artifact_is_refused(
    client: TestClient, workspace: Workspace, project: dict[str, Any]
) -> None:
    """Saved from a CSV table, checked now rather than on reopening."""
    job_id = workspace.jobs(project["id"])[0]["id"]
    _, envelope = workspace.artifacts(project["id"], job_id)
    other = next(
        (i for i, a in enumerate(envelope.artifacts) if not a.path.lower().endswith(".csv")),
        None,
    )
    if other is None:
        pytest.skip("this run published only CSV artifacts")
    refused = client.post(
        f"/api/projects/{project['id']}/views",
        json=body({"job": job_id, "index": other}),
    )
    assert refused.status_code == 400


def test_an_unknown_chart_kind_is_refused_at_the_boundary(client: TestClient, project: dict[str, Any]) -> None:
    """The engine draws more kinds than the workbench does. A view is reopened
    *in* the workbench, so a kind it cannot draw would save and never open."""
    table = a_table(client, project["id"])
    refused = client.post(
        f"/api/projects/{project['id']}/views",
        json=body(table, settings={"kind": "sunburst"}),
    )
    assert refused.status_code == 422


def test_an_unknown_setting_is_refused_rather_than_stored(client: TestClient, project: dict[str, Any]) -> None:
    """Storing an unrecognised key would make a later reader believe it did
    something."""
    table = a_table(client, project["id"])
    payload = body(table)
    payload["settings"] = {**SETTINGS, "logY": True}
    assert client.post(f"/api/projects/{project['id']}/views", json=payload).status_code == 422


def test_a_column_name_with_edge_spaces_is_refused(client: TestClient, project: dict[str, Any]) -> None:
    table = a_table(client, project["id"])
    refused = client.post(f"/api/projects/{project['id']}/views", json=body(table, settings={"y": " freq1"}))
    assert refused.status_code == 422


# ---------------------------------------------------- duplicating, deleting --


def test_duplicating_leaves_the_original_editable(client: TestClient, project: dict[str, Any]) -> None:
    table = a_table(client, project["id"])
    view = client.post(f"/api/projects/{project['id']}/views", json=body(table, "Trend")).json()
    copy = client.post(f"/api/projects/{project['id']}/views/{view['id']}/duplicate").json()

    assert copy["name"] == "Trend (copy)"
    assert copy["id"] != view["id"]
    assert copy["settings"] == view["settings"]
    assert copy["revision"] == 1, "a copy is a new view, not another revision of the old one"

    client.post(
        f"/api/projects/{project['id']}/views/{copy['id']}",
        json=body(table, "Trend (copy)", settings={"kind": "line"}),
    )
    assert client.get(f"/api/projects/{project['id']}/views/{view['id']}").json()["settings"]["kind"] == "bar"


def test_duplicating_twice_finds_another_free_name(client: TestClient, project: dict[str, Any]) -> None:
    table = a_table(client, project["id"])
    view = client.post(f"/api/projects/{project['id']}/views", json=body(table, "Trend")).json()
    first = client.post(f"/api/projects/{project['id']}/views/{view['id']}/duplicate").json()
    second = client.post(f"/api/projects/{project['id']}/views/{view['id']}/duplicate").json()
    assert {first["name"], second["name"]} == {"Trend (copy)", "Trend (copy 2)"}


def test_a_copied_name_stays_within_the_length_limit() -> None:
    assert len(free_name("x" * 120, set())) <= 120


def test_deleting_a_view_removes_only_that_one(client: TestClient, project: dict[str, Any]) -> None:
    table = a_table(client, project["id"])
    doomed = client.post(f"/api/projects/{project['id']}/views", json=body(table, "Doomed")).json()
    kept = client.post(f"/api/projects/{project['id']}/views", json=body(table, "Kept")).json()
    assert client.post(f"/api/projects/{project['id']}/views/{doomed['id']}/delete").status_code == 200
    assert [v["id"] for v in client.get(f"/api/projects/{project['id']}/views").json()] == [kept["id"]]
    assert client.get(f"/api/projects/{project['id']}/views/{doomed['id']}").status_code == 404


def test_deleting_a_view_leaves_its_run_alone(
    client: TestClient, workspace: Workspace, project: dict[str, Any]
) -> None:
    """Views are drafts over results. Deleting a draft must not touch the result."""
    table = a_table(client, project["id"])
    view = client.post(f"/api/projects/{project['id']}/views", json=body(table)).json()
    client.post(f"/api/projects/{project['id']}/views/{view['id']}/delete")
    assert workspace.artifact(project["id"], table["job"], table["index"]).is_file()


def test_a_project_cannot_be_filled_with_unlimited_views(
    client: TestClient, project: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused at the limit, and refused in words rather than by a stack trace."""
    monkeypatch.setattr("desktop_backend.store.MAX_VIEWS_PER_PROJECT", 2)
    table = a_table(client, project["id"])
    for index in range(2):
        assert client.post(f"/api/projects/{project['id']}/views", json=body(table, f"View {index}")).status_code == 200
    refused = client.post(f"/api/projects/{project['id']}/views", json=body(table, "One too many"))
    assert refused.status_code == 400
    assert "limit" in refused.json()["detail"]
    assert MAX_VIEWS_PER_PROJECT >= 100, "the real limit must leave room for ordinary use"


# ------------------------------------------------------- telling the truth --


def test_a_view_whose_source_changed_says_so(client: TestClient, workspace: Workspace, project: dict[str, Any]) -> None:
    """Run directories are immutable, but restores and stray hands are not.

    A view that quietly charts different numbers than the ones it was saved
    over is worse than one that admits it has gone stale.
    """
    table = a_table(client, project["id"])
    view = client.post(f"/api/projects/{project['id']}/views", json=body(table)).json()
    assert view["source"]["state"] == "ok"

    path = workspace.artifact(project["id"], table["job"], table["index"])
    path.write_text(DATASET.replace("120", "121"), encoding="utf-8")

    reopened = client.get(f"/api/projects/{project['id']}/views/{view['id']}").json()
    assert reopened["source"]["state"] == "changed"
    assert table["path"] in reopened["source"]["detail"]
    assert reopened["settings"] == SETTINGS, "the settings survive; only the claim about the file changes"


def test_a_view_whose_table_is_gone_says_so(client: TestClient, workspace: Workspace, project: dict[str, Any]) -> None:
    table = a_table(client, project["id"])
    view = client.post(f"/api/projects/{project['id']}/views", json=body(table)).json()
    workspace.artifact(project["id"], table["job"], table["index"]).unlink()

    reopened = client.get(f"/api/projects/{project['id']}/views/{view['id']}").json()
    assert reopened["source"]["state"] == "missing"
    assert reopened["settings"] == SETTINGS


def test_one_dangling_view_does_not_empty_the_list(
    client: TestClient, workspace: Workspace, project: dict[str, Any]
) -> None:
    table = a_table(client, project["id"])
    doomed = client.post(f"/api/projects/{project['id']}/views", json=body(table, "Doomed")).json()
    client.post(f"/api/projects/{project['id']}/views", json=body(table, "Fine"))
    workspace.artifact(project["id"], table["job"], table["index"]).unlink()

    listed = client.get(f"/api/projects/{project['id']}/views").json()
    assert len(listed) == 2
    assert {v["id"] for v in listed} >= {doomed["id"]}


# --------------------------------------------- the preview/publish contract --


def test_the_ordering_the_engine_cannot_reproduce_is_named() -> None:
    """The default preview sorts largest-first; the engine sorts by category.

    Same rows, different sequence -- and until this existed, publishing said
    nothing about it at all.
    """
    gaps = publication_gaps(ViewSettings(**{**SETTINGS, "sort": "high"}))
    assert len(gaps) == 1
    assert "category order" in gaps[0]


def test_an_ordering_the_engine_does_reproduce_is_not_flagged() -> None:
    assert publication_gaps(ViewSettings(**{**SETTINGS, "sort": "table"})) == []


def test_a_kind_with_no_category_axis_is_not_flagged() -> None:
    """A scatter's x is a number line; "largest first" means nothing there, so
    there is no promise for publishing to break."""
    scatter = {**SETTINGS, "kind": "scatter", "x": "freq2", "sort": "high"}
    assert publication_gaps(ViewSettings(**scatter)) == []


def test_a_saved_view_carries_its_own_publication_gaps(client: TestClient, project: dict[str, Any]) -> None:
    table = a_table(client, project["id"])
    view = client.post(f"/api/projects/{project['id']}/views", json=body(table)).json()
    assert view["publication"] == publication_gaps(ViewSettings(**SETTINGS))
    assert view["publication"], "the default settings do diverge, so this assertion is not vacuous"


def test_the_contract_is_served_to_the_interface(client: TestClient) -> None:
    """Shipped rather than restated in TypeScript: the /api/tools lesson."""
    served = client.get("/api/chart-contract")
    assert served.status_code == 200
    assert served.json() == chart_contract()
    assert served.json()["bar"]["high"], "a bar chart sorted high does diverge"


def test_every_contract_note_is_a_sentence_not_a_code() -> None:
    for kind, notes in chart_contract().items():
        for sort, note in notes.items():
            assert note.endswith("."), (kind, sort)
            assert len(note.split()) > 8, "this is shown to a reader, not logged"


# ------------------------------------------------------- backup and restore --


def restored_copy(workspace: Workspace, project_id: str) -> str:
    from desktop_backend.archives import backup, restore

    archive = workspace.root / "views-backup.nlpsuite"
    backup(workspace, project_id, archive)
    return str(restore(workspace, archive)["id"])


def test_views_survive_a_backup_and_restore(client: TestClient, workspace: Workspace, project: dict[str, Any]) -> None:
    table = a_table(client, project["id"])
    client.post(
        f"/api/projects/{project['id']}/views",
        json=body(table, "Trend", filters=[{"column": "word", "equals": "liberty"}]),
    )
    copy = restored_copy(workspace, project["id"])

    views = workspace.views(copy)
    assert [v["name"] for v in views] == ["Trend"]
    assert views[0]["settings"] == SETTINGS
    assert views[0]["filters"] == [{"column": "word", "equals": "liberty"}]


def test_a_restored_view_points_at_the_restored_run(
    client: TestClient, workspace: Workspace, project: dict[str, Any]
) -> None:
    """Restoring renumbers every job. A view carrying the old ID would address
    a run this project does not have -- and the file it names would be gone."""
    table = a_table(client, project["id"])
    client.post(f"/api/projects/{project['id']}/views", json=body(table, "Trend"))
    copy = restored_copy(workspace, project["id"])

    view = workspace.views(copy)[0]
    assert view["job"] != table["job"], "the restored run has a new ID"
    assert view["job"] in {job["id"] for job in workspace.jobs(copy)}
    assert view["source"]["state"] == "ok", view["source"]["detail"]
    assert workspace.artifact(copy, view["job"], view["index"]).is_file()


def test_a_restored_view_keeps_its_revision_count(
    client: TestClient, workspace: Workspace, project: dict[str, Any]
) -> None:
    table = a_table(client, project["id"])
    view = client.post(f"/api/projects/{project['id']}/views", json=body(table, "Trend")).json()
    client.post(
        f"/api/projects/{project['id']}/views/{view['id']}", json=body(table, "Trend", settings={"kind": "line"})
    )
    copy = restored_copy(workspace, project["id"])
    assert workspace.views(copy)[0]["revision"] == 2


def test_a_project_with_no_views_still_backs_up(
    client: TestClient, workspace: Workspace, project: dict[str, Any]
) -> None:
    assert workspace.views(restored_copy(workspace, project["id"])) == []


def test_an_archive_from_before_views_still_restores(workspace: Workspace, project: dict[str, Any]) -> None:
    """Version 1 archives predate saved views and carry none. Refusing to read
    one would make this change destroy backups people already have."""
    import json as _json
    import zipfile

    from desktop_backend.archives import backup, restore

    archive = workspace.root / "v1.nlpsuite"
    backup(workspace, project["id"], archive)

    older = workspace.root / "older.nlpsuite"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(older, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "project.json":
                manifest = _json.loads(data)
                manifest["version"] = 1
                manifest.pop("views")
                data = _json.dumps(manifest, ensure_ascii=False).encode("utf-8")
            target.writestr(item.filename, data)

    assert workspace.views(str(restore(workspace, older)["id"])) == []


def test_a_restored_view_with_impossible_settings_is_refused(
    client: TestClient, workspace: Workspace, project: dict[str, Any]
) -> None:
    """A hand-edited or newer-build view fails during the restore, where
    somebody is watching, not as a broken chart weeks later."""
    import json as _json
    import zipfile

    from desktop_backend.archives import backup, restore

    table = a_table(client, project["id"])
    client.post(f"/api/projects/{project['id']}/views", json=body(table, "Trend"))
    archive = workspace.root / "views-backup.nlpsuite"
    backup(workspace, project["id"], archive)

    tampered = workspace.root / "tampered.nlpsuite"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(tampered, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "project.json":
                manifest = _json.loads(data)
                manifest["views"][0]["settings"]["kind"] = "sunburst"
                data = _json.dumps(manifest, ensure_ascii=False).encode("utf-8")
            target.writestr(item.filename, data)

    with pytest.raises(ValueError, match="Trend"):
        restore(workspace, tampered)
