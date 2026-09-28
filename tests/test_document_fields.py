"""Document details in a project: detected, overridden, selected by, carried into runs and backups."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from desktop_backend.archives import backup, restore
from desktop_backend.comparisons import ComparisonBody, comparisons, save_comparison
from desktop_backend.fields import (
    FieldRow,
    FieldSettings,
    details,
    field_names,
    project_axis,
    resolved_axis,
    save_settings,
    set_fields,
)
from desktop_backend.project_corpus import load_corpus
from desktop_backend.selection import CorpusSelection, resolve_selection
from desktop_backend.store import Workspace

NAMES = [
    "1946-01-21_harry s truman_sotu.txt",
    "1949-01-20_harry s truman_ina.txt",
    "1953-02-02_dwight d eisenhower_sotu.txt",
    "1957-01-21_dwight d eisenhower_ina.txt",
]


@pytest.fixture
def project(tmp_path: Path) -> tuple[Workspace, str, dict[str, str]]:
    workspace = Workspace(tmp_path / "ws")
    created = workspace.create("Mixed")
    ids = {}
    for name in NAMES:
        ids[name] = workspace.import_document(created["id"], name, f"Words of {name}.".encode())["id"]
    return workspace, created["id"], ids


def test_file_names_give_details_with_no_clicks(project: tuple[Workspace, str, dict[str, str]]) -> None:
    workspace, project_id, _ids = project
    documents = workspace.documents(project_id)
    assert documents[0]["fields"] == {"Date": "1946-01-21", "Speaker": "Harry S Truman", "Kind": "sotu"}
    assert documents[0]["field_sources"] == {"Date": "filename", "Speaker": "filename", "Kind": "filename"}
    names = {item["name"]: item for item in field_names(workspace, project_id)}
    assert list(names) == ["Date", "Kind", "Speaker"]
    assert names["Kind"]["distinct"] == 2 and names["Speaker"]["count"] == 4


def test_a_typed_value_wins_and_the_hidden_one_is_listed(project: tuple[Workspace, str, dict[str, str]]) -> None:
    workspace, project_id, ids = project
    target = ids["1953-02-02_dwight d eisenhower_sotu.txt"]
    set_fields(workspace, project_id, [FieldRow(document_id=target, name="Kind", value="address")])
    found = details(workspace, project_id, workspace.documents(project_id))[target]["Kind"]
    assert found == {"value": "address", "source": "user", "overridden": [{"value": "sotu", "source": "filename"}]}
    # A spreadsheet value sits under the typed one.
    set_fields(workspace, project_id, [FieldRow(document_id=target, name="Kind", value="state", source="csv")])
    assert details(workspace, project_id, workspace.documents(project_id))[target]["Kind"]["value"] == "address"
    # Clearing the typed value goes back to the next source down.
    set_fields(workspace, project_id, [FieldRow(document_id=target, name="kind", value="")])
    assert details(workspace, project_id, workspace.documents(project_id))[target]["Kind"]["value"] == "state"


def test_the_axis_is_resolved_for_the_shape_line(project: tuple[Workspace, str, dict[str, str]]) -> None:
    workspace, project_id, ids = project
    documents = workspace.documents(project_id)
    found = details(workspace, project_id, documents)
    assert resolved_axis(workspace, project_id, documents, found) == {
        "kind": "time",
        "noun": "Year",
        "placed": 4,
        "first": "1946-01-21",
        "last": "1957-01-21",
    }
    # Chapters typed in, and the order axis chosen with its noun.
    for position, name in enumerate(NAMES, 1):
        set_fields(workspace, project_id, [FieldRow(document_id=ids[name], name="Order", value=str(position * 10))])
    save_settings(workspace, project_id, FieldSettings(axis="order", order_label="Session"))
    found = details(workspace, project_id, documents)
    axis = resolved_axis(workspace, project_id, documents, found)
    assert axis == {"kind": "order", "noun": "Session", "placed": 4, "first": "10", "last": "40"}
    assert project_axis(workspace, project_id) == {"kind": "order", "noun": "Session"}


def test_a_run_records_the_axis_and_its_tables_carry_positions_and_details(
    project: tuple[Workspace, str, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run, not only the executor: submitted with the project's axis, and every table says where each document sits."""
    import json

    import pandas as pd

    from desktop_backend.runner import Runner, run_job

    workspace, project_id, ids = project
    for position, name in enumerate(NAMES, 1):
        set_fields(workspace, project_id, [FieldRow(document_id=ids[name], name="Order", value=str(position))])
    save_settings(workspace, project_id, FieldSettings(axis="order", order_label="Session"))
    monkeypatch.setenv("NLP_SUITE_RUN_FIGURES", "0")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    try:
        job = runner.submit(project_id, "text_statistics", {}, "spacy")
        run_job(workspace.root, job["id"])
    finally:
        runner.close()
    [done] = [j for j in workspace.jobs(project_id) if j["id"] == job["id"]]
    assert done["state"] == "DONE", done["diagnostics"]
    assert json.loads(done["request"])["axis"] == {"kind": "order", "noun": "Session"}
    run = workspace.project_dir(project_id) / done["run_dir"]
    per_document = next(
        frame
        for frame in (pd.read_csv(path) for path in run.glob("*.csv"))
        if "Document ID" in frame.columns and "Position" in frame.columns
    )
    assert sorted(per_document["Position label"]) == ["Session 1", "Session 2", "Session 3", "Session 4"]
    assert {"Speaker", "Kind", "Order"} <= set(per_document.columns)


def test_selection_by_detail(project: tuple[Workspace, str, dict[str, str]]) -> None:
    workspace, project_id, _ids = project
    documents = workspace.documents(project_id)
    chosen = resolve_selection(documents, CorpusSelection(fields={"Kind": ["ina"]}))
    assert [doc["name"] for doc in chosen] == [NAMES[1], NAMES[3]]
    both = resolve_selection(documents, CorpusSelection(fields={"kind": ["SOTU"], "Speaker": ["Harry S Truman"]}))
    assert [doc["name"] for doc in both] == [NAMES[0]]
    with pytest.raises(ValueError, match="Details here: Date, Kind, Speaker"):
        resolve_selection(documents, CorpusSelection(fields={"Party": ["Democratic"]}))


def test_selection_by_an_order_window(project: tuple[Workspace, str, dict[str, str]]) -> None:
    """ "Chapters 2-3" and the same window over Order values (plan 1.5)."""
    workspace, project_id, ids = project
    set_fields(
        workspace,
        project_id,
        # The last document keeps no Order: a prologue or a stray the corpus
        # does not line up.
        [FieldRow(document_id=ids[name], name="Order", value=str(i)) for i, name in enumerate(NAMES[:3], start=1)],
    )
    documents = workspace.documents(project_id)
    middle = resolve_selection(documents, CorpusSelection(order_from=2.0, order_to=3.0))
    assert [doc["name"] for doc in middle] == [NAMES[1], NAMES[2]]
    assert all(doc["document_order"] is not None for doc in middle)
    with_unordered = resolve_selection(documents, CorpusSelection(order_from=2.0, order_to=3.0, include_unordered=True))
    assert [doc["name"] for doc in with_unordered] == [NAMES[1], NAMES[2], NAMES[3]]
    with pytest.raises(ValueError, match="order_from"):
        CorpusSelection(order_from=3.0, order_to=2.0).order_bounds()


def test_a_typed_date_moves_the_document_for_selections_and_runs(
    project: tuple[Workspace, str, dict[str, str]],
) -> None:
    workspace, project_id, ids = project
    target = ids[NAMES[0]]
    set_fields(workspace, project_id, [FieldRow(document_id=target, name="Date", value="1960")])
    documents = workspace.documents(project_id)
    late = resolve_selection(documents, CorpusSelection(date_from="1955-01-01"))
    assert NAMES[0] in {doc["name"] for doc in late}
    [meta] = [doc for doc in late if doc["name"] == NAMES[0]]
    assert meta["document_date"] == "1960-01-01" and meta["date_source"] == "user"
    corpus, _ = load_corpus(workspace, project_id, resolve_selection(documents, None))
    [moved] = [doc for doc in corpus.docs if doc.label == NAMES[0]]
    assert moved.date == date(1960, 1, 1) and moved.details["Speaker"] == "Harry S Truman"


def test_an_old_request_without_details_still_reads_dates_from_names(
    project: tuple[Workspace, str, dict[str, str]],
) -> None:
    workspace, project_id, _ids = project
    items = [
        {k: v for k, v in doc.items() if k not in ("fields", "field_sources")}
        for doc in workspace.documents(project_id)
    ]
    corpus, _ = load_corpus(workspace, project_id, items)
    assert corpus.docs[0].date == date(1946, 1, 21)
    assert corpus.docs[0].fields == ()


def test_renamed_parts_keep_their_reading(project: tuple[Workspace, str, dict[str, str]]) -> None:
    workspace, project_id, _ids = project
    save_settings(workspace, project_id, FieldSettings(part_names={"Kind": "Genre", "Date": "Delivered"}))
    fields = workspace.documents(project_id)[0]["fields"]
    assert fields == {"Delivered": "1946-01-21", "Speaker": "Harry S Truman", "Genre": "sotu"}
    save_settings(workspace, project_id, FieldSettings(filename_detection=False))
    assert workspace.documents(project_id)[0]["fields"] == {}


def test_a_changed_detail_makes_the_glance_key_new(project: tuple[Workspace, str, dict[str, str]]) -> None:
    """The glance's sentences read details and the axis, so its cache must too."""
    from desktop_backend.glance import current_key

    workspace, project_id, ids = project
    before = current_key(workspace, project_id)
    set_fields(workspace, project_id, [FieldRow(document_id=ids[NAMES[0]], name="Party", value="Democratic")])
    assert current_key(workspace, project_id) != before
    save_settings(workspace, project_id, FieldSettings(axis="order", order_label="Session"))
    assert current_key(workspace, project_id) != before


def test_names_the_suite_uses_are_refused(project: tuple[Workspace, str, dict[str, str]]) -> None:
    workspace, project_id, ids = project
    with pytest.raises(ValueError, match="already use"):
        set_fields(workspace, project_id, [FieldRow(document_id=ids[NAMES[0]], name="Document ID", value="7")])
    with pytest.raises(ValueError, match="cannot name a detail"):
        set_fields(workspace, project_id, [FieldRow(document_id=ids[NAMES[0]], name="?!", value="7")])


def test_details_settings_and_comparisons_survive_backup(project: tuple[Workspace, str, dict[str, str]]) -> None:
    workspace, project_id, ids = project
    set_fields(workspace, project_id, [FieldRow(document_id=ids[NAMES[0]], name="Party", value="Democratic")])
    save_settings(workspace, project_id, FieldSettings(order_label="Session"))
    definition = {
        "sides": [
            {"name": "Addresses", "project_id": project_id, "selection": {"fields": {"Kind": ["sotu"]}}},
            {"name": "Inaugurals", "project_id": project_id, "selection": {"fields": {"Kind": ["ina"]}}},
        ]
    }
    save_comparison(workspace, project_id, ComparisonBody(name="Two kinds", definition=definition))
    archive = workspace.root / "mixed.nlpsuite"
    backup(workspace, project_id, archive)
    restored = restore(workspace, archive)["id"]
    parties = {doc["name"]: doc["fields"].get("Party") for doc in workspace.documents(restored)}
    assert parties[NAMES[0]] == "Democratic" and parties[NAMES[1]] is None
    [saved] = comparisons(workspace, restored)
    assert {side["project_id"] for side in saved["definition"]["sides"]} == {restored}


def test_purging_a_document_removes_its_details(project: tuple[Workspace, str, dict[str, str]]) -> None:
    workspace, project_id, ids = project
    target = ids[NAMES[0]]
    set_fields(workspace, project_id, [FieldRow(document_id=target, name="Party", value="Democratic")])
    workspace.trash_document(project_id, target, True)
    workspace.purge_document(project_id, target)
    with workspace.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM document_fields WHERE document_id=?", (target,)).fetchone()[0] == 0


def test_project_events_are_stored_and_positioned(project: tuple[Workspace, str, dict[str, str]]) -> None:
    """Dated events save with the project and resolve to decimal years (section 4)."""
    workspace, project_id, _ = project
    from desktop_backend.fields import ProjectEvent, project_events

    save_settings(
        workspace,
        project_id,
        FieldSettings(
            events=[
                ProjectEvent(name="Pearl Harbor", date="1941-12-07"),
                ProjectEvent(name="Korea", date="1950.5"),
                ProjectEvent(name="Unreadable", date="not a date"),
            ]
        ),
    )
    found = project_events(workspace, project_id)
    # The unreadable date is dropped, never placed at the origin.
    assert [event["name"] for event in found] == ["Pearl Harbor", "Korea"]
    assert abs(found[0]["position"] - 1941.93) < 0.01
    assert found[1]["position"] == 1950.5
    # The stored settings keep every event, so the Corpus page can still show
    # and fix the one that could not be positioned.
    from desktop_backend.fields import settings

    assert len(settings(workspace, project_id)["settings"]["events"]) == 3


def test_duplicate_event_names_are_refused(project: tuple[Workspace, str, dict[str, str]]) -> None:
    from pydantic import ValidationError

    from desktop_backend.fields import ProjectEvent

    with pytest.raises(ValidationError):
        FieldSettings(events=[ProjectEvent(name="Korea", date="1950"), ProjectEvent(name="korea", date="1951")])
