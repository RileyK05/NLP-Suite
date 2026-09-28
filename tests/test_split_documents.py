"""A book stored as its chapters (docs/internal/PLAN_0.5.0.md 2.4, 2.9).

The split writes one document per section, each holding exactly its slice of
the book, with Work / Order / Chapter / Title details and a record of where
it came from. The book is never changed, goes to Trash by default, and may be
purged later without losing the record.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from desktop_backend.archives import backup, restore
from desktop_backend.fields import FieldSettings, save_settings, settings
from desktop_backend.runner import overlapping_documents
from desktop_backend.sections import SplitBody, public_plan, split_document
from desktop_backend.server import create_app
from desktop_backend.store import Workspace

BOOKS = Path(__file__).parent / "fixtures" / "books"


@pytest.fixture
def book(tmp_path: Path) -> tuple[Workspace, str, dict[str, object]]:
    workspace = Workspace(tmp_path / "ws")
    project_id = workspace.create("Novels")["id"]
    document = workspace.import_document(project_id, "Two Rivers.txt", (BOOKS / "two_books.txt").read_bytes())
    return workspace, project_id, document


def _read(workspace: Workspace, project_id: str, document: dict[str, object]) -> str:
    return (workspace.project_dir(project_id) / "corpus" / str(document["stored_name"])).read_text(encoding="utf-8")


def test_the_preview_writes_nothing(book: tuple[Workspace, str, dict[str, object]]) -> None:
    workspace, project_id, document = book
    plan = public_plan(workspace, project_id, str(document["id"]), SplitBody())
    assert plan["rule"] == "headings" and plan["level"] == "Chapter" and len(plan["sections"]) == 6
    assert plan["sections"][4]["parents"] == [["Book", "II"]]
    assert plan["sections"][0]["opening"].startswith("BOOK I CHAPTER 1. The First Letter")
    assert [doc["id"] for doc in workspace.documents(project_id)] == [document["id"]]


def test_each_chapter_is_a_document_holding_exactly_its_slice(book: tuple[Workspace, str, dict[str, object]]) -> None:
    workspace, project_id, document = book
    whole = _read(workspace, project_id, document)
    made = split_document(workspace, project_id, str(document["id"]), SplitBody())
    assert len(made) == 6
    texts = [_read(workspace, project_id, chapter) for chapter in made]
    # Nothing lost or repeated: the chapters are the book from its first heading on.
    assert "".join(texts) == whole[whole.index("BOOK I") :]
    first = made[0]
    assert first["name"] == f"Two Rivers {chr(0x2013)} 001 Chapter 1 The First Letter.txt"
    assert first["fields"] == {
        "Book": "I",
        "Chapter": "1",
        "Order": "1",
        "Title": "The First Letter",
        "Work": "Two Rivers",
    }
    assert set(first["field_sources"].values()) == {"split"}
    assert made[5]["fields"]["Book"] == "II"
    derivation = json.loads(str(first["derivation"]))
    assert derivation["source_sha256"] == document["sha256"] and derivation["rule"] == "headings"
    assert first["derived_from"] == document["id"]


def test_the_book_goes_to_trash_and_the_axis_becomes_chapters(book: tuple[Workspace, str, dict[str, object]]) -> None:
    workspace, project_id, document = book
    split_document(workspace, project_id, str(document["id"]), SplitBody())
    assert [doc["id"] for doc in workspace.documents(project_id, trashed=True)] == [document["id"]]
    chosen = settings(workspace, project_id)["settings"]
    assert (chosen["axis"], chosen["order_label"]) == ("order", "Chapter")


def test_a_chosen_axis_is_left_alone(book: tuple[Workspace, str, dict[str, object]]) -> None:
    workspace, project_id, document = book
    save_settings(workspace, project_id, FieldSettings(axis="none"))
    split_document(workspace, project_id, str(document["id"]), SplitBody())
    assert settings(workspace, project_id)["settings"]["axis"] == "none"


def test_keeping_the_whole_book_warns_every_run_that_reads_both(
    book: tuple[Workspace, str, dict[str, object]],
) -> None:
    workspace, project_id, document = book
    split_document(workspace, project_id, str(document["id"]), SplitBody(keep_whole=True))
    documents = workspace.documents(project_id)
    assert len(documents) == 7
    notes = overlapping_documents(documents)
    assert [note.code for note in notes] == ["CORPUS_OVERLAPPING_DOCUMENTS"]
    assert overlapping_documents([doc for doc in documents if doc["id"] != document["id"]]) == []


def test_a_changed_book_is_not_split_from_a_stale_preview(book: tuple[Workspace, str, dict[str, object]]) -> None:
    workspace, project_id, document = book
    with pytest.raises(ValueError, match="changed since the preview"):
        split_document(workspace, project_id, str(document["id"]), SplitBody(expected_sha256="0" * 64))
    assert len(workspace.documents(project_id)) == 1


def test_a_chapter_is_not_split_again(book: tuple[Workspace, str, dict[str, object]]) -> None:
    workspace, project_id, document = book
    made = split_document(workspace, project_id, str(document["id"]), SplitBody())
    with pytest.raises(ValueError, match="already a section"):
        public_plan(workspace, project_id, str(made[0]["id"]), SplitBody())


def test_purging_the_book_keeps_each_chapters_record(book: tuple[Workspace, str, dict[str, object]]) -> None:
    workspace, project_id, document = book
    made = split_document(workspace, project_id, str(document["id"]), SplitBody())
    workspace.purge_document(project_id, str(document["id"]))
    chapter = next(doc for doc in workspace.documents(project_id) if doc["id"] == made[0]["id"])
    assert json.loads(str(chapter["derivation"]))["source_sha256"] == document["sha256"]


def test_a_backup_keeps_which_book_each_chapter_came_from(
    book: tuple[Workspace, str, dict[str, object]], tmp_path: Path
) -> None:
    workspace, project_id, document = book
    split_document(workspace, project_id, str(document["id"]), SplitBody())
    archive = tmp_path / "novels.zip"
    backup(workspace, project_id, archive)
    restored = restore(Workspace(tmp_path / "other"), archive)
    other = Workspace(tmp_path / "other")
    books = other.documents(restored["id"], trashed=True)
    chapters = other.documents(restored["id"])
    assert len(books) == 1 and len(chapters) == 6
    assert {chapter["derived_from"] for chapter in chapters} == {books[0]["id"]}
    assert chapters[0]["fields"]["Work"] == "Two Rivers"


def test_the_routes_preview_then_apply(book: tuple[Workspace, str, dict[str, object]]) -> None:
    workspace, project_id, document = book
    client = TestClient(create_app(workspace, "split-token"))
    client.headers["Authorization"] = "Bearer split-token"
    base = f"/api/projects/{project_id}/documents/{document['id']}/sections"
    preview = client.post(f"{base}/preview", json={"rule": "words", "block_words": 500})
    assert preview.status_code == 200, preview.text
    assert preview.json()["rule"] == "words"
    assert "SECTIONS_NOT_CHAPTERS" in [d["code"] for d in preview.json()["diagnostics"]]
    applied = client.post(f"{base}/apply", json={"expected_sha256": document["sha256"]})
    assert applied.status_code == 200, applied.text
    assert len(applied.json()["documents"]) == 6
    refused = client.post(f"{base}/preview", json={"rule": "chapters"})
    assert refused.status_code == 422
    # The bytes of the book are untouched (R3).
    assert base64.b64encode((BOOKS / "two_books.txt").read_bytes()) == base64.b64encode(
        (workspace.project_dir(project_id) / "corpus" / str(document["stored_name"])).read_bytes()
    )
