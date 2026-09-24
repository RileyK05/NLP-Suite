"""One research session end to end, over real speeches, through the API.

The unit tests establish that each piece is correct. This establishes that the
pieces are wired to each other: load, ask, page, save, update, reopen, publish
-- the sequence a researcher actually performs, with the same HTTP calls the
desktop makes.

It runs against the real corpus because every wiring defect it covers was a
mismatch between what one step recorded and what the next step read, and a
two-document fixture agrees with itself too easily. The parse is shared with
the other real-corpus tests through the annotation cache; see ``real_snapshot``
in the root conftest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import prime_annotations, real_corpus_dir

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from desktop_backend.runner import Runner
from desktop_backend.server import create_app
from desktop_backend.store import Workspace

pytestmark = [
    pytest.mark.model_integration,
    pytest.mark.skipif(real_corpus_dir() is None, reason="no real corpus on this machine"),
]

#: A phrase whose tokenization is the whole point: the regular expression that
#: used to split queries cannot find it, and this corpus is full of it.
SUBJECT = "U.S."


@pytest.fixture
def session(tmp_path: Path, real_documents: list[Path], real_snapshot: Any, monkeypatch: Any):
    """A workspace holding the real documents, with the parse already cached."""
    submitted: list[dict[str, Any]] = []

    def record(  # noqa: PLR0913 - it stands in for Runner.submit, whose signature this is
        self: Runner,
        project_id: str,
        tool: str,
        params: dict[str, Any],
        parser: str,
        *,
        selection: Any = None,
    ) -> dict[str, Any]:
        """Publishing spawns a worker that reparses the corpus.

        What this test is about is the request that worker receives, so the
        submission is observed rather than performed. ``_question_plan``
        validates it against the publisher's own parameter specification, so a
        parameter the publisher would reject still fails here.
        """
        from desktop_backend.runner import _question_plan

        _question_plan(tool, params).unwrap()
        submitted.append({"tool": tool, "params": params, "parser": parser, "selection": selection})
        # Shaped like a real job row, because the endpoint summarizes one.
        return {
            "id": "job-1",
            "project_id": project_id,
            "tool": tool,
            "state": "QUEUED",
            "request": json.dumps(
                {
                    "params": {tool: params},
                    "parser": parser,
                    "documents": [{"id": item} for item in (selection.document_ids if selection else [])],
                    "selection": {"document_ids": selection.document_ids if selection else None},
                }
            ),
        }

    monkeypatch.setattr(Runner, "submit", record)

    workspace = Workspace(tmp_path / "workspace")
    with TestClient(create_app(workspace, "tok")) as client:
        client.headers["Authorization"] = "Bearer tok"
        project = workspace.create("State of the Union")
        for path in real_documents:
            workspace.import_document(project["id"], path.name, path.read_bytes())
        yield client, str(project["id"]), submitted, workspace


@pytest.fixture
def loaded(session: Any, real_snapshot: Any, real_parse_cache: Path):
    client, project, submitted, workspace = session
    # The parse this corpus already has, put where this workspace looks for it.
    prime_annotations(workspace.root, real_snapshot.key, real_parse_cache)
    state = client.post(f"/api/projects/{project}/live/warm", json={"parser": "spacy"}).json()
    assert state["state"] == "ready", state
    return client, project, submitted, state


def _ask(client: Any, project: str, snapshot: str, **body: Any) -> dict[str, Any]:
    response = client.post(
        f"/api/projects/{project}/live/phrase",
        json={"text": SUBJECT, "snapshot_id": snapshot, **body},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _save_body(state: dict[str, Any], answer: dict[str, Any], name: str, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "snapshot_id": answer["snapshot_id"],
        "parser": state["parser"],
        "document_ids": state["selection"]["ids"],
        "specification": {
            "schema_version": 1,
            "kind": "phrase_distribution",
            "text": answer["question"]["subject"]["text"],
            "comparison_text": None,
            "case_sensitive": False,
            "position_bins": 10,
            "evidence_year": None,
            "evidence_document_id": None,
        },
        "matching": {
            "version": answer["question"]["matching_profile"]["version"],
            "tokenizer": answer["question"]["matching_profile"]["tokenizer"],
            "source": answer["question"]["matching_profile"]["source"],
            "tokens": answer["question"]["subject"]["tokens"],
            "comparison_tokens": [],
        },
        **extra,
    }


class TestTheSessionHoldsTogether:
    def test_the_workspace_snapshot_is_the_corpus_snapshot(self, loaded: Any, real_snapshot: Any) -> None:
        """Imported documents describe the same corpus as the files on disk.

        If this ever stops holding, the other tests here still pass but pay a
        full reparse, and a saved question would cite a snapshot the publisher
        could not reproduce.
        """
        _client, _project, _submitted, state = loaded
        assert state["snapshot_id"] == real_snapshot.key
        assert state["documents"] == len(real_snapshot.corpus.docs)

    def test_asking_paging_and_drilling_keep_one_subject(self, loaded: Any) -> None:
        client, project, _submitted, state = loaded
        snapshot = state["snapshot_id"]
        first = _ask(client, project, snapshot, evidence_offset=0, evidence_limit=25)
        assert first["summary"]["occurrences"] > 0, f"{SUBJECT} was not found in these speeches"
        assert first["question"]["matching_profile"]["source"] == "snapshot"

        second = _ask(client, project, snapshot, evidence_offset=25, evidence_limit=25)
        assert second["question"]["subject"] == first["question"]["subject"]
        assert second["summary"] == first["summary"]
        assert [row["id"] for row in second["evidence"]["rows"]] != [row["id"] for row in first["evidence"]["rows"]]

        busiest = max(first["documents"], key=lambda row: row["occurrences"])
        drilled = _ask(client, project, snapshot, evidence_document_id=busiest["document_id"])
        assert drilled["summary"] == first["summary"]
        assert drilled["evidence"]["filtered_total"] == busiest["occurrences"]
        assert {row["document_id"] for row in drilled["evidence"]["rows"]} == {busiest["document_id"]}

    def test_a_question_about_documents_that_are_not_loaded_is_refused(self, loaded: Any) -> None:
        client, project, _submitted, _state = loaded
        response = client.post(
            f"/api/projects/{project}/live/phrase",
            json={"text": SUBJECT, "snapshot_id": "b" * 64},
        )
        assert response.status_code == 409
        assert response.json()["stale_snapshot"] is True


class TestSavingRecordsWhatWasAsked:
    def test_a_saved_question_keeps_the_tokens_it_was_answered_with(self, loaded: Any) -> None:
        client, project, _submitted, state = loaded
        answer = _ask(client, project, state["snapshot_id"])
        saved = client.post(f"/api/projects/{project}/questions", json=_save_body(state, answer, "US over time"))
        assert saved.status_code == 200, saved.text
        record = saved.json()
        assert record["matching"]["tokens"] == answer["question"]["subject"]["tokens"]
        assert record["matching"]["tokens"] == [SUBJECT]
        assert record["matching"]["source"] == "snapshot"

        # And it comes back the same way from a fresh read.
        listed = client.get(f"/api/projects/{project}/questions").json()
        assert listed[0]["matching"] == record["matching"]

    def test_a_question_whose_scope_does_not_hold_together_is_refused(self, loaded: Any) -> None:
        client, project, _submitted, state = loaded
        answer = _ask(client, project, state["snapshot_id"])

        wrong_snapshot = _save_body(state, answer, "Wrong snapshot")
        wrong_snapshot["snapshot_id"] = "c" * 64
        assert client.post(f"/api/projects/{project}/questions", json=wrong_snapshot).status_code == 400

        wrong_parser = _save_body(state, answer, "Wrong parser")
        wrong_parser["parser"] = "stanza"
        assert client.post(f"/api/projects/{project}/questions", json=wrong_parser).status_code == 400

        wrong_documents = _save_body(state, answer, "Wrong documents")
        wrong_documents["document_ids"] = state["selection"]["ids"][:5]
        assert client.post(f"/api/projects/{project}/questions", json=wrong_documents).status_code == 400

    def test_an_update_aimed_at_a_superseded_revision_is_refused(self, loaded: Any) -> None:
        client, project, _submitted, state = loaded
        answer = _ask(client, project, state["snapshot_id"])
        record = client.post(
            f"/api/projects/{project}/questions", json=_save_body(state, answer, "US over time")
        ).json()
        assert record["revision"] == 1

        first = client.post(
            f"/api/projects/{project}/questions/{record['id']}",
            json=_save_body(state, answer, "US over time", expected_revision=1),
        )
        assert first.status_code == 200, first.text
        assert first.json()["revision"] == 2

        # A second editor still holding revision 1 must not overwrite revision 2.
        stale = client.post(
            f"/api/projects/{project}/questions/{record['id']}",
            json=_save_body(state, answer, "US over time", expected_revision=1),
        )
        assert stale.status_code == 400
        assert "revision 2" in stale.json()["detail"]
        assert client.get(f"/api/projects/{project}/questions").json()[0]["revision"] == 2


class TestReopeningAndPublishingAskTheSavedQuestion:
    def test_a_reopened_question_answers_what_it_answered_when_saved(self, loaded: Any) -> None:
        client, project, _submitted, state = loaded
        answer = _ask(client, project, state["snapshot_id"])
        record = client.post(
            f"/api/projects/{project}/questions", json=_save_body(state, answer, "US over time")
        ).json()

        # Cold: forget the corpus, as closing and reopening the app would.
        client.post(f"/api/projects/{project}/live/cool")
        assert client.get(f"/api/projects/{project}/live").json()["state"] == "cold"

        warmed = client.post(
            f"/api/projects/{project}/live/warm",
            json={"selection": {"document_ids": record["document_ids"]}, "parser": record["parser"]},
        ).json()
        assert warmed["snapshot_id"] == record["snapshot_id"]
        again = _ask(client, project, record["snapshot_id"])
        assert again["summary"] == answer["summary"]
        assert again["question"]["subject"] == answer["question"]["subject"]

    def test_publishing_hands_the_publisher_the_saved_tokens(self, loaded: Any) -> None:
        client, project, submitted, state = loaded
        answer = _ask(client, project, state["snapshot_id"])
        record = client.post(
            f"/api/projects/{project}/questions", json=_save_body(state, answer, "US over time")
        ).json()

        published = client.post(f"/api/projects/{project}/questions/{record['id']}/publish")
        assert published.status_code == 200, published.text
        request = submitted[-1]
        assert request["tool"] == "phrase_distribution"
        assert request["params"]["snapshot-id"] == record["snapshot_id"]
        assert json.loads(request["params"]["resolved-tokens"]) == [SUBJECT]
        assert json.loads(request["params"]["comparison-tokens"]) == []
        assert request["selection"].document_ids == record["document_ids"]

    def test_a_question_saved_before_the_rules_were_recorded_publishes_without_tokens(self, loaded: Any) -> None:
        """The migration rule, stated: no recorded tokens means the parser
        resolves the text again, and the published record says so."""
        client, project, submitted, state = loaded
        answer = _ask(client, project, state["snapshot_id"])
        body = _save_body(state, answer, "Older question")
        del body["matching"]
        record = client.post(f"/api/projects/{project}/questions", json=body).json()
        assert record["matching"]["source"] == "unrecorded"
        assert record["matching"]["tokens"] == []

        client.post(f"/api/projects/{project}/questions/{record['id']}/publish")
        assert json.loads(submitted[-1]["params"]["resolved-tokens"]) == []
