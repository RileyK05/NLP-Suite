"""A transcript cut by speaker (docs/PLAN_0.5.0.md 2.5, 2.9).

The fixture is a three-speaker town hall with a role in parentheses, turns
broken across lines, a speaker who talks twice in a row, and four kinds of
stage direction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.file_ops.sections import detect_turns
from desktop_backend.sections import SplitBody, public_turns, split_document
from desktop_backend.store import Workspace

TRANSCRIPT = Path(__file__).parent / "fixtures" / "books" / "transcript_town_hall.txt"


def test_speakers_are_found_with_their_turns_and_words() -> None:
    plan = detect_turns(TRANSCRIPT.read_text(encoding="utf-8")).unwrap()
    assert [(name, turns) for name, turns, _words in plan.speakers] == [
        ("Moderator", 3),
        ("Mayor Hale", 2),
        ("Councillor", 2),
    ]
    # Two lines in a row from the mayor are one turn.
    assert "state fund" in plan.turns[1].text and "levy" in plan.turns[1].text
    assert plan.turns[3].role == "DISTRICT 2"
    assert plan.preamble_words == 12


def test_stage_directions_are_counted_and_left_out() -> None:
    plan = detect_turns(TRANSCRIPT.read_text(encoding="utf-8")).unwrap()
    assert dict(plan.directions) == {"(APPLAUSE)": 1, "(LAUGHTER)": 1, "(inaudible)": 1, "[CROSSTALK]": 1}
    everything = " ".join(turn.text for turn in plan.turns)
    assert "APPLAUSE" not in everything and "CROSSTALK" not in everything and "inaudible" not in everything


def test_prose_is_not_a_transcript() -> None:
    result = detect_turns("It was a quiet evening. Nobody spoke at all.\n")
    assert result.value is None and result.diagnostics[0].code == "TURNS_NO_SPEAKERS"


@pytest.fixture
def hall(tmp_path: Path) -> tuple[Workspace, str, str]:
    workspace = Workspace(tmp_path / "ws")
    project_id = workspace.create("Hearings")["id"]
    document = workspace.import_document(project_id, "2024-05-02_town hall.txt", TRANSCRIPT.read_bytes())
    return workspace, project_id, str(document["id"])


def test_one_document_per_speaker_without_the_moderator(hall: tuple[Workspace, str, str]) -> None:
    workspace, project_id, document_id = hall
    body = SplitBody(transcript=True, speakers=["Mayor Hale", "Councillor"])
    preview = public_turns(workspace, project_id, document_id, body)
    assert preview["documents"] == 2
    assert [s["kept"] for s in preview["speakers"]] == [False, True, True]
    made = split_document(workspace, project_id, document_id, body)
    assert [doc["fields"]["Speaker"] for doc in made] == ["Mayor Hale", "Councillor"]
    mayor = made[0]
    assert mayor["fields"]["Turns"] == "2" and mayor["fields"]["Event"] == "2024-05-02_town hall"
    assert mayor["fields"]["Date"] == "2024-05-02"
    text = (workspace.project_dir(project_id) / "corpus" / mayor["stored_name"]).read_text(encoding="utf-8")
    assert "\n\n" in text and "APPLAUSE" not in text and "Moderator" not in text


def test_one_document_per_turn_lines_the_turns_up(hall: tuple[Workspace, str, str]) -> None:
    workspace, project_id, document_id = hall
    made = split_document(workspace, project_id, document_id, SplitBody(transcript=True, per_turn=True))
    assert [doc["fields"]["Order"] for doc in made] == [str(n) for n in range(1, 8)]
    assert made[0]["fields"]["Speaker"] == "Moderator"


def test_keeping_nobody_is_refused(hall: tuple[Workspace, str, str]) -> None:
    workspace, project_id, document_id = hall
    with pytest.raises(ValueError, match="at least one speaker"):
        split_document(workspace, project_id, document_id, SplitBody(transcript=True, speakers=[]))
