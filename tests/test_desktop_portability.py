"""Portable imports, untrusted backups and honest environment reporting."""

from contextlib import contextmanager
from pathlib import Path

import pytest

from core.profiler.registry import get_tool
from desktop_backend.archives import backup, restore, safe_member, validate_members
from desktop_backend.environment import availability
from desktop_backend.paths import portable_key
from desktop_backend.store import Workspace


@pytest.mark.parametrize("name", ["CON.txt", "nul.txt", "aux.csv", "lpt1.txt", "COM¹.txt", "é" * 110 + ".txt"])
def test_import_names_are_portable(tmp_path: Path, name: str) -> None:
    workspace = Workspace(tmp_path)
    project = workspace.create("Portable")
    data = b"text\nhello world\n" if name.endswith(".csv") else b"A readable sentence."
    document = workspace.import_document(project["id"], name, data)
    assert safe_member(document["stored_name"])
    assert workspace.preview(project["id"], document["id"])["text"]


def test_case_collisions_survive_backup_restore(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create("Case")
    first = workspace.import_document(project["id"], "A.txt", b"First text.")
    second = workspace.import_document(project["id"], "a.txt", b"Other text.")
    assert portable_key(first["stored_name"]) != portable_key(second["stored_name"])
    archive = tmp_path / "project.nlpsuite"
    backup(workspace, project["id"], archive)
    assert restore(workspace, archive)["documents"] == 2


@pytest.mark.parametrize(
    "name",
    [".", "../escape", "C:/drive", "x\\y", "a/NUL.txt", "a/trailing.", "a/trailing ", "a/stream:ads", "a/\x00.txt"],
)
def test_unsafe_portable_member_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        safe_member(name)


@pytest.mark.parametrize("names", [["A.txt", "a.txt"], ["a/x.txt", "A/y.txt"], ["a", "a/b"], ["é.txt", "e\u0301.txt"]])
def test_aliases_and_parent_conflicts_rejected(names: list[str]) -> None:
    with pytest.raises(ValueError):
        validate_members(names)


def test_restore_commit_failure_rolls_back_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create("Original")
    workspace.import_document(project["id"], "a.txt", b"Preserve this document.")
    archive = tmp_path / "project.nlpsuite"
    backup(workspace, project["id"], archive)
    original_connect = workspace.connect

    @contextmanager
    def fail_commit():
        with original_connect() as db:
            yield db
            raise OSError("Simulated commit failure")

    with monkeypatch.context() as patch:
        patch.setattr(workspace, "connect", fail_commit)
        with pytest.raises(OSError, match="commit failure"):
            restore(workspace, archive)
    assert len(workspace.projects()) == 1
    assert [path.name for path in (workspace.root / "projects").iterdir()] == [project["id"]]


def test_missing_transformers_are_not_advertised_ready() -> None:
    spec = get_tool("bert_extract")
    assert spec is not None
    result = availability(spec, {"spacy": True, "en_core_web_sm": True})
    assert result["state"] == "needs_setup"
    assert "transformers" in result["missing"]


def test_bert_topics_requires_torch_not_just_transformers() -> None:
    """Advertising this tool without torch would offer a run that cannot finish.

    ``TransformerBackend.embed`` does ``import torch`` and runs the model under
    ``torch.no_grad()``, so transformers alone is not enough: the tool would
    appear ready in the desktop and then fail at embed time.
    """
    spec = get_tool("bert_topics")
    assert spec is not None
    ready = {"spacy": True, "en_core_web_sm": True, "transformers": True, "torch": True}
    assert availability(spec, ready)["state"] == "available"
    without_torch = availability(spec, {**ready, "torch": False})
    assert without_torch["state"] == "needs_setup"
    assert "torch" in without_torch["missing"]


def test_lexicon_requirement_is_visible() -> None:
    spec = get_tool("nrc")
    assert spec is not None
    # nrclex ships the bundled lexicon; without it the tool is setup-blocked,
    # not input-blocked (no asset file to pick).
    assert availability(spec, {"spacy": True, "en_core_web_sm": True})["state"] == "needs_setup"
    assert availability(spec, {"spacy": True, "en_core_web_sm": True, "nrclex": True})["state"] == "available"
