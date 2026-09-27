"""Stage directions are not the author's words, and are not guessed at (plan 5.2)."""

from __future__ import annotations

from pathlib import Path

from core.io.cleaning import count_stage_directions, strip_stage_directions


class TestStripStageDirections:
    def test_the_known_reactions_go(self) -> None:
        text = "We shall fight (Applause.) on the beaches [Laughter] and inaudible bits (Inaudible)."
        cleaned, removed = strip_stage_directions(text)
        assert cleaned == "We shall fight  on the beaches  and inaudible bits ."
        assert removed == 3

    def test_a_span_that_carries_meaning_stays(self) -> None:
        text = "The treaty (see page 12) was signed [sic] on the third."
        cleaned, removed = strip_stage_directions(text)
        assert cleaned == text
        assert removed == 0

    def test_only_reaction_words_are_dropped(self) -> None:
        """ "(the beginning)" is prose in brackets; "(booing)" is a sound."""
        text = "It began (the beginning) badly (Booing.)"
        cleaned, removed = strip_stage_directions(text)
        assert "(the beginning)" in cleaned
        assert "(Booing.)" not in cleaned
        assert removed == 1

    def test_a_reactions_mix_is_dropped_but_a_sentence_stays(self) -> None:
        cleaned, removed = strip_stage_directions("Then (applause and laughter). Later (he said applause matters).")
        assert "(applause and laughter)" not in cleaned
        assert "(he said applause matters)" in cleaned
        assert removed == 1

    def test_more_than_four_words_is_left_alone(self) -> None:
        text = "(sustained applause that continues for many minutes)"
        cleaned, removed = strip_stage_directions(text)
        assert cleaned == text and removed == 0

    def test_the_projects_own_words_count(self) -> None:
        text = "Then (table banging) and (gavel)."
        cleaned, removed = strip_stage_directions(text, extra=["table banging"])
        assert cleaned == "Then  and (gavel)."
        assert removed == 1

    def test_empty_text_is_empty(self) -> None:
        assert strip_stage_directions("") == ("", 0)

    def test_removing_is_idempotent(self) -> None:
        text = "Hello (Applause.) world (Laughter.)"
        once, removed = strip_stage_directions(text)
        twice, again = strip_stage_directions(once)
        assert twice == once and again == 0 and removed == 2


class TestCount:
    def test_counts_and_shows_examples_without_changing_anything(self) -> None:
        texts = ["Brave (Applause.) words (Laughter.)", "Still (applause) and (Crosstalk) here"]
        found, samples = count_stage_directions(texts)
        assert found == 4
        assert samples[0] == "(Applause.)" and "(Laughter.)" in samples and "(Crosstalk)" in samples
        assert texts[0] == "Brave (Applause.) words (Laughter.)"


class TestWhatTheToolsRead:
    """The clean reaches every tool, because every tool reads load_corpus (plan 5.2)."""

    def _workspace(self, tmp_path: Path):
        from desktop_backend.store import Workspace

        workspace = Workspace(tmp_path / "ws")
        project = workspace.create("Transcript")  # new projects: cleaning on (D6)
        workspace.import_document(
            project["id"], "town_hall.txt", b"Welcome all (Applause.) to the hall (Laughter.) tonight."
        )
        return workspace, project["id"]

    def test_a_new_project_reads_without_the_stage_directions(self, tmp_path: Path) -> None:
        from desktop_backend.project_corpus import load_corpus

        workspace, project_id = self._workspace(tmp_path)
        corpus, diagnostics = load_corpus(workspace, project_id, workspace.documents(project_id))
        assert "(Applause.)" not in corpus.docs[0].text
        assert corpus.docs[0].text.strip() == "Welcome all  to the hall  tonight."
        assert [d.code for d in diagnostics] == ["STAGE_DIRECTIONS_REMOVED"]

    def test_an_existing_project_opts_in_and_sees_the_counts(self, tmp_path: Path) -> None:
        from desktop_backend.fields import FieldSettings, TextCleaning, save_settings, settings
        from desktop_backend.project_corpus import load_corpus

        workspace, project_id = self._workspace(tmp_path)
        save_settings(
            workspace,
            project_id,
            FieldSettings(text_cleaning=TextCleaning(stage_directions=False)),
        )
        assert settings(workspace, project_id)["settings"]["text_cleaning"]["stage_directions"] is False
        corpus, diagnostics = load_corpus(workspace, project_id, workspace.documents(project_id))
        assert "(Applause.)" in corpus.docs[0].text
        assert diagnostics == [] or all(d.code != "STAGE_DIRECTIONS_REMOVED" for d in diagnostics)

    def test_the_imported_file_is_never_changed(self, tmp_path: Path) -> None:
        from desktop_backend.project_corpus import load_corpus

        workspace, project_id = self._workspace(tmp_path)
        [document] = workspace.documents(project_id)
        load_corpus(workspace, project_id, [document])
        stored = workspace.project_dir(project_id) / "corpus" / document["stored_name"]
        assert b"(Applause.)" in stored.read_bytes()
