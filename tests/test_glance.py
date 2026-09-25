"""Corpus at a glance: one job, one parse, the recipe's child runs, and a summary.

The end-to-end test parses real (tiny) documents with spaCy and runs the
whole recipe, so it is skipped where the English model is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from conftest import has_spacy_model
from core.insight.glance import GLANCE_FIGURES, RECIPE, glance_key, summarize
from desktop_backend.store import Workspace

SPEECHES = {
    "1934-01-03_franklin d roosevelt_sotu.txt": (
        "The nation faces hard times. Banks have failed and farms are lost. We will put people to work. "
        "Congress must act now to restore confidence in our banks."
    ),
    "1962-01-11_john f kennedy_sotu.txt": (
        "We choose to explore space. The world watches Berlin. Our economy must grow. "
        "Peace requires strength, and strength requires sacrifice from every citizen."
    ),
    "1994-01-25_william j clinton_sotu.txt": (
        "Every family deserves health care. Our schools must teach every child to read. "
        "The economy has added millions of jobs. We will balance the budget."
    ),
    "2014-01-28_barack obama_sotu.txt": (
        "Opportunity is who we are. Health care costs have slowed. Jobs are growing again. "
        "America must lead the world on climate and on trade."
    ),
}


class TestRecipe:
    def test_every_recipe_tool_is_registered_and_needs_no_choice(self) -> None:
        from core.profiler.plan import build_plan

        plan = build_plan(list(RECIPE), RECIPE)
        assert plan.ok, plan.diagnostics
        assert set(GLANCE_FIGURES) == set(RECIPE)

    def test_the_key_is_the_contents_not_the_order(self) -> None:
        assert glance_key(["b", "a"]) == glance_key(["a", "b"])
        assert glance_key(["a"]) != glance_key(["a", "c"])

    def test_the_summary_says_something_about_each_table(self) -> None:
        stats = pd.DataFrame(
            {"Document": ["1934_roosevelt.txt", "1994_clinton.txt"], "Tokens": [2000, 9000], "Sentences": [90, 400]}
        )
        tone = pd.DataFrame({"Document": ["1934_roosevelt.txt", "1994_clinton.txt"], "Compound": [0.3, 0.1]})
        lines = summarize(
            {"text_statistics": {"text_statistics.csv": stats}, "sentiment_vader_anew": {"vader.csv": tone}}
        )
        assert lines[0].startswith("2 documents, 11,000 words")
        # Both positive: the lower one is "least positive", not "most negative".
        assert "least positive" in lines[1]

    def test_missing_tables_are_skipped_not_fatal(self) -> None:
        assert summarize({}) == []


@pytest.mark.skipif(not has_spacy_model(), reason="needs the spaCy English model to parse")
class TestGlanceJob:
    def run(self, workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Any]:
        from desktop_backend.glance import GLANCE_TOOL, status
        from desktop_backend.runner import Runner, run_job

        project_id = workspace.create("Glance")["id"]
        for name, text in SPEECHES.items():
            workspace.import_document(project_id, name, text.encode("utf-8"))
        runner = Runner(workspace)
        monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
        job = runner.submit(project_id, GLANCE_TOOL, {}, "spacy")
        run_job(workspace.root, job["id"])
        runner.close()
        return project_id, status(workspace, project_id)

    def test_one_job_runs_the_recipe_and_summarises_it(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        workspace = Workspace(tmp_path / "workspace")
        project_id, answer = self.run(workspace, monkeypatch)
        assert answer["state"] == "ready", answer.get("job")
        text = " ".join(answer["summary"])
        assert "4 documents" in text
        assert "Easiest to read" in text and "Most positive in tone" in text
        assert answer["figures"], "the glance drew no figure"
        from desktop_backend.glance import figure_file

        first = figure_file(workspace, project_id, answer["figures"][0]["path"])
        assert first.read_bytes()[:4] == b"\x89PNG"
        with pytest.raises(KeyError):
            figure_file(workspace, project_id, "../../outside.png")

    def test_a_changed_corpus_makes_the_glance_stale(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from desktop_backend.glance import status

        workspace = Workspace(tmp_path / "workspace")
        project_id, answer = self.run(workspace, monkeypatch)
        assert answer["state"] == "ready"
        workspace.import_document(project_id, "2020-02-04_donald trump_sotu.txt", b"The economy is strong.")
        assert status(workspace, project_id)["state"] == "stale"

    def test_no_glance_yet(self, tmp_path: Path) -> None:
        from desktop_backend.glance import status

        workspace = Workspace(tmp_path / "workspace")
        project_id = workspace.create("Empty")["id"]
        assert status(workspace, project_id)["state"] == "none"
