"""Notebooks: the stored shape, their store, "Run and save", the templates and the AI guide."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import zipfile

import pandas as pd
import pytest

from core.script.reference import corpus_summary, guide, library_functions
from core.script.templates import TEMPLATES
from desktop_backend.notebook_export import export_notebook
from desktop_backend.notebooks import NotebookBody, new_notebook, normalize_notebook, notebook_from_template
from desktop_backend.runner import Runner, run_job
from desktop_backend.store import Workspace

SPEECHES = {
    "1946-01-21_harry s truman_sotu.txt": "Immigrants came to the border. The border was closed. We spoke of war and peace.",
    "1955-01-06_dwight d eisenhower_sotu.txt": "Peace is our aim. Refugees need homes. The economy is strong.",
    "1995-01-24_william j clinton_sotu.txt": "Border security matters. Deportation of aliens. Immigration law is broken.",
    "2007-01-23_george w bush_sotu.txt": "Secure borders and fair laws. The war goes on. Peace will come.",
}


@pytest.fixture
def project(tmp_path: Path) -> tuple[Workspace, str]:
    workspace = Workspace(tmp_path / "ws")
    created = workspace.create("Speeches")
    for name, text in SPEECHES.items():
        workspace.import_document(created["id"], name, text.encode("utf-8"))
    return workspace, created["id"]


def _run(
    workspace: Workspace, project_id: str, content: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    notebook = workspace.save_notebook(
        project_id, NotebookBody(name=f"nb {len(workspace.notebooks(project_id))}", content=content)
    )
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    try:
        job = runner.submit_notebook(project_id, notebook["id"], "spacy")
        run_job(workspace.root, job["id"])
    finally:
        runner.close()
    return next(j for j in workspace.jobs(project_id) if j["id"] == job["id"])


class TestShape:
    def test_jupyter_files_are_cleaned_to_code_and_text(self) -> None:
        content = normalize_notebook(
            {
                "nbformat": 4,
                "nbformat_minor": 2,
                "metadata": {"widgets": {}},
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["x = 1\n", "x"],
                        "outputs": [{"output_type": "stream"}],
                        "execution_count": 3,
                    },
                    {"cell_type": "raw", "source": "notes"},
                ],
            }
        )
        code, text = content["cells"]
        assert code["source"] == "x = 1\nx" and code["outputs"] == [] and code["execution_count"] is None
        assert text["cell_type"] == "markdown"
        assert content["metadata"]["nlpsuite"]["created_by"] == "import"

    def test_a_pasted_cell_keeps_its_mark_and_nothing_else(self) -> None:
        content = normalize_notebook(
            {
                "nbformat": 4,
                "cells": [
                    {"cell_type": "code", "source": "x", "metadata": {"nlpsuite": {"origin": "pasted"}, "tags": ["a"]}},
                    {"cell_type": "code", "source": "y", "metadata": {"nlpsuite": {"origin": "trusted-by-me"}}},
                ],
            }
        )
        pasted, other = content["cells"]
        assert pasted["metadata"] == {"nlpsuite": {"origin": "pasted"}}
        assert other["metadata"] == {}

    def test_what_is_not_a_notebook_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nbformat 4"):
            normalize_notebook({"cells": []})
        with pytest.raises(ValueError, match="code or markdown"):
            normalize_notebook({"nbformat": 4, "cells": [{"cell_type": "sql", "source": ""}]})


class TestStore:
    def test_save_update_duplicate_delete(self, project: tuple[Workspace, str]) -> None:
        workspace, project_id = project
        saved = workspace.save_notebook(
            project_id, NotebookBody(name="Immigration", content=new_notebook([("code", "1")]))
        )
        updated = workspace.update_notebook(
            project_id,
            saved["id"],
            NotebookBody(name="Immigration", content=new_notebook([("code", "2")]), expected_revision=1),
        )
        assert updated["revision"] == 2 and updated["content"]["cells"][0]["source"] == "2"
        with pytest.raises(ValueError, match="moved on to revision 2"):
            workspace.update_notebook(
                project_id, saved["id"], NotebookBody(name="Immigration", content=new_notebook([]), expected_revision=1)
            )
        copy = workspace.duplicate_notebook(project_id, saved["id"])
        assert copy["name"] == "Immigration (copy)" and copy["revision"] == 1
        with pytest.raises(ValueError, match="already has a notebook"):
            workspace.save_notebook(project_id, NotebookBody(name="Immigration", content=new_notebook([])))
        workspace.delete_notebook(project_id, saved["id"])
        assert [n["name"] for n in workspace.notebooks(project_id)] == ["Immigration (copy)"]

    def test_notebooks_survive_backup_and_restore(self, project: tuple[Workspace, str]) -> None:
        from desktop_backend.archives import backup, restore

        workspace, project_id = project
        content = notebook_from_template("word-group-over-time")
        workspace.save_notebook(project_id, NotebookBody(name="Immigration", content=content))
        archive = workspace.root / "notebooks.nlpsuite"
        backup(workspace, project_id, archive)
        restored = restore(workspace, archive)
        [notebook] = workspace.notebooks(restored["id"])
        assert notebook["name"] == "Immigration"
        assert [c["source"] for c in notebook["content"]["cells"]] == [c["source"] for c in content["cells"]]


class TestRunAndSave:
    def test_a_saved_run_holds_tables_figures_and_the_executed_notebook(
        self, project: tuple[Workspace, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace, project_id = project
        monkeypatch.setenv("NLP_SUITE_RUN_FIGURES", "0")
        job = _run(workspace, project_id, notebook_from_template("word-group-over-time"), monkeypatch)
        assert job["state"] == "DONE", job["diagnostics"]
        run = workspace.project_dir(project_id) / job["run_dir"]
        assert (run / "tables" / "Rate_in_each_document.csv").is_file()
        assert (run / "data" / "rates_per_document.csv").is_file()
        assert (run / "figures" / "word_group_over_time.png").is_file()
        executed = json.loads((run / "notebook.ipynb").read_text(encoding="utf-8"))
        assert executed["cells"][1]["outputs"], "the executed notebook keeps what each cell showed"
        provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
        assert [call["function"] for call in provenance["calls"]][:2] == ["corpus", "term_rates"]
        assert any(item["path"] == "tables/Rate_in_each_document.csv" for item in workspace.tables(project_id))

    def test_a_failing_cell_stops_the_run_and_keeps_what_came_before(
        self, project: tuple[Workspace, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace, project_id = project
        content = new_notebook(
            [
                ("code", "import nlpsuite as nlp\nnlp.show(nlp.corpus().documents, name='documents')"),
                ("code", "raise ValueError('bad column')"),
                ("code", "nlp.note('never reached')"),
            ]
        )
        job = _run(workspace, project_id, content, monkeypatch)
        assert job["state"] == "PARTIAL"
        assert "Cell 2" in job["diagnostics"][0]["message"] and "bad column" in job["diagnostics"][0]["message"]
        run = workspace.project_dir(project_id) / job["run_dir"]
        assert pd.read_csv(run / "tables" / "documents.csv").shape[0] == 4
        assert not (run / "notes.md").exists()


def test_run_and_save_waits_until_pasted_code_is_read(project: tuple[Workspace, str]) -> None:
    workspace, project_id = project
    content = new_notebook([("code", "import nlpsuite as nlp")])
    content["cells"].append({**content["cells"][0], "id": "pasted1", "metadata": {"nlpsuite": {"origin": "pasted"}}})
    saved = workspace.save_notebook(project_id, NotebookBody(name="Pasted", content=content))
    runner = Runner(workspace)
    try:
        with pytest.raises(ValueError, match="Cell 2 was pasted in"):
            runner.submit_notebook(project_id, saved["id"], "spacy")
        assert not workspace.jobs(project_id)
    finally:
        runner.close()


class TestExport:
    def test_the_zip_holds_the_notebook_its_data_the_guide_and_a_readme(
        self, project: tuple[Workspace, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace, project_id = project
        job = _run(workspace, project_id, notebook_from_template("word-group-over-time"), monkeypatch)
        [notebook] = workspace.notebooks(project_id)
        name, data = export_notebook(workspace, project_id, notebook["id"])
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
            readme = archive.read("README.md").decode("utf-8")
        assert name == "nb 0.zip"
        assert {"nb 0.ipynb", "GUIDE_FOR_AI.md", "nlpsuite_offline.py", "README.md"} <= names
        assert "data/Rate_in_each_document.csv" in names and "data/rates_per_document.csv" in names
        assert "figures/word_group_over_time.png" in names
        assert job["created"][:10] in readme

    def test_a_notebook_never_saved_as_a_run_says_its_data_is_empty(self, project: tuple[Workspace, str]) -> None:
        workspace, project_id = project
        saved = workspace.save_notebook(project_id, NotebookBody(name="Draft", content=new_notebook([("code", "1")])))
        _, data = export_notebook(workspace, project_id, saved["id"])
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            assert "has not been run with Run and save" in archive.read("README.md").decode("utf-8")
            assert not [n for n in archive.namelist() if n.startswith("data/")]

    def test_the_drawing_cell_runs_without_the_suite(
        self, project: tuple[Workspace, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The export's claim, tested: with only pandas and matplotlib, the figure cell draws."""
        from core.script.templates import _FIGURE

        workspace, project_id = project
        _run(workspace, project_id, notebook_from_template("word-group-over-time"), monkeypatch)
        [notebook] = workspace.notebooks(project_id)
        _, data = export_notebook(workspace, project_id, notebook["id"])
        folder = tmp_path / "export"
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            archive.extractall(folder)
        # The run's own figure is in the export; remove it so only the redraw can put it back.
        shutil.rmtree(folder / "figures")
        yearly = next(p.stem for p in (folder / "data").glob("Yearly*.csv"))
        script = "\n".join(
            [
                "import sys",
                "import matplotlib",
                "matplotlib.use('Agg')",
                "import nlpsuite_offline as nlp",
                "rates = nlp.load('Rate_in_each_document')",
                f"yearly = nlp.load({yearly!r})",
                "rate = 'immigration per 1000'",
                _FIGURE,
                "assert not [m for m in sys.modules if m == 'core' or m.startswith('core.')], 'the suite was imported'",
                "try:\n    nlp.corpus()\nexcept RuntimeError as exc:\n    assert 'needs the NLP Suite' in str(exc)",
            ]
        )
        (folder / "redraw.py").write_text(script, encoding="utf-8")
        done = subprocess.run(
            [sys.executable, "redraw.py"],
            cwd=folder,
            env={**os.environ, "PYTHONPATH": str(folder)},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        assert (folder / "figures" / "word_group_over_time.png").stat().st_size > 1000


class TestLint:
    def test_points_at_lines_worth_reading(self) -> None:
        from core.script.lint import lint

        code = "\n".join(
            [
                "import nlpsuite as nlp",
                "import os, subprocess",
                "import requests",
                "n = text.lower().split().count('border')",
                "hits = re.findall(r'border', text)",
                "table.to_csv('out.csv')",
                "open('x.txt')",
            ]
        )
        found = {item["line"]: item["message"] for item in lint(code)}
        assert 1 not in found
        assert "running other programs" in found[2] or "files and programs" in found[2]
        assert "the network" in found[3]
        assert "nlp.term_rates" in found[4] and "nlp.term_rates" in found[5]
        assert "nlp.save" in found[6]
        assert "opens a file" in found[7]

    def test_library_code_passes_clean(self) -> None:
        from core.script.lint import lint

        for template in TEMPLATES:
            for kind, source in template.cells:
                if kind == "code":
                    assert lint(source) == [], (template.id, source)

    def test_broken_code_is_named_not_raised(self) -> None:
        from core.script.lint import lint

        [warning] = lint("for x in")
        assert warning["line"] == 1 and "not valid Python" in str(warning["message"])


def test_every_function_has_an_insertable_call() -> None:
    for item in library_functions():
        if item["kind"] == "function":
            assert item["insert"].startswith(f"nlp.{item['name']}(")
            compile(item["insert"], "<insert>", "eval")  # parses; never run


@pytest.mark.parametrize("template", [t.id for t in TEMPLATES])
def test_every_template_runs_green(
    template: str, project: tuple[Workspace, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("spacy")
    workspace, project_id = project
    job = _run(workspace, project_id, notebook_from_template(template), monkeypatch)
    assert job["state"] == "DONE", job["diagnostics"]


class TestGuide:
    def test_names_every_function_and_tool(self) -> None:
        from core.script.api import _tool_names

        text = guide(version="full")
        for item in library_functions():
            assert item["name"] in text
        for tool in _tool_names():
            assert f"`{tool}`" in text
        short = guide(version="short")
        assert len(short) < len(text)
        assert "Worked examples" not in short

    def test_is_deterministic(self) -> None:
        assert guide(version="full") == guide(version="full")

    def test_says_what_the_corpus_has_and_never_its_text(self, project: tuple[Workspace, str]) -> None:
        from core.script.api import Corpus
        from core.script.session import Session
        from desktop_backend.project_source import ProjectSource

        workspace, project_id = project
        documents = Corpus(Session(ProjectSource(workspace, project_id)), None).documents
        text = guide(corpus=corpus_summary(documents, name="Speeches"))
        assert "4 documents" in text and "from 1946 to 2007" in text
        assert "1946-01-21_harry s truman_sotu.txt" in text
        for speech in SPEECHES.values():
            for sentence in speech.split(". "):
                assert sentence.strip(".") not in text

    def test_every_template_request_appears_as_an_example(self) -> None:
        text = guide(version="full")
        for template in TEMPLATES:
            assert template.request in text
