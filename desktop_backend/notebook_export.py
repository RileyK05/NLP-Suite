"""A notebook exported as a zip that makes sense away from the app.

:func:`export_notebook` holds the notebook, the tables and figures its latest
"Run and save" kept, the AI guide its code was written against, a README, and
``nlpsuite_offline.py`` -- the source in :data:`OFFLINE_MODULE`. With it,
``import nlpsuite_offline as nlp`` gives a colleague who has only pandas and
matplotlib the functions that *read and draw*: ``load`` reads
``data/<name>.csv``, ``show`` prints a table, ``figure`` saves into
``figures/``, ``save`` writes a CSV. The functions that *analyse* (``corpus``,
``run``, ``term_rates`` ...) need the suite and say so.

The stand-in is kept as text rather than a module the app imports because it
is written out, not run: it must not import anything from the suite, and a
packaged app does not ship ``.py`` sources to copy. tests/test_notebooks.py
runs it in a clean process and checks that ``core`` is never imported.

The zip is built in memory and handed to the caller; nothing is written to
the workspace (R3).
"""

from __future__ import annotations

from datetime import UTC, datetime
import io
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any
import zipfile

if TYPE_CHECKING:  # pragma: no cover - typing only
    from desktop_backend.store import Workspace

__all__ = ["OFFLINE_MODULE", "README", "export_notebook", "latest_saved_run", "safe_file_name"]

_UNSAFE = re.compile(r"[^A-Za-z0-9 _.-]+")


def safe_file_name(name: str, fallback: str = "notebook") -> str:
    """*name* with anything a file system might refuse replaced."""
    return _UNSAFE.sub("_", name).strip(" .") or fallback


def latest_saved_run(workspace: Workspace, project_id: str, notebook_id: str) -> dict[str, Any] | None:
    """The newest finished "Run and save" of this notebook, if it has one."""
    from desktop_backend.notebooks import NOTEBOOK_TOOL

    for job in workspace.jobs(project_id):  # newest first
        if job["tool"] != NOTEBOOK_TOOL or job["state"] not in ("DONE", "PARTIAL") or not job.get("run_dir"):
            continue
        with_request = json.loads(job.get("request") or "{}")
        if with_request.get("notebook_id") == notebook_id:
            return job
    return None


def export_notebook(workspace: Workspace, project_id: str, notebook_id: str) -> tuple[str, bytes]:
    """The notebook as ``(file name, zip bytes)``."""
    from core.script.reference import corpus_summary, guide

    notebook = workspace.notebook(project_id, notebook_id)
    stem = safe_file_name(notebook["name"])
    run = latest_saved_run(workspace, project_id, notebook_id)
    buffer = io.BytesIO()
    data_names: list[str] = []
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{stem}.ipynb", json.dumps(notebook["content"], indent=1, ensure_ascii=False))
        if run is not None:
            run_dir: Path = workspace.project_dir(project_id) / run["run_dir"]
            for folder, target in (("tables", "data"), ("data", "data"), ("figures", "figures")):
                for path in sorted((run_dir / folder).glob("*")):
                    if path.is_file() and path.suffix.lower() in (".csv", ".png", ".svg"):
                        archive.write(path, f"{target}/{path.name}")
                        if target == "data":
                            data_names.append(path.stem)
        summary = None
        if workspace.documents(project_id):
            from core.script.api import Corpus
            from core.script.session import Session
            from desktop_backend.project_source import ProjectSource

            source = ProjectSource(workspace, project_id)
            summary = corpus_summary(Corpus(Session(source), None).documents, name=source.name)
        archive.writestr("GUIDE_FOR_AI.md", guide(version="full", corpus=summary))
        archive.writestr("nlpsuite_offline.py", OFFLINE_MODULE)
        if run is not None:
            data_note = (
                f"the tables of its latest Run and save ({run['created'][:10]}, "
                f"revision {json.loads(run['request']).get('revision', '?')})."
            )
        else:
            data_note = "empty: the notebook has not been run with Run and save, which is what keeps its tables."
        archive.writestr(
            "README.md",
            README.format(
                name=notebook["name"],
                exported=datetime.now(UTC).strftime("%Y-%m-%d"),
                file=stem,
                revision=notebook["revision"],
                data_note=data_note,
                example=data_names[0] if data_names else "table_name",
            ),
        )
    return f"{stem}.zip", buffer.getvalue()


OFFLINE_MODULE = '''"""A stand-in for the NLP Suite's notebook library, for reading and drawing only.

Use it in place of the real library:

    import nlpsuite_offline as nlp

load/show/save/figure/note work with the files in this folder. Anything that
analyses text (corpus, run, term_rates, ...) needs the NLP Suite itself.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"


def load(name, table=None):
    """A table this export carries: data/<name>.csv."""
    stem = str(table or name)
    stem = stem[:-4] if stem.endswith(".csv") else stem
    path = DATA / f"{stem}.csv"
    if not path.is_file():
        have = ", ".join(sorted(p.stem for p in DATA.glob("*.csv"))) or "none"
        raise FileNotFoundError(f"No data/{stem}.csv in this export. Tables here: {have}.")
    import pandas as pd

    return pd.read_csv(path)


def tables():
    """The tables this export carries."""
    return sorted(p.stem for p in DATA.glob("*.csv"))


def show(table, **_chart):
    """Print a table (the app would also draw a chart beside it)."""
    try:
        from IPython.display import display
    except ImportError:
        print(table)
    else:
        display(table)


def save(table, name):
    """Write a table to data/<name>.csv."""
    DATA.mkdir(exist_ok=True)
    table.to_csv(DATA / f"{name}.csv", index=False)


def figure(fig=None, name="figure"):
    """Save a matplotlib figure as figures/<name>.png and .svg."""
    import matplotlib.pyplot as plt

    fig = fig if fig is not None else plt.gcf()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")


def note(text):
    print(text)


def _needs_the_suite(name):
    def missing(*_args, **_kwargs):
        raise RuntimeError(
            f"nlp.{name} analyses text and needs the NLP Suite. This export can only read "
            "the tables it carries (nlp.load) and draw them."
        )

    missing.__name__ = name
    return missing


for _name in ("corpus", "run", "term_rates", "passages", "measures", "sentiment",
              "entities", "topics", "keyness", "similar", "tools", "describe", "use_folder"):
    globals()[_name] = _needs_the_suite(_name)
'''

README = """# {name}

Exported from the NLP Suite on {exported}.

- `{file}.ipynb`: the notebook, revision {revision}.
- `data/`: {data_note}
- `figures/`: the figures that run drew.
- `GUIDE_FOR_AI.md`: the guide to the suite's library the code was written against.
- `nlpsuite_offline.py`: a stand-in library for reading and drawing without the suite.

## Running it

**With the NLP Suite:** import the notebook on the Scripts page ("Import .ipynb"),
or open it in Jupyter with the suite installed. It runs as it is.

**Without it:** the cells that *draw* run anywhere with pandas and matplotlib;
the cells that *analyse* text need the suite. Replace the first line,

    import nlpsuite as nlp

with

    import nlpsuite_offline as nlp

and read the tables from `data/` instead of computing them, for example:

    rates = nlp.load("{example}")

Then the matplotlib cells work as written, and `nlp.figure(fig, "name")` saves
into `figures/`.
"""
