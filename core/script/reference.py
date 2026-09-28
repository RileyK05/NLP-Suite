"""The library reference, and the guide people paste into their own AI chatbot.

Both are generated from the code -- the public functions' signatures and
docstrings, the tool registry, the templates -- never written by hand, so
neither can promise a call the library does not have. Both are deterministic:
the same library and the same corpus give the same bytes, so two releases'
guides can be compared with a diff (tests/test_notebooks.py::TestGuide checks
this).

The app itself contains no AI. The guide is how a researcher gets an assistant
they already use to write notebook code that calls the suite's functions
instead of reimplementing them (docs/internal/PLAN_0.5.0.md 4.8).
"""

from __future__ import annotations

from collections.abc import Mapping
import inspect
from typing import Any

__all__ = ["GUIDE_VERSIONS", "corpus_summary", "guide", "library_functions", "reference_markdown", "tools_markdown"]

GUIDE_VERSIONS = ("short", "full")
LIBRARY_VERSION = "1"
#: Sample values shown per document column in the guide's corpus section.
_SAMPLES = 10
#: Every column a corpus has wherever it comes from. Anything beyond these is
#: a project's own document detail (Speaker, Party, ...): it exists inside the
#: app, and a folder read with ``nlp.use_folder`` outside it has none.
_CORE_COLUMNS = ("Document ID", "Document", "Date", "Year", "Words")


def library_functions() -> list[dict[str, str]]:
    """Every public name in ``nlpsuite``: its kind, signature and docstring."""
    import nlpsuite

    found: list[dict[str, str]] = []
    for name in nlpsuite.__all__:
        value = getattr(nlpsuite, name)
        if inspect.isclass(value):
            doc = inspect.getdoc(value) or ""
            methods = [
                f"{name}.{member}{inspect.signature(getattr(value, member))}"
                for member in sorted(vars(value))
                if not member.startswith("_") and callable(getattr(value, member))
            ]
            properties = [
                f"{name}.{member}"
                for member in sorted(vars(value))
                if isinstance(getattr(value, member, None), property)
            ]
            found.append(
                {
                    "name": name,
                    "kind": "class",
                    "signature": name,
                    "doc": doc,
                    "members": "\n".join([*properties, *methods]),
                }
            )
        elif callable(value):
            signature = inspect.signature(value)
            required = [
                param.name
                for param in signature.parameters.values()
                if param.default is inspect.Parameter.empty
                and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            found.append(
                {
                    "name": name,
                    "kind": "function",
                    "signature": f"nlp.{name}{signature}",
                    "doc": inspect.getdoc(value) or "",
                    "members": "",
                    # What the page's "Insert" types: the call with the arguments it cannot do without.
                    "insert": f"nlp.{name}({', '.join(required)})",
                }
            )
    return found


def reference_markdown() -> str:
    """The library reference, for the Scripts page's side panel and the guide."""
    parts = []
    for item in library_functions():
        if item["kind"] == "class":
            parts.append(f"### {item['name']}\n\n{item['doc']}")
            if item["members"]:
                parts.append("```\n" + item["members"] + "\n```")
        else:
            parts.append(f"### `{item['signature']}`\n\n{item['doc']}")
    return "\n\n".join(parts)


def tools_markdown(*, short: bool) -> str:
    """Every tool ``nlp.run`` accepts: one line each (short) or with parameters and tables (full)."""
    from core.profiler.labels import tool_description, tool_label
    from core.profiler.registry import get_tool
    from core.script.api import _tool_names

    lines: list[str] = []
    for name in _tool_names():
        spec = get_tool(name)
        assert spec is not None  # noqa: S101 - _tool_names only lists registered tools
        summary = tool_description(name, spec.description)
        if short:
            lines.append(f"- `{name}`: {tool_label(name)}. {summary}")
            continue
        lines.append(f"#### `{name}`: {tool_label(name)}\n\n{summary}")
        params = [
            f"  - `{param.name.replace('-', '_')}` ({param.type}, default `{param.default!r}`"
            + (f", one of {list(param.choices)}" if param.choices else "")
            + f"): {param.help}"
            for param in spec.params
        ]
        lines.append("- Parameters:\n" + ("\n".join(params) if params else "  - (none)"))
        tables = [output for output in spec.outputs if output.endswith(".csv")]
        lines.append("- Tables: " + (", ".join(f"`{t}`" for t in tables) or "(none)"))
        lines.append("- Needs parsing: " + ("yes" if spec.requires_parse else "no"))
    return "\n".join(lines) if short else "\n\n".join(lines)


def corpus_summary(documents: Any, *, name: str = "") -> dict[str, Any]:
    """What the guide may say about a corpus: counts, columns and sample values. Never text.

    ``details`` names the columns that are a project's own document fields
    (``Speaker``, ``Party``, ...). They are present inside the app and absent
    from a folder read with ``nlp.use_folder`` outside it, so the guide says
    which is which rather than letting a script that relies on ``Speaker``
    break the moment it runs anywhere but the app.
    """
    import pandas as pd

    frame: pd.DataFrame = documents
    columns: dict[str, list[str]] = {}
    for column in frame.columns:
        if column in ("Document ID", "Words"):
            continue
        values = [str(v) for v in pd.unique(frame[column].dropna())]
        columns[str(column)] = sorted(values)[:_SAMPLES] if column != "Document" else values[:_SAMPLES]
    years = frame["Year"].dropna() if "Year" in frame.columns else pd.Series(dtype=float)
    return {
        "name": name,
        "documents": len(frame),
        "words": int(frame["Words"].sum()) if "Words" in frame.columns else 0,
        "years": [int(years.min()), int(years.max())] if len(years) else None,
        "dated": len(years),
        "columns": [str(c) for c in frame.columns],
        "details": [str(c) for c in frame.columns if str(c) not in _CORE_COLUMNS],
        "samples": columns,
    }


_INSTRUCTIONS = """You are writing Python for a notebook cell inside the NLP Suite, a desktop app for studying collections of texts. The researcher will paste your code into a cell and run it. Follow these rules:

1. Use only `nlpsuite` (imported as `nlp`), pandas, numpy, matplotlib, seaborn, scipy and scikit-learn. Nothing else is installed and nothing can be installed.
2. Prefer a library function to writing it yourself. Never count words with `re.findall` or `str.count` when `nlp.term_rates` does it; never split sentences yourself when `nlp.passages` or `corpus.sentences()` does it. For any tool without its own function, use `nlp.run("tool_name", corpus, ...)`.
3. Show every table with `nlp.show(table)` (it draws a chart beside it; pass `x=` and `y=` to choose one), keep figures with `nlp.figure(fig, "name")`, and keep data with `nlp.save(table, "name")`. For an interactive chart -- a sankey flow, a sunburst or treemap, a radar, or any kind the quick chart does not draw -- use `nlp.chart(table, kind=..., x=..., y=...)`. Never write files: the notebook's folder is scratch space that is thrown away.
4. Documents differ in length, so compare rates (per 1,000 or 10,000 words), never raw counts. To average over a year, average the per-document rates.
5. Say in a comment which counting rule you used (`match="lemma"`, `"form"` or `"exact-lowercase"`) and why; they give different numbers.
6. Use the column names listed under "The researcher's corpus". If the request needs a detail the corpus does not have (a speaker, a party), say so in a comment instead of inventing a column.
7. Split the work into a few short cells, each starting with a one-line comment saying what it does. Start the first cell with `import nlpsuite as nlp`.
8. Do not claim findings. Write code that shows the evidence; the researcher reads it."""

_NOTEBOOKS = """Cells run top to bottom in one Python process; variables persist between cells. A cell's last line is displayed if it is a value (a table shows with a chart, a figure is drawn). `print` works. "Stop" ends the process and forgets every variable. The corpus is parsed the first time a function needs it and reused afterwards; `exact-lowercase` counting needs no parsing and is instant."""

_OUTSIDE = """These cells normally run inside the NLP Suite, where the corpus is connected for you. They can also run outside it -- in Jupyter, VS Code, or a plain script -- with the suite installed as a Python package:

    pip install nlp-suite-ng        # the library alone; add [spacy] for parsing, [plotly] for charts, [all] for everything

Two things differ outside the app, and both break a script that ignores them:

1. **Point the library at your texts.** Inside the app `nlp.corpus()` finds the project's documents by itself; outside it there is no project, so call `nlp.use_folder("path/to/texts")` once (a directory of `.txt` files, read recursively) before anything else. `nlp.corpus()` then returns those documents.
2. **A folder has fewer document columns.** Outside the app `corpus.documents` has `Document ID`, `Document`, `Date`, `Year` and `Words` only. A project inside the app adds its own details (`Speaker`, `Party`, `Kind`, ...) as extra columns; a folder read with `nlp.use_folder` has none of them, because a filename carries no such fields. A script that groups by `Speaker` works in the app and fails outside it -- guard it (check `"Speaker" in corpus.documents.columns`) or derive the value from the filename yourself."""

_MISTAKES = """- **Counting by hand.** Wrong: `text.lower().count("border")`. Right: `nlp.term_rates(corpus, ["border"], match="exact-lowercase")`, which counts whole words and gives rates.
- **Raw counts across documents of different lengths.** Wrong: plotting `border count`. Right: plotting `border per 1000`.
- **Summing rates.** Wrong: `rates.groupby("Year").sum()` on a rate column. Right: `.mean()` of the per-document rates, or `by="year"` for pooled words.
- **Writing files.** Wrong: `table.to_csv("out.csv")`. Right: `nlp.save(table, "out")`.
- **Assuming dates.** Check `corpus.documents["Year"]` has values before grouping by year; a corpus of chapters or essays may have none.
- **Lemma lists with inflections.** With `match="lemma"`, list dictionary forms ("immigrant", not "immigrants"). With `match="exact-lowercase"`, list every form you want counted.
- **Merging `Year` back onto a tool's table.** Nearly every per-document table a tool writes already carries the documents' own columns -- `Document`, `Date`, `Year`, and each detail (`Speaker`, `Kind`, ...) -- beside its measure, so `table.merge(corpus.documents)` makes `Year_x`/`Year_y` and then a `KeyError` on `Year`. Read or group by the columns the table already has (print `table.columns`), and merge only the columns it lacks."""


def guide(*, version: str = "short", corpus: Mapping[str, Any] | None = None) -> str:
    """The document a researcher pastes into their chatbot before asking for a script."""
    from core.script.templates import TEMPLATES

    if version not in GUIDE_VERSIONS:
        raise ValueError(f"version must be one of {GUIDE_VERSIONS}")
    sections = [
        f"# NLP Suite scripting guide for AI assistants (library version {LIBRARY_VERSION}, {version})",
        "## 1. Instructions for you, the assistant\n\n" + _INSTRUCTIONS,
        "## 2. How notebooks work\n\n" + _NOTEBOOKS,
        "## 3. Running a script outside the app\n\n" + _OUTSIDE,
        "## 4. The library (`import nlpsuite as nlp`)\n\n" + reference_markdown(),
        "## 5. Tools for `nlp.run`\n\n"
        + ('Call `nlp.describe("tool")` in a cell to see a tool\'s parameters.\n\n' if version == "short" else "")
        + tools_markdown(short=version == "short"),
    ]
    if version == "full":
        examples = []
        for template in TEMPLATES:
            code = "\n\n".join(text for kind, text in template.cells if kind == "code")
            examples.append(f'### Request: "{template.request}"\n\n```python\n{code}\n```')
        sections.append("## 6. Worked examples\n\n" + "\n\n".join(examples))
        sections.append("## 7. Common mistakes\n\n" + _MISTAKES)
    else:
        sections.append("## 6. Common mistakes\n\n" + _MISTAKES)
    if corpus is not None:
        lines = [f"- {corpus['documents']} documents, {corpus['words']:,} words."]
        if corpus.get("years"):
            first, last = corpus["years"]
            lines.append(f"- {corpus['dated']} are dated, from {first} to {last}.")
        else:
            lines.append("- No document has a date.")
        lines.append("- `corpus.documents` columns: " + ", ".join(f"`{c}`" for c in corpus["columns"]))
        details = corpus.get("details") or []
        if details:
            # Named, and said to be app-only: a script that leans on Speaker
            # works inside the app and breaks on nlp.use_folder outside it.
            named = ", ".join(f"`{c}`" for c in details)
            portable = ", ".join(f"`{c}`" for c in corpus["columns"] if c not in details)
            lines.append(
                f"- {named} "
                + ("are" if len(details) > 1 else "is")
                + " a project detail the app supplies, not something a filename carries. "
                "A folder read with `nlp.use_folder(...)` outside the app has only "
                + portable
                + "; do not assume a detail exists if the code must run outside the app."
            )
        for column, values in corpus["samples"].items():
            lines.append(f"- `{column}` values include: " + ", ".join(f"`{v}`" for v in values))
        title = f" ({corpus['name']})" if corpus.get("name") else ""
        sections.append(f"## The researcher's corpus{title}\n\n" + "\n".join(lines))
    sections.append("---\n\nNow ask the researcher what they want to find out, then write the cells.")
    return "\n\n".join(sections) + "\n"
