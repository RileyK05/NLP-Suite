"""Lines worth reading in code that came from outside the app (docs/internal/PLAN_0.5.0.md 4.8.3).

A reading aid, not security. Python cannot be sandboxed from inside Python
(desktop_backend/kernel.py explains why), so nothing here stops code from
running. What it does is point at the handful of lines a careful reader
would want to look at before pressing Run on code a chatbot wrote: what it
imports, whether it touches files, the network or other programs, and
whether it counts words by hand when the suite already does that with its
own tokenizer.

The code is parsed with :mod:`ast` and never run.
"""

from __future__ import annotations

import ast

__all__ = ["lint"]

#: Modules a notebook is expected to use. Anything else is worth a look.
EXPECTED_MODULES = frozenset(
    {
        "nlpsuite",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
        "sklearn",
        "math",
        "re",
        "collections",
        "itertools",
        "statistics",
        "datetime",
        "json",
        "functools",
        "string",
        "textwrap",
    }
)

#: Modules that reach outside the notebook: files, programs, the network.
_REACHING = {
    "os": "files and programs on this computer",
    "sys": "the Python process itself",
    "shutil": "copying and deleting files",
    "pathlib": "files on this computer",
    "subprocess": "running other programs",
    "socket": "the network",
    "urllib": "the network",
    "requests": "the network",
    "http": "the network",
    "ftplib": "the network",
    "smtplib": "sending email",
    "ctypes": "native code",
    "pickle": "loading code disguised as data",
    "importlib": "importing modules by name",
}

_DYNAMIC = {
    "eval": "runs text as code",
    "exec": "runs text as code",
    "compile": "turns text into code",
    "__import__": "imports a module by name",
    "open": "opens a file (the notebook's folder is scratch; nlp.save keeps a table)",
}

_WRITES = {
    "to_csv": "nlp.save(table, name)",
    "to_excel": "nlp.save(table, name)",
    "to_parquet": "nlp.save(table, name)",
    "savefig": "nlp.figure(fig, name)",
}

_HAND_COUNTING = (
    "counts words by hand; nlp.term_rates does this with the suite's tokenizer and gives a rate per 1,000 words"
)


def _root(name: str) -> str:
    return name.split(".", maxsplit=1)[0]


def _chain_names(node: ast.AST) -> list[str]:
    """Attribute and call names along a chain like ``text.lower().split()``."""
    names: list[str] = []
    while True:
        if isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
            node = node.value
        else:
            return names


def lint(code: str) -> list[dict[str, object]]:
    """Warnings for *code*, one per line and kind, in line order.

    Each is ``{"line": n, "message": text}``; line 0 means the whole cell.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [{"line": exc.lineno or 0, "message": f"This is not valid Python: {exc.msg}."}]
    found: dict[tuple[int, str], str] = {}

    def warn(node: ast.AST, message: str) -> None:
        found.setdefault((getattr(node, "lineno", 0), message), message)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for module in modules:
                root = _root(module)
                if root in _REACHING:
                    warn(node, f"Imports {root}, which reaches {_REACHING[root]}. Read what it is used for.")
                elif root and root not in EXPECTED_MODULES:
                    warn(node, f"Imports {root}, which notebooks do not usually need.")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _DYNAMIC:
                warn(node, f"Calls {func.id}, which {_DYNAMIC[func.id]}.")
            elif isinstance(func, ast.Attribute):
                chain = _chain_names(func)
                if func.attr in _WRITES:
                    warn(
                        node,
                        f"Writes a file with .{func.attr}, into a scratch folder that is thrown away; "
                        f"{_WRITES[func.attr]} keeps it.",
                    )
                elif (
                    isinstance(func.value, ast.Name) and func.value.id == "re" and func.attr in ("findall", "finditer")
                ) or (func.attr == "count" and "split" in chain[1:]):
                    warn(node, f"This line {_HAND_COUNTING}.")
                elif isinstance(func.value, ast.Name) and func.value.id in _REACHING:
                    warn(node, f"Uses {func.value.id}.{func.attr}, which reaches {_REACHING[func.value.id]}.")
    return [{"line": line, "message": message} for (line, message) in sorted(found)]
