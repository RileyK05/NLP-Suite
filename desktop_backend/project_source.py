"""A notebook's documents: one project, read the way a published run reads it.

The script library (:mod:`core.script`) knows nothing about workspaces; it asks
a :class:`~core.script.session.Source` for documents and parses. This is the
source the app hands it: the project's documents, checked against their import
hashes, parsed through the same cache as runs and the Interactive page, and
the project's earlier result tables for ``nlp.load``.

*documents*, when given, freezes the document list (a saved run of a notebook
records exactly which documents it read, like any run). Without it the
source reads the project as it is now, which is what an open notebook wants:
import a document and the next cell sees it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from core.io.reader import Corpus
from core.script.session import ParsedTable, SuiteError
from desktop_backend.project_corpus import load_corpus, parse_cached, resolve_parser
from desktop_backend.store import Workspace

__all__ = ["ProjectSource"]


class ProjectSource:
    """One project's documents, parses and earlier tables."""

    def __init__(
        self,
        workspace: Workspace,
        project_id: str,
        *,
        parser: str = "spacy",
        documents: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.workspace = workspace
        self.project_id = project_id
        self.parser = parser
        self.name = str(workspace.project(project_id)["name"])
        self._frozen = None if documents is None else [dict(item) for item in documents]
        self._pipeline: Any = None

    def _rows(self) -> list[dict[str, Any]]:
        return self._frozen if self._frozen is not None else self.workspace.documents(self.project_id)

    def documents(self) -> list[dict[str, Any]]:
        return [
            {"id": str(row["id"]), "name": str(row["name"]), "fields": row.get("fields") or {}} for row in self._rows()
        ]

    def corpus(self, ids: Sequence[str] | None) -> Corpus:
        rows = self._rows()
        if not rows:
            raise SuiteError("This project has no documents yet. Import some on the Corpus page.")
        if ids is not None:
            by_id = {str(row["id"]): row for row in rows}
            missing = [i for i in ids if i not in by_id]
            if missing:
                raise SuiteError("A document this notebook asked for is no longer in the project.")
            rows = [by_id[i] for i in ids]
        try:
            corpus, _ = load_corpus(self.workspace, self.project_id, rows)
        except ValueError as exc:
            raise SuiteError(str(exc)) from exc
        return corpus

    def parse(self, corpus: Corpus) -> ParsedTable:
        if self._pipeline is None:
            resolved = resolve_parser(self.parser)
            if resolved.value is None:
                raise SuiteError(
                    "No English parser is ready. Open Settings for the parser's status.", resolved.diagnostics
                )
            self._pipeline = resolved.unwrap()
        try:
            parsed = parse_cached(self.workspace.root, corpus, self._pipeline)
        except ValueError as exc:
            raise SuiteError(f"The documents could not be parsed: {exc}") from exc
        return ParsedTable(parsed.table, parsed.tokenizer, parsed.diagnostics)

    def tables(self) -> list[dict[str, Any]]:
        return self.workspace.tables(self.project_id)

    def read_table(self, job: str, index: int) -> pd.DataFrame:
        try:
            path = self.workspace.artifact(self.project_id, job, index)
        except (KeyError, ValueError, OSError) as exc:
            raise SuiteError(f"That earlier table could not be opened: {exc}") from exc
        return pd.read_csv(path)
