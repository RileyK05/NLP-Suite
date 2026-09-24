"""FR-3.4 structured search — TXT corpus, CSV rows, CoNLL rows, one CLI.

TXT search core already existed; CSV-row search is new (table_search is
CoNLL-schema-gated by design). Provenance columns keep document/sentence
identity on every hit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.file_ops.search import search_csv_frame, search_in_text


class TestCsvSearch:
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"Name": ["Alice", "Bob", "Alicia"], "City": ["Rome", "Paris", "Rome"]},
            columns=["Name", "City"],
        )

    def test_contains_with_row_numbers(self) -> None:
        hits = search_csv_frame(self._frame(), "ali").unwrap().to_frame()
        assert hits["Row"].tolist() == [1, 3]
        assert hits["Column"].tolist() == ["Name", "Name"]

    def test_regex_and_case(self) -> None:
        hits = search_csv_frame(self._frame(), "^A.*e$", use_regex=True).unwrap().to_frame()
        assert hits["Row"].tolist() == [1]
        hits_cs = search_csv_frame(self._frame(), "ALICE", case_sensitive=True).unwrap().to_frame()
        assert hits_cs.empty

    def test_bad_regex_and_empty_query(self) -> None:
        assert any(
            d.code == "SEARCH_BAD_REGEX" for d in search_csv_frame(self._frame(), "([", use_regex=True).diagnostics
        )
        assert any(d.code == "SEARCH_EMPTY_QUERY" for d in search_csv_frame(self._frame(), "").diagnostics)

    def test_text_search_still_ok(self) -> None:
        assert search_in_text("hello\nworld", "o").unwrap() == [(1, "hello"), (2, "world")]


class TestCli:
    def test_text_mode(self, tmp_path: Path) -> None:
        from tools.search import main

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text("the cat sat\non the mat", encoding="utf-8")
        out = tmp_path / "out"
        assert main(["text", str(corpus), str(out), "--query", "cat"]) == 0
        run_dir = next(out.iterdir())
        hits = pd.read_csv(run_dir / "text_hits.csv")
        assert len(hits) == 1 and hits.iloc[0]["Document"] == "a.txt"

    def test_csv_mode(self, tmp_path: Path) -> None:
        from tools.search import main

        src = tmp_path / "src"
        src.mkdir()
        csv = src / "data.csv"
        pd.DataFrame({"Name": ["Alice", "Bob"]}).to_csv(csv, index=False)
        out = tmp_path / "out"
        assert main(["csv", str(csv), str(out), "--query", "ali"]) == 0
        assert (next(out.iterdir()) / "csv_hits.csv").is_file()

    def test_missing_input(self, tmp_path: Path) -> None:
        from tools.search import main

        assert main(["text", str(tmp_path / "ghost"), str(tmp_path / "out"), "--query", "x"]) == 2
        assert main(["csv", str(tmp_path / "ghost.csv"), str(tmp_path / "out"), "--query", "x"]) == 2
