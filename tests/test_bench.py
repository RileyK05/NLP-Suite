"""FR-9.7 — benchmark contract tests (schema, never budgets)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.bench import main

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini-corpus"


class TestBench:
    def test_bench_writes_schema(self, tmp_path: Path) -> None:
        target = tmp_path / "bench.json"
        code = main([str(FIXTURE), "--out", str(target), "--analyses", "readability,lexical_diversity"])
        assert code == 0
        record = json.loads(target.read_text(encoding="utf-8"))
        assert record["docs"] > 0 and record["tokens"] > 0
        assert set(record["stages"]) == {"read_corpus", "parse"}
        for name in ("readability", "lexical_diversity"):
            assert record["analyses"][name]["seconds"] >= 0

    def test_unknown_analysis_recorded_not_crash(self, tmp_path: Path) -> None:
        target = tmp_path / "bench.json"
        assert main([str(FIXTURE), "--out", str(target), "--analyses", "nope"]) == 0
        record = json.loads(target.read_text(encoding="utf-8"))
        assert record["analyses"]["nope"]["error"] == "BENCH_UNKNOWN_ANALYSIS"

    def test_missing_corpus_fails(self, tmp_path: Path) -> None:
        assert main([str(tmp_path / "nope"), "--out", str(tmp_path / "b.json")]) == 1
