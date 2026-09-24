"""FR-7 child 4 (C2) — batch envelope + CLI selection contract tests.

Per-tool run dirs, a parent batch.json linking children explicitly (never
globbed), failures recorded without child dirs. Offline except the marked
CLI replay. Fails until core/profiler/batch.py lands (C3).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.profiler.executor import BatchResult, ToolOutcome
from core.result import Diagnostic


def _batch() -> BatchResult:
    ok_frame = pd.DataFrame([{"Word": "cat", "Count": 2}])
    return BatchResult(
        outcomes=(
            ToolOutcome(name="readability", ok=True, frames={"readability.csv": ok_frame}, seconds=0.5),
            ToolOutcome(
                name="nrc",
                ok=False,
                seconds=1.0,
                diagnostics=(Diagnostic.error("NRC_LEXICON_MISSING", "no lexicon"),),
            ),
        )
    )


class TestBatchEnvelopes:
    def test_children_linked_explicitly(self, tmp_path: Path) -> None:
        from core.profiler import batch as batch_mod

        plan_params = {"readability": {}, "nrc": {}}
        report = batch_mod.write_batch(
            tmp_path,
            batch_mod.BatchRequest(selection=("readability", "nrc"), params=plan_params, batch=_batch()),
        )
        assert report.ok, report.diagnostics
        info = report.unwrap()
        assert (info.run_dir / "batch.json").is_file()
        assert (info.run_dir / "result.json").is_file()
        manifest = json.loads((info.run_dir / "batch.json").read_text(encoding="utf-8"))
        assert [c["tool"] for c in manifest["children"]] == ["readability", "nrc"]
        assert [c["ok"] for c in manifest["children"]] == [True, False]
        # every linked path resolves to a real file (output-root-relative, no globbing needed)
        for child in manifest["children"]:
            for artifact in child["artifacts"]:
                assert (tmp_path / artifact).is_file(), artifact
        # failed tool has no child dir, but its error is recorded
        assert manifest["children"][1]["diagnostics"][0]["code"] == "NRC_LEXICON_MISSING"
        assert info.child_dirs["readability"].is_dir()
        assert "nrc" not in info.child_dirs

    def test_reuse_links_prior_record_without_rerun(self, tmp_path: Path) -> None:
        from core.profiler import batch as batch_mod

        prior = {
            "tool": "readability",
            "ok": True,
            "seconds": 2.0,
            "run_dir": "readability__old",
            "artifacts": ["readability__old/readability.csv"],
            "diagnostics": [],
            "reuse_key": "k",
        }
        report = batch_mod.write_batch(
            tmp_path,
            batch_mod.BatchRequest(
                selection=("readability",),
                params={"readability": {}},
                batch=BatchResult(outcomes=()),
                reuse={"readability": prior},
            ),
        )
        assert report.ok, report.diagnostics
        info = report.unwrap()
        manifest = json.loads((info.run_dir / "batch.json").read_text(encoding="utf-8"))
        (child,) = manifest["children"]
        assert child["reused"] is True and child["run_dir"] == "readability__old"
        assert info.child_dirs == {}

    def test_empty_batch(self, tmp_path: Path) -> None:
        from core.profiler import batch as batch_mod

        report = batch_mod.write_batch(
            tmp_path, batch_mod.BatchRequest(selection=(), params={}, batch=BatchResult(outcomes=()))
        )
        assert report.ok, report.diagnostics
        manifest = json.loads((report.unwrap().run_dir / "batch.json").read_text(encoding="utf-8"))
        assert manifest["children"] == []
