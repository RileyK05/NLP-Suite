"""FR-7 child 5 (C2) — resume/cache + report contract tests.

Fingerprint stability, reuse-key sensitivity, manifest-based reuse, and a
failure-transparent report. Offline (hand-built corpora/manifests).
Fails until core/profiler/resume.py + report.py land (C3).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.io.reader import Corpus, Document, corpus_fingerprint, hash_text
from core.profiler.executor import BatchResult, ToolOutcome


def _corpus(texts: dict[str, str]) -> Corpus:
    base = Path("/corpus")
    docs = tuple(
        Document(doc_id=i, path=base / name, text=text, sha256=hash_text(text))
        for i, (name, text) in enumerate(texts.items(), start=1)
    )
    return Corpus(docs=docs, sha256=corpus_fingerprint(docs))


def _manifest(tmp_path: Path, corpus: Corpus, params: dict) -> Path:
    outcomes = (
        ToolOutcome(name="readability", ok=True, frames={"readability.csv": pd.DataFrame([{"a": 1}])}, seconds=0.5),
        ToolOutcome(name="nrc", ok=False, seconds=1.0, diagnostics=()),
    )
    from core.profiler import batch as batch_mod

    report = batch_mod.write_batch(
        tmp_path,
        batch_mod.BatchRequest(
            selection=("readability", "nrc"), params=params, batch=BatchResult(outcomes=outcomes), corpus=corpus
        ),
    )
    assert report.ok, report.diagnostics
    return report.unwrap().run_dir / "batch.json"


class TestResume:
    def test_fingerprint_stable_and_sensitive(self) -> None:
        first = corpus_fingerprint(_corpus({"a.txt": "hello", "b.txt": "world"}).docs)
        assert first == corpus_fingerprint(_corpus({"b.txt": "world", "a.txt": "hello"}).docs)
        assert first != corpus_fingerprint(_corpus({"a.txt": "hello!"}).docs)
        assert _corpus({"a.txt": "hello", "b.txt": "world"}).sha256 == first

    def test_reuse_key_sensitivity(self) -> None:
        from core.profiler import resume as resume_mod

        key = resume_mod.reuse_key("readability", "1", {}, "abc")
        assert key != resume_mod.reuse_key("readability", "1", {"seed": 1}, "abc")
        assert key != resume_mod.reuse_key("readability", "2", {}, "abc")
        assert key != resume_mod.reuse_key("readability", "1", {}, "abd")

    def test_find_reusable_matches(self, tmp_path: Path) -> None:
        from core.profiler import resume as resume_mod

        corpus = _corpus({"a.txt": "hello world"})
        params = {"readability": {}, "nrc": {}}
        manifest = _manifest(tmp_path, corpus, params)
        reuse = resume_mod.find_reusable(manifest, ["readability", "nrc"], params, corpus)
        assert set(reuse) == {"readability"}  # failed nrc is never reused
        assert reuse["readability"]["ok"] is True

    def test_find_reusable_rejects_changed_params(self, tmp_path: Path) -> None:
        from core.profiler import resume as resume_mod

        corpus = _corpus({"a.txt": "hello world"})
        manifest = _manifest(tmp_path, corpus, {"readability": {}})
        assert resume_mod.find_reusable(manifest, ["readability"], {"readability": {"seed": 9}}, corpus) == {}

    def test_find_reusable_rejects_changed_corpus(self, tmp_path: Path) -> None:
        from core.profiler import resume as resume_mod

        manifest = _manifest(tmp_path, _corpus({"a.txt": "hello world"}), {"readability": {}})
        assert (
            resume_mod.find_reusable(manifest, ["readability"], {"readability": {}}, _corpus({"a.txt": "changed"}))
            == {}
        )

    def test_find_reusable_missing_manifest(self, tmp_path: Path) -> None:
        from core.profiler import resume as resume_mod

        assert resume_mod.find_reusable(tmp_path / "nope.json", ["readability"], {}, _corpus({"a": "b"})) == {}


class TestReport:
    def test_failures_are_visible(self) -> None:
        from core.profiler import report as report_mod

        manifest = {
            "tool": "profiler",
            "selection": ["readability", "nrc"],
            "children": [
                {
                    "tool": "readability",
                    "ok": True,
                    "seconds": 0.5,
                    "run_dir": "readability__ts",
                    "artifacts": ["readability__ts/readability.csv"],
                    "diagnostics": [],
                },
                {
                    "tool": "nrc",
                    "ok": False,
                    "seconds": 1.0,
                    "run_dir": None,
                    "artifacts": [],
                    "diagnostics": [
                        {"severity": "ERROR", "code": "NRC_LEXICON_MISSING", "message": "no lexicon", "context": {}}
                    ],
                },
            ],
        }
        text = report_mod.render_report(manifest)
        assert "readability" in text and "nrc" in text
        assert "FAILED" in text and "NRC_LEXICON_MISSING" in text
        assert "readability__ts/readability.csv" in text
        assert "0.5" in text

    def test_reused_children_marked(self) -> None:
        from core.profiler import report as report_mod

        manifest = {
            "tool": "profiler",
            "selection": ["readability"],
            "children": [
                {
                    "tool": "readability",
                    "ok": True,
                    "seconds": 0.0,
                    "run_dir": "readability__old",
                    "artifacts": ["readability__old/readability.csv"],
                    "diagnostics": [],
                    "reused": True,
                }
            ],
        }
        assert "reused" in report_mod.render_report(manifest).lower()
