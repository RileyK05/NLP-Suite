"""Regressions found in the post-implementation review (no model downloads)."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import pytest

from core.conll.schema import Col
from core.io.reader import Corpus, Document, corpus_fingerprint, hash_text
from core.profiler import executor, resume
from core.profiler.batch import BatchRequest, write_batch
from core.profiler.executor import BatchResult, ToolOutcome
from core.profiler.plan import build_plan
from core.profiler.registry import get_tool, validate_specs
from core.result import Diagnostic, Result


def _corpus() -> Corpus:
    docs = (Document(doc_id=1, path=Path("/corpus/a.txt"), text="Hello world.", sha256=hash_text("Hello world.")),)
    return Corpus(docs=docs, sha256=corpus_fingerprint(docs))


def _batch(root: Path) -> Path:
    outcome = ToolOutcome("readability", True, {"readability.csv": pd.DataFrame({"score": [1]})})
    return (
        write_batch(
            root,
            BatchRequest(
                selection=("readability",),
                params={"readability": {}},
                batch=BatchResult((outcome,)),
                corpus=_corpus(),
            ),
        )
        .unwrap()
        .run_dir
    )


def test_background_cli_records_completion_after_submitter_exits(tmp_path: Path) -> None:
    from core.jobs import read_status

    root = tmp_path / "jobs"
    submitted = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "tools.jobs", "--jobs-root", str(root), "submit", "doctor", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    job_id = submitted.stdout.strip()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = read_status(job_id, jobs_root=str(root)).unwrap()
        if status.state != "RUNNING":
            break
        time.sleep(0.05)
    assert status.state == "DONE"
    assert status.exit_code == 0
    assert isinstance(status.argv, tuple)
    assert "usage:" in (root / job_id / "stdout.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("job_id", ["..", "../other", "C:\\other", "a\\b", "/absolute"])
def test_job_ids_cannot_escape_root(tmp_path: Path, job_id: str) -> None:
    from core.jobs import read_status

    assert not read_status(job_id, jobs_root=str(tmp_path)).ok


def test_jobs_start_failure_is_not_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import jobs
    from tools.jobs import main

    def fail(*args, **kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr(jobs.subprocess, "Popen", fail)
    assert main(["--jobs-root", str(tmp_path), "submit", "doctor", "--help"]) == 1
    assert jobs.list_jobs(jobs_root=str(tmp_path)).unwrap()[0].state == "FAILED"


def test_malformed_job_status_returns_diagnostic(tmp_path: Path) -> None:
    from core.jobs import read_status

    (tmp_path / "job").mkdir()
    (tmp_path / "job" / "status.json").write_text("[]", encoding="utf-8")
    assert not read_status("job", jobs_root=str(tmp_path)).ok


@pytest.mark.parametrize("damage", ["delete", "modify", "envelope", "escape"])
def test_resume_rejects_damaged_children(tmp_path: Path, damage: str) -> None:
    prior = _batch(tmp_path)
    manifest_path = prior / "batch.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["children"][0]
    assert resume.find_reusable(prior, ["readability"], {"readability": {}}, _corpus())
    artifact = tmp_path / record["artifacts"][0]
    if damage == "delete":
        artifact.unlink()
    elif damage == "modify":
        artifact.write_text("score\n9\n", encoding="utf-8")
    elif damage == "envelope":
        (tmp_path / record["run_dir"] / "result.json").write_text("{}", encoding="utf-8")
    else:
        record["run_dir"] = "../outside"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert resume.find_reusable(prior, ["readability"], {"readability": {}}, _corpus()) == {}


def test_resume_rejects_other_output_root(tmp_path: Path) -> None:
    prior = _batch(tmp_path / "old")
    assert (
        resume.find_reusable(prior, ["readability"], {"readability": {}}, _corpus(), output_root=tmp_path / "new") == {}
    )


@pytest.mark.parametrize("children", [None, 3, {}, [{"tool": [], "ok": True}]])
def test_resume_malformed_manifest_is_cache_miss(tmp_path: Path, children: object) -> None:
    (tmp_path / "batch.json").write_text(json.dumps({"children": children}), encoding="utf-8")
    assert resume.find_reusable(tmp_path, ["readability"], {}, _corpus()) == {}


def test_resume_tracks_file_contents(tmp_path: Path) -> None:
    wordlist = tmp_path / "words.txt"
    wordlist.write_text("cat", encoding="utf-8")
    params = {"wordlist": str(wordlist)}
    first = resume.current_key("spellcheck", params, _corpus())
    wordlist.write_text("dog", encoding="utf-8")
    assert first is not None and first != resume.current_key("spellcheck", params, _corpus())


def test_resume_requires_known_input_and_model_identity() -> None:
    assert resume.current_key("readability", {}, None) is None
    assert resume.current_key("nrc", {}, _corpus()) is None
    assert resume.current_key("sentence_complexity", {}, _corpus()) is None


def test_profiler_resume_never_runs_cached_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools import profiler

    corpus_dir = tmp_path / "input"
    corpus_dir.mkdir()
    (corpus_dir / "a.txt").write_text("Hello world.", encoding="utf-8")
    out = tmp_path / "out"
    calls = []

    def spy(ctx, params):
        calls.append("readability")
        return Result.success({"readability.csv": pd.DataFrame({"score": [1]})})

    monkeypatch.setitem(executor.ADAPTERS, "readability", spy)
    argv = [str(corpus_dir), str(out), "--analyses", "readability"]
    assert profiler.main(argv) == 0
    prior = next(out.glob("profiler__*/batch.json"))
    assert profiler.main([*argv, "--resume-from", str(prior)]) == 0
    assert calls == ["readability"]
    assert profiler.main([*argv, "--resume-from", str(tmp_path / "missing")]) == 0
    assert calls == ["readability", "readability"]


def test_default_profiler_selection_is_valid() -> None:
    from tools.profiler import _default_selection

    assert build_plan(_default_selection()).ok


def test_search_conll_plan_requires_parse() -> None:
    assert build_plan(["search"], {"search": {"mode": "conll", "query": "x"}}).unwrap().needs_parse
    assert not build_plan(["search"], {"search": {"mode": "text", "query": "x"}}).unwrap().needs_parse


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_plan_rejects_nonfinite_parameters(value: float) -> None:
    assert not build_plan(["doc_similarity"], {"doc_similarity": {"threshold": value}}).ok


def test_registry_rejects_duplicate_widget_keys() -> None:
    spec = get_tool("sentiment_vader_anew")
    assert spec is not None
    assert len(spec.params) == len({param.name for param in spec.params})
    bad = replace(spec, params=(*spec.params, spec.params[0]))
    assert any(diag.code == "REGISTRY_DUPLICATE_PARAM" for diag in validate_specs((bad,)))


def test_partial_analysis_is_failed_but_artifacts_survive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def partial(ctx, params):
        return Result.success(
            {"readability.csv": pd.DataFrame({"score": [1]})}, Diagnostic.error("PARTIAL_TEST", "one document failed")
        )

    monkeypatch.setitem(executor.ADAPTERS, "readability", partial)
    batch = executor.execute(build_plan(["readability"]).unwrap(), corpus=_corpus())
    assert not batch.outcomes[0].ok
    report = write_batch(tmp_path, BatchRequest(("readability",), {}, batch, _corpus())).unwrap()
    manifest = json.loads((report.run_dir / "batch.json").read_text(encoding="utf-8"))
    assert manifest["children"][0]["ok"] is False
    assert (report.child_dirs["readability"] / "readability.csv").is_file()
    assert report.envelope.diagnostics[0].code == "PARTIAL_TEST"
    assert resume.find_reusable(report.run_dir, ["readability"], {}, _corpus()) == {}


@pytest.mark.parametrize(
    "tool,function,path_key",
    [
        ("nrc", "emotions", "lexicon"),
        ("sentiment_swn_hedono", "hedonometer", "hedonometer-lexicon"),
    ],
)
def test_batch_lexicon_options_reach_analysis(monkeypatch, tool, function, path_key) -> None:
    calls = []

    def spy(frame, **kwargs):
        calls.append(kwargs)
        return Result.success(pd.DataFrame())

    monkeypatch.setattr(executor, function, spy)
    monkeypatch.setattr(executor, "sentiwordnet", lambda *args, **kwargs: Result.success(pd.DataFrame()))
    plan = build_plan([tool], {tool: {"field": "form", path_key: Path("custom.json")}}).unwrap()
    assert executor.execute(plan, table=pd.DataFrame()).outcomes[0].ok
    assert calls == [{"field": Col.FORM, "lexicon": "custom.json"}]


def test_batch_vader_options_reach_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    analyzer = object()

    def load(path):
        calls.append(("load", path))
        return Result.success(analyzer)

    def score(frame, **kwargs):
        calls.append(kwargs)
        return Result.success(pd.DataFrame())

    monkeypatch.setattr(executor, "load_vader_analyzer", load)
    for name in ("vader", "vader_sentences", "anew"):
        monkeypatch.setattr(executor, name, score)
    plan = build_plan(
        ["sentiment_vader_anew"],
        {
            "sentiment_vader_anew": {
                "field": "lemma",
                "anew-field": "form",
                "vader-lexicon": "vader.txt",
                "anew-lexicon": "anew.csv",
            }
        },
    ).unwrap()
    assert executor.execute(plan, table=pd.DataFrame()).outcomes[0].ok
    assert calls == [
        ("load", "vader.txt"),
        {"field": Col.LEMMA, "analyzer": analyzer},
        {"field": Col.LEMMA, "analyzer": analyzer},
        {"field": Col.FORM, "lexicon": "anew.csv"},
    ]


def test_blocked_rename_chain_preserves_all_sources(tmp_path: Path) -> None:
    from core.file_ops.filenames import RenamePlan, RenameRow, apply_renames

    for name in ("a", "b", "c", "d"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    plan = RenamePlan(
        tuple(RenameRow(old, new, "", "rename") for old, new in (("a", "b"), ("b", "c"), ("d", "e"))), tmp_path
    )
    result = apply_renames(plan, dry_run=False)
    assert result.unwrap() == [("d", "e")]
    assert [(tmp_path / name).read_text(encoding="utf-8") for name in ("a", "b", "c", "e")] == ["a", "b", "c", "d"]
    assert len(result.warnings) == 2


def test_duplicate_rename_destinations_are_rejected(tmp_path: Path) -> None:
    from core.file_ops.filenames import RenamePlan, RenameRow, apply_renames

    for name in ("a", "b"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    plan = RenamePlan((RenameRow("a", "c", "", "rename"), RenameRow("b", "c", "", "rename")), tmp_path)
    assert not apply_renames(plan, dry_run=False).ok
    assert (tmp_path / "a").read_text(encoding="utf-8") == "a"
    assert (tmp_path / "b").read_text(encoding="utf-8") == "b"


def test_unordered_tolerance_uses_complete_matching() -> None:
    from core.compare import CompareConfig, compare_frames

    first = pd.DataFrame({"value": [0.1, 0.0]})
    second = pd.DataFrame({"value": [0.05, 0.15]})
    config = CompareConfig(rtol=0, atol=0.11)
    assert compare_frames(first, second, config).ok
    assert compare_frames(first, second.iloc[::-1], config).ok


def test_run_comparison_cannot_pass_csv_schema_error(tmp_path: Path) -> None:
    from core.compare import compare_runs
    from core.io.writer import OutputWriter

    dirs = []
    for column in ("expected", "wrong"):
        writer = OutputWriter(tmp_path, tool="test", params={})
        assert writer.write_table(pd.DataFrame({column: [1]}), "table.csv").ok
        assert writer.finalize().ok
        dirs.append(writer.run_dir)
    result = compare_runs(*dirs)
    assert not result.ok
    assert result.value is None or not result.unwrap().passed
    assert not any(diag.code == "COMPARE_MATCH" for diag in result.diagnostics)


def test_writer_reserves_envelope_filename(tmp_path: Path) -> None:
    from core.io.writer import OutputWriter

    writer = OutputWriter(tmp_path, tool="test", params={})
    assert not writer.write_json({}, "result.json").ok
    assert not writer.write_text("oops", "./result.json").ok
    assert writer.finalize().ok
    assert writer.artifacts == ()


def test_writer_never_overwrites_existing_sidecar(tmp_path: Path) -> None:
    from core.conll.schema import CoNLLSchema
    from core.io.writer import OutputWriter

    writer = OutputWriter(tmp_path, tool="test", params={})
    assert writer.write_text("original", "table.csv.schema.json").ok
    assert not writer.write_table(pd.DataFrame({"x": [1]}), "table.csv", schema=CoNLLSchema()).ok
    assert (writer.run_dir / "table.csv.schema.json").read_text(encoding="utf-8") == "original"
    assert not (writer.run_dir / "table.csv").exists()
    writer.abandon()


def test_partial_parse_marks_only_dependent_outcomes_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    def score(ctx, params):
        return Result.success({"score.csv": pd.DataFrame({"score": [1]})})

    for name in ("readability", "sentence_complexity"):
        monkeypatch.setitem(executor.ADAPTERS, name, score)
    batch = executor.execute(
        build_plan(["readability", "sentence_complexity"]).unwrap(),
        corpus=_corpus(),
        table=pd.DataFrame(),
        parse_diagnostics=(Diagnostic.error("PARSE_FAILED", "document 2 failed"),),
    )
    raw, parsed = batch.outcomes
    assert raw.ok and raw.diagnostics == ()
    assert not parsed.ok and parsed.frames
    assert parsed.diagnostics[0].code == "PARSE_FAILED"


def test_profiler_cli_accepts_required_tool_params(tmp_path: Path) -> None:
    from tools.profiler import main

    corpus = tmp_path / "input"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Hello World.", encoding="utf-8")
    params = tmp_path / "params.json"
    params.write_text(
        json.dumps({"search": {"mode": "text", "query": "World", "case-sensitive": True}}), encoding="utf-8"
    )
    out = tmp_path / "out"
    assert main([str(corpus), str(out), "--analyses", "search", "--params", str(params)]) == 0
    output = next(out.glob("search__*/text_hits.csv"))
    assert "Hello World." in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("with_child", [False, True])
def test_batch_invalid_output_returns_diagnostic(tmp_path: Path, with_child: bool) -> None:
    invalid = tmp_path / "file"
    invalid.write_text("keep me", encoding="utf-8")
    outcomes = (ToolOutcome("readability", True, {"readability.csv": pd.DataFrame()}),) if with_child else ()
    result = write_batch(
        invalid,
        BatchRequest(
            selection=("readability",) if with_child else (),
            params={},
            batch=BatchResult(outcomes),
        ),
    )
    assert not result.ok
    assert result.diagnostics[0].code == "BATCH_WRITE_FAILED"
    assert invalid.read_text(encoding="utf-8") == "keep me"
