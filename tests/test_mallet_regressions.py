"""Regression coverage for MALLET's launcher and 2.0.8 output contract."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from core.analysis import mallet


def _fake_mallet_run(monkeypatch: pytest.MonkeyPatch, output: Path, calls: list[dict[str, object]]) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append({"argv": argv, **kwargs})
        if argv[1] == "train-topics":
            (output / "mallet_topic_keys.txt").write_text("0\t0.5\talpha beta\n1\t0.5\tgamma delta\n", encoding="utf-8")
            (output / "mallet_doc_topics.txt").write_text(
                "0\tfile:/C:/temp/0001_first.txt\t0.75\t0.25\n", encoding="utf-8"
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(mallet.subprocess, "run", run)


class TestMalletCommandContract:
    def test_import_precedes_train_and_preserves_sequence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_dir = tmp_path / "input"
        output = tmp_path / "output"
        input_dir.mkdir()
        output.mkdir()
        calls: list[dict[str, object]] = []
        _fake_mallet_run(monkeypatch, output, calls)

        result = mallet.train_topics(input_dir, output, n_topics=2, seed=7, binary=str(tmp_path / "mallet"))

        assert result.ok, result.diagnostics
        assert [call["argv"][1] for call in calls] == ["import-dir", "train-topics"]
        import_argv = calls[0]["argv"]
        train_argv = calls[1]["argv"]
        assert "--keep-sequence" in import_argv
        assert "--remove-stopwords" in import_argv
        assert train_argv[train_argv.index("--input") + 1] == str(output / "corpus.mallet")
        assert train_argv[train_argv.index("--optimize-interval") + 1] == "20"
        assert train_argv[train_argv.index("--num-iterations") + 1] == "1000"

    def test_mallet_home_install_is_discovered_and_passed_to_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "mallet"
        launcher_dir = home / "bin"
        launcher_dir.mkdir(parents=True)
        (launcher_dir / "mallet.bat").write_text("@echo off\n", encoding="utf-8")
        monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
        monkeypatch.setenv("MALLET_HOME", str(home))
        output = tmp_path / "output"
        output.mkdir()
        calls: list[dict[str, object]] = []
        _fake_mallet_run(monkeypatch, output, calls)

        assert mallet.mallet_binary() == str(launcher_dir / "mallet.bat")
        result = mallet.train_topics(tmp_path, output, n_topics=2)

        assert result.ok, result.diagnostics
        assert all(call["env"]["MALLET_HOME"] == str(home) for call in calls)


class TestMallet208DocumentTopics:
    def test_reads_two_topic_proportions_in_topic_order(self, tmp_path: Path) -> None:
        source = tmp_path / "mallet_doc_topics.txt"
        source.write_text("0\tfile:/C:/temp/0001_franklin%20d%20roosevelt.txt\t0.857\t0.143\n", encoding="utf-8")

        result = mallet.parse_doc_topics(source, n_topics=2, document_map={1: "franklin d roosevelt.txt"})

        assert result.ok, result.diagnostics
        row = result.unwrap().iloc[0]
        assert row["Document"] == "franklin d roosevelt.txt"
        assert int(row["Dominant topic"]) == 0
        assert float(row["Contribution"]) == 0.857
        assert row["Topic proportions"] == "0:0.857, 1:0.143"

    def test_reads_nine_topic_proportions_without_index_shift(self, tmp_path: Path) -> None:
        source = tmp_path / "mallet_doc_topics.txt"
        proportions = [0.02, 0.03, 0.04, 0.7, 0.05, 0.04, 0.03, 0.05, 0.04]
        source.write_text(
            "0\tfile:/C:/temp/0001_nine.txt\t" + "\t".join(str(value) for value in proportions) + "\n",
            encoding="utf-8",
        )

        result = mallet.parse_doc_topics(source, n_topics=9)

        assert result.ok, result.diagnostics
        row = result.unwrap().iloc[0]
        assert row["Document"] == "nine.txt"
        assert int(row["Dominant topic"]) == 3
        assert float(row["Contribution"]) == 0.7


@pytest.mark.model_integration
def test_real_mallet_cli_round_trip(tmp_path: Path) -> None:
    """Exercise import-dir, training, 2.0.8 parsing, and name mapping together."""
    binary = mallet.mallet_binary()
    if binary is None:
        pytest.skip("MALLET is not installed (set MALLET_HOME or add MALLET_HOME/bin to PATH)")

    corpus = tmp_path / "corpus"
    output = tmp_path / "output"
    corpus.mkdir()
    (corpus / "first.txt").write_text(
        "government policy economy public service nation people law reform " * 8, encoding="utf-8"
    )
    (corpus / "second.txt").write_text(
        "science research health education climate technology university " * 8, encoding="utf-8"
    )
    (corpus / "third.txt").write_text(
        "government science policy research public education economy health " * 8, encoding="utf-8"
    )

    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "tools.lda_mallet", str(corpus), str(output), "--topics", "2", "--seed", "42"],
        cwd=Path(__file__).parents[1],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    artifacts = list(output.glob("lda_mallet__*/topics_dominant.csv"))
    assert len(artifacts) == 1
    frame = pd.read_csv(artifacts[0])
    assert set(frame["Document"]) == {"first.txt", "second.txt", "third.txt"}
    for proportions in frame["Topic proportions"]:
        total = sum(float(part.split(":", 1)[1]) for part in str(proportions).split(", "))
        assert total == pytest.approx(1.0, abs=0.001)
