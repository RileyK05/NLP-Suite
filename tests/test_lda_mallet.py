"""lda_mallet: the MALLET half of the LDA comparison (CAP-TOPIC-03).

MALLET is a Java binary outside this suite, so most of this is the contract
around it: the tool is registered and separate from lda_gensim (HW2 grades
them against each other), a missing binary fails loudly with its install
pointer (never a silent substitution), and ``--output-doc-topics`` is read
into the same dominant-topic shape the Gensim tool emits, so the comparison
is a table comparison and not a format puzzle.

The live-run path for the binary is covered in tests/test_backends.py
(binary absent, bad output dir, bad keys). Here: the document-composition
parser and the registration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.analysis.mallet import parse_doc_topics
from core.profiler.registry import get_tool

FIXTURES = Path(__file__).parent / "fixtures" / "topics"


class TestDocTopicsParsing:
    def test_reads_topic_proportion_pairs(self) -> None:
        # MALLET 2.0.x style: source name then topic:proportion pairs.
        result = parse_doc_topics(FIXTURES / "mallet_doc_topics_pairs.txt")
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert list(frame.columns) == [
            "Document",
            "Dominant topic",
            "Contribution",
            "Topic proportions",
        ]
        first = frame.iloc[0]
        assert first["Document"].endswith("first.txt")
        assert int(first["Dominant topic"]) == 0
        assert float(first["Contribution"]) == 0.9032
        second = frame.iloc[1]
        assert int(second["Dominant topic"]) == 1
        assert float(second["Contribution"]) == 0.75

    def test_reads_alternating_topic_proportion_columns(self) -> None:
        # Older MALLETs alternate bare numbers: doc index, name, then
        # topic proportion pairs without colons.
        result = parse_doc_topics(FIXTURES / "mallet_doc_topics_columns.txt")
        assert result.ok, result.diagnostics
        row = result.unwrap().iloc[0]
        assert row["Document"] == "docs/third.txt"
        assert int(row["Dominant topic"]) == 0
        assert float(row["Contribution"]) == 0.6

    def test_skips_comments_and_blank_lines(self) -> None:
        result = parse_doc_topics(FIXTURES / "mallet_doc_topics_commented.txt")
        assert result.ok
        assert len(result.unwrap()) == 1

    def test_an_unreadable_format_is_a_diagnostic_not_an_empty_table(self) -> None:
        # "No document leans on any topic" and "we could not read the file"
        # are different findings; HW2 would read an empty table as the first.
        result = parse_doc_topics(FIXTURES / "mallet_doc_topics_unreadable.txt")
        assert not result.ok
        assert result.diagnostics[0].code == "MALLET_BAD_DOCS"

    def test_a_missing_file_is_a_diagnostic(self) -> None:
        result = parse_doc_topics(FIXTURES / "mallet_doc_topics_absent.txt")
        assert not result.ok
        assert result.diagnostics[0].code == "MALLET_BAD_DOCS"


class TestRegistration:
    def test_mallet_and_gensim_are_two_tools_in_the_topics_family(self) -> None:
        mallet = get_tool("lda_mallet")
        gensim = get_tool("lda_gensim")
        assert mallet is not None and gensim is not None
        assert mallet.family == gensim.family == "topics"
        assert mallet.name != gensim.name

    def test_mallet_needs_no_parse_and_says_so_in_its_outputs(self) -> None:
        # MALLET tokenizes its own input: the binary is the prerequisite, not
        # the CoNLL table the Gensim path reads.
        mallet = get_tool("lda_mallet")
        assert mallet is not None
        assert mallet.requires_parse is False
        assert mallet.outputs == ("topics.csv", "topics_dominant.csv")

    def test_gensim_carries_the_two_graded_views(self) -> None:
        gensim = get_tool("lda_gensim")
        assert gensim is not None
        assert "terms_by_relevance.csv" in gensim.outputs
        assert "intertopic_map.html" in gensim.outputs
        lam = next(p for p in gensim.params if p.name == "lambda")
        assert lam.minimum == 0.0 and lam.maximum == 1.0
        assert lam.default == 0.6


def test_the_desktop_run_names_documents_as_the_corpus_does(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The desktop's dominant-topic rows say "speech.txt", not the staged "0001_speech.txt.txt"."""
    import pandas as pd

    from core.analysis import mallet
    from core.io.reader import Document
    from core.result import Result
    from desktop_backend.runner import run_mallet

    def fake_train(input_dir: Path, output_dir: Path, **_kwargs: object) -> Result[pd.DataFrame]:
        staged = sorted(Path(input_dir).iterdir())
        rows = [f"{i}\t{path.as_uri()}\t0.7\t0.3" for i, path in enumerate(staged)]
        (Path(output_dir) / "mallet_doc_topics.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        return Result.success(pd.DataFrame({"Topic": [0, 1], "Keys": ["a", "b"]}))

    monkeypatch.setattr(mallet, "train_topics", fake_train)

    class Jobs:
        def __init__(self) -> None:
            self.updates: list[dict[str, object]] = []

        def update_job(self, _job_id: str, **fields: object) -> None:
            self.updates.append(fields)

    stored = tmp_path / "stored"
    stored.mkdir()
    for name, text in (("speech.txt", "one two"), ("notes.txt", "three four")):
        (stored / name).write_text(text, encoding="utf-8")
    documents = [
        Document(doc_id=1, path=stored / "speech.txt", text="one two", sha256="a", label="speech.txt"),
        Document(doc_id=2, path=stored / "notes.txt", text="three four", sha256="b", label="notes.txt"),
    ]
    jobs = Jobs()
    project_dir = tmp_path / "project"
    (project_dir / "runs").mkdir(parents=True)
    assert run_mallet(jobs, {"id": "j"}, {"topics": 2}, documents, project_dir) == 0  # type: ignore[arg-type]
    run_dir = project_dir / str(jobs.updates[-1]["run_dir"])
    dominant = pd.read_csv(run_dir / "topics_dominant.csv")
    assert sorted(dominant["Document"]) == ["notes.txt", "speech.txt"]


def test_a_desktop_run_without_document_topics_is_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MALLET's keys without its document composition is not "Results ready"."""
    import pandas as pd

    from core.analysis import mallet
    from core.io.reader import Document
    from core.result import Result
    from desktop_backend.runner import run_mallet

    def keys_only(_input_dir: Path, _output_dir: Path, **_kwargs: object) -> Result[pd.DataFrame]:
        return Result.success(pd.DataFrame({"Topic": [0], "Keys": ["a"]}))

    monkeypatch.setattr(mallet, "train_topics", keys_only)
    updates: list[dict[str, object]] = []

    class Jobs:
        def update_job(self, _job_id: str, **fields: object) -> None:
            updates.append(fields)

    (tmp_path / "speech.txt").write_text("one two", encoding="utf-8")
    documents = [Document(doc_id=1, path=tmp_path / "speech.txt", text="one two", sha256="a")]
    (tmp_path / "project" / "runs").mkdir(parents=True)
    assert run_mallet(Jobs(), {"id": "j"}, {"topics": 1}, documents, tmp_path / "project") == 1  # type: ignore[arg-type]
    assert updates[-1]["state"] == "PARTIAL"
    assert updates[-1]["diagnostics"], "the reason the document topics are missing is kept"
