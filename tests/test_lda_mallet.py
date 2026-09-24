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
