"""How a result table names a document, and why it never names it by full path.

Found by reading a real ``doc_similarity`` run, whose readout said:

    The closest pair is C:/Users/.../Temp/claude/.../scratchpad/corp/copy.txt
    and C:/Users/.../Temp/claude/.../scratchpad/corp/long.txt.

Three problems in one line. It is unreadable; it publishes the machine's
directory layout into output that gets shared and handed in; and it was
inconsistent -- the parser backends, ``readability`` and ``lexical_diversity``
had always written the basename, while ``doc_similarity`` and ``spellcheck``
wrote absolute paths and ``doc_duplicates`` wrote ``id:absolute/path`` when a
basename was ambiguous.

A short name is not enough on its own, though: a corpus of ``2020/report.txt``
and ``2021/report.txt`` must not collapse into one label, or every table keyed
by document name silently merges two documents. So the rule is the shortest
label that is still unique, and both halves are tested here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.analysis import doc_duplicates, doc_similarity, lexical_diversity, readability, spellcheck
from core.io.reader import Corpus, Document, display_names, hash_text

TEXTS = (
    "The harvest was late that year and the soil stayed wet right through the spring months.",
    "Markets in the valley reported lower prices for grain, for hay, and for winter feed.",
    "Farmers argued about drainage, about debt, and about the rising price of certified seed.",
)


def _corpus(paths: list[str], texts: tuple[str, ...] = TEXTS) -> Corpus:
    docs = tuple(
        Document(doc_id=i + 1, path=Path(path), text=texts[i % len(texts)], sha256=hash_text(str(i)))
        for i, path in enumerate(paths)
    )
    return Corpus(docs=docs, sha256="corpus")


class TestDisplayNames:
    def test_a_unique_basename_is_the_whole_label(self) -> None:
        corpus = _corpus(["C:/corpora/study/alpha.txt", "C:/corpora/study/beta.txt"])
        assert list(display_names(corpus.docs).values()) == ["alpha.txt", "beta.txt"]

    def test_a_repeated_basename_keeps_enough_path_to_stay_distinct(self) -> None:
        corpus = _corpus(["C:/corpora/2020/report.txt", "C:/corpora/2021/report.txt"])
        assert list(display_names(corpus.docs).values()) == ["2020/report.txt", "2021/report.txt"]

    def test_only_the_ambiguous_documents_pay_for_the_ambiguity(self) -> None:
        corpus = _corpus(
            ["C:/corpora/2020/report.txt", "C:/corpora/2021/report.txt", "C:/corpora/notes.txt"],
        )
        assert list(display_names(corpus.docs).values()) == ["2020/report.txt", "2021/report.txt", "notes.txt"]

    def test_the_corpus_root_is_stripped_however_deep_it_is(self) -> None:
        corpus = _corpus(
            ["C:/a/b/c/d/x/r.txt", "C:/a/b/c/d/y/r.txt", "C:/a/b/c/d/z/r.txt"],
        )
        assert list(display_names(corpus.docs).values()) == ["x/r.txt", "y/r.txt", "z/r.txt"]

    def test_paths_with_no_shared_root_keep_their_full_path(self) -> None:
        """Two drives have nothing to strip, and a wrong label is worse than a long one."""
        corpus = _corpus(["C:/one/r.txt", "D:/two/r.txt"])
        assert set(display_names(corpus.docs).values()) == {"C:/one/r.txt", "D:/two/r.txt"}

    def test_the_same_file_listed_twice_is_still_two_labels(self) -> None:
        corpus = _corpus(["C:/corpora/r.txt", "C:/corpora/r.txt"])
        labels = list(display_names(corpus.docs).values())
        assert labels == ["r.txt (#1)", "r.txt (#2)"]
        assert len(set(labels)) == 2

    def test_a_single_document_is_named_by_its_basename(self) -> None:
        corpus = _corpus(["C:/very/deeply/nested/only.txt"], texts=(TEXTS[0],))
        assert display_names(corpus.docs) == {1: "only.txt"}

    def test_labels_are_unique_whatever_the_corpus(self) -> None:
        corpus = _corpus(
            [
                "C:/c/a/r.txt",
                "C:/c/b/r.txt",
                "C:/c/r.txt",
                "C:/c/a/notes.txt",
                "C:/c/a/r.txt",
            ]
        )
        labels = list(display_names(corpus.docs).values())
        assert len(set(labels)) == len(labels), labels

    def test_the_same_corpus_always_gets_the_same_labels(self) -> None:
        """R6: a label is a pure function of the corpus, not of call order."""
        paths = ["C:/c/2020/r.txt", "C:/c/2021/r.txt", "C:/c/n.txt"]
        assert display_names(_corpus(paths).docs) == display_names(_corpus(paths).docs)


def _document_columns(frame: pd.DataFrame) -> list[str]:
    """Columns that name documents, whatever the tool calls them."""
    return [str(c) for c in frame.columns if "Document" in str(c) or str(c) == "Members"]


@pytest.fixture
def absolute_corpus(tmp_path: Path) -> Corpus:
    """A corpus read from absolute paths, which is what every real run has."""
    root = tmp_path / "corpus"
    root.mkdir()
    docs = []
    for index, text in enumerate(TEXTS):
        path = root / f"doc{index + 1}.txt"
        path.write_text(text, encoding="utf-8")
        docs.append(Document(doc_id=index + 1, path=path, text=text, sha256=hash_text(text)))
    return Corpus(docs=tuple(docs), sha256="corpus")


class TestNoResultTableLeaksAPath:
    """The guard for the class of defect, not just the instance of it.

    Every one of these ran on a corpus whose documents sit at absolute paths,
    and no cell in any of their document columns may contain one.
    """

    def _assert_clean(self, frame: pd.DataFrame, root: Path) -> None:
        columns = _document_columns(frame)
        assert columns, f"no document column found in {list(frame.columns)}"
        for column in columns:
            for value in frame[column].tolist():
                text = str(value)
                assert str(root) not in text, f"{column} leaks the corpus path: {text!r}"
                assert "/Temp/" not in text and ":\\" not in text, f"{column} holds a filesystem path: {text!r}"

    def test_doc_similarity(self, absolute_corpus: Corpus, tmp_path: Path) -> None:
        frame = doc_similarity.pairwise_similarity(absolute_corpus).unwrap().to_frame()
        self._assert_clean(frame, tmp_path)

    def test_doc_similarity_empty_document_warning(self, tmp_path: Path) -> None:
        """The diagnostic text is read by people too."""
        path = tmp_path / "corpus" / "blank.txt"
        docs = (
            Document(doc_id=1, path=tmp_path / "corpus" / "a.txt", text=TEXTS[0], sha256=hash_text("a")),
            Document(doc_id=2, path=path, text="   ", sha256=hash_text("")),
            Document(doc_id=3, path=tmp_path / "corpus" / "b.txt", text=TEXTS[1], sha256=hash_text("b")),
        )
        result = doc_similarity.pairwise_similarity(Corpus(docs=docs, sha256="c"))
        warning = next(d for d in result.diagnostics if d.code == "SIM_EMPTY_DOC")
        assert "blank.txt is empty" in warning.message
        assert str(tmp_path) not in warning.message

    def test_doc_duplicates_groups(self, tmp_path: Path) -> None:
        root = tmp_path / "corpus"
        docs = tuple(
            Document(doc_id=i + 1, path=root / name, text=TEXTS[0], sha256=hash_text("same"))
            for i, name in enumerate(["a.txt", "b.txt"])
        )
        frame = doc_duplicates.exact_groups(Corpus(docs=docs, sha256="c")).unwrap().to_frame()
        assert frame.iloc[0]["Members"] == "a.txt, b.txt"
        self._assert_clean(frame, tmp_path)

    def test_doc_duplicates_disambiguates_without_a_full_path(self, tmp_path: Path) -> None:
        """This used to read ``1:C:/long/absolute/2020/report.txt``."""
        root = tmp_path / "corpus"
        docs = tuple(
            Document(doc_id=i + 1, path=root / year / "report.txt", text=TEXTS[0], sha256=hash_text("same"))
            for i, year in enumerate(["2020", "2021"])
        )
        frame = doc_duplicates.exact_groups(Corpus(docs=docs, sha256="c")).unwrap().to_frame()
        assert frame.iloc[0]["Members"] == "2020/report.txt, 2021/report.txt"
        self._assert_clean(frame, tmp_path)

    def test_spellcheck(self, absolute_corpus: Corpus, tmp_path: Path) -> None:
        vocab = frozenset({"the", "harvest", "was", "late", "that", "year"})
        frame = spellcheck.run(absolute_corpus, vocab).unwrap().to_frame()
        assert not frame.empty, "expected findings against a deliberately small wordlist"
        self._assert_clean(frame, tmp_path)

    def test_readability(self, absolute_corpus: Corpus, tmp_path: Path) -> None:
        frame = readability.run(absolute_corpus).unwrap().to_frame()
        self._assert_clean(frame, tmp_path)

    def test_lexical_diversity(self, absolute_corpus: Corpus, tmp_path: Path) -> None:
        frame = lexical_diversity.run(absolute_corpus).unwrap().to_frame()
        self._assert_clean(frame, tmp_path)

    def test_the_guard_would_catch_a_leak(self) -> None:
        """A check that cannot fail proves nothing."""
        leaked = pd.DataFrame({"Document": ["C:\\Users\\someone\\corpus\\a.txt"], "Score": [1]})
        with pytest.raises(AssertionError, match="holds a filesystem path"):
            self._assert_clean(leaked, Path("D:/elsewhere"))


class TestTheParsedTableKeepsDocumentsApart:
    """The CoNLL Document column is what every parser-backed tool groups by.

    A bare basename made ``2020/report.txt`` and ``2021/report.txt`` into one
    label, so any table grouped or charted by Document silently merged them.
    Exercised through the fake Stanza pipeline, so it needs no model.
    """

    @staticmethod
    def _parse(paths: list[str]):  # type: ignore[no-untyped-def]
        from test_stanza_fake import FakePipeline, _pipeline

        docs = tuple(
            Document(doc_id=i + 1, path=Path(path), text="Barack Obama visited", sha256=hash_text(path))
            for i, path in enumerate(paths)
        )
        return _pipeline(FakePipeline()).parse(Corpus(docs=docs, sha256="x")).unwrap()

    def test_same_basename_in_two_folders_stays_two_documents(self) -> None:
        frame = self._parse(["C:/corpus/2020/report.txt", "C:/corpus/2021/report.txt"])
        assert sorted(set(frame["Document"])) == ["2020/report.txt", "2021/report.txt"]

    def test_unique_basenames_are_unchanged(self) -> None:
        """The common case must still read exactly as it always has."""
        frame = self._parse(["C:/corpus/alpha.txt", "C:/corpus/beta.txt"])
        assert sorted(set(frame["Document"])) == ["alpha.txt", "beta.txt"]

    def test_the_parsed_table_never_carries_an_absolute_path(self) -> None:
        frame = self._parse(["C:/corpus/deep/a.txt", "C:/corpus/other/b.txt"])
        assert all(":" not in str(v) and "/" not in str(v) for v in set(frame["Document"]))
