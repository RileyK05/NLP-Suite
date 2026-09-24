"""Pytest bootstrap and shared fixtures.

Lives at the repo root so that pytest puts the repo root on ``sys.path``,
which is what makes ``import core.result`` resolve inside tests without the
package needing to be installed first.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import pandas as pd
import pytest

from core.pipelines.spacy_backend import spacy_model_name

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# This machine's system ``pytest-of-<user>`` temp directory can reject the
# ``tmp_path`` factory (PermissionError creating it under the system TEMP),
# which fails every ``tmp_path`` test at setup and quietly costs the suite
# that coverage. Point ``tempfile`` at a SHORT fresh subdirectory of the
# system temp instead: the fixture then works everywhere and the poisoned
# ``pytest-of-<user>`` directory is bypassed. It must stay short -- it is a
# prefix on every run directory path, and Windows MAX_PATH (260) is already
# close once ``runs/<tool>__<timestamp>-<uuid>/<artifact>`` is appended.
# Import-time on purpose -- ``tmp_path_factory`` resolves its base through
# ``tempfile.gettempdir()`` after conftest import but before any test body.
_SYSTEM_TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or tempfile.gettempdir())
WORKSPACE_TMP = _SYSTEM_TEMP / "nlp-tmp"
WORKSPACE_TMP.mkdir(exist_ok=True)
tempfile.tempdir = str(WORKSPACE_TMP)


def pytest_configure(config: pytest.Config) -> None:
    # Apple's Accelerate BLAS (numpy's default on macOS arm64) raises the
    # floating-point "invalid" flag on finite matrix products, so numpy warns
    # "invalid value encountered in dot" for correct results, and
    # filterwarnings=error fails the NMF tests on macOS only. The tests' own
    # sign and repeatability assertions still fail on a real NaN.
    if sys.platform == "darwin":
        config.addinivalue_line("filterwarnings", "ignore:invalid value encountered in dot:RuntimeWarning")


def has_spacy_model() -> bool:
    """Whether the default spaCy model is actually loadable.

    The suite hard-fails on missing parser models (no silent degradation),
    so integration tests that need a real parse skip with a pointer to the
    fix instead of failing on an environment problem.
    """
    try:
        import spacy  # allowed late: optional dep, probe runs at test time

        spacy.load(spacy_model_name("en"))
        return True
    except Exception:
        return False


#: The corpus the desktop is actually developed against: 87 State of the Union
#: addresses, 1934-2024. Point this at another folder of .txt files to run the
#: real-corpus tests over different material.
REAL_CORPUS_ENV = "NLP_SUITE_REAL_CORPUS"
#: Where the parse of it is kept between runs. Parsing 600,000 tokens takes
#: most of a minute; every test that needs a real parse shares one, and the
#: cache survives the session so a second run costs a parquet read.
REAL_CACHE_ENV = "NLP_SUITE_REAL_CORPUS_CACHE"


def real_corpus_dir() -> Path | None:
    """The folder of real documents, if this machine has one.

    Fixtures are clean by construction, and the defects these tests exist for
    were all invisible to clean fixtures: a tokenizer disagreement needs text
    containing ``U.S.``, a document-relative position needs documents with many
    sentences, and an alignment failure needs the punctuation real editors use.
    """
    configured = os.environ.get(REAL_CORPUS_ENV)
    candidate = Path(configured) if configured else Path.home() / "Downloads" / "POTUS State of the Union 1934-2024"
    return candidate if candidate.is_dir() and any(candidate.glob("*.txt")) else None


@pytest.fixture(scope="session")
def real_documents() -> list[Path]:
    folder = real_corpus_dir()
    if folder is None:
        pytest.skip(f"No real corpus; set {REAL_CORPUS_ENV} to a folder of .txt documents")
    return sorted(folder.glob("*.txt"))


@pytest.fixture(scope="session")
def real_corpus(real_documents: list[Path]) -> Any:
    from core.io.reader import Corpus, Document, corpus_fingerprint, date_from_filename, hash_text, read_text

    docs = []
    for index, path in enumerate(real_documents, 1):
        text = read_text(path).unwrap()
        docs.append(Document(index, path, text, date_from_filename(path), hash_text(text), f"doc-{index}", path.name))
    frozen = tuple(docs)
    return Corpus(frozen, corpus_fingerprint(frozen))


@pytest.fixture(scope="session")
def real_parse_cache() -> Path:
    configured = os.environ.get(REAL_CACHE_ENV)
    root = Path(configured) if configured else Path(tempfile.gettempdir()) / "nlp-suite-real-parses"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session")
def real_snapshot(real_corpus: Any, real_parse_cache: Path) -> Any:
    """The real corpus, parsed once, with the tokenizer that parsed it."""
    if not has_spacy_model():
        pytest.skip("spaCy English model is not installed")
    from desktop_backend.live import Bench

    return Bench(real_parse_cache).warm(real_corpus, "spacy")


def prime_annotations(workspace_root: Path, key: str, cache_root: Path) -> bool:
    """Put the shared parse where a workspace's own Bench will find it.

    The cache is content-addressed, so this is the same table the workspace
    would have produced -- it just does not spend a minute producing it again.
    """
    source = cache_root / "annotations" / f"{key}.parquet"
    if not source.is_file():
        return False
    target = workspace_root / "annotations"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target / source.name)
    return True


def build_frame(rows: list[list[object]], columns: list[str]) -> pd.DataFrame:
    """A CoNLL-shaped DataFrame with the given header order."""
    return pd.DataFrame(rows, columns=columns)


class HashEmbeddingBackend:
    """Deterministic offline test double for the embedding protocol.

    MD5 vectors of ``word|context`` — context-sensitive and stable, but
    untrained. Lives here (never in production) so every test module
    shares one fake without cross-test imports.
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def dimension(self) -> int:
        return self._dim

    def embed(self, words: Sequence[str], contexts: Sequence[str]) -> list[list[float]]:
        return [_hash_vec(f"{word}|{context}", self._dim) for word, context in zip(words, contexts, strict=True)]


def _hash_vec(text: str, dim: int) -> list[float]:
    # Test-only determinism, not security: first 16 bytes of SHA-256.
    digest = hashlib.sha256(text.encode("utf-8")).digest()[:16]
    vals = [(digest[i % len(digest)] / 127.5) - 1.0 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


CANONICAL_MINIMAL: list[str] = [
    "ID",
    "Form",
    "Lemma",
    "POS",
    "NER",
    "Head",
    "DepRel",
    "Record ID",
    "Sentence ID",
    "Document ID",
    "Document",
]


@pytest.fixture
def conll_frame() -> pd.DataFrame:
    """Two documents, three sentences, thirteen tokens.

    ``Document ID`` is ``'1'`` (not ``'1.0'``) on purpose: that is the exact
    input that made the legacy division loop emit a phantom empty sentence,
    because its "previous document" sentinel was hard-coded to ``'1.0'``.
    """
    rows: list[list[object]] = [
        # document 1, sentence 1 — "The president went to Italy ."
        [1, "The", "the", "DT", "O", 2, "det", 1, 1, "1", "doc_a.txt"],
        [2, "president", "president", "NN", "O", 3, "nsubj", 2, 1, "1", "doc_a.txt"],
        [3, "went", "go", "VBD", "O", 0, "root", 3, 1, "1", "doc_a.txt"],
        [4, "to", "to", "IN", "O", 5, "case", 4, 1, "1", "doc_a.txt"],
        [5, "Italy", "Italy", "NNP", "S-GPE", 3, "obl", 5, 1, "1", "doc_a.txt"],
        [6, ".", ".", ".", "O", 3, "punct", 6, 1, "1", "doc_a.txt"],
        # document 1, sentence 2 — "He stayed ."
        [7, "He", "he", "PRP", "O", 2, "nsubj", 7, 2, "1", "doc_a.txt"],
        [8, "stayed", "stay", "VBD", "O", 0, "root", 8, 2, "1", "doc_a.txt"],
        [9, ".", ".", ".", "O", 8, "punct", 9, 2, "1", "doc_a.txt"],
        # document 2, sentence 1 — "Ladies and gentlemen ."
        [10, "Ladies", "lady", "NNS", "O", 3, "nsubj", 10, 1, "2", "doc_b.txt"],
        [11, "and", "and", "CC", "O", 3, "cc", 11, 1, "2", "doc_b.txt"],
        [12, "gentlemen", "gentleman", "NNS", "O", 1, "conj", 12, 1, "2", "doc_b.txt"],
        [13, ".", ".", ".", "O", 3, "punct", 13, 1, "2", "doc_b.txt"],
    ]
    return build_frame(rows, CANONICAL_MINIMAL)


@pytest.fixture
def universal_frame() -> pd.DataFrame:
    """A raw Stanza-shaped table: CoNLL-U ``feats`` column, Universal POS tags."""
    rows: list[list[object]] = [
        [1, "The", "the", "DET", "O", 2, "det", "_", 1, 1, "1", "doc_a.txt", "Definite=Def|PronType=Art"],
        [2, "cats", "cat", "NOUN", "O", 3, "nsubj", "_", 2, 1, "1", "doc_a.txt", "Number=Plur"],
        [3, "ran", "run", "VERB", "O", 0, "root", "_", 3, 1, "1", "doc_a.txt", "Tense=Past|VerbForm=Fin"],
        [4, "fast", "fast", "ADV", "O", 3, "advmod", "_", 4, 1, "1", "doc_a.txt", "Degree=Cmp"],
    ]
    columns = [
        "ID",
        "Form",
        "Lemma",
        "POS",
        "NER",
        "Head",
        "DepRel",
        "Deps",
        "Record ID",
        "Sentence ID",
        "Document ID",
        "Document",
        "feats",
    ]
    return build_frame(rows, columns)


@pytest.fixture
def without_wordnet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make WordNet genuinely unreachable, however it is installed here.

    The three tests that check "the corpus is missing, and the message says
    how to fix it" used to depend on the machine not having WordNet. Once it
    was installed they stopped testing anything, and did it in two different
    ways, neither of which announced itself:

    * ``tests/test_wordnet.py`` emptied ``nltk.data.path``, which isolates
      nothing once a corpus has been loaded. NLTK's ``LazyCorpusLoader``
      materialises into a reader on first use and that reader keeps answering
      from the files it already opened, so the test passed alone and failed
      after anything else in the session had touched WordNet.
    * ``tests/test_nominalization.py`` and ``tests/test_symbolic.py`` skipped
      themselves when ``nltk.data.find("corpora/wordnet")`` succeeded. The
      downloader leaves a ``wordnet.zip``, not an unzipped directory, so that
      probe raises ``LookupError`` on a perfectly working install: the guard
      decided the data was missing, declined to skip, and then asserted a
      failure against code that resolved the corpus from the zip.

    Forcing the condition is better than skipping on it. These tests now run
    everywhere, and they run against the path they are about.
    """
    nltk = pytest.importorskip("nltk")
    # Evict the corpus package so the next import rebuilds its lazy loaders
    # instead of handing back readers that have already opened their files.
    for name in [m for m in list(sys.modules) if m == "nltk.corpus" or m.startswith("nltk.corpus.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(nltk.data, "path", [])
