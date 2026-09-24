"""Pipeline protocol — the single abstraction over parser backends.

ARCHITECTURE.md invariants:

* **R5** — models load once via :class:`PipelineCache`. This protocol is what
  the cache stores.
* **R6** — configuration is passed explicitly; a pipeline never reads global
  state to decide which language it supports.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import pandas as pd

from core.config import supports_language
from core.io.reader import Corpus
from core.result import Result

__all__ = ["SUPPORTED_BACKENDS", "Pipeline", "PipelineBackend", "supports"]


PipelineBackend = str

SUPPORTED_BACKENDS: frozenset[str] = frozenset({"spacy", "stanza", "corenlp"})


@runtime_checkable
class Pipeline(Protocol):
    """A parser that can turn a :class:`Corpus` into a CoNLL table."""

    @property
    def backend(self) -> str: ...

    @property
    def language(self) -> str: ...

    @property
    def tasks(self) -> frozenset[str]: ...

    def parse(self, corpus: Corpus) -> Result[pd.DataFrame]: ...

    def supports(self, language: str, tasks: Iterable[str] = ()) -> bool: ...

    def tokenize(self, text: str) -> tuple[str, ...]:
        """Split a short string the way this backend splits a document.

        Asking a corpus a question about words means deciding what a word is
        twice: once when the corpus was parsed, once when the question was
        typed. A second opinion is not a smaller error than a wrong answer --
        it *is* the wrong answer. A query tokenizer that splits ``U.S.`` into
        four tokens finds nothing in a table whose parser produced one, and
        reports that a phrase plainly present in the text never occurs.

        Short input, so this must not re-run a document pipeline: only the
        step that decides token boundaries.
        """
        ...


def supports(backend: str, language: str, tasks: Iterable[str] = ()) -> bool:
    """Whether *backend* has a confirmed model for *language*.

    *tasks* is accepted for forward-compatibility; the current gate is
    language-only. The legacy suite's gate was ``if True``.
    """
    return supports_language(backend, language)
