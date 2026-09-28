"""``import nlpsuite as nlp``: the NLP Suite's scripting library.

The functions live in :mod:`core.script.api`; this package is only the name a
notebook imports them under, so a script reads as ``nlp.term_rates(...)``
rather than ``core.script.api.term_rates(...)``, and so the name can outlive a
reorganisation of ``core``.
"""

from core.script import viz
from core.script.api import (
    Corpus,
    RunResult,
    SuiteError,
    chart,
    corpus,
    describe,
    entities,
    figure,
    keyness,
    load,
    measures,
    note,
    passages,
    run,
    save,
    sentiment,
    show,
    similar,
    tables,
    term_rates,
    tools,
    topics,
    use_folder,
)

__all__ = [
    "Corpus",
    "RunResult",
    "SuiteError",
    "chart",
    "corpus",
    "describe",
    "entities",
    "figure",
    "keyness",
    "load",
    "measures",
    "note",
    "passages",
    "run",
    "save",
    "sentiment",
    "show",
    "similar",
    "tables",
    "term_rates",
    "tools",
    "topics",
    "use_folder",
    "viz",
]
