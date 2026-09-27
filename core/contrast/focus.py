"""How often each side uses the focus words (plan 3.4 "Topic focus rates").

Counted by the same rules as ``nlp.term_rates`` and the lexicon tool: lemma,
form, or exact lowercase words in the raw text. Two tables:

- per document: ``Document ID, Document, Side, Group, Date, Year, Words counted,
  <g> count, <g> per 10k`` for each word group;
- per group and side: the same counts pooled, the pooled rate, and the mean of
  the documents' rates. The pooled rate answers "how much of this side's text
  is about it"; the mean of rates keeps one long speech from speaking for all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from core.analysis.lexicon_series import lexicon_series, raw_text_series
from core.conll.schema import Col
from core.contrast.sides import Alignment, side_of
from core.io.reader import Corpus
from core.result import Diagnostic, Result

__all__ = ["focus_rates"]

_PER = 10_000


def focus_rates(
    table: pd.DataFrame | None,
    corpus: Corpus,
    lexicon: Mapping[str, Sequence[str]],
    alignment: Alignment,
    *,
    match: str = "lemma",
) -> Result[tuple[pd.DataFrame, pd.DataFrame]]:
    """``(per document, per group and side)`` focus rates."""
    if not lexicon:
        return Result.failure(Diagnostic.error("CONTRAST_NO_FOCUS", "focus rates need at least one word group"))
    if match == "exact-lowercase":
        counted = raw_text_series({str(doc.doc_id): doc.text for doc in corpus.docs}, lexicon)
    else:
        if table is None:
            return Result.failure(Diagnostic.error("CONTRAST_NEEDS_PARSE", "lemma and form matching need the parse"))
        ids = dict(zip(table[Col.DOCUMENT.value].astype(str), table["Document ID"].astype(str), strict=False))
        counted = lexicon_series(table, lexicon, ids, field=Col.LEMMA if match == "lemma" else Col.FORM)
    if counted.value is None:
        return Result[tuple[pd.DataFrame, pd.DataFrame]](None, counted.diagnostics)
    long = counted.unwrap().rename(columns={"Facet": "Document ID"})
    wide = long.pivot_table(index="Document ID", columns="Category", values="Occurrences", aggfunc="sum")
    wide = wide.reindex(columns=list(lexicon)).fillna(0).astype(int)
    words = long.groupby("Document ID")["Tokens"].first()

    rows = []
    for doc in corpus.docs:
        key = str(doc.doc_id)
        if doc.doc_id not in alignment.groups or key not in wide.index:
            continue
        size = int(words.get(key, 0))
        row: dict[str, object] = {
            "Document ID": key,
            "Document": doc.name,
            "Side": side_of(doc),
            "Group": alignment.groups[doc.doc_id],
            "Date": doc.date.isoformat() if doc.date else "",
            "Year": doc.date.year if doc.date else None,
            "Words counted": size,
        }
        for name in lexicon:
            count = int(wide.at[key, name])
            row[f"{name} count"] = count
            row[f"{name} per 10k"] = round(count * _PER / size, 4) if size else None
        rows.append(row)
    per_document = pd.DataFrame(rows)
    if per_document.empty:
        return Result.failure(Diagnostic.error("CONTRAST_NO_DOCUMENTS", "no compared document had words to count"))

    pooled_rows = []
    for (group, side), part in per_document.groupby(["Group", "Side"], sort=False):
        total = int(part["Words counted"].sum())
        entry: dict[str, object] = {"Group": group, "Side": side, "Documents": len(part), "Words counted": total}
        for name in lexicon:
            count = int(part[f"{name} count"].sum())
            entry[f"{name} count"] = count
            entry[f"{name} per 10k"] = round(count * _PER / total, 4) if total else None
            entry[f"{name} mean of document rates"] = round(float(part[f"{name} per 10k"].mean()), 4)
        pooled_rows.append(entry)
    return Result.success((per_document, pd.DataFrame(pooled_rows)), *counted.diagnostics)
