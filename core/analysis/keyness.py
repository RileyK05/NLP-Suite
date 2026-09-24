"""Keyness lab — G2 log-likelihood over two document groups of one corpus.

Splits the corpus into two text groups with a regex over document names
(group A matches, group B is the remainder), counts the field tokens per
word in each group, and scores every word through the shared
:func:`core.analysis.stats_categorical.log_likelihood` engine (Mode A),
so the G2/p-value/Log Ratio/Pct Diff/BIC semantics and validation come
from the one audited implementation (FR-2.3/C6-5). The two group labels in
the output are ``Group A`` / ``Group B``; ``--top-n`` bounds the output
frame after scoring (0 keeps everything) so a 50k-word corpus does not
write a 50k-row CSV by accident.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from core.analysis.stats_categorical import log_likelihood
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["keyness"]

_TOP_N_CAP = 100_000


def keyness(
    frame: pd.DataFrame,
    group_pattern: str,
    *,
    field: Col = Col.LEMMA,
    smoothing: float = 0.5,
    top_n: int = 200,
) -> Result[pd.DataFrame]:
    """G2 keyness of tokens in group A (name matches) against group B (rest)."""
    if not group_pattern.strip():
        return Result.failure(Diagnostic.error("KEYNESS_EMPTY_PATTERN", "group-pattern must be non-empty"))
    try:
        pattern = re.compile(group_pattern)
    except re.error as exc:
        return Result.failure(Diagnostic.error("KEYNESS_BAD_REGEX", f"invalid group-pattern regex: {exc}"))
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 0 <= top_n <= _TOP_N_CAP:
        return Result.failure(Diagnostic.error("KEYNESS_BAD_TOP_N", f"top-n must be 0..{_TOP_N_CAP}, got {top_n!r}"))
    if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)) or float(smoothing) <= 0:
        return Result.failure(Diagnostic.error("KEYNESS_BAD_SMOOTHING", "smoothing must be a positive number"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(
            Diagnostic.error("KEYNESS_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}")
        )
    if frame.empty:
        return Result.failure(Diagnostic.error("KEYNESS_EMPTY", "frame is empty"))
    if Col.DOCUMENT.value not in frame.columns:
        return Result.failure(Diagnostic.error("KEYNESS_MISSING_DOCUMENT", "frame needs the Document column"))

    counts_a: Counter[str] = Counter()
    counts_b: Counter[str] = Counter()
    seen_a = False
    seen_b = False
    for doc_name, group in frame.groupby(Col.DOCUMENT.value, sort=False):
        target = counts_a if pattern.search(str(doc_name)) else counts_b
        if target is counts_a:
            seen_a = True
        else:
            seen_b = True
        for token in group[field.value].tolist():
            text = str(token).lower()
            if text.strip() and text.isalpha():
                target[text] += 1
    if not seen_a or not seen_b:
        return Result.failure(
            Diagnostic.error(
                "KEYNESS_ONE_GROUP",
                "both groups need at least one document; the pattern matched "
                f"{'all' if seen_a else 'none'} of {len(frame.groupby(Col.DOCUMENT.value))} document(s)",
            )
        )

    # sorted(): the union of two Counter key views is a set, and set iteration
    # order over strings varies with the process hash seed. Feeding that into
    # the scorer made the frame itself differ between runs.
    rows = [
        {"Word": word, "Group A": counts_a.get(word, 0), "Group B": counts_b.get(word, 0)}
        for word in sorted(counts_a.keys() | counts_b.keys())
    ]
    combined = pd.DataFrame(rows, columns=["Word", "Group A", "Group B"])
    result = log_likelihood(
        combined,
        "Word",
        "Group A",
        "Group B",
        smoothing=float(smoothing),
    )
    if result.value is None:
        return Result[pd.DataFrame](None, result.diagnostics)
    out = result.unwrap().frame
    if top_n:
        out = out.head(top_n)
    else:
        out = out.copy()
    label_a, label_b = "Group A", "Group B"
    out = out.rename(
        columns={
            f"Freq {label_a}": f"Freq {label_a} (pattern docs)",
            f"Freq {label_b}": f"Freq {label_b} (other docs)",
        }
    )
    return Result.success(out, *result.diagnostics)
