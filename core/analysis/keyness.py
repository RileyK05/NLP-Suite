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
from collections.abc import Mapping

import pandas as pd

from core.analysis.stats_categorical import log_likelihood
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["detail_groups", "keyness"]

_TOP_N_CAP = 100_000


def _values(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def detail_groups(
    details: Mapping[str, Mapping[str, str]], field_name: str, group_a: str, group_b: str = ""
) -> Result[tuple[dict[str, str], tuple[str, str]]]:
    """Groups A and B from a document detail: ``Kind = sotu`` against ``Kind = ina``.

    *details* maps each document name to its details. *group_a* and
    *group_b* are comma-separated values, matched without regard to case;
    an empty *group_b* means every other document. Returns the mapping
    :func:`keyness` takes and the two groups' names for its columns
    ("Kind = sotu", "all others"). A detail nobody has, or a value no
    document has, is refused with the names or values that do exist.
    """
    wanted = field_name.strip().casefold()
    of_document = {
        name: next((str(value).strip() for key, value in fields.items() if key.casefold() == wanted), "")
        for name, fields in details.items()
    }
    present = sorted({value for value in of_document.values() if value}, key=str.casefold)
    if not present:
        known = sorted({key for fields in details.values() for key in fields}, key=str.casefold)
        return Result.failure(
            Diagnostic.error(
                "KEYNESS_UNKNOWN_DETAIL",
                f"no document has a detail called {field_name!r}; details here: {', '.join(known) or 'none'}",
            )
        )
    a_values, b_values = _values(group_a), _values(group_b)
    if not a_values:
        return Result.failure(
            Diagnostic.error(
                "KEYNESS_NO_GROUP_A", f"name the {field_name} value(s) of group A; values: {', '.join(present)}"
            )
        )
    lowered = {value.casefold() for value in present}
    unknown = [value for value in (*a_values, *b_values) if value.casefold() not in lowered]
    if unknown:
        return Result.failure(
            Diagnostic.error(
                "KEYNESS_UNKNOWN_VALUE",
                f"no document has {field_name} = {', '.join(unknown)}; values: {', '.join(present)}",
            )
        )
    a_set = {value.casefold() for value in a_values}
    b_set = {value.casefold() for value in b_values}
    groups: dict[str, str] = {}
    for name, value in of_document.items():
        if value.casefold() in a_set:
            groups[name] = "A"
        elif (value.casefold() in b_set) if b_set else True:
            groups[name] = "B"
    label_a = f"{field_name} = {', '.join(a_values)}"
    label_b = f"{field_name} = {', '.join(b_values)}" if b_values else "all others"
    return Result.success((groups, (label_a, label_b)))


def keyness(
    frame: pd.DataFrame,
    group_pattern: str = "",
    *,
    groups: Mapping[str, str] | None = None,
    labels: tuple[str, str] = ("Group A", "Group B"),
    field: Col = Col.LEMMA,
    smoothing: float = 0.5,
    top_n: int = 200,
) -> Result[pd.DataFrame]:
    """G2 keyness of tokens in group A against group B.

    The groups are either a regex over document names (A matches, B is the
    rest) or, with *groups*, an explicit mapping of document name to ``"A"``
    or ``"B"`` (a document in neither is left out) -- how a comparison of two
    sides, or of ``Kind = sotu`` against ``Kind = ina``, says which is which
    without anyone writing a regex. *labels* name the two groups in the
    output columns; the regex path keeps its historical names.
    """
    pattern: re.Pattern[str] | None = None
    if groups is None:
        if not group_pattern.strip():
            return Result.failure(Diagnostic.error("KEYNESS_EMPTY_PATTERN", "group-pattern must be non-empty"))
        try:
            pattern = re.compile(group_pattern)
        except re.error as exc:
            return Result.failure(Diagnostic.error("KEYNESS_BAD_REGEX", f"invalid group-pattern regex: {exc}"))
    label_a, label_b = labels
    if not label_a.strip() or not label_b.strip() or label_a == label_b:
        return Result.failure(Diagnostic.error("KEYNESS_BAD_LABELS", "the two groups need two different names"))
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
        if groups is not None:
            side = groups.get(str(doc_name))
            if side not in ("A", "B"):
                continue
            target = counts_a if side == "A" else counts_b
        else:
            assert pattern is not None  # noqa: S101 - set whenever groups is None
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
        if groups is not None:
            missing = label_a if not seen_a else label_b
            return Result.failure(
                Diagnostic.error("KEYNESS_ONE_GROUP", f"{missing} has no documents in this table; keyness needs both")
            )
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
        {"Word": word, label_a: counts_a.get(word, 0), label_b: counts_b.get(word, 0)}
        for word in sorted(counts_a.keys() | counts_b.keys())
    ]
    combined = pd.DataFrame(rows, columns=["Word", label_a, label_b])
    result = log_likelihood(
        combined,
        "Word",
        label_a,
        label_b,
        smoothing=float(smoothing),
    )
    if result.value is None:
        return Result[pd.DataFrame](None, result.diagnostics)
    out = result.unwrap().frame
    if top_n:
        out = out.head(top_n)
    else:
        out = out.copy()
    if groups is None:
        out = out.rename(
            columns={
                f"Freq {label_a}": f"Freq {label_a} (pattern docs)",
                f"Freq {label_b}": f"Freq {label_b} (other docs)",
            }
        )
    return Result.success(out, *result.diagnostics)
