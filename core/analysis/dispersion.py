"""Lexical dispersion: how evenly a word is spread across a corpus (CAP-STATS-13).

Frequency on its own is misleading, and the failure is systematic. Two words
can occur 100 times each in the same corpus and mean completely different
things: one appears in every document a couple of times, and is core
vocabulary; the other appears 100 times in a single document, and is that
document's topic. Ranked by frequency they are indistinguishable. This is the
single most common way a word-frequency table misleads a reader.

A dispersion measure separates them. Four are reported, because they answer
slightly different questions and the field has not settled on one:

* **Range** -- how many parts contain the word at all. Crude but immediately
  interpretable, and the first thing to check.
* **Gries' DP** (Deviation of Proportions) -- half the total absolute
  difference between where the word's occurrences actually fall and where the
  corpus's own size distribution says they should fall. ``0`` is perfectly
  even, approaching ``1`` is entirely concentrated. It handles unequal part
  sizes correctly, which is why it is the modern default.
* **DP norm** -- DP divided by its theoretical maximum for this corpus, so
  values are comparable between corpora with different part structures.
* **Juilland's D** -- the classic measure, ``1 - CV / sqrt(n - 1)``, where CV
  is the coefficient of variation of the word's size-normalised frequency
  across parts. ``1`` is perfectly even, ``0`` maximally concentrated, so it
  runs in the opposite direction to DP.

**Adjusted Frequency** (Juilland's U, ``F * D``) is also reported: a frequency
discounted by how clumped the word is. Ranking by U rather than raw frequency
is the standard way to build a defensible core-vocabulary list.

Parts are documents by default. That is the natural unit for a document
collection, but for a single long text you can split into equal-sized chunks
instead, which is what makes dispersion meaningful inside one book.

This capability is new: the legacy suite's
``statistics_corpus_word_frequency_util`` reported raw and relative frequency
only, with no dispersion of any kind, so there is no legacy behaviour to
reproduce here.
"""

from __future__ import annotations

from collections import Counter
import math

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["PART_MODES", "dispersion"]

PART_MODES: tuple[str, ...] = ("document", "chunk")

_COLUMNS = [
    "Term",
    "Frequency",
    "Range",
    "Parts",
    "Gries DP",
    "DP norm",
    "Juilland D",
    "Adjusted Frequency",
]

_MIN_PARTS = 2


def _parts_by_document(frame: pd.DataFrame, field_col: str, min_length: int) -> list[list[str]]:
    parts: list[list[str]] = []
    for _document_id, group in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        parts.append(_tokens(group[field_col], min_length))
    return parts


def _tokens(values: pd.Series, min_length: int) -> list[str]:
    kept: list[str] = []
    for value in values:
        token = str(value).casefold()
        if token.isalpha() and len(token) >= min_length:
            kept.append(token)
    return kept


def _parts_by_chunk(frame: pd.DataFrame, field_col: str, min_length: int, chunks: int) -> list[list[str]]:
    """Split the whole corpus, in row order, into *chunks* equal-sized parts.

    The remainder is spread over the leading chunks rather than dumped into the
    last one, so part sizes differ by at most one token and no chunk is a
    short outlier that would distort the variance-based measures.
    """
    tokens = _tokens(frame[field_col], min_length)
    if not tokens:
        return []
    size, remainder = divmod(len(tokens), chunks)
    parts: list[list[str]] = []
    start = 0
    for index in range(chunks):
        width = size + (1 if index < remainder else 0)
        if width == 0:
            continue
        parts.append(tokens[start : start + width])
        start += width
    return parts


def _gries_dp(counts: list[int], sizes: list[int], total_count: int, total_size: int) -> tuple[float, float]:
    """Gries' DP and its corpus-normalised form.

    DP is half the summed absolute difference between the word's observed
    distribution over parts and the expected distribution implied by part
    sizes. Its theoretical maximum for a corpus is ``1 - min(expected)``,
    which is what DP norm divides by.
    """
    expected = [size / total_size for size in sizes]
    observed = [count / total_count for count in counts]
    deviation = 0.5 * sum(abs(o - e) for o, e in zip(observed, expected, strict=True))
    ceiling = 1.0 - min(expected)
    return deviation, (deviation / ceiling if ceiling > 0 else 0.0)


def _juilland_d(counts: list[int], sizes: list[int]) -> float:
    """``1 - CV / sqrt(n - 1)`` over size-normalised part frequencies.

    Normalising by part size first is what makes D valid on unequal parts. The
    population standard deviation is used, matching Juilland's definition.
    """
    parts = len(counts)
    normalized = [count / size for count, size in zip(counts, sizes, strict=True)]
    mean = sum(normalized) / parts
    if mean == 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in normalized) / parts
    coefficient = math.sqrt(variance) / mean
    # For non-negative counts the coefficient of variation cannot exceed
    # sqrt(n - 1), so D is mathematically bounded below by 0 and a negative
    # value here is floating-point error at the maximally-clumped extreme
    # (it prints as -0.0). Clamp rather than report a value D cannot take.
    return max(0.0, 1.0 - coefficient / math.sqrt(parts - 1))


def dispersion(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    parts: str = "document",
    chunks: int = 10,
    min_count: int = 5,
    min_length: int = 1,
    top_n: int | None = None,
) -> Result[pd.DataFrame]:
    """One row per term, with frequency and four dispersion measures.

    Sorted by frequency descending then term alphabetically, so ties never
    reorder between runs (R6).
    """
    if parts not in PART_MODES:
        return Result.failure(Diagnostic.error("DISP_BAD_PARTS", f"parts must be one of {PART_MODES}, got {parts!r}"))
    if chunks < _MIN_PARTS:
        return Result.failure(Diagnostic.error("DISP_BAD_CHUNKS", f"chunks must be >= {_MIN_PARTS}, got {chunks}"))
    if min_count < 1:
        return Result.failure(Diagnostic.error("DISP_BAD_MIN_COUNT", f"min_count must be >= 1, got {min_count}"))
    if min_length < 1:
        return Result.failure(Diagnostic.error("DISP_BAD_MIN_LENGTH", f"min_length must be >= 1, got {min_length}"))
    if top_n is not None and top_n < 1:
        return Result.failure(Diagnostic.error("DISP_BAD_TOP_N", f"top_n must be >= 1, got {top_n}"))
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("DISP_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_COLUMNS))

    if parts == "document":
        split = _parts_by_document(frame, field.value, min_length)
    else:
        split = _parts_by_chunk(frame, field.value, min_length, chunks)
    split = [part for part in split if part]
    if len(split) < _MIN_PARTS:
        return Result.success(
            pd.DataFrame(columns=_COLUMNS),
            Diagnostic.warning(
                "DISP_TOO_FEW_PARTS",
                f"dispersion needs at least {_MIN_PARTS} non-empty parts, got {len(split)}; "
                "use --parts chunk to split a single document",
            ),
        )

    sizes = [len(part) for part in split]
    total_size = sum(sizes)
    per_part: list[Counter[str]] = [Counter(part) for part in split]
    totals: Counter[str] = Counter()
    for counter in per_part:
        totals.update(counter)

    rows: list[dict[str, object]] = []
    for term in sorted(totals):
        total_count = totals[term]
        if total_count < min_count:
            continue
        counts = [counter[term] for counter in per_part]
        deviation, normalised = _gries_dp(counts, sizes, total_count, total_size)
        juilland = _juilland_d(counts, sizes)
        rows.append(
            {
                "Term": term,
                "Frequency": int(total_count),
                "Range": sum(1 for count in counts if count > 0),
                "Parts": len(split),
                "Gries DP": round(deviation, 6),
                "DP norm": round(normalised, 6),
                "Juilland D": round(juilland, 6),
                "Adjusted Frequency": round(total_count * juilland, 4),
            }
        )
    if not rows:
        return Result.success(
            pd.DataFrame(columns=_COLUMNS),
            Diagnostic.warning("DISP_NO_TERMS", f"no term reached min_count={min_count}"),
        )
    out = (
        pd.DataFrame(rows, columns=_COLUMNS)
        .sort_values(["Frequency", "Term"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    diagnostics: list[Diagnostic] = []
    if top_n is not None and len(out) > top_n:
        diagnostics.append(Diagnostic.info("DISP_TRUNCATED", f"showing the top {top_n} of {len(out)} terms"))
        out = out.head(top_n).reset_index(drop=True)
    return Result.success(out, *diagnostics)
