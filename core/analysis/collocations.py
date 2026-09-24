"""Collocation association measures over a parsed corpus (CAP-NGRAM-04).

Which word pairs occur together more often than chance would predict? Raw
co-occurrence frequency answers badly: "of the" is the most frequent bigram in
almost any English corpus and tells you nothing about that corpus. An
association measure compares the observed co-occurrence against what
independence predicts, so a pair is scored on how *surprising* it is.

Seven measures, all from one contingency table per pair, because they disagree
in ways that matter:

* **PMI** -- ``log2(O / E)``. Maximally sensitive to rare pairs: a bigram
  occurring once, whose words each occur once, gets the highest possible
  score. Use it with ``min_count`` above 1, never alone.
* **PPMI** -- PMI clamped at 0. The standard input to distributional-semantic
  vector spaces, where negative association is noise.
* **t-score** -- ``(O - E) / sqrt(O)``. Favours frequent, well-attested pairs;
  the conventional complement to PMI in corpus linguistics.
* **z-score** -- ``(O - E) / sqrt(E)``. Like t-score but standardised on the
  expected count, so it is less forgiving of low-frequency inflation.
* **Log-likelihood (G2)** -- Dunning's statistic, summed over all four cells
  of the contingency table. The measure that behaves best on sparse data and
  the one to sort by when you must pick one.
* **Dice** -- ``2O / (f1 + f2)``. A set-overlap coefficient, independent of
  corpus size, good for comparing across corpora of different sizes.
* **Log Dice** -- ``14 + log2(Dice)``. Sketch Engine's default, on a readable
  scale where 14 is the theoretical maximum.

Two span modes. ``adjacent`` counts only immediate bigrams (w[i], w[i+1]) and
is the strict reading of "collocation". ``window`` counts every pair within
``window`` tokens of each other in the same sentence, unordered, which
recovers verb-object pairs separated by modifiers. Neither crosses a sentence
boundary.

Legacy note: ``NGrams_collocation_statistics_util.py`` computed PMI, t-score,
Dice and chi-square, and a "log-likelihood" that was only the first cell of
the G2 sum (``2 * O * ln(O/E)``) rather than the four-cell statistic. That
single-cell form is not Dunning's G2, is not chi-square distributed and cannot
be compared against a critical value, so it is not reproduced here; the G2
column is the real statistic, computed by the same engine that backs
``keyness`` and ``stats_categorical`` so the suite reports one number for one
concept.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from itertools import pairwise
import math

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["MAX_DISTINCT_PAIRS", "SPAN_MODES", "collocations"]

SPAN_MODES: tuple[str, ...] = ("adjacent", "window")

_COLUMNS = [
    "Word 1",
    "Word 2",
    "Pair",
    "Co-occurrences",
    "Word 1 Freq",
    "Word 2 Freq",
    "Expected",
    "PMI",
    "PPMI",
    "T-Score",
    "Z-Score",
    "G2 (log-likelihood)",
    "Dice",
    "Log Dice",
]

_MAX_WINDOW = 25
# Ceiling on distinct pairs held in memory. Window mode has no useful
# frequency bound (see _count_pairs), so without this a large corpus at a wide
# window exhausts memory and the desktop worker dies with no explanation.
# ~2M pairs is roughly 230 MB including key tuples: large enough for real
# corpora, small enough to fail before the machine does.
MAX_DISTINCT_PAIRS = 2_000_000
_LOG_DICE_OFFSET = 14.0  # Sketch Engine's constant: puts the maximum at 14.


class _TooManyPairs(Exception):
    """Internal signal: the pair table hit MAX_DISTINCT_PAIRS mid-count."""


def _sentences(frame: pd.DataFrame, field_col: str) -> Iterator[list[str]]:
    """Token strings per sentence, in row order. Pairs never cross a sentence."""
    for _key, group in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        yield [str(x) for x in group[field_col].tolist()]


def _keep(token: str, stopwords: frozenset[str], min_length: int) -> bool:
    """Alphabetic, long enough, not a stopword. Matches the keyness filter."""
    return token.isalpha() and len(token) >= min_length and token not in stopwords


def _count_pairs(
    sentences: list[list[str]],
    *,
    span: str,
    window: int,
    stopwords: frozenset[str],
    min_length: int,
    case_sensitive: bool,
    min_count: int,
) -> tuple[Counter[str], Counter[tuple[str, str]], int]:
    """Unigram counts, pair counts and the token total N.

    In ``window`` mode a pair is unordered and recorded once per
    co-occurrence, so ``(a, b)`` and ``(b, a)`` accumulate into the same key.
    In ``adjacent`` mode order is preserved, because "New York" and "York New"
    are not the same collocation.

    Memory invariant: two passes, because storing every pair is what makes
    this tool fall over on a real corpus. The first pass counts unigrams; the
    second stores only pairs that could still reach *min_count*.

    The bound differs by span, and getting it wrong silently corrupts counts:

    * **adjacent** -- each occurrence of the pair consumes one occurrence of
      each word, so the count is at most ``min(f(a), f(b))``. A word rarer
      than *min_count* cannot appear in any qualifying pair.
    * **window** -- one occurrence of ``a`` pairs with *every* ``b`` inside
      the window, so in "a b b b" the pair (a, b) occurs three times while
      ``f(a)`` is 1. An occurrence also pairs with the window on both sides,
      so the bound is ``min(f(a), f(b)) * 2 * window``. At any realistic
      ``min_count`` that threshold is 1 and nothing is excluded, so window
      mode relies on ``MAX_DISTINCT_PAIRS`` instead. Both factors matter:
      dropping the ``2`` silently loses real pairs at small windows.

    Both filters are exact: surviving counts are identical to an unfiltered
    pass, which ``test_filter_is_exact_across_settings`` checks directly. How
    much the adjacent filter actually saves depends on the corpus's hapax
    rate, so it is a useful reduction rather than a guarantee -- on synthetic
    Zipfian text at 1M tokens it removed about 4% of stored pairs. The
    dependable protection is ``MAX_DISTINCT_PAIRS``.
    """
    unigrams: Counter[str] = Counter()
    total = 0
    kept_sentences: list[list[str]] = []
    for sentence in sentences:
        tokens = [t if case_sensitive else t.casefold() for t in sentence]
        kept = [t for t in tokens if _keep(t, stopwords, min_length)]
        total += len(kept)
        unigrams.update(kept)
        kept_sentences.append(kept)

    reach = 1 if span == "adjacent" else 2 * window
    threshold = math.ceil(min_count / reach)
    frequent = {word for word, count in unigrams.items() if count >= threshold}
    pairs: Counter[tuple[str, str]] = Counter()
    for kept in kept_sentences:
        if span == "adjacent":
            for left, right in pairwise(kept):
                if left in frequent and right in frequent:
                    pairs[(left, right)] += 1
            continue
        for i, left in enumerate(kept):
            if left not in frequent:
                continue
            for right in kept[i + 1 : i + 1 + window]:
                if right not in frequent:
                    continue
                pairs[(left, right) if left <= right else (right, left)] += 1
                if len(pairs) > MAX_DISTINCT_PAIRS:
                    raise _TooManyPairs
    return unigrams, pairs, total


def _g2(observed: float, f1: float, f2: float, total: float) -> float:
    """Dunning's log-likelihood over all four cells of the 2x2 table.

    Cells with a zero observed count contribute nothing (the limit of
    ``o * ln(o/e)`` as ``o -> 0`` is 0), which is what lets G2 stay usable on
    the sparse tables that collocation work produces.
    """
    o11 = observed
    o12 = f1 - observed
    o21 = f2 - observed
    o22 = total - f1 - f2 + observed
    row1, row2 = o11 + o12, o21 + o22
    col1, col2 = o11 + o21, o12 + o22
    if total <= 0 or row1 <= 0 or row2 <= 0 or col1 <= 0 or col2 <= 0:
        return 0.0
    expected = (
        row1 * col1 / total,
        row1 * col2 / total,
        row2 * col1 / total,
        row2 * col2 / total,
    )
    accumulated = 0.0
    for observed_cell, expected_cell in zip((o11, o12, o21, o22), expected, strict=True):
        if observed_cell > 0 and expected_cell > 0:
            accumulated += observed_cell * math.log(observed_cell / expected_cell)
    return 2.0 * accumulated


def collocations(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    span: str = "adjacent",
    window: int = 5,
    min_count: int = 3,
    min_length: int = 1,
    top_n: int | None = None,
    stopwords: frozenset[str] = frozenset(),
    case_sensitive: bool = False,
) -> Result[pd.DataFrame]:
    """One row per word pair, scored by seven association measures.

    Sorted by G2 descending then by the pair alphabetically, so the ordering is
    total and the output is byte-identical across runs (R6). ``top_n`` truncates
    after that sort.
    """
    if span not in SPAN_MODES:
        return Result.failure(Diagnostic.error("COLLOC_BAD_SPAN", f"span must be one of {SPAN_MODES}, got {span!r}"))
    if window < 1 or window > _MAX_WINDOW:
        return Result.failure(Diagnostic.error("COLLOC_BAD_WINDOW", f"window must be 1..{_MAX_WINDOW}, got {window}"))
    if min_count < 1:
        return Result.failure(Diagnostic.error("COLLOC_BAD_MIN_COUNT", f"min_count must be >= 1, got {min_count}"))
    if min_length < 1:
        return Result.failure(Diagnostic.error("COLLOC_BAD_MIN_LENGTH", f"min_length must be >= 1, got {min_length}"))
    if top_n is not None and top_n < 1:
        return Result.failure(Diagnostic.error("COLLOC_BAD_TOP_N", f"top_n must be >= 1, got {top_n}"))
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("COLLOC_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_COLUMNS))

    normalized_stopwords = frozenset(w if case_sensitive else w.casefold() for w in stopwords)
    try:
        unigrams, pairs, total = _count_pairs(
            list(_sentences(frame, field.value)),
            span=span,
            window=window,
            stopwords=normalized_stopwords,
            min_length=min_length,
            case_sensitive=case_sensitive,
            min_count=min_count,
        )
    except _TooManyPairs:
        # A named refusal beats an out-of-memory kill: the run stops with
        # advice instead of taking the worker process down with it.
        return Result.failure(
            Diagnostic.error(
                "COLLOC_TOO_MANY_PAIRS",
                f"more than {MAX_DISTINCT_PAIRS:,} distinct pairs at span={span!r}, window={window}; "
                "raise --min-count, narrow --window, use --span adjacent, or supply a stopword list",
                span=span,
                window=window,
                min_count=min_count,
            )
        )
    if total == 0:
        return Result.success(
            pd.DataFrame(columns=_COLUMNS),
            Diagnostic.warning(
                "COLLOC_NO_TOKENS",
                "no tokens survived filtering; check --min-length and the stopword list",
            ),
        )

    rows: list[dict[str, object]] = []
    for (word_a, word_b), observed in pairs.items():
        if observed < min_count:
            continue
        f1 = unigrams[word_a]
        f2 = unigrams[word_b]
        expected = f1 * f2 / total
        # Guarded rather than assumed: a pair can only be observed if both
        # words occur, so expected > 0 holds, but a caller-supplied frame with
        # duplicated rows should degrade to a zero score, not raise.
        pmi = math.log2(observed / expected) if expected > 0 else 0.0
        dice = 2.0 * observed / (f1 + f2) if (f1 + f2) > 0 else 0.0
        rows.append(
            {
                "Word 1": word_a,
                "Word 2": word_b,
                "Pair": f"{word_a} {word_b}",
                "Co-occurrences": int(observed),
                "Word 1 Freq": int(f1),
                "Word 2 Freq": int(f2),
                "Expected": round(expected, 4),
                "PMI": round(pmi, 4),
                "PPMI": round(max(pmi, 0.0), 4),
                "T-Score": round((observed - expected) / math.sqrt(observed), 4),
                "Z-Score": round((observed - expected) / math.sqrt(expected), 4) if expected > 0 else 0.0,
                "G2 (log-likelihood)": round(_g2(observed, f1, f2, total), 4),
                "Dice": round(dice, 6),
                "Log Dice": round(_LOG_DICE_OFFSET + math.log2(dice), 4) if dice > 0 else 0.0,
            }
        )
    if not rows:
        return Result.success(
            pd.DataFrame(columns=_COLUMNS),
            Diagnostic.warning(
                "COLLOC_NO_PAIRS",
                f"no pair reached min_count={min_count}; lower it or use a larger corpus",
            ),
        )
    out = (
        pd.DataFrame(rows, columns=_COLUMNS)
        .sort_values(["G2 (log-likelihood)", "Pair"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    diagnostics: list[Diagnostic] = []
    if top_n is not None and len(out) > top_n:
        diagnostics.append(Diagnostic.info("COLLOC_TRUNCATED", f"showing the top {top_n} of {len(out)} pairs by G2"))
        out = out.head(top_n).reset_index(drop=True)
    return Result.success(out, *diagnostics)
