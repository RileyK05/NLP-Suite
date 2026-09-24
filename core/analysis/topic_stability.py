"""Topic stability — does a topic come back when the model is refitted?

LDA is a randomized fit. A topic is only worth naming ("the economy topic")
if it reappears when the same corpus is fit again from a different random
start; a topic that only shows up under one seed is an artefact of that
seed's initialization, not a property of the corpus. Nothing else in the
suite answers that question: :mod:`core.analysis.lda` fits one model at one
seed and reports it as if it were the whole story.

This module reuses :func:`core.analysis.lda.fit_lda` (R10 forbids a second
LDA implementation) once per seed, and answers the stability question by
**matching topics across fits**, then measuring overlap:

1. The first seed is the REFERENCE. Its topics are the ones a reader will
   eventually name.
2. Every other seed's topics are matched one-to-one to the reference topics
   by maximising total Jaccard similarity of their top-N word sets — the
   Hungarian algorithm (:func:`scipy.optimize.linear_sum_assignment`) on the
   negated similarity matrix. One-to-one matters: a greedy "best match per
   reference topic" lets two reference topics both claim the one refitted
   topic that looks most like both of them, which overstates how much of the
   model actually reproduced. Topic numbers are arbitrary and gensim does not
   promise topic 3 means the same thing across two fits — matching by content
   is the only way to compare them at all.
3. A reference topic's stability is judged on its **worst** seed, not its
   average: a topic that matches beautifully under four seeds and falls apart
   under a fifth is exactly the case a reader needs flagged, and a mean would
   quietly average that failure away.

Pure function of (doc_tokens, seeds, ...): no filesystem, no plotting. The
randomness lives entirely inside gensim, seeded and already covered by
``tests/test_lda.py::TestLdaReference::test_determinism`` — the same seed
gives byte-identical topics on this environment, which is what makes
comparing seed against seed a meaningful measurement rather than noise on
noise.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from core.analysis.lda import fit_lda
from core.result import Diagnostic, Result, Severity

__all__ = ["StabilityResult", "topic_stability"]

TOPIC = "Topic"
WORD = "Word"
SEED = "Seed"
MATCHED_TOPIC = "Matched topic"
JACCARD = "Jaccard"
SHARED_WORDS = "Shared words"
REFERENCE_WORDS = "Reference words"
MEAN_JACCARD = "Mean Jaccard"
MIN_JACCARD = "Min Jaccard"
SEEDS = "Seeds"
STABLE = "Stable"
#: The threshold the Stable column was computed with, as a column. A boolean
#: outcome alone leaves a reader guessing where the line was drawn; the
#: panel draws a reference line at the run's own cutoff.
STABLE_AT = "Stable at"

#: The summary table's columns, in one place (the panel's requires reads
#: these, and the executor writes the CSV from this order).
SUMMARY_COLUMNS: tuple[str, ...] = (
    TOPIC,
    REFERENCE_WORDS,
    MEAN_JACCARD,
    MIN_JACCARD,
    SEEDS,
    STABLE,
    STABLE_AT,
)


@dataclass(frozen=True, slots=True)
class StabilityResult:
    """A stability run: how the reference seed's topics fared under refits.

    ``matches`` is one row per (reference topic, other seed) — the raw
    comparisons. ``summary`` collapses that to one row per reference topic,
    which is the shape a panel draws.
    """

    matches: pd.DataFrame  # Topic, Seed, Matched topic, Jaccard, Shared words, Reference words
    summary: pd.DataFrame  # Topic, Reference words, Mean Jaccard, Min Jaccard, Seeds, Stable
    reference_seed: int
    n_topics: int
    top_n: int
    stable_at: float


def topic_stability(
    doc_tokens: Mapping[str, Sequence[str]],
    *,
    n_topics: int,
    seeds: Sequence[int],
    top_n: int = 10,
    passes: int = 10,
    stable_at: float = 0.5,
) -> Result[StabilityResult]:
    """Refit ``doc_tokens`` at every seed and match topics back to the first.

    The first entry of ``seeds`` is the reference; every other seed's topics
    are one-to-one matched to it by top-``top_n`` word overlap (Jaccard). A
    reference topic is ``Stable`` when its worst (minimum) Jaccard across the
    other seeds is at least ``stable_at`` — see the module docstring for why
    minimum rather than mean.
    """
    seed_list = list(seeds)
    if len(seed_list) < 2:
        return Result.failure(
            Diagnostic.error(
                "STABILITY_TOO_FEW_SEEDS",
                f"need at least 2 seeds to compare a topic against another fit, got {len(seed_list)}; "
                "one fit alone cannot be checked for stability.",
                seeds=seed_list,
            )
        )
    duplicate_seeds = sorted(seed for seed, count in Counter(seed_list).items() if count > 1)
    if duplicate_seeds:
        return Result.failure(
            Diagnostic.error(
                "STABILITY_DUPLICATE_SEEDS",
                f"seeds must be distinct: a seed compared against itself always scores Jaccard 1.0 and "
                f"inflates stability; repeated seed(s): {duplicate_seeds}",
                duplicates=duplicate_seeds,
            )
        )
    if n_topics < 2:
        return Result.failure(
            Diagnostic.error(
                "STABILITY_BAD_K", f"n_topics must be at least 2 to have topics to compare, got {n_topics}"
            )
        )
    if top_n < 1:
        return Result.failure(Diagnostic.error("STABILITY_BAD_TOP_N", f"top_n must be at least 1, got {top_n}"))
    if not (0.0 < stable_at <= 1.0):
        return Result.failure(
            Diagnostic.error("STABILITY_BAD_THRESHOLD", f"stable_at must be in the range (0, 1], got {stable_at}")
        )

    carried: list[Diagnostic] = []
    fits: dict[int, pd.DataFrame] = {}
    for seed in seed_list:
        fitted = fit_lda(doc_tokens, n_topics=n_topics, top_n=top_n, seed=seed, passes=passes, coherence=False)
        if not fitted.ok:
            # The fit's own diagnostics say why; carry them through rather
            # than re-wrapping, so the real cause (e.g. TOPIC_BACKEND_MISSING,
            # TOPIC_EMPTY_VOCABULARY) is what a caller sees.
            return Result.failure(*carried, *fitted.diagnostics)
        carried.extend(diagnostic for diagnostic in fitted.diagnostics if diagnostic.severity is Severity.WARNING)
        fits[seed] = fitted.unwrap().topics

    reference_seed = seed_list[0]
    reference_words = _word_lists(fits[reference_seed], n_topics)

    match_rows: list[dict[str, object]] = []
    scores_by_topic: dict[int, list[float]] = {topic_id: [] for topic_id in range(n_topics)}
    for seed in seed_list[1:]:
        other_words = _word_lists(fits[seed], n_topics)
        similarity = _similarity_matrix(reference_words, other_words)
        assignment = _match_topics(similarity)
        for reference_id, other_id in enumerate(assignment):
            score = float(similarity[reference_id, other_id])
            scores_by_topic[reference_id].append(score)
            shared = [word for word in reference_words[reference_id] if word in other_words[other_id]]
            match_rows.append(
                {
                    TOPIC: reference_id,
                    SEED: seed,
                    MATCHED_TOPIC: other_id,
                    JACCARD: round(score, 4),
                    SHARED_WORDS: ", ".join(shared),
                    REFERENCE_WORDS: ", ".join(reference_words[reference_id]),
                }
            )

    matches = pd.DataFrame(match_rows, columns=[TOPIC, SEED, MATCHED_TOPIC, JACCARD, SHARED_WORDS, REFERENCE_WORDS])

    summary_rows: list[dict[str, object]] = []
    for topic_id in range(n_topics):
        scores = scores_by_topic[topic_id]
        mean_jaccard = float(np.mean(scores))
        min_jaccard = float(np.min(scores))
        summary_rows.append(
            {
                TOPIC: topic_id,
                REFERENCE_WORDS: ", ".join(reference_words[topic_id]),
                MEAN_JACCARD: round(mean_jaccard, 4),
                MIN_JACCARD: round(min_jaccard, 4),
                SEEDS: len(scores),
                # Judged on the worst seed on purpose (see module docstring):
                # one seed where the topic falls apart is the finding, and a
                # mean would let a single bad seed hide behind the good ones.
                STABLE: min_jaccard >= stable_at,
                # The run's own cutoff, in the row: the panel draws its
                # reference line at this, and the CSV says what "Stable"
                # meant without anyone opening a settings page.
                STABLE_AT: stable_at,
            }
        )
    summary = pd.DataFrame(summary_rows, columns=list(SUMMARY_COLUMNS))

    return Result.success(
        StabilityResult(
            matches=matches,
            summary=summary,
            reference_seed=reference_seed,
            n_topics=n_topics,
            top_n=top_n,
            stable_at=stable_at,
        ),
        *carried,
    )


def _word_lists(topics: pd.DataFrame, n_topics: int) -> list[list[str]]:
    """Each topic's top words, in the weight-descending order ``fit_lda`` wrote them."""
    return [topics.loc[topics[TOPIC] == topic_id, WORD].astype(str).tolist() for topic_id in range(n_topics)]


def _jaccard(a: set[str], b: set[str]) -> float:
    """Intersection over union of two word sets; 0.0 when both are empty."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _similarity_matrix(reference: Sequence[Sequence[str]], other: Sequence[Sequence[str]]) -> np.ndarray:
    """Reference topics (rows) against another fit's topics (columns), by Jaccard."""
    matrix = np.zeros((len(reference), len(other)))
    for i, ref_words in enumerate(reference):
        ref_set = set(ref_words)
        for j, other_words in enumerate(other):
            matrix[i, j] = _jaccard(ref_set, set(other_words))
    return matrix


def _match_topics(similarity: np.ndarray) -> list[int]:
    """One-to-one row-to-column match maximising total similarity.

    ``linear_sum_assignment`` (the Hungarian algorithm) on the negated matrix,
    not a greedy argmax per row: greedy lets two rows both claim the one
    column that looks most like both of them, so the same refitted topic gets
    credited as the match for two different reference topics and the reported
    stability is overstated. The Hungarian solution is the assignment with
    the highest possible total similarity, with no column reused.
    """
    row_indices, col_indices = linear_sum_assignment(-similarity)
    assignment = [0] * similarity.shape[0]
    for row, col in zip(row_indices, col_indices, strict=True):
        assignment[row] = int(col)
    return assignment
