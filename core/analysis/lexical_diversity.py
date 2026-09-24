"""Lexical diversity (FR-2.2) — Guiraud, MTLD, vocd-D on raw document text.

Parity basis: ``statistics_corpus_lexical_diversity_util.py`` (legacy).
Tokenization, TTR-family formulas, the MTLD factoring loop (threshold 0.72,
bidirectional mean, 10-token floor), vocd sampling geometry (35..50, 100
trials, D grid 10..200 step 0.5), rounding (TTR-family 4dp, MTLD/vocd 2dp),
and legacy output columns are ported verbatim.

Intentional differences from legacy:

* vocd is seeded (``seed`` parameter, default recorded in the envelope).
  Legacy drew from the unseeded global numpy RNG, so no two legacy runs
  agreed with each other either — reproducibility beats matching one draw.
  The sampler is ``random.Random``; only the TTR-of-sample distribution
  matters, never numpy's exact stream.
* Documents below the 10-token floor emit a zero row with a
  ``LEXDIV_SHORT_TEXT`` warning instead of vanishing silently; empty
  documents get ``LEXDIV_EMPTY_DOC``. Short-text behavior is explicit now.
* This module overlaps ``corpus_statistics`` (TTR/Root/Log live in both).
  Unifying the token source is follow-up work for the profiler pass, not
  this packet — touching ``corpus_statistics`` outputs here would bundle an
  unrelated behavior change into a review-row module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re

import numpy as np
import pandas as pd

from core.io.reader import Corpus, display_names
from core.result import Diagnostic, Result

__all__ = [
    "LEXDIV_DEFAULT_SEED",
    "LexDivResult",
    "expected_ttr",
    "log_ttr",
    "mtld",
    "root_ttr",
    "run",
    "tokenize",
    "ttr",
    "vocd",
]


LEXDIV_DEFAULT_SEED = 42
_MTLD_THRESHOLD = 0.72
_MTLD_MIN_TOKENS = 10
_VOCD_MIN_SAMPLE = 35
_VOCD_MAX_SAMPLE = 50
_VOCD_TRIALS = 100

_OUTPUT_COLUMNS = [
    "Document ID",
    "Document",
    "Total Tokens",
    "Unique Types",
    "TTR",
    "Root TTR (Guiraud)",
    "Log TTR (Herdan)",
    "MTLD",
    "vocd-D",
]


@dataclass(frozen=True, slots=True)
class LexDivResult:
    frame: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()


def tokenize(text: str) -> list[str]:
    """Lowercase whitespace split with unicode word filter.

    C6-8: ``str.isalpha()`` is already Unicode-aware (café, naïve, Cyrillic,
    CJK all pass). What the legacy filter DID silently discard was
    punctuation-attached words ("dog's" -> dropped). Here the apostrophe
    and intra-word marks stay part of the word: a token is any run of
    word characters minus digits/underscore, plus internal apostrophes
    and hyphens. Punctuation-only tokens are still excluded.
    """
    pattern = r"[^\W\d_]+(?:['’-][^\W\d_]+)*['’]?"  # noqa: RUF001 — curly quote is intentional
    return re.findall(pattern, text.lower())


def ttr(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def root_ttr(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / math.sqrt(len(tokens))


def log_ttr(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    return math.log(len(set(tokens))) / math.log(len(tokens))


def _mtld_forward(tokens: list[str], threshold: float) -> float:
    factor_count = 0.0
    types: set[str] = set()
    token_count = 0
    for tok in tokens:
        token_count += 1
        types.add(tok)
        if (len(types) / token_count) <= threshold:
            factor_count += 1
            types = set()
            token_count = 0
    if token_count > 0:
        remaining_ttr = len(types) / token_count
        if remaining_ttr < 1.0:
            factor_count += (1.0 - remaining_ttr) / (1.0 - threshold)
    if factor_count == 0:
        return float(len(tokens))
    return len(tokens) / factor_count


def mtld(tokens: list[str], threshold: float = _MTLD_THRESHOLD) -> float:
    if len(tokens) < _MTLD_MIN_TOKENS:
        return 0.0
    forward = _mtld_forward(tokens, threshold)
    backward = _mtld_forward(tokens[::-1], threshold)
    return (forward + backward) / 2.0


def expected_ttr(d_candidate: float, sample_size: int) -> float:
    """Malvern & Richards expected TTR for sample size ``sample_size`` at ``D``."""
    if d_candidate == 0:
        return 1.0
    return (d_candidate / sample_size) * (math.sqrt(1 + 2 * sample_size / d_candidate) - 1)


def vocd(tokens: list[str], seed: int = LEXDIV_DEFAULT_SEED) -> float:
    """Best-fit D over the 10..200 grid (step 0.5) against sampled mean TTRs."""
    if len(tokens) < _VOCD_MIN_SAMPLE:
        return 0.0
    rng = random.Random(seed)
    actual_max = min(_VOCD_MAX_SAMPLE, len(tokens))
    sample_sizes = list(range(_VOCD_MIN_SAMPLE, actual_max + 1))
    if not sample_sizes:
        return 0.0
    observed: dict[int, float] = {}
    for size in sample_sizes:
        draws = [ttr([tokens[i] for i in rng.sample(range(len(tokens)), size)]) for _ in range(_VOCD_TRIALS)]
        observed[size] = float(np.mean(draws))
    best_d = 50.0
    best_err = float("inf")
    grid = np.arange(10.0, 200.0 + 0.25, 0.5)  # C6-8: 200.0 endpoint INCLUDED
    for d_candidate in grid:
        err = sum((observed[size] - expected_ttr(float(d_candidate), size)) ** 2 for size in sample_sizes)
        if err < best_err:
            best_err = err
            best_d = float(d_candidate)
    return best_d


def run(corpus: Corpus, *, seed: int = LEXDIV_DEFAULT_SEED) -> Result[LexDivResult]:
    """Per-document lexical diversity over raw corpus texts (no parse needed).

    C6-8 short-text policy: measures that are mathematically defined keep
    being computed for short nonempty documents (tokens, types, TTR,
    Guiraud, Herdan). Only MTLD and vocd-D — whose definitions require the
    10-token floor / 35-token minimum — carry ``pd.NA`` below their
    minimums plus a LEXDIV_SHORT_TEXT warning. A missing value is never
    disguised as a genuine 0.0 score.
    """
    rows: list[dict[str, object]] = []
    diags: list[Diagnostic] = []
    short_docs: list[str] = []
    vocd_short_docs: list[str] = []
    names = display_names(corpus.docs)
    for doc in corpus.docs:
        name = names[doc.doc_id]
        tokens = tokenize(doc.text)
        vocd_short = False
        if not doc.text.strip():
            diags.append(Diagnostic.warning("LEXDIV_EMPTY_DOC", f"{name} has no text", doc_id=doc.doc_id, path=name))
            short = True
        elif len(tokens) < _MTLD_MIN_TOKENS:
            short_docs.append(name)
            short = True
        elif len(tokens) < _VOCD_MIN_SAMPLE:
            # 10..34 tokens: MTLD is defined (its floor is 10) but vocd-D is
            # NOT (35-token minimum). Publishing vocd()'s 0.0 would disguise
            # "undefined at this length" as a genuine minimum score — the
            # exact case the module docstring forbids (review finding S4).
            vocd_short_docs.append(name)
            short = False
            vocd_short = True
        else:
            short = False
        if short:
            rows.append(
                {
                    "Document ID": str(doc.doc_id),
                    "Document": name,
                    "Total Tokens": len(tokens),
                    "Unique Types": len(set(tokens)),
                    # Defined measures continue for short nonempty docs;
                    # empty docs have no tokens so all measures are NA.
                    "TTR": round(ttr(tokens), 4) if tokens else pd.NA,
                    "Root TTR (Guiraud)": round(root_ttr(tokens), 4) if tokens else pd.NA,
                    "Log TTR (Herdan)": round(log_ttr(tokens), 4) if tokens else pd.NA,
                    "MTLD": pd.NA,
                    "vocd-D": pd.NA,
                }
            )
            continue
        rows.append(
            {
                "Document ID": str(doc.doc_id),
                "Document": name,
                "Total Tokens": len(tokens),
                "Unique Types": len(set(tokens)),
                "TTR": round(ttr(tokens), 4),
                "Root TTR (Guiraud)": round(root_ttr(tokens), 4),
                "Log TTR (Herdan)": round(log_ttr(tokens), 4),
                "MTLD": round(mtld(tokens), 2),
                "vocd-D": pd.NA if vocd_short else round(vocd(tokens, seed=seed), 2),
            }
        )
    if short_docs:
        diags.append(
            Diagnostic.warning(
                "LEXDIV_SHORT_TEXT",
                f"{len(short_docs)} document(s) below the {_MTLD_MIN_TOKENS}-token floor; "
                "MTLD and vocd-D are NA there (defined measures still computed)",
                count=len(short_docs),
                documents=short_docs[:5],
            )
        )
    if vocd_short_docs:
        diags.append(
            Diagnostic.warning(
                "LEXDIV_VOCD_SHORT_TEXT",
                f"{len(vocd_short_docs)} document(s) between {_MTLD_MIN_TOKENS} and "
                f"{_VOCD_MIN_SAMPLE - 1} tokens; MTLD is computed there but vocd-D needs "
                f"{_VOCD_MIN_SAMPLE}+ tokens and is NA (never a fabricated 0.0)",
                count=len(vocd_short_docs),
                documents=vocd_short_docs[:5],
            )
        )
    frame = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
    return Result.success(LexDivResult(frame=frame), *diags)
