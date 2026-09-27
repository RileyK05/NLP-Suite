"""The words that set each side apart (plan 3.4, "Distinctive words").

The scoring is :func:`core.analysis.keyness.keyness` -- G2 log-likelihood with
the log ratio and BIC of the shared categorical engine -- with the sides as its
two groups. What this adds is the fairness rule of plan 3.6: every count comes
with its rate per 10,000 words, because a side three times the size of the
other uses nearly every word more often, and a G2 read alone rewards that.

One table shape whatever the number of sides: a row per word and the side it
is overrepresented in, against "the other side" (two sides) or "the other
sides" (three or more, each side against the rest).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import pandas as pd

from core.analysis.keyness import keyness
from core.conll.schema import Col
from core.contrast.sides import side_of
from core.io.reader import Corpus
from core.result import Diagnostic, Result

__all__ = ["distinctive_words"]

_PER = 10_000
_OTHERS = "the other sides"


def _totals(table: pd.DataFrame, names: dict[str, str], field: Col) -> Counter[str]:
    """Counted words per side, by the rule keyness counts with (alphabetic, lowercased)."""
    words = table[field.value].fillna("").astype(str).str.lower()
    counted = words.str.isalpha() & (words.str.strip() != "")
    sides = table[Col.DOCUMENT.value].astype(str).map(names)
    return Counter(sides[counted].dropna().tolist())


def distinctive_words(
    table: pd.DataFrame, corpus: Corpus, sides: Sequence[str], *, field: Col = Col.LEMMA, top_n: int = 60
) -> Result[pd.DataFrame]:
    """``Side, Compared with, Word, Count, Per 10k, Other count, Other per 10k, Log Ratio, G2, p-value``."""
    by_label = {doc.path.name: side_of(doc) for doc in corpus.docs}
    by_label.update({doc.name: side_of(doc) for doc in corpus.docs})
    names = {str(name): by_label.get(str(name), "") for name in table[Col.DOCUMENT.value].astype(str).unique()}
    totals = _totals(table, names, field)
    frames = []
    diagnostics: list[Diagnostic] = []
    pairs = [(sides[0], sides[1])] if len(sides) == 2 else [(side, _OTHERS) for side in sides]  # noqa: PLR2004
    for side, other in pairs:
        groups = {name: ("A" if owner == side else "B") for name, owner in names.items() if owner}
        if other != _OTHERS:
            groups = {name: ("A" if owner == side else "B") for name, owner in names.items() if owner in (side, other)}
        scored = keyness(table, groups=groups, labels=(side, other), field=field, top_n=0)
        diagnostics.extend(scored.diagnostics)
        if scored.value is None:
            continue
        frame = scored.unwrap()
        other_total = sum(totals.values()) - totals[side] if other == _OTHERS else totals[other]
        for owner, compared, own_total, rest_total in (
            (side, other, totals[side], other_total),
            (other, side, other_total, totals[side]),
        ):
            if owner == _OTHERS:
                continue
            mine = frame[frame["Overrepresented in"] == owner].head(top_n)
            if mine.empty:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "Side": owner,
                        "Compared with": compared,
                        "Word": mine["Word"],
                        "Count": mine[f"Freq {owner}"],
                        "Per 10k": (mine[f"Freq {owner}"] * _PER / max(own_total, 1)).round(3),
                        "Other count": mine[f"Freq {compared}"],
                        "Other per 10k": (mine[f"Freq {compared}"] * _PER / max(rest_total, 1)).round(3),
                        # Signed as "how many times more often here", whichever
                        # group the scorer called A.
                        "Log Ratio": mine["Log Ratio"] if owner == side else -mine["Log Ratio"],
                        "G2": mine["G2 (log-likelihood)"],
                        "p-value": mine["p-value"],
                    }
                )
            )
    if not frames:
        return Result[pd.DataFrame](
            None, (*diagnostics, Diagnostic.error("CONTRAST_NO_WORDS", "no side has distinctive words"))
        )
    out = pd.concat(frames, ignore_index=True)
    return Result.success(out, *diagnostics)
