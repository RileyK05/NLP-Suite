"""Per-document measures side by side, with how different the sides are (plan 3.4, "Size and style").

Each measure is a column some existing tool already computes per document
(readability, text statistics, lexical diversity, VADER); this module only
lines them up by side and says how far apart the sides are. The effect size is
Cliff's delta -- the chance a document from one side scores higher than one
from the other, minus the reverse -- with the bands and the Mann-Whitney test
of :mod:`core.analysis.stats_groups`, so "large" means what it means there.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
import math
from typing import cast

import pandas as pd
from scipy import stats as scipy_stats

from core.analysis.stats_groups import cliff_band
from core.contrast.sides import ALL, Alignment, side_of
from core.io.reader import Corpus

__all__ = ["MEASURES", "TONE", "Measure", "measure_table", "measure_tests"]

#: Fewer documents than this on a side: medians are shown, no test is run.
_MIN_TESTED = 3


@dataclass(frozen=True)
class Measure:
    tool: str
    table: str
    column: str
    #: What the measure is called in a sentence.
    label: str
    #: What a higher value is, as the object of "has": "longer sentences",
    #: "text that is easier to read".
    higher: str


MEASURES: tuple[Measure, ...] = (
    Measure(
        "text_statistics", "text_statistics.csv", "Avg Sentence Length", "sentence length (words)", "longer sentences"
    ),
    Measure("text_statistics", "text_statistics.csv", "Avg Word Length", "word length (letters)", "longer words"),
    Measure(
        "readability", "readability.csv", "Flesch Reading Ease", "Flesch reading ease", "text that is easier to read"
    ),
    Measure("readability", "readability.csv", "Flesch-Kincaid Grade", "Flesch-Kincaid grade", "a higher grade level"),
    Measure(
        "lexical_diversity", "lexical_diversity.csv", "MTLD", "vocabulary diversity (MTLD)", "more varied vocabulary"
    ),
)
TONE = Measure(
    "sentiment_vader_anew", "vader.csv", "Compound", "tone (VADER compound, -1 to 1)", "a more positive tone"
)


def measure_table(
    corpus: Corpus,
    frames: Mapping[str, pd.DataFrame],
    measures: Sequence[Measure],
    alignment: Alignment,
) -> pd.DataFrame:
    """One row per compared document: ``Document ID, Document, Side, Group, Date, Year`` and each measure."""
    rows = []
    for doc in corpus.docs:
        if doc.doc_id not in alignment.groups:
            continue
        rows.append(
            {
                "Document ID": str(doc.doc_id),
                "Document": doc.name,
                "Side": side_of(doc),
                "Group": alignment.groups[doc.doc_id],
                "Date": doc.date.isoformat() if doc.date else "",
                "Year": doc.date.year if doc.date else None,
            }
        )
    out = pd.DataFrame(rows, columns=["Document ID", "Document", "Side", "Group", "Date", "Year"])
    for measure in measures:
        frame = frames.get(measure.table)
        if frame is None or measure.column not in frame.columns:
            continue
        values = dict(
            zip(frame["Document ID"].astype(str), pd.to_numeric(frame[measure.column], errors="coerce"), strict=False)
        )
        out[measure.column] = out["Document ID"].map(values)
    return out


def _number(value: float) -> str:
    return f"{value:,.2f}" if abs(value) < 100 else f"{value:,.0f}"  # noqa: PLR2004 - digits a reader needs


def _compare(a: pd.Series, b: pd.Series) -> dict[str, object]:
    row: dict[str, object] = {
        "n A": len(a),
        "n B": len(b),
        "Median A": round(float(a.median()), 4) if len(a) else None,
        "Median B": round(float(b.median()), 4) if len(b) else None,
    }
    if len(a) >= _MIN_TESTED and len(b) >= _MIN_TESTED:
        stat_u, p_value = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        delta = 2.0 * float(stat_u) / (len(a) * len(b)) - 1.0
        row.update(
            {
                "Cliff's delta": round(delta, 4),
                "Effect": cliff_band(abs(delta)),
                "U": float(stat_u),
                "p-value": float(p_value),
            }
        )
    else:
        row.update({"Cliff's delta": None, "Effect": "too few documents to test", "U": None, "p-value": None})
    return row


def _medians(row: Mapping[str, object]) -> tuple[float, float]:
    """A compared row's two medians; only called on rows where both sides had values."""
    return cast("float", row["Median A"]), cast("float", row["Median B"])


def _reading(measure: Measure, side_a: str, side_b: str, row: Mapping[str, object]) -> str:
    median_a, median_b = row["Median A"], row["Median B"]
    if not isinstance(median_a, float) or not isinstance(median_b, float):
        return ""
    if math.isclose(median_a, median_b):
        return f"{side_a} and {side_b} have the same median {measure.label} ({_number(median_a)})."
    higher, lower = (side_a, side_b) if median_a > median_b else (side_b, side_a)
    return f"{higher} has {measure.higher} than {lower}: " + _figures(measure, side_a, side_b, row)


def _figures(measure: Measure, side_a: str, side_b: str, row: Mapping[str, object]) -> str:
    """The medians and the effect behind a reading, as the second half of its sentence."""
    median_a, median_b = _medians(row)
    text = f"median {measure.label} {_number(median_a)} for {side_a}, {_number(median_b)} for {side_b}"
    delta = row["Cliff's delta"]
    if isinstance(delta, float):
        p_value = row["p-value"]
        shown = "p < 0.001" if isinstance(p_value, float) and p_value < 0.001 else f"p = {p_value:.3f}"  # noqa: PLR2004
        text += f"; {row['Effect']} effect (Cliff's delta {delta:+.2f}, {shown})."
    else:
        text += f" ({row['Effect']})."
    return text


def measure_tests(
    table: pd.DataFrame, measures: Sequence[Measure], sides: Sequence[str], unit: str = "groups"
) -> pd.DataFrame:
    """For each measure and pair of sides: medians, effect size, test, and a sentence.

    With groups (an alignment), one row per group, and an "All groups" row
    saying in how many groups the difference points the same way -- the form
    of the answer to "per president, did the two settings differ?". ``unit``
    names the groups in that sentence ("Speaker groups", "decades").
    """
    rows: list[dict[str, object]] = []
    groups = list(dict.fromkeys(table["Group"]))
    for measure in measures:
        if measure.column not in table.columns:
            continue
        values = table[["Side", "Group", measure.column]].dropna()
        for side_a, side_b in combinations(sides, 2):
            per_group = []
            for group in groups:
                here = values[values["Group"] == group]
                row = _compare(
                    here[here["Side"] == side_a][measure.column], here[here["Side"] == side_b][measure.column]
                )
                if row["Median A"] is None or row["Median B"] is None:
                    continue
                per_group.append((group, row))
            if len(groups) > 1:
                rows.extend(
                    {
                        "Measure": measure.label,
                        "Column": measure.column,
                        "Group": group,
                        "Side A": side_a,
                        "Side B": side_b,
                        **row,
                        "Reading": _reading(measure, side_a, side_b, row),
                    }
                    for group, row in per_group
                )
            overall = _compare(
                values[values["Side"] == side_a][measure.column], values[values["Side"] == side_b][measure.column]
            )
            reading = _reading(measure, side_a, side_b, overall)
            if len(groups) > 1 and per_group:
                # Said from the side that is higher in more groups, so the
                # count reads as a majority ("13 of 14"), not its complement.
                medians = [_medians(r) for _g, r in per_group]
                higher_a = sum(1 for a, b in medians if a > b)
                higher_b = sum(1 for a, b in medians if b > a)
                leader, other, wins = (side_a, side_b, higher_a) if higher_a >= higher_b else (side_b, side_a, higher_b)
                # The direction once, per group; then the pooled figures.
                reading = (
                    f"{leader} has {measure.higher} than {other} in {wins} of {len(per_group)} {unit}. "
                    "Over all compared documents: " + _figures(measure, side_a, side_b, overall)
                )
            rows.append(
                {
                    "Measure": measure.label,
                    "Column": measure.column,
                    "Group": ALL if len(groups) <= 1 else "All groups",
                    "Side A": side_a,
                    "Side B": side_b,
                    **overall,
                    "Reading": reading,
                }
            )
    columns = [
        "Measure",
        "Column",
        "Group",
        "Side A",
        "Side B",
        "n A",
        "n B",
        "Median A",
        "Median B",
        "Cliff's delta",
        "Effect",
        "U",
        "p-value",
        "Reading",
    ]
    return pd.DataFrame(rows, columns=columns)
