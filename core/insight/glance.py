"""Corpus at a glance: one fixed recipe of existing analyses, and what it says.

No new analysis. The recipe runs seven tools the suite already has, over
one parse of the corpus, as one batch (each tool a child run with its own
tables and figures). :func:`summarize` then reads those tables and writes
the few sentences a reader wants first: how big the corpus is, which
documents are hardest and easiest to read, most and least varied, most
positive and most negative, which two read most alike and which stands
apart, who and what it names most, and its most distinctive words.

Nothing here needs a language model, so it runs on any install in about
the time of one parse. ``RECIPE_VERSION`` joins the corpus fingerprint in
the cache key: changing the recipe re-runs it once for every corpus.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib

import pandas as pd

from core.analysis.lda import STOPWORDS
from core.viz.panel_helpers import FUNCTION_WORDS, document_labels

__all__ = ["GLANCE_FIGURES", "RECIPE", "RECIPE_VERSION", "glance_key", "summarize"]

RECIPE_VERSION = "1"

#: Documents needed before "most" and "least" mean anything, and before a trend does.
_MIN_COMPARED = 2
_MIN_TREND = 8
#: A rank correlation this strong reads as a trend, weaker as none.
_TREND_RHO = 0.3
#: Documents needed before one can stand apart from the rest.
_MIN_APART = 3

#: Tool -> parameters. Tools that need nothing chosen (no groups, no model).
RECIPE: dict[str, dict[str, object]] = {
    "text_statistics": {},
    "tfidf": {"top-n": 10, "max-df-ratio": 0.8},
    "readability": {},
    "lexical_diversity": {},
    "sentiment_vader_anew": {"analysis": "vader"},
    "doc_similarity": {},
    "ner": {},
}

#: The figure to show for each tool, best first: the first one the run drew
#: wins (an undated corpus has no over-time figure, so the next is used).
GLANCE_FIGURES: dict[str, tuple[str, ...]] = {
    "text_statistics": ("text_statistics_over_time", "text_statistics_all_measures"),
    # tfidf's figures are one document at a time; the summary's line of
    # distinctive words is its corpus view.
    "tfidf": (),
    "readability": ("readability_by_decade", "readability_over_time", "readability_all_measures"),
    "lexical_diversity": ("lexical_diversity_over_time", "lexical_diversity_all_measures"),
    "sentiment_vader_anew": ("sentiment_vader_tone_over_time", "sentiment_vader_documents"),
    "doc_similarity": ("doc_similarity_map", "doc_similarity_heatmap"),
    "ner": ("ner_top_entities", "ner_entity_timeline"),
}


def glance_key(document_hashes: list[str]) -> str:
    """The cache key: the documents' contents (order-free) and the recipe version."""
    joined = "|".join(sorted(document_hashes)) + f"|recipe-{RECIPE_VERSION}"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _short(name: object) -> str:
    text = str(name)
    return document_labels([text]).get(text, text)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(dtype=float)


def _extremes(frame: pd.DataFrame, column: str) -> tuple[str, float, str, float] | None:
    values = _numeric(frame, column)
    if values.notna().sum() < _MIN_COMPARED or "Document" not in frame.columns:
        return None
    high, low = values.idxmax(), values.idxmin()
    return (
        _short(frame.loc[high, "Document"]),
        float(values[high]),
        _short(frame.loc[low, "Document"]),
        float(values[low]),
    )


def _size(frame: pd.DataFrame) -> str | None:
    tokens, sentences = _numeric(frame, "Tokens"), _numeric(frame, "Sentences")
    if tokens.empty:
        return None
    line = f"{len(frame):,} documents, {int(tokens.sum()):,} words and {int(sentences.sum()):,} sentences"
    if len(frame) > 1:
        line += f"; the median document is {int(tokens.median()):,} words"
        longest = _extremes(frame, "Tokens")
        if longest:
            line += (
                f", the longest {longest[0]} ({int(longest[1]):,}) and the shortest {longest[2]} ({int(longest[3]):,})"
            )
    return line + "."


def _readability(frame: pd.DataFrame) -> str | None:
    found = _extremes(frame, "Flesch Reading Ease")
    if not found:
        return None
    easy, easy_score, hard, hard_score = found
    return (
        f"Easiest to read: {easy} (Flesch {easy_score:.0f}); hardest: {hard} ({hard_score:.0f}). "
        "Higher is easier; 60-70 is plain English."
    )


def _diversity(frame: pd.DataFrame) -> str | None:
    found = _extremes(frame, "MTLD")
    if not found:
        return None
    return f"Most varied vocabulary: {found[0]} (MTLD {found[1]:.0f}); least: {found[2]} ({found[3]:.0f})."


def _sentiment(frame: pd.DataFrame) -> str | None:
    found = _extremes(frame, "Compound")
    if not found:
        return None
    # "Most negative: +0.17" reads as a contradiction when every document is
    # on the positive side; then it is the least positive.
    low_word = "most negative" if found[3] < 0 else "least positive"
    line = f"Most positive in tone: {found[0]} (VADER {found[1]:+.2f}); {low_word}: {found[2]} ({found[3]:+.2f})"
    years = _numeric(frame, "Year")
    tone = _numeric(frame, "Compound")
    both = pd.concat([years, tone], axis=1).dropna()
    if len(both) >= _MIN_TREND and both.iloc[:, 0].nunique() > 1:
        rho = float(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman"))
        trend = "rises" if rho >= _TREND_RHO else "falls" if rho <= -_TREND_RHO else "shows no clear trend"
        line += f"; over time it {trend} (Spearman {chr(0x3C1)} = {rho:+.2f})"
    return line + "."


def _similarity(pairs: pd.DataFrame) -> str | None:
    needed = {"Document A", "Document B", "Similarity"}
    if not needed <= set(pairs.columns) or pairs.empty:
        return None
    score = pd.to_numeric(pairs["Similarity"], errors="coerce")
    best = score.idxmax()
    line = (
        f"Most alike in wording: {_short(pairs.loc[best, 'Document A'])} and {_short(pairs.loc[best, 'Document B'])} "
        f"({score[best]:.0f}% TF-IDF similarity)"
    )
    long = pd.concat(
        [
            pd.DataFrame({"doc": pairs["Document A"], "score": score}),
            pd.DataFrame({"doc": pairs["Document B"], "score": score}),
        ]
    )
    means = long.groupby("doc")["score"].mean()
    if len(means) >= _MIN_APART:
        line += f"; the one that stands apart is {_short(means.idxmin())} (average {means.min():.0f}%)"
    return line + "."


def _entities(frame: pd.DataFrame) -> str | None:
    if not {"Entity", "NER Tag", "Count"} <= set(frame.columns) or frame.empty:
        return None
    counts = frame.assign(Count=_numeric(frame, "Count")).groupby(["NER Tag", "Entity"])["Count"].sum()
    parts = []
    for tag, word in (("PERSON", "people"), ("GPE", "places"), ("ORG", "organisations")):
        if tag in counts.index.get_level_values(0):
            top = counts.loc[tag].sort_values(ascending=False).head(4)
            parts.append(f"{word}: " + ", ".join(f"{name} ({int(n)})" for name, n in top.items()))
    return ("Named most often -- " + "; ".join(parts) + ".") if parts else None


def _terms(frame: pd.DataFrame) -> str | None:
    if not {"Term", "TF-IDF"} <= set(frame.columns) or frame.empty:
        return None
    # Summed over documents, frequent function words ("he", "us", "shall")
    # outweigh every distinctive one; they are left out.
    content = frame[~frame["Term"].astype(str).str.lower().isin(STOPWORDS | FUNCTION_WORDS)]
    content = content[content["Term"].astype(str).str.isalpha()]
    weights = content.assign(_w=_numeric(content, "TF-IDF")).groupby("Term")["_w"].sum().sort_values(ascending=False)
    top = [str(term) for term in weights.head(10).index]
    return f"Most distinctive words (summed TF-IDF): {', '.join(top)}." if top else None


def summarize(tables: Mapping[str, Mapping[str, pd.DataFrame]]) -> list[str]:
    """The glance's sentences, from each recipe tool's tables (tool -> name -> frame)."""

    def table(tool: str, name: str) -> pd.DataFrame:
        return tables.get(tool, {}).get(name, pd.DataFrame())

    lines = [
        _size(table("text_statistics", "text_statistics.csv")),
        _readability(table("readability", "readability.csv")),
        _diversity(table("lexical_diversity", "lexical_diversity.csv")),
        _sentiment(table("sentiment_vader_anew", "vader.csv")),
        _similarity(table("doc_similarity", "doc_pairs.csv")),
        _entities(table("ner", "entity_timeline.csv")),
        _terms(table("tfidf", "tfidf.csv")),
    ]
    return [line for line in lines if line]
