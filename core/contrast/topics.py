"""Which topics each side spends its words on (plan 3.4 "Shared and distinct topics").

**One** topic model over both sides, fitted by the ``lda_gensim`` tool, then
each topic's share of each side's text. Two models fitted separately and
compared by topic number compare nothing: topic 3 of one model and topic 3 of
another are unrelated, because the numbering is arbitrary.

The share comes from the model's own paragraph table (``topic_flow.csv``):
the words of each paragraph count towards the topic that dominates it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from core.contrast.sides import side_of
from core.io.reader import Corpus
from core.result import Diagnostic, Result

__all__ = ["topic_prevalence"]


def _number(value: object) -> int:
    try:
        return int(float(str(value)))
    except ValueError:
        return -1


def topic_prevalence(
    frames: Mapping[str, pd.DataFrame], corpus: Corpus, sides: Sequence[str]
) -> Result[tuple[pd.DataFrame, pd.DataFrame]]:
    """``(topics: Topic, Words; prevalence: Topic, Keywords, <side> share ..., Leans towards, Ratio)``."""
    flow = frames.get("topic_flow.csv")
    words = frames.get("topics.csv")
    if flow is None or words is None or flow.empty:
        return Result.failure(Diagnostic.error("CONTRAST_NO_TOPICS", "the topic model produced no paragraph table"))
    owner = {str(doc.doc_id): side_of(doc) for doc in corpus.docs}
    flow = flow.assign(Side=flow["Document ID"].astype(str).map(owner))
    flow = flow[flow["Side"].isin(sides)]
    tokens = pd.to_numeric(flow["Tokens"], errors="coerce").fillna(0)
    flow = flow.assign(Tokens=tokens)
    totals = flow.groupby("Side")["Tokens"].sum()
    by_topic = flow.groupby(["Dominant topic", "Side"])["Tokens"].sum().unstack(fill_value=0)
    keywords = words.groupby("Topic")["Word"].apply(lambda column: ", ".join(column.astype(str).head(8)))
    keywords.index = [_number(value) for value in keywords.index]
    rows = []
    for topic in sorted(by_topic.index, key=_number):
        # The paragraph table stores topic numbers as floats (a column with
        # gaps); a reader is shown "Topic 3", not "Topic 3.0".
        row: dict[str, object] = {"Topic": _number(topic), "Keywords": keywords.get(_number(topic), "")}
        shares = {}
        for side in sides:
            share = (
                float(by_topic.at[topic, side]) / float(totals.get(side, 0) or 1) if side in by_topic.columns else 0.0
            )
            shares[side] = share
            row[f"{side} share"] = round(share, 4)
        ranked = sorted(shares.items(), key=lambda item: item[1], reverse=True)
        top, second = ranked[0], ranked[1] if len(ranked) > 1 else (ranked[0][0], 0.0)
        row["Leans towards"] = top[0]
        row["Ratio"] = round(top[1] / second[1], 3) if second[1] else None
        rows.append(row)
    return Result.success((words.copy(), pd.DataFrame(rows)))
