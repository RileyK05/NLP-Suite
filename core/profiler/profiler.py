"""Profiler — orchestrates a minimal suite run."""

from __future__ import annotations

import pandas as pd

from core.analysis.corpus_statistics import run as corpus_stats
from core.analysis.sentiment_vader_anew import vader
from core.analysis.text_statistics import run as text_stats
from core.conll.schema import Col
from core.result import Diagnostic, Result

__all__ = ["profile"]


def profile(frame: pd.DataFrame) -> Result[pd.DataFrame]:
    """Run a small suite and return a wide summary per document."""
    if frame.empty:
        return Result.failure(Diagnostic.error("PROFILER_EMPTY", "frame is empty"))

    t_res = text_stats(frame)
    if not t_res.ok:
        return Result[pd.DataFrame](None, t_res.diagnostics)
    tdf = t_res.unwrap().to_frame().set_index("Document ID")

    # Corpus stats
    c_res = corpus_stats(frame, field=Col.FORM)
    if not c_res.ok:
        return Result[pd.DataFrame](None, c_res.diagnostics)
    cdf = c_res.unwrap().to_frame().set_index("Document ID")

    # Sentiment
    s_res = vader(frame, field=Col.FORM)
    if not s_res.ok:
        return Result[pd.DataFrame](None, s_res.diagnostics)
    sdf = s_res.unwrap().set_index("Document ID")

    # Join on Document ID
    try:
        joined = tdf.join(cdf, lsuffix="_txt", rsuffix="_corp", how="outer")
        joined = joined.join(sdf, how="outer", rsuffix="_sent")
    except Exception as exc:
        return Result.failure(Diagnostic.error("PROFILER_JOIN_FAILED", f"join failed: {exc}"))

    joined = joined.reset_index()
    # Flatten duplicate columns if any, keep first
    joined = joined.loc[:, ~joined.columns.duplicated()]
    return Result.success(joined)
