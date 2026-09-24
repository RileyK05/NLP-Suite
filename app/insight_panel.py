"""Shared Streamlit panel: what the result says, and what to chart.

One component so every page explains its output the same way, and so the
explanation lives next to the table instead of in a separate tool the analyst
has to know exists.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.insight.readout import readout
from core.insight.recommend import degenerate_reasons, recommend_charts

__all__ = ["render_axis_check", "render_insight_panel"]

# An x/y check needs two columns to choose between.
_MIN_COLUMNS_FOR_CHECK = 2


def render_insight_panel(frame: pd.DataFrame, *, tool: str, key: str) -> None:
    """Readout and chart recommendations for a finished result table."""
    if frame.empty:
        return
    reading = readout(frame, tool=tool)
    with st.expander("What this result says", expanded=True):
        st.markdown(reading.to_markdown())

    recommendations = recommend_charts(frame, tool=tool, limit=4)
    if not recommendations:
        return
    with st.expander("Charts worth drawing", expanded=False):
        st.caption(
            "Ranked by how much they can tell you about this table. Pairings that could only "
            "restate the data (a column against itself, or two columns that are the same quantity) "
            "are left out rather than offered."
        )
        for index, recommendation in enumerate(recommendations):
            st.markdown(f"**{recommendation.question}**")
            st.caption(recommendation.why)
            st.code(
                f"nlp-suite charts RESULT.csv OUT --kind {recommendation.spec.kind} "
                f"--x {recommendation.spec.x!r} --y {recommendation.spec.y!r}",
                language="bash",
            )
            if index < len(recommendations) - 1:
                st.divider()
    _render_axis_check(frame, key=key)


def _render_axis_check(frame: pd.DataFrame, *, key: str) -> None:
    """Let the analyst test a pairing of their own before drawing it."""
    columns = [str(column) for column in frame.columns]
    if len(columns) < _MIN_COLUMNS_FOR_CHECK:
        return
    with st.expander("Check a chart idea", expanded=False):
        st.caption("Pick two columns and find out whether plotting them can tell you anything.")
        left, right = st.columns(2)
        with left:
            x = st.selectbox("x axis", options=columns, key=f"{key}_x")
        with right:
            y = st.selectbox("y axis", options=columns, index=min(1, len(columns) - 1), key=f"{key}_y")
        render_axis_check(frame, x, y)


def render_axis_check(frame: pd.DataFrame, x: str, y: str) -> None:
    """Verdict for one x/y pairing."""
    reasons = degenerate_reasons(frame, x, y)
    if not reasons:
        st.success(f"{x} against {y} is worth drawing: the two columns vary independently.")
        return
    st.warning(f"{x} against {y} would not tell you anything:")
    for reason in reasons:
        st.markdown(f"- {reason}")
