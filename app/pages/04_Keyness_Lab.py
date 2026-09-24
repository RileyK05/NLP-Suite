"""NLP Suite NG — Keyness Lab (FR-8.3 extension).

A focused two-group comparison page: give one pattern over document names
(a regex, or a plain substring, which is a valid regex). Matching documents
are group A and the rest are group B; the G2 log-likelihood engine scores
the parsed corpus and the table filters by Log Ratio direction.
Runs synchronously like the Tools page — point at a small corpus or use
``nlp-suite keyness`` for large ones.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.insight_panel import render_insight_panel
from app.session import render_shared_sidebar
from app.state import resolve_corpus
from core.analysis.keyness import keyness
from core.conll.schema import Col

st.set_page_config(page_title="NLP Suite NG — Keyness Lab", layout="wide")
st.title("Keyness Lab")
st.caption(
    "Which words distinguish two groups of documents? Group A matches the first pattern; "
    "group B is everything else. Scored with G2 log-likelihood (Hardie Log Ratio, smoothing 0.5)."
)

state = render_shared_sidebar()
corpus_dir = state.corpus_dir
parser = state.parser
language = state.language

col_a, col_b = st.columns(2)
with col_a:
    pattern_a = st.text_input("Group A pattern (matches document names)", value="")
with col_b:
    field_choice = st.selectbox("Token field", options=["lemma", "form"], index=0)
top_n = st.number_input("Rows to keep", min_value=0, max_value=100000, value=200, help="0 keeps every word")

if st.button("Compute keyness", type="primary") and pattern_a.strip():
    resolved = resolve_corpus(corpus_dir)
    if resolved.value is None:
        for diag in resolved.diagnostics:
            st.error(str(diag))
        st.stop()
    from core.config import NLPConfig
    from core.io.reader import read_corpus
    from core.pipelines.cache import PipelineCache

    corpus_result = read_corpus(resolved.unwrap())
    if corpus_result.value is None:
        for diag in corpus_result.diagnostics:
            st.error(str(diag))
        st.stop()
    for diag in corpus_result.diagnostics:
        st.warning(str(diag))
    corpus = corpus_result.unwrap()
    try:
        config = NLPConfig(parser=parser, language=language)  # type: ignore[arg-type]
    except ValueError as exc:
        st.error(f"config error: {exc}")
        st.stop()
    cache = PipelineCache()
    from core.pipelines.resolve import resolve_pipeline

    # Falls back to another installed backend when this one's model
    # is missing or damaged, and says so rather than doing it quietly.
    pipeline_result = resolve_pipeline(cache, config)
    if pipeline_result.value is None:
        for diag in pipeline_result.diagnostics:
            st.error(str(diag))
        st.stop()
    for diag in pipeline_result.diagnostics:
        # A substituted parser changes the result, so it is shown, not logged.
        if diag.code == "PARSER_FALLBACK":
            st.warning(diag.message)
    with st.spinner("Parsing and scoring keyness ..."):
        table_result = pipeline_result.unwrap().parse(corpus)
        if table_result.value is None:
            for diag in table_result.diagnostics:
                st.error(str(diag))
            st.stop()
        result = keyness(
            table_result.unwrap(),
            pattern_a,
            field=Col.FORM if field_choice == "form" else Col.LEMMA,
            top_n=int(top_n),
        )
    if result.value is None:
        for diag in result.diagnostics:
            st.error(str(diag))
        st.stop()
    for diag in result.diagnostics:
        st.info(str(diag))
    frame: pd.DataFrame = result.unwrap()
    st.session_state["keyness_frame"] = frame

if "keyness_frame" in st.session_state:
    frame = st.session_state["keyness_frame"]
    st.subheader(f"{len(frame)} key word(s)")
    direction = st.radio("Direction", options=["all", "group A only", "group B only"], horizontal=True)
    shown = frame
    if direction == "group A only":
        shown = frame[frame["Log Ratio"] > 0]
    elif direction == "group B only":
        shown = frame[frame["Log Ratio"] < 0]
    st.dataframe(shown, use_container_width=True, height=480)
    render_insight_panel(frame, tool="keyness", key="keyness")

    csv = shown.to_csv(index=False).encode("utf-8")
    st.download_button("Download keyness.csv", data=csv, file_name="keyness.csv", mime="text/csv")
    st.caption("Group A over-represented words have positive Log Ratio; group B negative.")
