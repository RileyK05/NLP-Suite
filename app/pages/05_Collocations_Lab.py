"""NLP Suite NG — Collocations Lab (FR-8.3 extension).

An interactive view on CAP-NGRAM-04. The point of the page is that the seven
association measures *disagree*, and reading only one of them is how people
get collocation analysis wrong. So the corpus is parsed and scored once, and
the measure you sort by is a control on the result rather than a parameter of
the run: switching from G2 to PMI re-ranks instantly and shows you the same
pairs in a very different order.

Runs synchronously like the other lab pages — point at a small corpus, or use
``nlp-suite collocations`` for large ones.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.insight_panel import render_insight_panel
from app.session import render_shared_sidebar
from app.state import resolve_corpus
from core.analysis.collocations import collocations
from core.conll.schema import Col

st.set_page_config(page_title="NLP Suite NG — Collocations Lab", layout="wide")
st.title("Collocations Lab")
st.caption(
    "Which words genuinely belong together? Seven association measures over the same pairs, "
    "because they rank differently and the difference is the finding."
)

MEASURES = ("G2 (log-likelihood)", "PMI", "PPMI", "T-Score", "Z-Score", "Dice", "Log Dice", "Co-occurrences")

state = render_shared_sidebar()
corpus_dir = state.corpus_dir
parser = state.parser
language = state.language

col_a, col_b, col_c = st.columns(3)
with col_a:
    field_choice = st.selectbox("Token field", options=["lemma", "form"], index=0)
    span = st.selectbox(
        "Span", options=["adjacent", "window"], index=0, help="adjacent bigrams, or any pair inside a window"
    )
with col_b:
    window = st.number_input("Window size", min_value=1, max_value=25, value=5, help="used when span is 'window'")
    min_count = st.number_input(
        "Minimum co-occurrences",
        min_value=1,
        max_value=1000,
        value=3,
        help="association measures are unreliable below 3",
    )
with col_c:
    min_length = st.number_input("Minimum token length", min_value=1, max_value=20, value=1)
    top_n = st.number_input("Rows to keep", min_value=0, max_value=100000, value=200, help="0 keeps every pair")

stopword_text = st.text_area(
    "Stopwords (optional, one per line)",
    value="",
    height=80,
    help="The suite never applies a hidden stoplist: anything you want excluded goes here.",
)

if st.button("Find collocations", type="primary"):
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
    with st.spinner("Parsing and scoring collocations ..."):
        table_result = pipeline_result.unwrap().parse(corpus)
        if table_result.value is None:
            for diag in table_result.diagnostics:
                st.error(str(diag))
            st.stop()
        result = collocations(
            table_result.unwrap(),
            field=Col.FORM if field_choice == "form" else Col.LEMMA,
            span=span,
            window=int(window),
            min_count=int(min_count),
            min_length=int(min_length),
            top_n=int(top_n) or None,
            stopwords=frozenset(line.strip().casefold() for line in stopword_text.splitlines() if line.strip()),
        )
    if result.value is None:
        for diag in result.diagnostics:
            st.error(str(diag))
        st.stop()
    for diag in result.diagnostics:
        st.info(str(diag))
    st.session_state["collocations_frame"] = result.unwrap()

if "collocations_frame" in st.session_state:
    frame: pd.DataFrame = st.session_state["collocations_frame"]
    if frame.empty:
        st.warning("No pair met the minimum count. Lower it, or use a larger corpus.")
        st.stop()
    st.subheader(f"{len(frame)} pair(s)")
    measure = st.radio("Rank by", options=MEASURES, horizontal=True)
    shown = frame.sort_values([measure, "Pair"], ascending=[False, True], kind="stable").reset_index(drop=True)
    st.dataframe(shown, use_container_width=True, height=440)

    # The teaching point of the page: the top of the list changes with the
    # measure. Showing both top-tens side by side makes that concrete.
    if measure != "PMI":
        left, right = st.columns(2)
        with left:
            st.caption(f"Top 10 by {measure}")
            st.write(", ".join(shown["Pair"].head(10)))
        with right:
            by_pmi = frame.sort_values(["PMI", "Pair"], ascending=[False, True], kind="stable")
            st.caption("Top 10 by PMI")
            st.write(", ".join(by_pmi["Pair"].head(10)))
        overlap = len(set(shown["Pair"].head(10)) & set(by_pmi["Pair"].head(10)))
        st.caption(
            f"{overlap} of 10 pairs appear in both lists. "
            "PMI rewards rare tight pairs; G2 and t-score reward well-attested ones."
        )

    render_insight_panel(frame, tool="collocations", key="colloc")

    csv = shown.to_csv(index=False).encode("utf-8")
    st.download_button("Download collocations.csv", data=csv, file_name="collocations.csv", mime="text/csv")
    st.caption(
        "Sort by G2 when you need one answer. Read PMI alongside the count column: "
        "a pair seen twice can top the PMI list and mean nothing."
    )
