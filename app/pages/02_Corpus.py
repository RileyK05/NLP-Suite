"""NLP Suite NG — corpus selection and validation (FR-8.2).

Validates the shared-sidebar corpus with ``core.data.validation``
(the same check the ``corpus_validation`` CLI runs) and shows the
per-file report plus a status rollup. Fix problems here before running
tools: every tool page reads this same directory.
"""

from __future__ import annotations

import streamlit as st

from app.session import render_shared_sidebar
from app.state import resolve_corpus, summarize_validation
from core.data.validation import validate
from core.io.reader import read_corpus

st.set_page_config(page_title="Corpus", page_icon="📂")
st.title("Corpus selection and validation")

state = render_shared_sidebar()

resolved = resolve_corpus(state.corpus_dir)
if resolved.value is None:
    for diag in resolved.diagnostics:
        st.error(str(diag))
    st.stop()
corpus_dir = resolved.unwrap()
st.caption(f"Validating `{corpus_dir}` (.txt files, minimum 1 token)")

report = validate(corpus_dir)
if report.value is None:
    for diag in report.diagnostics:
        st.error(str(diag))
    st.stop()
frame = report.unwrap()
summary = summarize_validation(frame)
st.metric("Files", summary["files"])
st.metric("OK", summary["ok"])
st.metric("With issues", summary["issues"])
st.metric("Tokens", summary["tokens"])
st.subheader("Per-file report")
st.dataframe(frame, use_container_width=True)
for diag in report.diagnostics:
    st.info(str(diag))

loaded = read_corpus(corpus_dir)
if loaded.value is None:
    st.warning("The corpus reader could not load these files — see diagnostics.")
    for diag in loaded.diagnostics:
        st.info(str(diag))
else:
    st.success(f"Reader loaded {len(loaded.unwrap())} document(s) — tools will see the same set.")
