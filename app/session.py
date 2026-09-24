"""Shared sidebar (FR-8.1) — the Streamlit half of ``app.state``.

Importing this module needs a Streamlit runtime (like every page); the
state object and validation stay in ``app.state`` so tests cover them.
"""

from __future__ import annotations

import streamlit as st

from app.state import DEFAULT_STATE, AppState

__all__ = ["render_shared_sidebar"]


def render_shared_sidebar() -> AppState:
    """Render the corpus & environment sidebar; persist values across pages."""
    saved = st.session_state.get("app_state", DEFAULT_STATE)
    if not isinstance(saved, AppState):
        saved = DEFAULT_STATE
    with st.sidebar:
        st.header("Corpus & environment")
        corpus_dir = st.text_input("Corpus directory", value=saved.corpus_dir)
        output_root = st.text_input("Output root", value=saved.output_root)
        parser = st.selectbox("Parser", options=["spacy", "stanza"], index=["spacy", "stanza"].index(saved.parser))
        language = st.text_input("Language", value=saved.language)
    state = AppState(corpus_dir=corpus_dir, output_root=output_root, parser=parser, language=language)
    st.session_state["app_state"] = state
    return state
