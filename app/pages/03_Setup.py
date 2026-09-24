"""NLP Suite NG — guided setup (FR-8.6).

First-run readiness checklist over the shared-sidebar values: fast,
import-only checks with fixes inline. Anything deeper (parser models,
optional extras) belongs to `nlp-suite doctor`, linked below.
"""

from __future__ import annotations

import streamlit as st

from app.session import render_shared_sidebar
from app.state import setup_readiness

st.set_page_config(page_title="Setup", page_icon="🧭")
st.title("Guided setup")

state = render_shared_sidebar()

items = setup_readiness(state.corpus_dir, parser=state.parser)
ready = all(item.ok for item in items)
for item in items:
    if item.ok:
        st.success(item.label)
    else:
        st.error(item.label)
        if item.fix:
            st.caption(f"Fix: {item.fix}")

if ready:
    st.success("Ready — pick a tool on the Tools page or validate the corpus first.")
else:
    st.warning("Fix the rows above, then come back. Nothing here downloads models.")
st.info("For parser models and optional extras, run `nlp-suite doctor en` in a terminal.")
