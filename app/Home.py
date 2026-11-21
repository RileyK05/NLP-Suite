"""NLP Suite NG — Streamlit viewer (the gallery).

Scans run directories, groups envelopes by tool and date, and renders
artifacts by ``kind``. No per-tool code path — unknown kinds degrade to
download-only, so shipping a new tool never requires a viewer change.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from app.scanner import RunRecord, artifact_view, find_runs

DEFAULT_OUTPUT_ROOT = Path("out")
MAX_INPUTS_SHOWN = 5


@st.cache_data(ttl=10, show_spinner=False)
def _cached_runs(output_root_str: str, _nonce: int) -> list[RunRecord]:
    return find_runs(Path(output_root_str))


@st.cache_data(ttl=30, show_spinner=False)
def _cached_bytes(path_str: str, _mtime_ns: int, _size: int) -> bytes:
    return Path(path_str).read_bytes()


@st.cache_data(ttl=30, show_spinner=False)
def _cached_csv(path_str: str, _mtime_ns: int, _size: int) -> pd.DataFrame:
    return pd.read_csv(path_str)


def _file_bytes(target: Path) -> bytes:
    stat = target.stat()
    return _cached_bytes(str(target), stat.st_mtime_ns, stat.st_size)


def _render_table(run_dir: Path, artifact_path: str) -> None:
    target = run_dir / artifact_path
    if not target.is_file():
        st.warning(f"Missing table: {artifact_path}")
        return
    try:
        stat = target.stat()
        frame = _cached_csv(str(target), stat.st_mtime_ns, stat.st_size)
        raw_bytes = _cached_bytes(str(target), stat.st_mtime_ns, stat.st_size)
    except Exception as exc:
        st.error(f"Could not read {artifact_path}: {exc}")
        return
    st.dataframe(frame, use_container_width=True)
    st.download_button(
        label=f"Download {artifact_path}",
        data=raw_bytes,
        file_name=Path(artifact_path).name,
        mime="text/csv",
    )


def _render_markdown(run_dir: Path, artifact_path: str) -> None:
    target = run_dir / artifact_path
    if not target.is_file():
        st.warning(f"Missing report: {artifact_path}")
        return
    try:
        raw_bytes = _file_bytes(target)
        st.markdown(raw_bytes.decode("utf-8"))
    except Exception as exc:
        st.error(f"Could not read {artifact_path}: {exc}")
        return
    st.download_button(
        label=f"Download {artifact_path}",
        data=raw_bytes,
        file_name=Path(artifact_path).name,
        mime="text/markdown",
    )


def _render_html(run_dir: Path, artifact_path: str) -> None:
    target = run_dir / artifact_path
    if not target.is_file():
        st.warning(f"Missing HTML: {artifact_path}")
        return
    try:
        raw_bytes = _file_bytes(target)
        components.html(raw_bytes.decode("utf-8"), height=500, scrolling=True)
    except Exception as exc:
        st.error(f"Could not render {artifact_path}: {exc}")
        return
    st.download_button(
        label=f"Download {artifact_path}",
        data=raw_bytes,
        file_name=Path(artifact_path).name,
        mime="text/html",
    )


def _render_chart(run_dir: Path, artifact_path: str) -> None:
    target = run_dir / artifact_path
    if not target.is_file():
        st.warning(f"Missing chart: {artifact_path}")
        return
    try:
        raw_bytes = _file_bytes(target)
    except Exception as exc:
        st.error(f"Could not read {artifact_path}: {exc}")
        return
    if target.suffix == ".html":
        try:
            components.html(raw_bytes.decode("utf-8"), height=500, scrolling=True)
        except Exception as exc:
            st.error(f"Could not render {artifact_path}: {exc}")
    st.download_button(
        label=f"Download {artifact_path}",
        data=raw_bytes,
        file_name=Path(artifact_path).name,
    )


def _render_image(run_dir: Path, artifact_path: str) -> None:
    target = run_dir / artifact_path
    if not target.is_file():
        st.warning(f"Missing image: {artifact_path}")
        return
    st.image(str(target))
    st.download_button(
        label=f"Download {artifact_path}",
        data=_file_bytes(target),
        file_name=Path(artifact_path).name,
    )


def _render_unknown(run_dir: Path, artifact_path: str, kind: str) -> None:
    target = run_dir / artifact_path
    if not target.is_file():
        st.warning(f"Missing artifact ({kind}): {artifact_path}")
        return
    st.info(f"Unknown artifact kind {kind!r} — download only")
    st.download_button(
        label=f"Download {artifact_path}",
        data=_file_bytes(target),
        file_name=Path(artifact_path).name,
    )


st.set_page_config(page_title="NLP Suite NG", layout="wide")
st.title("NLP Suite NG — Runs")

query_output = st.query_params.get("output", None)
default_root = Path(query_output) if isinstance(query_output, str) else DEFAULT_OUTPUT_ROOT

with st.sidebar:
    st.header("Settings")
    output_root_str = st.text_input("Output root", value=str(default_root))
    output_root = Path(output_root_str)
    refresh_nonce = int(st.session_state.get("home_refresh_nonce", 0))
    if st.button("Refresh"):
        refresh_nonce += 1
        st.session_state["home_refresh_nonce"] = refresh_nonce
        st.rerun()

runs = _cached_runs(str(output_root), refresh_nonce)

if not runs:
    st.info(
        f"No runs found under {output_root}. Run a tool first: `python -m tools.conll_wordlist <corpus> {output_root}`"
    )
    st.stop()

# Group by tool for the selector
tool_names = sorted({record.envelope.tool for record in runs})
selected_tool = st.sidebar.selectbox("Tool", options=["(all)", *tool_names])

filtered = [
    record
    for record in runs
    if selected_tool == "(all)" or record.envelope.tool == selected_tool  # noqa: PLR1714
]

labels = [f"{record.envelope.tool} — {record.envelope.created} — {record.run_dir.name}" for record in filtered]
selected_label = st.selectbox("Run", options=labels)
selected = filtered[labels.index(selected_label)]

envelope = selected.envelope
st.subheader(f"{envelope.tool} — {envelope.created}")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"**Tool:** `{envelope.tool}`")
    st.markdown(f"**Created:** {envelope.created}")
    st.markdown(f"**Run dir:** `{selected.run_dir}`")
with col_b:
    st.markdown(f"**Inputs:** {len(envelope.inputs)} file(s)")
    for inp in envelope.inputs[:MAX_INPUTS_SHOWN]:
        st.caption(f"{inp.path} — {inp.sha256[:8]}")
    if len(envelope.inputs) > MAX_INPUTS_SHOWN:
        st.caption(f"... and {len(envelope.inputs) - MAX_INPUTS_SHOWN} more")
    st.markdown(f"**Params:** `{envelope.params}`")

if envelope.diagnostics:
    with st.expander(f"Diagnostics ({len(envelope.diagnostics)})", expanded=False):
        for diag in envelope.diagnostics:
            if diag.severity.value == "ERROR":
                st.error(str(diag))
            elif diag.severity.value == "WARNING":
                st.warning(str(diag))
            else:
                st.info(str(diag))

st.divider()
st.subheader("Artifacts")

for artifact in envelope.artifacts:
    st.markdown(f"**{artifact.path}** — *{artifact.kind}*")
    if artifact.description:
        st.caption(artifact.description)
    view = artifact_view(artifact.kind, artifact.path)
    if view == "table":
        _render_table(selected.run_dir, artifact.path)
    elif view == "markdown":
        _render_markdown(selected.run_dir, artifact.path)
    elif view == "html":
        _render_html(selected.run_dir, artifact.path)
    elif view == "image":
        _render_image(selected.run_dir, artifact.path)
    elif artifact.kind == "chart":
        _render_chart(selected.run_dir, artifact.path)
    else:
        _render_unknown(selected.run_dir, artifact.path, artifact.kind)
    st.divider()
