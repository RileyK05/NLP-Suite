"""NLP Suite NG — declarative tool forms (FR-8.3).

Renders one form per registry tool: widgets come from ``app.forms``
descriptors, validation from ``plan.build_plan``, execution from the shared
executor + batch writer. Runs are synchronous: long analyses block the
page, so point at a small corpus here and use ``nlp-suite jobs submit``
(or the tool CLI) for full runs. Results reopen in the gallery (Home).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from app.forms import WidgetDescriptor, widgets_for
from app.session import render_shared_sidebar
from app.state import resolve_corpus
from core.profiler.batch import BatchRequest, write_batch
from core.profiler.executor import execute
from core.profiler.plan import build_plan
from core.profiler.registry import get_tool, tool_names
from core.result import Diagnostic

DEFAULT_OUTPUT_ROOT = Path("out")


def _eligible_tools() -> list[str]:
    return [name for name in tool_names() if (spec := get_tool(name)) is not None and spec.profiler_eligible]


def _render_widget(descriptor: WidgetDescriptor) -> Any:
    """One Streamlit widget per descriptor; returns the submitted value."""
    key = f"param-{descriptor.key}"
    if descriptor.widget == "checkbox":
        return st.checkbox(descriptor.label, value=bool(descriptor.default), help=descriptor.help, key=key)
    if descriptor.widget == "selectbox":
        options = [str(choice) for choice in descriptor.choices]
        default = (
            str(descriptor.default) if descriptor.default in descriptor.choices else (options[0] if options else "")
        )
        index = options.index(default) if default in options else 0
        picked = st.selectbox(descriptor.label, options=options, index=index, help=descriptor.help, key=key)
        return next((choice for choice in descriptor.choices if str(choice) == picked), picked)
    if descriptor.widget == "number" and descriptor.default is not None:
        number_default = descriptor.default
        minimum = descriptor.minimum
        maximum = descriptor.maximum
        if isinstance(number_default, float) or isinstance(minimum, float) or isinstance(maximum, float):
            return st.number_input(
                descriptor.label,
                value=float(number_default),
                min_value=float(minimum) if minimum is not None else None,
                max_value=float(maximum) if maximum is not None else None,
                help=descriptor.help,
                key=key,
            )
        return st.number_input(
            descriptor.label,
            value=int(number_default),
            min_value=int(minimum) if minimum is not None else None,
            max_value=int(maximum) if maximum is not None else None,
            help=descriptor.help,
            key=key,
        )
    # text, path, and default-less numbers (coerced on submit)
    raw = st.text_input(
        descriptor.label,
        value="" if descriptor.default is None else str(descriptor.default),
        help=descriptor.help,
        key=key,
    )
    if descriptor.widget == "number":
        return None if raw == "" else raw
    return raw


def _coerce(descriptor: WidgetDescriptor, raw: Any) -> Any:
    """Empty inputs become None (registry defaults apply); numeric text parses."""
    if raw == "" and not descriptor.required and descriptor.widget in ("text", "path", "number"):
        return None
    if descriptor.widget == "number" and isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw  # plan validation rejects with a clear error
    return raw


st.set_page_config(page_title="NLP Suite NG — Tools", layout="wide")
st.title("Run an analysis")

tools = _eligible_tools()
if not tools:
    st.error("No batch-eligible tools in the registry.")
    st.stop()

selected = st.selectbox("Tool", options=tools)
spec = get_tool(selected)
if spec is None:  # unreachable: options came from the registry
    st.error(f"Unknown tool: {selected}")
    st.stop()
st.caption(spec.description)

state = render_shared_sidebar()
corpus_dir = state.corpus_dir
output_root = state.output_root
parser = state.parser
language = state.language

st.subheader("Parameters")
values: dict[str, Any] = {}
for descriptor in widgets_for(selected):
    values[descriptor.key] = _coerce(descriptor, _render_widget(descriptor))

if st.button("Run", type="primary"):
    planned = build_plan([selected], {selected: values})
    if planned.value is None:
        for diag in planned.diagnostics:
            st.error(str(diag))
        st.stop()
    plan = planned.unwrap()
    with st.spinner(f"Running {selected} … (synchronous; `nlp-suite jobs submit {selected}` for large corpora)"):
        from core.config import NLPConfig
        from core.io.reader import read_corpus
        from core.pipelines.cache import PipelineCache

        resolved = resolve_corpus(corpus_dir)
        if resolved.value is None:
            for diag in resolved.diagnostics:
                st.error(str(diag))
            st.stop()
        corpus_result = read_corpus(resolved.unwrap())
        # A bad document is a diagnostic, not an aborted run: keep the good
        # documents and surface the failures (matches tools/_cli.load_corpus).
        if corpus_result.value is None:
            for diag in corpus_result.diagnostics:
                st.error(str(diag))
            st.stop()
        for diag in corpus_result.diagnostics:
            st.warning(str(diag))
        corpus = corpus_result.unwrap()
        table = None
        parse_diagnostics: tuple[Diagnostic, ...] = ()
        if plan.needs_parse:
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
            table_result = pipeline_result.unwrap().parse(corpus)
            if table_result.value is None:
                for diag in table_result.diagnostics:
                    st.error(str(diag))
                st.stop()
            table = table_result.unwrap()
            parse_diagnostics = table_result.diagnostics
        batch = execute(plan, corpus=corpus, table=table, parse_diagnostics=parse_diagnostics)
        report = write_batch(
            Path(output_root),
            BatchRequest(selection=(selected,), params=plan.params, batch=batch, corpus=corpus),
        )
    if report.value is None:
        for diag in report.diagnostics:
            st.error(str(diag))
        st.stop()
    info = report.unwrap()
    failed = [o.name for o in batch.outcomes if not o.ok]
    if failed:
        st.warning(f"{selected} finished with errors — see diagnostics.")
    else:
        st.success(f"{selected} done → `{info.run_dir}`")
    for outcome in batch.outcomes:
        with st.expander(f"{outcome.name} — {'ok' if outcome.ok else 'FAILED'} ({outcome.seconds}s)"):
            for diag in outcome.diagnostics:
                st.info(str(diag))
            for artifact in info.envelope.artifacts if info.envelope is not None else []:
                st.caption(str(artifact))
    st.info("Reopen this run in the gallery (Home page).")
