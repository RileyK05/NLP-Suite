"""Shapes & networks — story shape, Sankey, network."""

from __future__ import annotations

import html as html_lib

import pandas as pd

from core.analysis.postags import is_noun_tag
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["network_gexf", "sankey_html", "story_shape"]


def story_shape(frame: pd.DataFrame) -> Result[pd.DataFrame]:
    """Per-sentence shape metrics: token count and noun/verb ratios.

    This is the deterministic core of shape_of_stories_* — no model, just
    the CoNLL parse grouped by sentence.
    """
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    for need in [Col.FORM.value, Col.POS.value, Col.SENTENCE_ID.value, Col.DOCUMENT_ID.value]:
        if need not in frame.columns:
            return Result.failure(Diagnostic.error("SHAPE_MISSING_COLUMN", f"missing {need!r}"))
    if frame.empty:
        return Result.success(
            pd.DataFrame(columns=["Document ID", "Sentence ID", "Tokens", "Noun Ratio", "Verb Ratio", "Sentence"])
        )

    rows: list[dict[str, object]] = []
    for (did, sid), g in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        toks = g[Col.FORM.value].tolist()
        pos = g[Col.POS.value].astype(str).tolist()
        n = len(toks)
        nouns = sum(1 for p in pos if is_noun_tag(p))
        verbs = sum(1 for p in pos if p.startswith("VB") or p in ("VERB", "AUX"))
        sent_text = " ".join(str(x) for x in toks)
        rows.append(
            {
                "Document ID": str(did),
                "Sentence ID": sid,
                "Tokens": n,
                "Noun Ratio": round(nouns / n, 4) if n else 0.0,
                "Verb Ratio": round(verbs / n, 4) if n else 0.0,
                "Sentence": sent_text,
            }
        )
    df = pd.DataFrame(rows, columns=["Document ID", "Sentence ID", "Tokens", "Noun Ratio", "Verb Ratio", "Sentence"])
    # First-appearance order from groupby(sort=False) is the document order;
    # sorting by the string Document ID would put "10" before "2".
    return Result.success(df)


def sankey_html(
    frame: pd.DataFrame,
    source: str,
    target: str,
    value: str = "",
    title: str = "",
) -> Result[str]:
    """Sankey as Plotly HTML if available, else a simple table fallback."""
    if frame.empty:
        return Result.failure(Diagnostic.error("SANKEY_EMPTY", "frame is empty"))
    missing = [c for c in (source, target) if c not in frame.columns]
    if value and value not in frame.columns:
        # A requested weight column must exist; silently dropping it would
        # make every link weight 1.0 while the caller believes otherwise.
        missing.append(value)
    if missing:
        return Result.failure(
            Diagnostic.error("SANKEY_BAD_COLUMN", f"column(s) not in frame: {missing}", missing=missing)
        )

    # Try Plotly
    try:
        import plotly.graph_objects as go

        # Map labels to indices
        labels = pd.concat([frame[source].astype(str), frame[target].astype(str)]).unique().tolist()
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        src_idx = frame[source].astype(str).map(label_to_idx).tolist()
        tgt_idx = frame[target].astype(str).map(label_to_idx).tolist()
        vals = pd.to_numeric(frame[value], errors="coerce").fillna(1).tolist() if value else [1.0] * len(frame)
        fig = go.Figure(
            data=[
                go.Sankey(
                    node={"label": labels},
                    link={"source": src_idx, "target": tgt_idx, "value": vals},
                )
            ]
        )
        fig.update_layout(title_text=title or "Sankey", font_size=10)
        html_str: str = fig.to_html(full_html=False, include_plotlyjs="cdn")
        return Result.success(html_str)
    except ImportError:
        pass
    except Exception as exc:
        return Result.failure(Diagnostic.error("SANKEY_FAILED", f"plotly failed: {exc}"))

    # Fallback
    try:
        title_html = f"<h3>{html_lib.escape(title)}</h3>" if title else ""
        rows_html = ""
        for _, row in frame.iterrows():
            s = html_lib.escape(str(row[source]))
            t = html_lib.escape(str(row[target]))
            v = html_lib.escape(str(row[value])) if value and value in frame.columns else "1"
            rows_html += f"<tr><td>{s}</td><td>→</td><td>{t}</td><td>{v}</td></tr>"
        html_str = (
            f"{title_html}<table border='1' cellpadding='4' style='border-collapse:collapse'>"
            f"<tr><th>{html_lib.escape(source)}</th><th></th><th>{html_lib.escape(target)}</th><th>{html_lib.escape(value) if value else 'Weight'}</th></tr>"
            f"{rows_html}</table>"
        )
        # Knowable degradation: the link list shows the same rows and weights
        # as the Sankey would, but it is a table, not a flow diagram — the
        # envelope must say so.
        return Result.success(
            html_str,
            Diagnostic.warning(
                "SANKEY_PLOTLY_UNAVAILABLE",
                "plotly is not installed; rendered a plain HTML link table with the same data instead. "
                "Fix: pip install plotly",
                fix="pip install plotly",
            ),
        )
    except Exception as exc:
        return Result.failure(Diagnostic.error("SANKEY_FAILED", f"fallback failed: {exc}"))


def network_gexf(
    frame: pd.DataFrame,
    source: str,
    target: str,
    weight: str | None = None,
) -> Result[str]:
    """Thin wrapper over Gephi GEXF for network graphs."""
    from core.viz.wordcloud_gephi import gephi_gexf

    return gephi_gexf(frame, source_col=source, target_col=target, weight_col=weight)
