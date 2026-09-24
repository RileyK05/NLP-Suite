"""Embedding visualizations — t-SNE scatter (word2vec, legacy parity).

Legacy reference: ``src/word2vec_tsne_plot_util.py`` — a Plotly scatter of the
2-D t-SNE projection with the word next to each point (2D; the legacy 3D
variant is deliberately not carried: 3-D scatters read poorly and the 2-D
projection is the research-useful form). Knowable degradation like every
viz module: without plotly, a plain sortable HTML table of the same
coordinates replaces the chart and the envelope says so.
"""

from __future__ import annotations

import html as html_lib

import pandas as pd

from core.result import Diagnostic, Result

__all__ = ["tsne_html"]


def tsne_html(frame: pd.DataFrame, title: str = "") -> Result[str]:
    """Interactive scatter of t-SNE coordinates (Word, X, Y) as HTML."""
    checked = _check(frame)
    if checked is not None:
        return Result[str](None, (checked,))

    try:
        import plotly.express as px

        fig = px.scatter(
            frame,
            x="X",
            y="Y",
            text="Word",
            hover_name="Word",
            template="plotly_white",
        )
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.update_layout(title_text=title or "t-SNE word vectors", font_size=11)
        html_str: str = fig.to_html(full_html=False, include_plotlyjs="cdn")
        return Result.success(html_str)
    except ImportError:
        pass
    except Exception as exc:
        return Result.failure(Diagnostic.error("TSNE_PLOT_FAILED", f"plotly failed: {exc}"))

    # Fallback: sortable coordinate table, same data as the chart would show.
    try:
        title_html = f"<h3>{html_lib.escape(title)}</h3>" if title else "<h3>t-SNE word vectors</h3>"
        rows_html = ""
        for _, row in frame.iterrows():
            rows_html += f"<tr><td>{html_lib.escape(str(row['Word']))}</td><td>{row['X']}</td><td>{row['Y']}</td></tr>"
        html_out = (
            f"{title_html}"
            "<style>"
            ".tsne-table{border-collapse:collapse;font-family:monospace;font-size:12px}"
            ".tsne-table th,.tsne-table td{border:1px solid #ccc;padding:3px 7px;text-align:left}"
            "</style>"
            "<table class='tsne-table'>"
            "<tr><th>Word</th><th>X</th><th>Y</th></tr>"
            f"{rows_html}"
            "</table>"
        )
        return Result.success(
            html_out,
            Diagnostic.warning(
                "TSNE_PLOTLY_UNAVAILABLE",
                "plotly is not installed; rendered a plain HTML coordinate table instead of "
                "the interactive scatter. Fix: pip install plotly",
                fix="pip install plotly",
            ),
        )
    except Exception as exc:
        return Result.failure(Diagnostic.error("TSNE_PLOT_FAILED", f"fallback failed: {exc}"))


def _check(frame: pd.DataFrame) -> Diagnostic | None:
    if frame.empty:
        return Diagnostic.error("TSNE_PLOT_EMPTY", "t-SNE frame is empty")
    missing = [c for c in ("Word", "X", "Y") if c not in frame.columns]
    if missing:
        return Diagnostic.error("TSNE_PLOT_BAD_COLUMN", f"column(s) not in frame: {missing}", missing=missing)
    return None
