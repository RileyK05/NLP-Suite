"""N-gram viewer chart — frequency over time as HTML (FR-1.6).

One line per N-gram, Year on x, Per Million on y (the culturomics rate axis;
raw counts stay in the CSV beside the chart). Knowable degradation like every
viz module: without plotly, a plain sortable HTML table of the same series
replaces the chart and the envelope says so. An empty series is not a failure —
it is the honest picture "no hits".
"""

from __future__ import annotations

import html as html_lib
from itertools import pairwise

import pandas as pd

from core.result import Diagnostic, Result

__all__ = ["ngram_viewer_html"]


def ngram_viewer_html(series: pd.DataFrame, *, title: str = "N-gram viewer") -> Result[str]:
    """Line chart of the n-gram series (N-gram, Year, Count, Per Million, ...)."""
    missing = [c for c in ("N-gram", "Year", "Per Million") if c not in series.columns]
    if series.empty:
        return Result.success(
            f"<p>{html_lib.escape(title)}: no hits to plot. "
            "Either the queried n-grams never occur in this corpus, or the run has no dated documents.</p>"
        )
    if missing:
        return Result.failure(
            Diagnostic.error("NG_VIEWER_PLOT_BAD_COLUMN", f"column(s) not in frame: {missing}", missing=missing)
        )

    try:
        import plotly.express as px

        # Plotly only breaks a line at an explicit null. Insert null-valued
        # sentinels for years with no corpus exposure; zero-hit exposure years
        # already arrive as real zero-valued rows from ngram_series.
        plot_series = series.copy()
        gap_rows: list[dict[str, object]] = []
        for _query, group in series.groupby("N-gram", sort=False):
            ordered = group.sort_values("Year", kind="stable")
            years = pd.to_numeric(ordered["Year"], errors="coerce").dropna().astype(int).tolist()
            template = ordered.iloc[0].to_dict()
            for left, right in pairwise(years):
                if right - left > 1:
                    gap = dict(template)
                    gap["Year"] = (left + right) / 2
                    for column in (
                        "Count",
                        "Per Million",
                        "Share of Documents",
                        "Raw Count",
                        "Raw Per Million",
                        "Corpus Tokens",
                        "Corpus Documents",
                        "Documents With Hit",
                    ):
                        if column in gap:
                            gap[column] = None
                    gap_rows.append(gap)
        if gap_rows:
            plot_series = pd.DataFrame([*series.to_dict("records"), *gap_rows])
            plot_series = plot_series.sort_values(["N-gram", "Year"], kind="stable")
        fig = px.line(
            plot_series,
            x="Year",
            y="Per Million",
            color="N-gram",
            markers=True,
            hover_data=[
                c
                for c in (
                    "Count",
                    "Share of Documents",
                    "Raw Count",
                    "Corpus Tokens",
                    "Corpus Documents",
                    "Documents With Hit",
                )
                if c in series.columns
            ],
            template="plotly_white",
        )
        fig.update_traces(connectgaps=False)
        fig.update_layout(title_text=title or "N-gram viewer", font_size=11)
        html_str: str = fig.to_html(full_html=False, include_plotlyjs="cdn")
        return Result.success(html_str)
    except ImportError:
        pass
    except Exception as exc:
        return Result.failure(Diagnostic.error("NG_VIEWER_PLOT_FAILED", f"plotly failed: {exc}"))

    # Fallback: plain table, same rows the chart would have drawn.
    try:
        columns = [c for c in series.columns]
        header = "".join(f"<th>{html_lib.escape(str(c))}</th>" for c in columns)
        rows_html = ""
        for _, row in series.iterrows():
            cells = "".join(f"<td>{html_lib.escape(str(row[c]))}</td>" for c in columns)
            rows_html += f"<tr>{cells}</tr>"
        html_out = (
            f"<h3>{html_lib.escape(title)}</h3>"
            "<style>"
            ".ngv-table{border-collapse:collapse;font-family:monospace;font-size:12px}"
            ".ngv-table th,.ngv-table td{border:1px solid #ccc;padding:3px 7px;text-align:left}"
            "</style>"
            f"<table class='ngv-table'><tr>{header}</tr>{rows_html}</table>"
        )
        return Result.success(
            html_out,
            Diagnostic.warning(
                "NG_VIEWER_PLOTLY_UNAVAILABLE",
                "plotly is not installed; rendered a plain HTML table of the series instead of "
                "the interactive line chart. Fix: pip install plotly",
                fix="pip install plotly",
            ),
        )
    except Exception as exc:
        return Result.failure(Diagnostic.error("NG_VIEWER_PLOT_FAILED", f"fallback failed: {exc}"))
