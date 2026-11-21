"""Charts — single Plotly wrapper."""

from __future__ import annotations

import html as html_lib

import pandas as pd

from core.result import Diagnostic, Result

__all__ = ["bar_chart_html"]


def bar_chart_html(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
) -> Result[str]:
    """Render a bar chart as HTML.

    If ``plotly`` is installed it is used lazily; otherwise a minimal
    HTML table fallback is emitted **with a WARNING diagnostic naming the
    degraded rendering** — the fallback shows the same numbers and relative
    bar widths, but it is a table, not an interactive chart, and the caller
    must be able to tell the difference from the envelope alone. No file is
    written — the caller decides where to put the HTML (typically via
    ``OutputWriter.write_html``).
    """
    if frame.empty:
        return Result.failure(Diagnostic.error("CHART_EMPTY", "frame is empty"))
    if x not in frame.columns or y not in frame.columns:
        return Result.failure(Diagnostic.error("CHART_BAD_COLUMN", f"columns {x!r}, {y!r} must be in frame"))

    # Try Plotly lazily (R1: no import at module load)
    try:
        import plotly.express as px

        fig = px.bar(frame, x=x, y=y, title=title or f"{y} by {x}")
        fig.update_layout(template="plotly_white", height=400)
        html_str: str = fig.to_html(full_html=False, include_plotlyjs="cdn")
        return Result.success(html_str)
    except ImportError:
        pass
    except Exception as exc:
        return Result.failure(Diagnostic.error("CHART_FAILED", f"plotly failed: {exc}"))

    # Fallback: simple HTML table + inline bar widths
    try:
        y_vals = pd.to_numeric(frame[y], errors="coerce").fillna(0).tolist()
        x_vals = frame[x].tolist()
        # Bar length shows magnitude. The old divisor was the largest signed
        # value, so an all-negative column drew its smallest value as a full
        # bar and a mixed column could exceed the track (a -8 against a 4 gave
        # -200%).
        largest = max((abs(float(value)) for value in y_vals), default=0.0)
        row_chunks: list[str] = []
        for x_raw, y_raw in zip(x_vals, y_vals, strict=True):
            # Reuse the coerced value: a per-row ``nan or 0`` fallback does not
            # work — NaN is truthy, so the "or" never fired.
            yv = float(y_raw) if pd.notna(y_raw) else 0.0
            xv = html_lib.escape(str(x_raw))
            pct = (abs(yv) / largest * 100) if largest else 0
            row_chunks.append(
                f'<tr><td>{xv}</td><td style="width:60%"><div style="background:#378ADD;height:16px;width:{pct:.1f}%"></div></td>'
                f"<td>{yv:.2f}</td></tr>"
            )
        rows_html = "".join(row_chunks)
        title_html = f"<h3>{html_lib.escape(title)}</h3>" if title else ""
        html_str = (
            f"{title_html}<table border='1' cellpadding='4' style='border-collapse:collapse;width:100%'>"
            f"<tr><th>{html_lib.escape(x)}</th><th>Bar</th><th>{html_lib.escape(y)}</th></tr>"
            f"{rows_html}</table>"
        )
        # Knowable degradation: the fallback is an honest approximation of the
        # same data (identical numbers, relative bar widths), but the caller
        # must see in the envelope that plotly was not used.
        return Result.success(
            html_str,
            Diagnostic.warning(
                "CHART_PLOTLY_UNAVAILABLE",
                "plotly is not installed; rendered a plain HTML table with the same data instead. "
                "Fix: pip install plotly",
                fix="pip install plotly",
            ),
        )
    except Exception as exc:
        return Result.failure(Diagnostic.error("CHART_FAILED", f"fallback failed: {exc}"))
