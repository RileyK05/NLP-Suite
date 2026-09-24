"""Explicit CSV workflows, sharing engine statistics and artifact publication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.profiler.registry import ParamSpec, ToolSpec
from core.result import Result
from core.viz.chartspec import CHART_KINDS
from desktop_backend.store import Workspace


def column(name: str) -> ParamSpec:
    return ParamSpec(name, "str", "", True, "Exact CSV column heading")


INPUT = ParamSpec("input", "path", None, True, "Choose the original CSV data table, not converted corpus text")
ALPHA = ParamSpec(
    "alpha", "float", 0.05, False, "Significance level (strictly between zero and one)", minimum=0, maximum=1
)


#: Which syllabus family each table workflow belongs to (R: tool taxonomy).
#: A CSV workflow is still a statistics or a visualization question; the
#: gallery groups by the question, not by the input format.
_TABLE_FAMILIES = {
    "keyness": "words",
    "charts": "visualization",
    "wordcloud_gephi": "visualization",
}


def table_spec(name: str, description: str, params: tuple[ParamSpec, ...]) -> ToolSpec:
    return ToolSpec(
        name="table_" + name,
        capability_ids=(),
        packet="FR-8.1",
        description=description,
        requires_parse=False,
        params=(INPUT, *params),
        outputs=(),
        version="1",
        input_kind="csv",
        family=_TABLE_FAMILIES.get(name, "corpus_statistics"),
        profiler_eligible=False,
    )


TABLE_TOOLS = {
    spec.name: spec
    for spec in (
        table_spec(
            "chi2",
            "Chi-square association, residuals and observed/expected counts",
            (column("col1"), column("col2"), ALPHA),
        ),
        table_spec("crosstab", "Cross-tabulated counts and row percentages", (column("col1"), column("col2"))),
        table_spec(
            "keyness",
            "Log-likelihood keyword comparison between two frequency columns",
            (column("word-col"), column("freq1"), column("freq2")),
        ),
        table_spec("mw", "Mann-Whitney comparison of two groups", (column("value-col"), column("group-col"), ALPHA)),
        table_spec(
            "kw",
            "Kruskal-Wallis group comparison with Dunn post-hoc tests",
            (
                column("value-col"),
                column("group-col"),
                ALPHA,
                ParamSpec(
                    "posthoc-method",
                    "str",
                    "holm",
                    False,
                    "Multiple-comparison correction",
                    choices=("holm", "bonferroni"),
                ),
            ),
        ),
        table_spec(
            "trend",
            "Mann-Kendall trend and Sen slope over dated observations",
            (column("date-col"), column("value-col")),
        ),
        table_spec(
            "rankcorr",
            "Spearman or Kendall rank correlation",
            (
                column("col-x"),
                column("col-y"),
                ParamSpec("method", "str", "spearman", False, "Correlation method", choices=("spearman", "kendall")),
            ),
        ),
        table_spec(
            "charts",
            "Publication-ready chart over an analyst CSV (bar/line/scatter/histogram/box/heatmap + more)",
            (
                # Derived, never restated: a hand-written copy here silently
                # withheld a chart kind the engine could draw (R6).
                ParamSpec("kind", "str", "bar", True, "chart kind", choices=CHART_KINDS),
                column("x"),
                column("y"),
                ParamSpec("group", "str", "", False, "group column (color/series; heatmap row axis)"),
                ParamSpec(
                    "agg",
                    "str",
                    "",
                    False,
                    "aggregation for duplicates: sum|mean|median|count",
                    choices=("", "sum", "mean", "median", "count"),
                ),
                ParamSpec("top-n", "int", None, False, "keep n categories by total", minimum=1),
                ParamSpec("normalize", "str", "none", False, "percent|share", choices=("none", "percent", "share")),
                ParamSpec("bins", "int", None, False, "histogram bin count", minimum=1),
                ParamSpec("title", "str", "", False, "chart title"),
                ParamSpec(
                    "format", "str", "html", False, "output format", choices=("html", "png", "svg", "pdf", "xlsx")
                ),
            ),
        ),
        table_spec(
            "wordcloud_gephi",
            "Wordcloud (HTML/PNG) or Gephi GEXF network over an analyst CSV",
            (
                ParamSpec("mode", "str", "wordcloud", True, "what to build", choices=("wordcloud", "gexf")),
                column("word-col"),
                column("weight-col"),
                ParamSpec("source-col", "str", "", False, "edge source column (gexf mode)"),
                ParamSpec("target-col", "str", "", False, "edge target column (gexf mode)"),
                ParamSpec("title", "str", "", False, "wordcloud title"),
                ParamSpec("max-words", "int", 100, False, "maximum words", minimum=1),
                ParamSpec("image", "bool", False, False, "also render a raster PNG wordcloud"),
                ParamSpec(
                    "shape", "str", "", False, "procedural mask shape (e.g. butterfly)", choices=("", "butterfly")
                ),
            ),
        ),
    )
}


def run_table(workspace: Workspace, job: dict[str, Any], params: dict[str, Any]) -> int:
    import pandas as pd

    from core.analysis.stats_categorical import chi_square, crosstab, log_likelihood
    from core.analysis.stats_groups import kruskal_wallis, mann_whitney
    from core.analysis.stats_trends import correlation_summary, mann_kendall
    from core.io.writer import OutputWriter

    path = Path(params["input"])
    frame = pd.read_csv(path, encoding="utf-8-sig")
    tool = job["tool"]
    if tool in ("table_charts", "table_wordcloud_gephi"):
        return run_table_visualization(workspace, job, tool, params, path, frame)
    result: Result[Any]
    members: tuple[str, ...]
    # Each branch invokes the same engine function as its existing thin CLI.
    if tool == "table_chi2":
        result = chi_square(frame, params["col1"], params["col2"], alpha=params["alpha"])
        members = ("summary", "residuals", "obs_exp")
    elif tool == "table_crosstab":
        result = crosstab(frame, params["col1"], params["col2"])
        members = ("counts", "row_pct")
    elif tool == "table_keyness":
        result = log_likelihood(frame, params["word-col"], params["freq1"], params["freq2"])
        members = ("to_frame",)
    elif tool == "table_mw":
        result = mann_whitney(frame, params["value-col"], params["group-col"], alpha=params["alpha"])
        members = ("summary", "medians")
    elif tool == "table_kw":
        result = kruskal_wallis(
            frame,
            params["value-col"],
            params["group-col"],
            alpha=params["alpha"],
            posthoc_method=params["posthoc-method"],
        )
        members = ("summary", "medians", "posthoc")
    elif tool == "table_trend":
        result = mann_kendall(frame, params["date-col"], params["value-col"])
        members = ("summary", "trend")
    elif tool == "table_rankcorr":
        result = correlation_summary(frame, params["col-x"], params["col-y"], params["method"])
        members = ()
    else:
        raise ValueError("Unknown table workflow")
    if not result.ok:
        workspace.update_job(
            job["id"],
            state="FAILED",
            stage="Check table and column settings",
            diagnostics=[d.to_dict() for d in result.diagnostics],
        )
        return 1
    value = result.unwrap()
    frames = (
        {name: getattr(value, name)() if name == "to_frame" else getattr(value, name) for name in members}
        if members
        else {"correlation": value}
    )
    project_dir = workspace.project_dir(job["project_id"])
    writer = OutputWriter(project_dir / "runs", tool=tool, params=params, inputs=(path,))
    try:
        for name, table in frames.items():
            written = writer.write_table(table, name + ".csv", kind="table")
            if not written.ok:
                raise ValueError("; ".join(d.message for d in written.diagnostics))
        writer.add_diagnostics(*result.diagnostics)
        envelope = writer.finalize()
        if not envelope.ok:
            raise ValueError("; ".join(d.message for d in envelope.diagnostics))
    except Exception:
        writer.abandon()
        raise
    workspace.update_job(
        job["id"],
        state="DONE",
        stage="Results ready",
        run_dir=writer.run_dir.relative_to(project_dir).as_posix(),
        diagnostics=[d.to_dict() for d in result.diagnostics],
    )
    return 0


def run_table_visualization(
    workspace: Workspace,
    job: dict[str, Any],
    tool: str,
    params: dict[str, Any],
    path: Path,
    frame: Any,
) -> int:
    """Charts / wordcloud / GEXF workflows over an analyst CSV (FR-6.5, FR-6.7)."""
    from core.io.writer import OutputWriter

    def fail(diagnostics: Any, stage: str) -> int:
        workspace.update_job(
            job["id"],
            state="FAILED",
            stage=stage,
            diagnostics=[d.to_dict() for d in diagnostics],
        )
        return 1

    project_dir = workspace.project_dir(job["project_id"])
    writer = OutputWriter(project_dir / "runs", tool=tool.removeprefix("table_"), params=params, inputs=(path,))
    diagnostics: list[Any] = []
    try:
        if tool == "table_charts":
            from core.viz.charts_excel import excel_chart_bytes
            from core.viz.chartspec import ChartSpec, coerce_date_axis, prepare_chart_data
            from core.viz.plotters import chart_image_bytes, render_chart

            kind = params["kind"]
            agg = params.get("agg") or None
            spec_params: dict[str, Any] = {
                "kind": kind,
                "x": params["x"],
                "y": params["y"],
                "group": params["group"] or None,
                "agg": agg,
                "normalize": params.get("normalize", "none"),
                "top_n": params.get("top-n"),
                "title": params.get("title", ""),
                "bins": None,
            }
            if kind == "histogram" and params.get("bins"):
                spec_params["bins"] = int(params["bins"])
            spec = ChartSpec(**spec_params)
            # A CSV has no dtypes, so a date column arrives as text. Kinds that
            # need a real date axis get one here, at the boundary where the
            # file is read; without it no calendar chart could be drawn from
            # any file at all.
            dated = coerce_date_axis(frame, spec)
            if dated.value is None:
                writer.abandon()
                return fail(dated.diagnostics, "Check chart settings")
            prepared = prepare_chart_data(dated.unwrap(), spec)
            if prepared.value is None:
                writer.abandon()
                return fail(prepared.diagnostics, "Check chart settings")
            chart = prepared.unwrap()
            rendered = render_chart(chart, spec)
            if rendered.value is None:
                writer.abandon()
                return fail(rendered.diagnostics, "Chart rendering failed")
            written = writer.write_html(
                rendered.unwrap(), "chart.html", kind="chart", description=f"{kind} chart {spec.y} by {spec.x}"
            )
            if not written.ok:
                writer.abandon()
                return fail(written.diagnostics, "Chart writing failed")
            written = writer.write_table(chart.data, "chart_data.csv", kind="table", description="prepared chart data")
            if not written.ok:
                writer.abandon()
                return fail(written.diagnostics, "Chart data writing failed")
            fmt = params.get("format", "html")
            if fmt != "html":
                if fmt == "xlsx":
                    image = excel_chart_bytes(chart, spec)
                    target = "chart.xlsx"
                    kind_str = "table"
                else:
                    image = chart_image_bytes(chart, spec, fmt)
                    target = f"chart.{fmt}"
                    kind_str = "image"
                if image.value is None:
                    writer.abandon()
                    return fail(image.diagnostics, f"{fmt} export failed")
                written = writer.write_bytes(image.unwrap(), target, kind=kind_str, description=f"{kind} chart ({fmt})")
                if not written.ok:
                    writer.abandon()
                    return fail(written.diagnostics, f"{fmt} export failed")
            diagnostics = list(rendered.diagnostics)
        else:  # wordcloud_gephi
            from core.viz.wordcloud_gephi import gephi_gexf, wordcloud_html, wordcloud_image

            word_col, weight_col = params["word-col"], params["weight-col"]
            if params["mode"] == "gexf":
                source_col, target_col = params.get("source-col"), params.get("target-col")
                if not source_col or not target_col:
                    writer.abandon()
                    return fail([], "GEXF mode needs source and target columns")
                built = gephi_gexf(frame, source_col=source_col, target_col=target_col, weight_col=weight_col)
                if built.value is None:
                    writer.abandon()
                    return fail(built.diagnostics, "GEXF build failed")
                written = writer.write_html(built.unwrap(), "graph.gexf", kind="gexf", description="Gephi edge list")
                if not written.ok:
                    writer.abandon()
                    return fail(written.diagnostics, "GEXF writing failed")
            else:
                built = wordcloud_html(
                    frame,
                    word_col=word_col,
                    weight_col=weight_col,
                    title=params.get("title", ""),
                    max_words=int(params.get("max-words", 100)),
                )
                if built.value is None:
                    writer.abandon()
                    return fail(built.diagnostics, "Wordcloud build failed")
                written = writer.write_html(built.unwrap(), "wordcloud.html", kind="chart", description="wordcloud")
                if not written.ok:
                    writer.abandon()
                    return fail(built.diagnostics, "Wordcloud writing failed")
                written = writer.write_table(
                    frame[[word_col, weight_col]],
                    "wordcloud_words.csv",
                    kind="table",
                    description="word-frequency pairs feeding the cloud",
                )
                if not written.ok:
                    writer.abandon()
                    return fail(written.diagnostics, "Wordcloud words writing failed")
                if params.get("image"):
                    image_path = workspace.project_dir(job["project_id"]) / "runs" / ".staging_wc.png"
                    rendered_png = wordcloud_image(
                        frame,
                        word_col=word_col,
                        weight_col=weight_col,
                        output_path=image_path,
                        max_words=int(params.get("max-words", 200)),
                        shape=params.get("shape") or None,
                    )
                    if rendered_png.value is not None:
                        png_bytes = rendered_png.unwrap().read_bytes()
                        written = writer.write_bytes(
                            png_bytes, "wordcloud.png", kind="image", description="raster wordcloud"
                        )
                        if not written.ok:
                            writer.abandon()
                            return fail(written.diagnostics, "Wordcloud image failed")
                        diagnostics = list(rendered_png.diagnostics)
                    else:
                        writer.abandon()
                        return fail(rendered_png.diagnostics, "Wordcloud image failed")
        writer.add_diagnostics(*diagnostics)
        envelope = writer.finalize()
        if not envelope.ok:
            raise ValueError("; ".join(d.message for d in envelope.diagnostics))
    except Exception as exc:
        writer.abandon()
        raise ValueError(str(exc)) from exc
    workspace.update_job(
        job["id"],
        state="DONE",
        stage="Results ready",
        run_dir=writer.run_dir.relative_to(project_dir).as_posix(),
        diagnostics=[d.to_dict() for d in diagnostics],
    )
    return 0
