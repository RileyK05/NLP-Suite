"""Batch report (FR-7.6) — failure-transparent markdown summary.

Renders a manifest (as written by ``batch.write_batch``) to markdown.
Failures are listed with their diagnostic codes, never folded into prose;
reused children are marked. Pure function: trivially testable, and the
CLI writes its output as ``report.md`` beside ``batch.json``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["render_report"]


def render_report(manifest: Mapping[str, Any]) -> str:
    """Render one batch manifest to a markdown report."""
    children = [c for c in manifest.get("children", []) if isinstance(c, dict)]
    selection = manifest.get("selection", [])
    lines = [
        "# Profiler batch report",
        "",
        f"Selection: {', '.join(str(name) for name in selection) or '(none)'}",
        "",
        "| Tool | Status | Seconds | Artifacts |",
        "|---|---|---|---|",
    ]
    failures: list[dict[str, Any]] = []
    for child in children:
        name = str(child.get("tool", "?"))
        ok = bool(child.get("ok", False))
        seconds = child.get("seconds", 0.0)
        artifacts = child.get("artifacts", [])
        status = "ok" if ok else "FAILED"
        if ok and child.get("reused"):
            status = "ok (reused)"
        artifact_cell = "<br>".join(str(a) for a in artifacts) if artifacts else "—"
        lines.append(f"| {name} | {status} | {seconds} | {artifact_cell} |")
        if not ok:
            failures.append(child)
    lines.append("")
    if failures:
        lines.extend(["## Failures", ""])
        for child in failures:
            lines.append(f"### {child.get('tool', '?')} (after {child.get('seconds', 0.0)}s)")
            lines.append("")
            diags = child.get("diagnostics", [])
            if not diags:
                lines.append("- no diagnostics recorded")
            for diag in diags:
                if isinstance(diag, dict):
                    lines.append(f"- `{diag.get('code', '?')}`: {diag.get('message', '')}")
                else:
                    lines.append(f"- {diag}")
            lines.append("")
    else:
        lines.extend(["All tools succeeded.", ""])
    return "\n".join(lines)
