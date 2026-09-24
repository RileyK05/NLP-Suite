"""Run scanning — pure, testable, no Streamlit import."""

from __future__ import annotations

from pathlib import Path

from core.artifacts.envelope import Envelope

__all__ = ["RunRecord", "artifact_view", "find_runs"]


from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_dir: Path
    envelope: Envelope


def find_runs(output_root: Path) -> list[RunRecord]:
    """Find all run directories under *output_root* that contain an envelope."""
    root = Path(output_root)
    if not root.is_dir():
        return []
    records: list[RunRecord] = []
    for result_file in root.rglob("result.json"):
        envelope_result = Envelope.read(result_file.parent)
        if envelope_result.value is not None:
            records.append(RunRecord(run_dir=result_file.parent, envelope=envelope_result.unwrap()))
    records.sort(key=lambda record: record.envelope.created, reverse=True)
    return records


def artifact_view(kind: str, path: str) -> str:
    """How the gallery renders one artifact: table | markdown | html | image | download.

    Reports (batch ``report.md``, GEXF notes) render as markdown, HTML
    artifacts (Plotly charts, wordclouds) inline, CSV tables as frames,
    raster images (PNG wordclouds) inline; everything else — manifests,
    GEXF graphs, unknown kinds — is download-only. Pure so tests cover
    the mapping.
    """
    suffix = Path(path).suffix.lower()
    if kind == "table" and suffix == ".csv":
        return "table"
    if suffix == ".md":
        return "markdown"
    if suffix == ".html":
        return "html"
    if kind == "image" and suffix in (".png", ".jpg", ".jpeg", ".svg"):
        return "image"
    return "download"
