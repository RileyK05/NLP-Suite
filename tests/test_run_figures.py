"""A finished run publishes its figures beside its tables.

Through the run's own writer, so every figure is sealed, hashed and listed in
the envelope like any other artifact; and never at the tables' expense -- a
figure that cannot be drawn is a warning, and the run still publishes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.io.reader import hash_file
from core.profiler.batch import BatchRequest, write_batch
from core.profiler.executor import BatchResult, ToolOutcome

pytest.importorskip("seaborn")

KEYNESS = pd.DataFrame(
    {
        "Word": ["freedom", "the", "tariff", "labor", "war"],
        "Freq Group A (pattern docs)": [120, 9000, 1, 60, 80],
        "Freq Group B (other docs)": [12, 8700, 19, 5, 20],
        "G2 (log-likelihood)": [88.4, 41.2, 12.1, 50.3, 30.2],
        "p-value": [0.0, 0.0, 0.001, 0.0, 0.0],
        "Log Ratio": [2.31, 0.04, -3.2, 2.9, 1.4],
        "Overrepresented in": ["Group A", "Group A", "Group B", "Group A", "Group A"],
    }
)


def publish(root: Path, *, figures: bool, frames: dict[str, pd.DataFrame] | None = None) -> Path:
    outcome = ToolOutcome(name="keyness", ok=True, frames=frames or {"keyness.csv": KEYNESS})
    report = write_batch(root, BatchRequest(("keyness",), {"keyness": {}}, BatchResult((outcome,)), figures=figures))
    assert report.ok, [str(d) for d in report.diagnostics]
    return report.unwrap().child_dirs["keyness"]


def test_a_run_with_figures_carries_them_sealed_in_its_envelope(tmp_path: Path) -> None:
    from core.artifacts.envelope import Envelope

    run = publish(tmp_path, figures=True)
    listed = {artifact.path for artifact in Envelope.read(run).unwrap().artifacts}
    for name in ("figures/keyness_volcano.png", "figures/keyness_volcano.svg", "figures/README.md"):
        assert (run / name).is_file(), name
        assert name in listed, "sealed and listed like every other artifact"
    assert (run / "figures/keyness_volcano.png").read_bytes()[:4] == b"\x89PNG"
    svg = (run / "figures/keyness_volcano.svg").read_text(encoding="utf-8")
    # The caption names the exact table it was drawn from, by its hash.
    assert hash_file(run / "keyness.csv")[:12] in svg


def test_a_cli_batch_writes_only_its_tables(tmp_path: Path) -> None:
    run = publish(tmp_path, figures=False)
    assert not (run / "figures").exists()


def test_a_figure_that_cannot_be_drawn_does_not_cost_the_run(tmp_path: Path) -> None:
    """A table the figure refuses (every G2 missing) still publishes."""
    broken = KEYNESS.assign(**{"G2 (log-likelihood)": ["x"] * len(KEYNESS)})
    run = publish(tmp_path, figures=True, frames={"keyness.csv": broken})
    assert (run / "keyness.csv").is_file()
