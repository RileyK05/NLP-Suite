"""Keep learning content aligned with the actual selectable desktop catalog."""

import json
from pathlib import Path

from desktop_backend.runner import DESKTOP_TOOLS


def test_every_desktop_tool_has_learning_content() -> None:
    path = Path(__file__).resolve().parents[1] / "desktop/src/toolGuides.json"
    guides = json.loads(path.read_text(encoding="utf-8"))
    assert set(guides) == set(DESKTOP_TOOLS)
    for guide in guides.values():
        assert set(guide) == {"what", "how", "question", "interpretation", "limits"}
        assert all(isinstance(value, str) and len(value) > 60 for value in guide.values())
