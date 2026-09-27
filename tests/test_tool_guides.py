"""The Learn page's tool guides must not promise what the library lacks.

The examples in desktop/src/toolGuides.json are written by hand next to code
that keeps moving; this test is the leash. It checks every guide carries every
section with real content, that examples only call names in ``nlpsuite``, and
that ``nlp.run("...")`` only names tools ``run`` accepts (the table_* and
lda_mallet guides deliberately teach the scipy/UI path instead).
"""

from __future__ import annotations

import json
from pathlib import Path
import re

GUIDES = json.loads(
    (Path(__file__).resolve().parents[1] / "desktop" / "src" / "toolGuides.json").read_text(encoding="utf-8")
)
FIELDS = ("what", "how", "formula", "question", "interpretation", "limits", "settings", "example")


def test_every_guide_has_every_section() -> None:
    assert len(GUIDES) == 61
    for name, guide in GUIDES.items():
        assert list(guide) == list(FIELDS), name
        for field, text in guide.items():
            assert len(text) > 60, f"{name}.{field} is thinner than a guide section should be"


def test_examples_only_call_real_library_names() -> None:
    import nlpsuite

    library = set(nlpsuite.__all__)
    for name, guide in GUIDES.items():
        for function in re.findall(r"nlp\.([a-z_]+)\(", guide["example"]):
            assert function in library, f"{name}'s example calls nlp.{function}, which does not exist"


def test_examples_only_run_tool_names_run_accepts() -> None:
    from core.script.api import _tool_names

    runnable = set(_tool_names())
    for name, guide in GUIDES.items():
        for tool in re.findall(r'nlp\.run\("([a-z0-9_]+)"', guide["example"]):
            assert tool in runnable, f"{name}'s example runs {tool!r}, which nlp.run does not accept"
