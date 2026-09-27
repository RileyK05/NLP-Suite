"""File-name detection of document details (docs/PLAN_0.5.0.md 1.4.1).

The real workspace's names (names only; tests/fixtures/names/real_projects.txt)
are the coverage set: the detector must read all of them, and must agree with
the speaker rule every figure has used until now.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.io.filename_fields import Template, apply_template, detect_template, order_value
from core.viz.panel_helpers import speaker_of

NAMES = Path(__file__).parent / "fixtures" / "names" / "real_projects.txt"


def projects() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    current = ""
    for line in NAMES.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            current = line.strip("[]")
            found[current] = []
        else:
            found[current].append(line)
    return found


def test_the_fixture_is_read() -> None:
    sizes = {name: len(names) for name, names in projects().items()}
    assert sizes == {"Inaugural speech (31)": 31, "SOTU (87)": 87, "SOTU (68)": 68}


@pytest.mark.parametrize("project", ["Inaugural speech (31)", "SOTU (87)", "SOTU (68)"])
def test_every_real_name_gives_date_speaker_and_kind(project: str) -> None:
    names = projects()[project]
    template = detect_template(names)
    assert template is not None
    assert template.parts == ("Date", "Speaker", "Kind")
    assert template.fits == len(names) and not template.misses
    for name in names:
        fields = apply_template(name, template)
        assert fields["Date"] == name[:10]
        assert fields["Speaker"]


def test_the_mixed_project_has_two_kinds() -> None:
    names = projects()["SOTU (68)"]
    template = detect_template(names)
    assert template is not None
    assert {apply_template(name, template)["Kind"] for name in names} == {"sotu", "ina"}


def test_the_detector_agrees_with_the_speaker_every_figure_used() -> None:
    """Moving the rule must change no figure: same speaker for every name speaker_of reads."""
    for names in projects().values():
        template = detect_template(names)
        assert template is not None
        for name in names:
            if speaker_of(name):
                assert apply_template(name, template)["Speaker"] == speaker_of(name), name


def test_example_readings() -> None:
    fields = apply_template(
        "1934-01-03_franklin d roosevelt_sotu.txt",
        detect_template(["1934-01-03_franklin d roosevelt_sotu.txt", "1935-01-04_franklin d roosevelt_sotu.txt"])
        or Template((), "", 0, 0),
    )
    assert fields == {"Date": "1934-01-03", "Speaker": "Franklin D Roosevelt", "Kind": "sotu"}

    template = detect_template(["1934_roosevelt.txt", "1950_truman.txt", "1961_kennedy.txt"])
    assert template is not None and template.parts == ("Date", "Speaker")
    assert apply_template("1950_truman.txt", template) == {"Date": "1950", "Speaker": "Truman"}

    chapters = detect_template(["ch01_the_beginning.txt", "ch02_a_ball.txt"])
    assert chapters is None  # the title has underscores too: no single shape fits, so no guess

    numbered = detect_template(["01 - The Beginning.txt", "02 - A Ball.txt", "03 - Letters.txt"])
    assert numbered is not None and numbered.parts == ("Order", "Title")
    assert apply_template("02 - A Ball.txt", numbered) == {"Order": "2", "Title": "A Ball"}

    single = detect_template(["Chapter 1.txt", "Chapter 2.txt", "Chapter 10.txt"])
    assert single is not None and single.parts == ("Order",)
    assert apply_template("Chapter 10.txt", single) == {"Order": "10"}

    books = detect_template(["austen_pride_1813.txt", "austen_emma_1815.txt", "bronte_jane_1847.txt"])
    assert books is not None and books.parts == ("Author", "Title", "Date")


def test_names_without_a_common_shape_give_nothing() -> None:
    assert detect_template(["notes.txt", "my essay final v2.txt", "draft.txt", "interview-3.txt"]) is None
    assert detect_template(["1934-01-03_roosevelt_sotu.txt"]) is None  # one name shows no pattern


def test_a_template_lists_the_names_it_does_not_fit() -> None:
    names = [f"19{n}0-01-0{n}_speaker {n}_sotu.txt" for n in range(1, 9)] + ["readme.txt"]
    template = detect_template(names)
    assert template is not None and template.misses == ("readme.txt",) and template.fits == 8
    assert template.describe() == "Date, then Speaker, then Kind separated by _ (fits 8 of 9)"
    too_many_misses = [*names[:3], "a.txt", "b.txt", "c.txt"]
    assert detect_template(too_many_misses) is None


@pytest.mark.parametrize(
    ("text", "value"),
    [("1", "1"), ("01", "1"), ("ch1", "1"), ("Chapter 12", "12"), ("IV", "4"), ("XLII", "42"), ("chapter", None)],
)
def test_order_values(text: str, value: str | None) -> None:
    assert order_value(text) == value
