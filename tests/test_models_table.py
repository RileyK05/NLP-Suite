"""tools/models.py reads the doctor's model checks, which return (ok, detail, damaged)."""

from __future__ import annotations


def test_the_model_table_builds_with_the_doctors_three_part_checks() -> None:
    from tools.models import model_table

    table = model_table()
    assert {"Backend", "Language", "Installed", "Detail", "Fix"} <= set(table.columns)
