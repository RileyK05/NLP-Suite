"""FR-9.3 — model inventory contract tests (read-only, never downloads)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.models import model_table


class TestModels:
    def test_table_covers_configured_backends(self) -> None:
        frame = model_table()
        assert list(frame.columns) == ["Backend", "Language", "Model", "Installed", "Detail", "Fix"]
        assert set(frame["Backend"].unique()) == {"stanza", "spacy"}
        assert "en" in set(frame["Language"].unique())
        assert frame["Fix"].notna().all()

    def test_cli_writes_inventory(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from tools.models import main

        code = main([str(tmp_path / "out")])
        assert code in (0, 1)  # 1 when models are missing here — both honest
        assert (tmp_path / "out").exists()
        out = capsys.readouterr().out
        assert "stanza" in out and "spacy" in out
