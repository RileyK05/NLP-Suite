"""Bounded status reads during Windows atomic publication."""

import json
import os
from pathlib import Path
from unittest.mock import mock_open

import pytest

from core import jobs


@pytest.mark.parametrize(
    "code", [5, 32, 33, pytest.param(None, marks=pytest.mark.skipif(os.name != "nt", reason="Windows CRT EACCES"))]
)
def test_read_retries_windows_sharing_errors(monkeypatch: pytest.MonkeyPatch, code: int | None) -> None:
    error = PermissionError("brief sharing conflict")
    error.winerror = code
    reader = mock_open(read_data='{"id":"test"}')
    reader.side_effect = [error, reader.return_value]
    waits = []
    monkeypatch.setattr("builtins.open", reader)
    monkeypatch.setattr(jobs.time, "sleep", waits.append)
    assert jobs._read_status_payload(Path("status.json")) == {"id": "test"}
    assert waits == [0.01]


def test_read_retries_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    error = PermissionError("persistent denial")
    error.winerror = 5
    reader = mock_open()
    reader.side_effect = error
    monkeypatch.setattr("builtins.open", reader)
    monkeypatch.setattr(jobs.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        jobs._read_status_payload(Path("status.json"))
    assert reader.call_count == jobs._STATUS_REPLACE_ATTEMPTS


def test_invalid_json_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = mock_open(read_data="not JSON")
    monkeypatch.setattr("builtins.open", reader)
    with pytest.raises(json.JSONDecodeError):
        jobs._read_status_payload(Path("status.json"))
    assert reader.call_count == 1
