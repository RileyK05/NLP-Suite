"""Details from a spreadsheet the user already has (docs/internal/PLAN_0.5.0.md 1.4.2).

The engine half reads the table and matches its rows to documents; the route
previews that match, then stores it. The case the plan's done-when names is a
hand-made sheet of parties for a few presidents: keyed by the president, not
by the file name, it must still give every one of their speeches a Party
that "group by" can use.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from core.io.metadata_table import match_metadata, proposed_target, read_table
from desktop_backend.server import create_app
from desktop_backend.store import Workspace

NAMES = [
    "1946-01-21_harry s truman_sotu.txt",
    "1949-01-20_harry s truman_ina.txt",
    "1953-02-02_dwight d eisenhower_sotu.txt",
    "1957-01-21_dwight d eisenhower_ina.txt",
]


def _table(text: str, filename: str = "details.csv") -> pd.DataFrame:
    return read_table(text.encode("utf-8"), filename).unwrap()


# ------------------------------------------------------------------ reading --


class TestReading:
    def test_every_cell_stays_text(self) -> None:
        frame = _table("file,chapter,year\na.txt,01,1934\n")
        assert frame.loc[0, "chapter"] == "01" and frame.loc[0, "year"] == "1934"

    def test_the_delimiter_is_sniffed(self) -> None:
        assert list(_table("file;party\na.txt;Democratic\n").columns) == ["file", "party"]
        assert list(_table("file\tparty\na.txt\tDemocratic\n", "d.tsv").columns) == ["file", "party"]

    def test_a_workbook_reads_dates_as_days(self) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        from datetime import datetime

        book = openpyxl.Workbook()
        sheet = book.active
        sheet.append(["file", "when", "chapter"])
        sheet.append(["a.txt", datetime(1934, 1, 3), 3.0])
        buffer = io.BytesIO()
        book.save(buffer)
        frame = read_table(buffer.getvalue(), "details.xlsx").unwrap()
        assert frame.loc[0, "when"] == "1934-01-03" and frame.loc[0, "chapter"] == "3"

    def test_a_single_column_is_refused_with_why(self) -> None:
        result = read_table(b"file\na.txt\n", "d.csv")
        assert result.value is None and "at least one column of details" in result.diagnostics[0].message

    def test_other_formats_are_refused(self) -> None:
        assert read_table(b"x", "notes.docx").diagnostics[0].code == "METADATA_BAD_FORMAT"

    def test_headers_propose_the_axis_details(self) -> None:
        assert proposed_target("Year") == "Date"
        assert proposed_target("chapter") == "Order"
        assert proposed_target("party") == "Party"
        assert proposed_target("  vote share (%) ") == "Vote share"


# ----------------------------------------------------------------- matching --


class TestMatching:
    def test_names_match_exactly_without_extension_and_ignoring_case(self) -> None:
        sheet = _table(
            "document,party\n"
            "1946-01-21_harry s truman_sotu.txt,Democratic\n"
            "1949-01-20_harry s truman_ina,Democratic\n"
            "1953-02-02_DWIGHT D EISENHOWER_sotu,Republican\n"
            "nobody.txt,Whig\n"
        )
        report = match_metadata(sheet, NAMES).unwrap()
        assert report.name_column == "document" and report.matched_on == ""
        assert report.matched_by == {
            NAMES[0]: "exact",
            NAMES[1]: "without extension",
            NAMES[2]: "ignoring case",
        }
        assert report.values[NAMES[2]] == {"Party": "Republican"}
        assert report.unmatched_rows == ("nobody.txt",)
        assert report.unmatched_documents == (NAMES[3],)

    def test_a_sheet_keyed_by_a_detail_reaches_every_document_with_that_value(self) -> None:
        """Ten presidents' parties: one row per president gives each of their speeches a Party."""
        details = {
            NAMES[0]: {"Speaker": "Harry S Truman"},
            NAMES[1]: {"Speaker": "Harry S Truman"},
            NAMES[2]: {"Speaker": "Dwight D Eisenhower"},
            NAMES[3]: {"Speaker": "Dwight D Eisenhower"},
        }
        sheet = _table("president,party\nharry s truman,Democratic\nDwight D Eisenhower,Republican\n")
        report = match_metadata(sheet, NAMES, details=details).unwrap()
        assert report.matched_on == "Speaker"
        assert {name: values["Party"] for name, values in report.values.items()} == {
            NAMES[0]: "Democratic",
            NAMES[1]: "Democratic",
            NAMES[2]: "Republican",
            NAMES[3]: "Republican",
        }
        assert report.unmatched_documents == ()

    def test_a_column_of_repeated_values_is_never_the_key(self) -> None:
        """Found in the real app: once Party existed, the same sheet keyed itself on its party column."""
        details = {
            NAMES[0]: {"Speaker": "Harry S Truman", "Party": "Democratic"},
            NAMES[1]: {"Speaker": "Harry S Truman", "Party": "Democratic"},
            NAMES[2]: {"Speaker": "Dwight D Eisenhower", "Party": "Republican"},
            NAMES[3]: {"Speaker": "Dwight D Eisenhower", "Party": "Republican"},
        }
        sheet = _table(
            "president,party\nharry s truman,Democratic\nfranklin d roosevelt,Democratic\n"
            "dwight d eisenhower,Republican\n"
        )
        report = match_metadata(sheet, NAMES, details=details).unwrap()
        assert (report.name_column, report.matched_on) == ("president", "Speaker")
        assert report.values[NAMES[2]] == {"Party": "Republican"}

    def test_names_win_a_tie_with_a_detail(self) -> None:
        details = {name: {"Label": name} for name in NAMES}
        report = match_metadata(_table("file,party\n" + NAMES[0] + ",Democratic\n"), NAMES, details=details).unwrap()
        assert report.matched_on == ""

    def test_nothing_matching_is_refused_with_an_example_of_each_side(self) -> None:
        result = match_metadata(_table("who,party\nlincoln,Republican\n"), NAMES)
        assert result.value is None
        message = result.diagnostics[0].message
        assert "'lincoln'" in message and NAMES[0] in message

    def test_a_left_out_column_is_not_stored_and_a_renamed_one_is(self) -> None:
        sheet = _table("file,party,notes\n" + NAMES[0] + ",Democratic,long speech\n")
        report = match_metadata(sheet, NAMES, targets={"notes": "", "party": "Affiliation"}).unwrap()
        assert report.values[NAMES[0]] == {"Affiliation": "Democratic"}

    def test_two_columns_cannot_become_one_detail(self) -> None:
        sheet = _table("file,year,date\n" + NAMES[0] + ",1946,1946-01-21\n")
        assert match_metadata(sheet, NAMES).diagnostics[0].code == "METADATA_SAME_TARGET"

    def test_a_date_column_the_date_detail_cannot_read_is_flagged(self) -> None:
        sheet = _table("file,date\n" + NAMES[0] + ",1/21/1946\n")
        codes = [d.code for d in match_metadata(sheet, NAMES).diagnostics]
        assert "METADATA_DATE_FORMAT" in codes

    def test_a_later_row_for_the_same_document_is_reported_not_used(self) -> None:
        sheet = _table(f"file,party\n{NAMES[0]},Democratic\n{NAMES[0]},Whig\n")
        report = match_metadata(sheet, NAMES).unwrap()
        assert report.values[NAMES[0]] == {"Party": "Democratic"}
        assert report.duplicate_rows == (NAMES[0],)


# -------------------------------------------------------------------- route --


@pytest.fixture
def client(tmp_path: Path) -> tuple[TestClient, Workspace, str]:
    workspace = Workspace(tmp_path / "ws")
    project_id = workspace.create("Mixed")["id"]
    for name in NAMES:
        workspace.import_document(project_id, name, f"Words of {name}.".encode())
    app = TestClient(create_app(workspace, "import-token"))
    app.headers["Authorization"] = "Bearer import-token"
    return app, workspace, project_id


def _send(app: TestClient, project_id: str, text: str, **extra: object) -> dict[str, object]:
    body = {"filename": "parties.csv", "content": base64.b64encode(text.encode()).decode(), **extra}
    response = app.post(f"/api/projects/{project_id}/fields/import", json=body)
    assert response.status_code == 200, response.text
    return response.json()


PARTIES = "president,party\nHarry S Truman,Democratic\nDwight D Eisenhower,Republican\nAbraham Lincoln,Republican\n"


def test_the_preview_stores_nothing_and_says_what_would_happen(client: tuple[TestClient, Workspace, str]) -> None:
    app, workspace, project_id = client
    preview = _send(app, project_id, PARTIES)
    assert preview["applied"] is False
    assert preview["matched_on"] == "Speaker" and preview["matched"] == 4
    assert preview["unmatched_rows"] == ["Abraham Lincoln"]
    assert [column["target"] for column in preview["columns"]] == ["Party"]
    assert all("Party" not in doc["fields"] for doc in workspace.documents(project_id))


def test_applying_gives_every_speech_its_party_from_the_spreadsheet(client: tuple[TestClient, Workspace, str]) -> None:
    app, workspace, project_id = client
    applied = _send(app, project_id, PARTIES, dry_run=False)
    assert applied["applied"] is True
    documents = {doc["name"]: doc for doc in workspace.documents(project_id)}
    assert documents[NAMES[1]]["fields"]["Party"] == "Democratic"
    assert documents[NAMES[3]]["field_sources"]["Party"] == "csv"


def test_a_corrected_sheet_replaces_the_last_import(client: tuple[TestClient, Workspace, str]) -> None:
    app, workspace, project_id = client
    _send(app, project_id, PARTIES, dry_run=False)
    _send(app, project_id, "president,party\nHarry S Truman,Democrat\n", dry_run=False)
    parties = {doc["name"]: doc["fields"].get("Party") for doc in workspace.documents(project_id)}
    # Eisenhower is not in the corrected sheet, so his old imported value goes.
    assert parties == {NAMES[0]: "Democrat", NAMES[1]: "Democrat", NAMES[2]: None, NAMES[3]: None}


def test_a_typed_value_still_wins_over_an_import(client: tuple[TestClient, Workspace, str]) -> None:
    app, workspace, project_id = client
    target = next(doc for doc in workspace.documents(project_id) if doc["name"] == NAMES[0])
    app.post(
        f"/api/projects/{project_id}/fields",
        json={"rows": [{"document_id": target["id"], "name": "Party", "value": "Independent"}]},
    )
    _send(app, project_id, PARTIES, dry_run=False)
    assert next(d for d in workspace.documents(project_id) if d["name"] == NAMES[0])["fields"]["Party"] == "Independent"


def test_a_name_the_suite_reserves_is_flagged_in_the_preview_and_refused_on_apply(
    client: tuple[TestClient, Workspace, str],
) -> None:
    app, _workspace, project_id = client
    sheet = "president,words\nHarry S Truman,many\n"
    preview = _send(app, project_id, sheet)
    assert "already use" in preview["columns"][0]["problem"]
    body = {"filename": "p.csv", "content": base64.b64encode(sheet.encode()).decode(), "dry_run": False}
    refused = app.post(f"/api/projects/{project_id}/fields/import", json=body)
    assert refused.status_code == 400 and "already use" in refused.json()["detail"]


def test_a_damaged_upload_is_refused_plainly(client: tuple[TestClient, Workspace, str]) -> None:
    app, _workspace, project_id = client
    response = app.post(
        f"/api/projects/{project_id}/fields/import", json={"filename": "p.csv", "content": "not base64!!"}
    )
    assert response.status_code == 400 and "choose it again" in response.json()["detail"]
