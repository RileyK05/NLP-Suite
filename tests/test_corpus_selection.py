from __future__ import annotations

from pydantic import ValidationError
import pytest

from desktop_backend.selection import CorpusSelection, document_metadata, resolve_selection


def docs() -> list[dict[str, object]]:
    return [
        {"id": "a", "name": "speech_2020-01-15.txt", "words": 3},
        {"id": "b", "name": "speech_2021-02-28.txt", "words": 4},
        {"id": "c", "name": "undated.txt", "words": 5},
    ]


def test_default_selects_all_in_original_order_and_adds_provenance() -> None:
    original = docs()
    selected = resolve_selection(original, None)
    assert [row["id"] for row in selected] == ["a", "b", "c"]
    assert selected[0]["document_date"] == "2020-01-15"
    assert selected[0]["date_source"] == "filename"
    assert selected[-1]["document_date"] is None
    assert original == docs()


def test_ids_are_intersection_of_date_filter_and_preserve_order() -> None:
    selection = CorpusSelection(document_ids=["c", "a"], date_from="2020-01-01", date_to="2020-12-31")
    assert [row["id"] for row in resolve_selection(docs(), selection)] == ["a"]


def test_date_bounds_are_inclusive_and_undated_can_be_included() -> None:
    selection = CorpusSelection(date_from="2020-01-15", date_to="2021-02-28", include_undated=True)
    assert [row["id"] for row in resolve_selection(docs(), selection)] == ["a", "b", "c"]


def test_filtered_selection_excludes_undated_by_default() -> None:
    selection = CorpusSelection(date_from="2020-01-01")
    assert [row["id"] for row in resolve_selection(docs(), selection)] == ["a", "b"]


@pytest.mark.parametrize("value", ["2021-02-29", "2020-02-30", "2020/01/01", "20200101", "2020-1-01"])
def test_dates_are_strict_real_iso_dates(value: str) -> None:
    with pytest.raises(ValidationError):
        CorpusSelection(date_from=value)


def test_reversed_bounds_unknown_ids_and_empty_effective_selection_fail() -> None:
    with pytest.raises(ValueError, match="on or before"):
        resolve_selection(docs(), CorpusSelection(date_from="2021-01-01", date_to="2020-01-01"))
    with pytest.raises(ValueError, match="Unknown"):
        resolve_selection(docs(), CorpusSelection(document_ids=["missing"]))
    with pytest.raises(ValueError, match="empty"):
        resolve_selection(docs(), CorpusSelection(document_ids=[]))
    with pytest.raises(ValueError, match="empty"):
        resolve_selection(docs(), CorpusSelection(date_from="2022-01-01"))


def test_ids_are_nonempty_unique_and_bounded() -> None:
    with pytest.raises(ValidationError):
        CorpusSelection(document_ids=[""])
    with pytest.raises(ValidationError):
        CorpusSelection(document_ids=["a", "a"])
    with pytest.raises(ValidationError):
        CorpusSelection(document_ids=[str(i) for i in range(2001)])


def test_extra_fields_are_rejected_and_metadata_does_not_mutate() -> None:
    with pytest.raises(ValidationError):
        CorpusSelection(question="all")
    doc = {"id": "a", "name": "report_2024-01-01.txt"}
    result = document_metadata(doc)
    assert result["document_date"] == "2024-01-01"
    assert doc == {"id": "a", "name": "report_2024-01-01.txt"}
