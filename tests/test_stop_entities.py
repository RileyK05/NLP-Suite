"""Person entities that are probably not people: named, never removed (plan 5.2)."""

from __future__ import annotations

from core.analysis.stop_entities import suspect_persons


class TestSuspects:
    def test_a_common_noun_is_a_suggestion(self) -> None:
        assert suspect_persons(["Speaker", "Alice"]) == ["Speaker"]

    def test_an_entity_in_most_documents_is_a_suggestion(self) -> None:
        names = suspect_persons(["Alice", "Bob"], {"Alice": 0.8, "Bob": 0.2})
        assert names == ["Alice"]

    def test_an_occasional_person_is_not_suggested(self) -> None:
        assert suspect_persons(["Alice"], {"Alice": 0.3}) == []

    def test_suggestions_come_most_widespread_first(self) -> None:
        names = suspect_persons(["Chamber", "Speaker", "Boo"], {"Chamber": 0.6, "Speaker": 0.9, "Boo": 0.1})
        assert names == ["Speaker", "Chamber", "Boo"]

    def test_nothing_is_removed_so_an_empty_list_stays_empty(self) -> None:
        assert suspect_persons([]) == []
