"""Topic prevalence over time: dates parsed conservatively, shares that add up.

The builder is called directly here rather than through ``prepare_panel``.
The ``stream`` shape has a renderer now, but this panel is registered
separately, and calling the builder keeps these tests about what the panel
*claims* rather than about the registry's gate, which ``tests/test_panels.py``
already covers. Parameters are therefore passed as a plain dict, already in
the form the registry would have validated them into.

Two properties carry most of the weight.

**Evidence resolves against the published table.** A bucket is a derived
value, so the builder adds a ``Period`` column to ``PreparedPanel.data`` and
filters on that. ``TestEvidenceResolves`` takes every mark's filters back to
that table and asserts they find exactly the documents the mark counted --
including that the resolved row count equals ``evidence.count``, which is the
number the figure puts in front of a reader.

**An undated document is never guessed at.** A year bucketed as zero, or a
name matched loosely enough to find "1999" in ``report_1999_v2``, produces a
plot that is wrong in a way nobody can see. Both are tested.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from core.result import Result, Severity
from core.viz.panels_lda_prevalence import (
    CONTRIBUTION,
    DOCUMENT,
    DOCUMENT_ID,
    DOMINANT_TOPIC,
    LDA_PREVALENCE,
    PERIOD,
    TOPIC_KEYWORDS,
    lda_prevalence,
)
from core.viz.panelspec import PreparedPanel, Provenance

# Real corpus names: 87 State of the Union addresses, 1934-2024.
_ROWS: list[tuple[str, int, float]] = [
    ("1934-01-03_franklin d roosevelt_sotu.txt", 0, 0.80),
    ("1935-01-04_franklin d roosevelt_sotu.txt", 0, 0.60),
    ("1938-01-03_franklin d roosevelt_sotu.txt", 1, 0.40),
    ("1945-01-06_franklin d roosevelt_sotu.txt", 1, 0.90),
    ("1947-01-06_harry s truman_sotu.txt", 1, 0.70),
    ("2024-03-07_joseph r biden_sotu.txt", 2, 0.50),
]


def frame(rows: list[tuple[str, int, float]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                DOCUMENT_ID: f"d{index}",
                DOCUMENT: name,
                DOMINANT_TOPIC: topic,
                CONTRIBUTION: contribution,
                TOPIC_KEYWORDS: f"word{topic}, other{topic}, third{topic}",
            }
            for index, (name, topic, contribution) in enumerate(rows if rows is not None else _ROWS)
        ]
    )


def build(params: dict[str, Any] | None = None, rows: list[tuple[str, int, float]] | None = None) -> Result[Any]:
    """The builder with defaults filled in, as the registry would have."""
    settings = LDA_PREVALENCE.defaults() | (params or {})
    return lda_prevalence(frame(rows), settings, Provenance(tool="lda_gensim", panel="lda_prevalence", source="d.csv"))


def ok(params: dict[str, Any] | None = None, rows: list[tuple[str, int, float]] | None = None) -> PreparedPanel:
    result = build(params, rows)
    assert result.ok, [str(d) for d in result.diagnostics]
    return result.unwrap()


def codes(result: Result[Any]) -> list[str]:
    return [d.code for d in result.diagnostics]


def find(result: Result[Any], code: str) -> Any:
    matching = [d for d in result.diagnostics if d.code == code]
    assert len(matching) == 1, f"expected one {code}, got {codes(result)}"
    return matching[0]


class TestEvidenceResolves:
    def test_filters_select_exactly_the_documents_the_mark_counted(self) -> None:
        """The load-bearing test. A bucket is derived, so the builder
        publishes a Period column and filters on it; if that column were
        missing or misnamed the filters would silently select nothing."""
        panel = ok()
        assert PERIOD in panel.data.columns, "the derived bucket must be in the published table"
        for mark in panel.marks:
            selected = panel.data
            for column, value in mark.evidence.filters:
                assert column in panel.data.columns, f"{mark.key}: filter names a column the table lacks"
                selected = selected[selected[column].astype(str) == value]
            assert len(selected) == mark.evidence.count, (
                f"{mark.key}: filters found {len(selected)} documents but the mark claims {mark.evidence.count}"
            )
            assert len(selected) > 0

    def test_the_counts_add_up_to_the_documents_that_were_plotted(self) -> None:
        panel = ok()
        assert sum(mark.evidence.count for mark in panel.marks) == len(_ROWS)

    def test_evidence_is_scoped_to_documents(self) -> None:
        """A band stands for speeches, which are things you can go and read."""
        panel = ok()
        assert {mark.evidence.scope for mark in panel.marks} == {"documents"}


class TestDates:
    def test_an_undated_document_is_dropped_and_named(self) -> None:
        rows = [*_ROWS, ("undated_speech.txt", 0, 0.5)]
        result = build(rows=rows)
        assert result.ok, "one bad name does not fail a corpus that mostly parses"
        notice = find(result, "PANEL_NO_DATE")
        assert notice.severity is Severity.WARNING
        assert notice.context["dropped"] == 1
        assert "undated_speech.txt" in notice.message, "the reader has to see which name failed"

    def test_a_corpus_with_no_dates_at_all_fails_with_the_expected_form(self) -> None:
        rows = [("speech_one.txt", 0, 0.5), ("speech_two.txt", 1, 0.5)]
        result = build(rows=rows)
        assert not result.ok
        message = find(result, "PANEL_NO_DATES").message
        assert "YYYY" in message, "the failure must show what a usable name looks like"
        assert "speech_one.txt" in message

    def test_a_year_in_the_middle_of_a_name_is_not_a_date(self) -> None:
        """`report_1999_v2` is not a 1999 document, and a regex loose enough
        to say it is produces a plot that is wrong invisibly."""
        rows = [("report_1999_v2.txt", 0, 0.5), ("2001-01-01_someone_sotu.txt", 1, 0.5)]
        result = build(rows=rows)
        assert result.ok
        assert find(result, "PANEL_NO_DATE").context["dropped"] == 1
        panel = result.unwrap()
        assert set(panel.data[DOCUMENT]) == {"2001-01-01_someone_sotu.txt"}

    def test_a_bare_year_is_accepted(self) -> None:
        rows = [("1999_state_of_the_union.txt", 0, 0.5), ("2001-01-01_someone_sotu.txt", 1, 0.5)]
        result = build(rows=rows)
        assert result.ok
        assert "PANEL_NO_DATE" not in codes(result)

    def test_a_five_digit_number_is_not_a_year(self) -> None:
        rows = [("19999_odd.txt", 0, 0.5), ("2001-01-01_someone_sotu.txt", 1, 0.5)]
        result = build(rows=rows)
        assert result.ok
        assert find(result, "PANEL_NO_DATE").context["dropped"] == 1


class TestBuckets:
    def test_decades_and_years_group_differently(self) -> None:
        decades = {mark.x for mark in ok({"bucket": "decade"}).marks}
        years = {mark.x for mark in ok({"bucket": "year"}).marks}
        assert decades == {1930.0, 1940.0, 2020.0}
        assert 1934.0 in years and 1935.0 in years
        assert len(years) > len(decades), "a year axis is finer than a decade axis"

    def test_the_axis_label_follows_the_bucket(self) -> None:
        assert ok({"bucket": "decade"}).x_label == "Decade"
        assert ok({"bucket": "year"}).x_label == "Year"

    def test_a_one_document_bucket_is_called_out(self) -> None:
        """A 100% band drawn from a single speech is not a trend."""
        result = build({"bucket": "decade"})
        notice = find(result, "PANEL_THIN_BUCKET")
        assert notice.severity is Severity.INFO
        assert "2020" in notice.message, "the 2020s hold one address in this fixture"

    def test_a_corpus_with_no_thin_buckets_says_nothing(self) -> None:
        rows = [
            ("1934-01-03_a_sotu.txt", 0, 0.5),
            ("1935-01-03_b_sotu.txt", 1, 0.5),
            ("1944-01-03_c_sotu.txt", 0, 0.5),
            ("1945-01-03_d_sotu.txt", 1, 0.5),
        ]
        assert "PANEL_THIN_BUCKET" not in codes(build({"bucket": "decade"}, rows))


class TestMeasures:
    def test_counting_documents_and_summing_contribution_are_different(self) -> None:
        """Two different quantities, which is why there is no blended default."""
        documents = {m.key: m.y for m in ok({"measure": "documents", "normalize": False}).marks}
        contribution = {m.key: m.y for m in ok({"measure": "contribution", "normalize": False}).marks}
        assert documents != contribution
        assert documents["1930:0"] == 2.0, "two addresses in the 1930s are dominant in topic 0"
        assert contribution["1930:0"] == pytest.approx(1.40), "0.80 + 0.60 of contribution mass"

    def test_normalising_makes_each_bucket_a_share_of_one(self) -> None:
        panel = ok({"normalize": True})
        totals: dict[float, float] = {}
        for mark in panel.marks:
            totals[mark.x] = totals.get(mark.x, 0.0) + mark.y
        for period, total in totals.items():
            assert total == pytest.approx(1.0), f"{period} sums to {total}"

    def test_raw_totals_are_not_shares(self) -> None:
        panel = ok({"normalize": False, "measure": "documents"})
        totals: dict[float, float] = {}
        for mark in panel.marks:
            totals[mark.x] = totals.get(mark.x, 0.0) + mark.y
        assert totals[1930.0] == 3.0, "three addresses in the 1930s"

    def test_the_y_label_names_the_quantity_actually_drawn(self) -> None:
        assert "Share" in ok({"normalize": True, "measure": "documents"}).y_label
        assert "Number" in ok({"normalize": False, "measure": "documents"}).y_label
        assert "contribution" in ok({"normalize": False, "measure": "contribution"}).y_label.lower()

    def test_a_broken_contribution_is_dropped_only_where_it_is_read(self) -> None:
        """`documents` never reads Contribution, so a bad value there must not
        remove a speech from a count it plays no part in."""
        rows = [*_ROWS]
        bad = frame(rows)
        # Cast first: writing text into a float column is a pandas
        # FutureWarning, which would make this test fail for a reason that
        # has nothing to do with the panel.
        bad[CONTRIBUTION] = bad[CONTRIBUTION].astype(object)
        bad.loc[0, CONTRIBUTION] = "not a number"

        settings = LDA_PREVALENCE.defaults() | {"measure": "contribution"}
        provenance = Provenance(tool="lda_gensim", panel="lda_prevalence", source="d.csv")
        weighted = lda_prevalence(bad, settings, provenance)
        assert weighted.ok
        assert find(weighted, "PANEL_BAD_NUMERIC").context["dropped"] == 1

        counted = lda_prevalence(bad, LDA_PREVALENCE.defaults() | {"measure": "documents"}, provenance)
        assert counted.ok
        assert "PANEL_BAD_NUMERIC" not in codes(counted), "a count does not read Contribution"
        assert sum(m.evidence.count for m in counted.unwrap().marks) == len(rows)


class TestShape:
    def test_bands_are_ordered_by_topic_number_not_by_appearance(self) -> None:
        """A colour must mean the same topic on every run."""
        panel = ok()
        assert [group.split(":")[0] for group in panel.groups] == ["Topic 0", "Topic 1", "Topic 2"]

    def test_every_mark_belongs_to_a_declared_band(self) -> None:
        panel = ok()
        assert {mark.group for mark in panel.marks} <= set(panel.groups)

    def test_the_panel_declares_the_stream_shape(self) -> None:
        assert ok().shape == "stream"
        assert LDA_PREVALENCE.shape == "stream"

    def test_mark_keys_identify_a_bucket_and_a_topic(self) -> None:
        panel = ok()
        assert "1930:0" in {mark.key for mark in panel.marks}


class TestNotes:
    def test_the_dominant_topic_caveat_is_stated(self) -> None:
        """A speech about several things is counted once, and a reader
        building an argument on these bands needs to know that."""
        notes = " ".join(ok().notes).lower()
        assert "dominant" in notes
        assert "mixture" in notes or "several" in notes

    def test_the_definition_and_the_panel_agree_on_their_limits(self) -> None:
        assert LDA_PREVALENCE.notes
        assert ok().notes == LDA_PREVALENCE.notes

    def test_every_parameter_explains_itself(self) -> None:
        for param in LDA_PREVALENCE.params:
            assert param.help.strip(), f"{param.name} has no help"
