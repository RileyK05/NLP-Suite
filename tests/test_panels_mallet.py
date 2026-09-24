"""MALLET's actual document-topic output makes a truthful composition panel."""

from pathlib import Path

import pandas as pd

from core.analysis.mallet import parse_doc_topics
from core.viz.panels import prepare_panel
from core.viz.panelspec import Source


def test_document_ribbons_are_saved_topic_shares_with_source_rows(tmp_path: Path) -> None:
    source = tmp_path / "mallet_doc_topics.txt"
    source.write_text("0 address_1930.txt 0:0.6 1:0.4\n1 address_1940.txt 0:0.2 1:0.8\n", encoding="utf-8")
    frame = parse_doc_topics(source).unwrap()
    assert list(frame.columns) == ["Document", "Dominant topic", "Contribution", "Topic proportions"]
    result = prepare_panel("mallet_document_topics", frame, source=Source(path="topics_dominant.csv"))
    panel = result.unwrap()
    assert panel.shape == "ribbon"
    assert panel.groups == ("Topic 0", "Topic 1")
    assert [(mark.x, mark.size) for mark in panel.marks[:2]] == [(0, 0.6), (0.6, 0.4)]
    for mark in panel.marks:
        rows = frame
        for column, value in mark.evidence.filters:
            rows = rows[rows[column].astype(str) == value]
        assert len(rows) == mark.evidence.count == 1


def test_bad_mixture_is_named_and_omitted() -> None:
    frame = pd.DataFrame(
        [
            {"Document": "valid.txt", "Topic proportions": "0:0.7, 1:0.3"},
            {"Document": "broken.txt", "Topic proportions": "0:0.7, 1:-0.3"},
        ]
    )
    result = prepare_panel("mallet_document_topics", frame, source=Source(path="topics_dominant.csv"))
    panel = result.unwrap()
    assert {mark.label for mark in panel.marks} == {"valid.txt"}
    assert any(d.code == "PANEL_BAD_PROPORTIONS" for d in result.diagnostics)


def test_topic_terms_preserve_rank_without_inventing_word_weights() -> None:
    frame = pd.DataFrame(
        [
            {"Topic": 0, "Weight": 44.0, "Words": "war peace defense treaty"},
            {"Topic": 1, "Weight": 18.0, "Words": "jobs wages union labor"},
        ]
    )
    panel = prepare_panel(
        "mallet_topic_terms",
        frame,
        {"topic": 1, "top-n": 3},
        source=Source(path="topics.csv"),
    ).unwrap()

    assert panel.shape == "positions"
    assert [mark.label for mark in panel.marks] == ["jobs", "wages", "union"]
    assert [mark.x for mark in panel.marks] == [1.0, 2.0, 3.0]
    assert panel.y_categories == ("jobs", "wages", "union")
    assert "not a probability" in " ".join(panel.notes)
    for mark in panel.marks:
        assert mark.evidence.filters == (
            ("Topic", "1"),
            ("Rank", str(int(mark.x))),
            ("Word", mark.label),
        )
        evidence_rows = panel.data
        for column, value in mark.evidence.filters:
            evidence_rows = evidence_rows[evidence_rows[column].astype(str) == value]
        assert len(evidence_rows) == 1, "each mark links to its exact ranked-term row"
        assert mark.evidence.phrase == mark.label


def test_topic_terms_names_unknown_topic() -> None:
    frame = pd.DataFrame([{"Topic": 3, "Weight": 12.0, "Words": "peace law"}])
    result = prepare_panel("mallet_topic_terms", frame, {"topic": 0}, source=Source(path="topics.csv"))
    assert result.value is None
    assert result.diagnostics[0].code == "PANEL_TOPIC_NOT_FOUND"
