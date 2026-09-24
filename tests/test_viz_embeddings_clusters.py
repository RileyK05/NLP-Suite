"""core.viz.embeddings (t-SNE scatter) and core.viz.shape_clusters (CAP-VIZ-08)."""

from __future__ import annotations

import pandas as pd
import pytest

from core.conll.schema import Col
from core.viz.embeddings import tsne_html
from core.viz.shape_clusters import cluster_shapes, suggest_n_clusters
from core.viz.shapes import story_shape


def _tsne_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{"Word": w, "X": float(i), "Y": float(-i)} for i, w in enumerate(["dog", "cat", "run", "walk"])]
    )


class TestTsneHtml:
    def test_scatter_html_contains_data(self) -> None:
        result = tsne_html(_tsne_frame(), title="test")
        assert result.ok
        html = result.unwrap()
        assert len(html) > 500
        assert "dog" in html or "Word" in html

    def test_no_diagnostics_with_plotly(self) -> None:
        pytest.importorskip("plotly")
        result = tsne_html(_tsne_frame())
        assert result.ok and not result.diagnostics

    def test_empty_frame_is_an_error(self) -> None:
        result = tsne_html(pd.DataFrame(columns=["Word", "X", "Y"]))
        assert not result.ok
        assert result.diagnostics[0].code == "TSNE_PLOT_EMPTY"

    def test_missing_columns_named(self) -> None:
        result = tsne_html(pd.DataFrame({"A": [1]}))
        assert not result.ok
        assert result.diagnostics[0].code == "TSNE_PLOT_BAD_COLUMN"
        assert "Word" in result.diagnostics[0].message


def _shape_frame() -> pd.DataFrame:
    rows = []
    for did, name, slope in (
        ("1", "rise.txt", 1.0),
        ("2", "rise2.txt", 1.0),
        ("3", "fall.txt", -1.0),
        ("4", "fall2.txt", -1.0),
    ):
        for sid in range(1, 21):
            frac = sid
            rows.append(
                {
                    "Document ID": did,
                    "Document": name,
                    "Sentence ID": sid,
                    "Tokens": int(10 + 5 * slope * frac),
                    "Noun Ratio": 0.5 + 0.1 * slope * frac,
                    "Verb Ratio": 0.3 - 0.1 * slope * frac,
                }
            )
    return pd.DataFrame(rows)


class TestShapeClusters:
    def test_kmeans_separates_shape_families(self) -> None:
        result = cluster_shapes(_shape_frame())
        assert result.ok
        assignments, k = result.unwrap()
        clusters = dict(zip(assignments["Document"], assignments["Cluster"], strict=True))
        assert clusters["rise.txt"] == clusters["rise2.txt"]
        assert clusters["fall.txt"] == clusters["fall2.txt"]
        assert clusters["rise.txt"] != clusters["fall.txt"]
        assert k >= 2

    def test_deterministic_for_fixed_seed(self) -> None:
        a = cluster_shapes(_shape_frame(), seed=42).unwrap()[0]
        b = cluster_shapes(_shape_frame(), seed=42).unwrap()[0]
        assert a.equals(b)

    def test_explicit_k_honored(self) -> None:
        assignments, k = cluster_shapes(_shape_frame(), n_clusters=2).unwrap()
        assert k == 2
        assert set(assignments["Cluster"]) <= {0, 1}

    def test_too_small_corpus_refused(self) -> None:
        frame = _shape_frame()
        result = cluster_shapes(frame[frame["Document ID"] == "1"])
        assert not result.ok
        assert result.diagnostics[0].code == "SHAPE_CLUSTER_TOO_SMALL"

    def test_bad_columns_named(self) -> None:
        result = cluster_shapes(pd.DataFrame({"A": [1]}))
        assert not result.ok
        assert result.diagnostics[0].code == "SHAPE_CLUSTER_BAD_COLUMN"

    def test_suggested_k_floor_is_two(self) -> None:
        import numpy as np

        matrix = np.random.RandomState(0).rand(6, 96)
        result = suggest_n_clusters(matrix)
        assert result.ok
        assert result.unwrap() >= 2

    def test_integrates_with_story_shape(self) -> None:
        """The real story_shape output feeds the clusterer directly."""
        rows = []
        record = 0
        for did, slope in (("1", 1.0), ("2", -1.0), ("3", 0.5)):
            for sid in range(1, 13):
                record += 1
                rows.append(
                    {
                        Col.ID.value: sid,
                        Col.FORM.value: f"w{sid}",
                        Col.LEMMA.value: f"w{sid}",
                        Col.POS.value: "NOUN" if (sid + int(slope * 10)) % 2 else "VERB",
                        Col.NER.value: "O",
                        Col.HEAD.value: 0,
                        Col.DEPREL.value: "root",
                        Col.DEPS.value: "_",
                        Col.CLAUSE_TAG.value: "",
                        Col.RECORD_ID.value: record,
                        Col.SENTENCE_ID.value: sid,
                        Col.DOCUMENT_ID.value: did,
                        Col.DOCUMENT.value: f"doc{did}.txt",
                    }
                )
        shape = story_shape(pd.DataFrame(rows))
        assert shape.ok
        result = cluster_shapes(shape.unwrap())
        assert result.ok
        assignments, _k = result.unwrap()
        assert len(assignments) == 3
