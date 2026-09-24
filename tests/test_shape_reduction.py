"""shape_reduction — HC, SVD and NMF over one shared story-shape matrix.

These tests freeze the thing HW5 grades: the three data-reduction algorithms
see the SAME matrix, so their answers can be compared, and each answer's
shape (tree / axes with variance / non-negative parts) is the difference the
rubric asks about. Heavy backends are exercised through the installed
scipy/sklearn stack but only on tiny in-memory matrices.
"""

from __future__ import annotations

import pandas as pd

from core.analysis.shape_reduction import (
    build_shape_matrix,
    hierarchical_cluster,
    matrix_frame,
    nmf_reduce,
    svd_reduce,
)
from core.viz.shape_reduction import components_html, dendrogram_html

SHAPE_COLUMNS = ["Document ID", "Sentence ID", "Tokens", "Noun Ratio", "Verb Ratio", "Sentence"]


def shapes(rows: list[tuple[str, object, float, float, float]]) -> pd.DataFrame:
    """Per-sentence shape rows: (doc_id, sent_id, tokens, noun_ratio, verb_ratio)."""
    return pd.DataFrame(
        [
            {
                "Document ID": did,
                "Sentence ID": sid,
                "Tokens": tokens,
                "Noun Ratio": noun,
                "Verb Ratio": verb,
                "Sentence": f"sentence {sid}",
            }
            for did, sid, tokens, noun, verb in rows
        ],
        columns=SHAPE_COLUMNS,
    )


def corpus() -> pd.DataFrame:
    """Three documents with visibly different trajectories."""
    rows: list[tuple[str, object, float, float, float]] = []
    # doc 1: falls apart (fewer tokens, more verbs) toward the end
    for i, (tokens, noun, verb) in enumerate([(40, 0.5, 0.1), (30, 0.4, 0.2), (10, 0.2, 0.5)], start=1):
        rows.append(("1", i, float(tokens), noun, verb))
    # doc 2: flat and calm
    for i, (tokens, noun, verb) in enumerate([(25, 0.3, 0.2)] * 4, start=1):
        rows.append(("2", i, float(tokens), noun, verb))
    # doc 3: rises (more tokens, more nouns) — the mirror of doc 1
    for i, (tokens, noun, verb) in enumerate([(10, 0.2, 0.4), (30, 0.4, 0.2), (50, 0.6, 0.1)], start=1):
        rows.append(("3", i, float(tokens), noun, verb))
    return shapes(rows)


class TestBuildMatrix:
    def test_one_row_per_document_with_stable_feature_names(self) -> None:
        matrix = build_shape_matrix(corpus(), resample=8).unwrap()
        assert matrix.documents == ("1", "2", "3")
        assert matrix.values.shape == (3, 24)  # 3 metrics x 8 points
        assert matrix.feature_names[0] == "Tokens 01"
        assert matrix.feature_names[-1] == "Verb Ratio 08"

    def test_document_order_is_appearance_not_alphabetical(self) -> None:
        frame = shapes(
            [
                ("10", 1, 1.0, 0.1, 0.1),
                ("2", 1, 2.0, 0.2, 0.2),
            ]
        )
        matrix = build_shape_matrix(frame, resample=4).unwrap()
        assert matrix.documents == ("10", "2")

    def test_empty_input_is_an_empty_typed_matrix(self) -> None:
        matrix = build_shape_matrix(shapes([]), resample=4).unwrap()
        assert matrix.documents == ()
        frame = matrix_frame(matrix)
        assert frame.empty
        assert list(frame.columns)[:2] == ["Document ID", "Document"]

    def test_one_document_cannot_be_reduced_loudly(self) -> None:
        frame = shapes([("1", 1, 10.0, 0.1, 0.1)])
        result = build_shape_matrix(frame)
        assert not result.ok
        assert result.diagnostics[0].code == "SHAPE_REDUCTION_TOO_SMALL"

    def test_a_bad_resample_is_refused(self) -> None:
        result = build_shape_matrix(corpus(), resample=1)
        assert not result.ok
        assert result.diagnostics[0].code == "SHAPE_REDUCTION_BAD_RESAMPLE"

    def test_missing_columns_fail_loudly(self) -> None:
        result = build_shape_matrix(pd.DataFrame({"x": [1]}))
        assert not result.ok
        assert result.diagnostics[0].code == "SHAPE_REDUCTION_MISSING_COLUMN"


class TestHierarchical:
    def test_assignments_carry_every_document(self) -> None:
        matrix = build_shape_matrix(corpus(), resample=8).unwrap()
        result = hierarchical_cluster(matrix, n_clusters=2).unwrap()
        assert list(result.assignments.columns) == ["Document ID", "Document", "Cluster", "Distance"]
        assert set(result.assignments["Cluster"]) <= {1, 2}
        assert list(result.merges.columns) == ["Merge", "Left", "Right", "Height", "Left id", "Right id"]
        assert len(result.merges) == 2  # n_docs - 1 merges

    def test_the_opposite_stories_land_in_different_groups(self) -> None:
        matrix = build_shape_matrix(corpus(), resample=8).unwrap()
        result = hierarchical_cluster(matrix, n_clusters=2).unwrap()
        cluster_of = dict(zip(result.assignments["Document ID"], result.assignments["Cluster"], strict=True))
        # doc 1 falls and doc 3 rises: the tree must tell them apart.
        assert cluster_of["1"] != cluster_of["3"]

    def test_the_same_input_twice_gives_the_same_tree(self) -> None:
        matrix = build_shape_matrix(corpus(), resample=8).unwrap()
        first = hierarchical_cluster(matrix, method="average", n_clusters=2).unwrap()
        second = hierarchical_cluster(matrix, method="average", n_clusters=2).unwrap()
        pd.testing.assert_frame_equal(first.assignments, second.assignments)
        pd.testing.assert_frame_equal(first.merges, second.merges)

    def test_a_bad_k_names_the_valid_range(self) -> None:
        matrix = build_shape_matrix(corpus(), resample=8).unwrap()
        result = hierarchical_cluster(matrix, n_clusters=9)
        assert not result.ok
        assert result.diagnostics[0].code == "SHAPE_HC_BAD_K"

    def test_an_unknown_method_is_refused(self) -> None:
        matrix = build_shape_matrix(corpus(), resample=8).unwrap()
        result = hierarchical_cluster(matrix, method="ward-ish")
        assert not result.ok
        assert result.diagnostics[0].code == "SHAPE_HC_BAD_METHOD"


class TestSvd:
    def test_scores_loadings_and_variance_share(self) -> None:
        matrix = build_shape_matrix(corpus(), resample=8).unwrap()
        result = svd_reduce(matrix, n_components=2, seed=7).unwrap()
        assert list(result.scores.columns) == ["Document ID", "Document", "Component 1", "Component 2"]
        assert list(result.loadings.columns) == ["Feature", "Component 1", "Component 2"]
        assert list(result.explained.columns) == ["Component", "Explained variance", "Cumulative"]
        assert abs(float(result.explained["Cumulative"].iloc[-1]) - 1.0) < 1e-6
        assert len(result.loadings) == len(matrix.feature_names)

    def test_a_fixed_seed_repeats_exactly(self) -> None:
        matrix = build_shape_matrix(corpus(), resample=8).unwrap()
        first = svd_reduce(matrix, n_components=2, seed=7).unwrap()
        second = svd_reduce(matrix, n_components=2, seed=7).unwrap()
        pd.testing.assert_frame_equal(first.scores, second.scores)
        pd.testing.assert_frame_equal(first.loadings, second.loadings)

    def test_a_bad_k_names_the_valid_range(self) -> None:
        matrix = build_shape_matrix(corpus(), resample=8).unwrap()
        result = svd_reduce(matrix, n_components=99)
        assert not result.ok
        assert result.diagnostics[0].code == "SHAPE_SVD_BAD_K"


class TestNmf:
    def test_parts_are_non_negative_and_the_shift_is_reported(self) -> None:
        matrix = build_shape_matrix(corpus(), resample=8).unwrap()
        result = nmf_reduce(matrix, n_components=2, seed=7).unwrap()
        assert list(result.scores.columns) == ["Document ID", "Document", "Component 1", "Component 2"]
        assert (result.scores[["Component 1", "Component 2"]].to_numpy() >= -1e-9).all()
        assert (result.loadings[["Component 1", "Component 2"]].to_numpy() >= -1e-9).all()
        assert list(result.explained.columns) == ["Component", "Share of mass", "Cumulative"]

    def test_negative_values_are_shifted_and_said_so(self) -> None:
        frame = corpus()
        frame.loc[frame.index[:3], "Noun Ratio"] = [-5.0, -4.0, -3.0]
        matrix = build_shape_matrix(frame, resample=8).unwrap()
        result = nmf_reduce(matrix, n_components=2, seed=7)
        assert result.ok, result.diagnostics
        assert result.value.shift > 0
        assert any(d.code == "SHAPE_NMF_SHIFT" for d in result.diagnostics)

    def test_a_fixed_seed_repeats_exactly(self) -> None:
        matrix = build_shape_matrix(corpus(), resample=8).unwrap()
        first = nmf_reduce(matrix, n_components=2, seed=7).unwrap()
        second = nmf_reduce(matrix, n_components=2, seed=7).unwrap()
        pd.testing.assert_frame_equal(first.scores, second.scores)

    def test_a_bad_k_names_the_valid_range(self) -> None:
        matrix = build_shape_matrix(corpus(), resample=8).unwrap()
        result = nmf_reduce(matrix, n_components=99)
        assert not result.ok
        assert result.diagnostics[0].code == "SHAPE_NMF_BAD_K"

    def test_too_many_iterations_is_refused_loudly(self) -> None:
        matrix = build_shape_matrix(corpus(), resample=8).unwrap()
        result = nmf_reduce(matrix, max_iter=2)
        assert not result.ok
        assert result.diagnostics[0].code == "SHAPE_NMF_BAD_ITER"


class TestCharts:
    def test_the_dendrogram_renders_without_a_plotting_backend(self) -> None:
        matrix = build_shape_matrix(corpus(), resample=8).unwrap()
        result = hierarchical_cluster(matrix, n_clusters=2).unwrap()
        html = dendrogram_html(result).unwrap()
        assert "<svg" in html and "clusters" in html

    def test_component_profiles_render_one_panel_per_metric(self) -> None:
        matrix = build_shape_matrix(corpus(), resample=8).unwrap()
        result = svd_reduce(matrix, n_components=2, seed=7).unwrap()
        html = components_html(result).unwrap()
        assert "<svg" in html
        assert "Tokens" in html and "Noun Ratio" in html and "Verb Ratio" in html

    def test_nmf_charts_carry_the_shift_note(self) -> None:
        frame = corpus()
        frame.loc[frame.index[:3], "Noun Ratio"] = [-5.0, -4.0, -3.0]
        matrix = build_shape_matrix(frame, resample=8).unwrap()
        result = nmf_reduce(matrix, n_components=2, seed=7).unwrap()
        html = components_html(result).unwrap()
        assert "shifted" in html
