"""FR-2.6 sentence complexity â€” dependency distance, depth, subordination, Yngve/Frazier.

Yngve/Frazier oracle: the legacy ``tree.py`` + ``sentence_complexity_node_util.py``
directly (both are stdlib-only). Dependency metrics are hand-computed on the
shared ``conll_frame`` fixture. Yngve/Frazier need constituency trees, which
only CoreNLP can supply (documented parser requirement, FR-5.2).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

from conftest import has_spacy_model
from core.analysis import sentence_complexity as C

LEGACY_SRC = Path(__file__).resolve().parent.parent.parent / "NLP-Suite-1.6.38" / "src"
needs_oracle = pytest.mark.skipif(not (LEGACY_SRC / "tree.py").is_file(), reason="legacy oracle checkout absent")


def load_legacy_nodes():  # type: ignore[no-untyped-def]
    """Import legacy tree.py as ``tree`` (as its sibling expects) plus the Node module."""
    import importlib.util

    tree_spec = importlib.util.spec_from_file_location("tree", LEGACY_SRC / "tree.py")
    assert tree_spec is not None and tree_spec.loader is not None
    tree_mod = importlib.util.module_from_spec(tree_spec)
    sys.modules["tree"] = tree_mod
    tree_spec.loader.exec_module(tree_mod)
    node_spec = importlib.util.spec_from_file_location(
        "legacy_complexity_node", LEGACY_SRC / "sentence_complexity_node_util.py"
    )
    assert node_spec is not None and node_spec.loader is not None
    node_mod = importlib.util.module_from_spec(node_spec)
    node_spec.loader.exec_module(node_mod)
    return tree_mod, node_mod


TINY_TREE = "(S (NP x) (VP y))"
RICH_TREE = "(S (NP (DT The) (NN president)) (VP (VBD went) (PP (IN to) (NP (NNP Italy)))) (. .))"


class TestDependencyDistance:
    def test_hand_case(self) -> None:
        # ids 1..3, heads [2, 0, 2]: distances 1, -, 1
        assert C.mean_dependency_distance([1, 2, 3], [2, 0, 2]) == pytest.approx(1.0)
        assert C.mean_dependency_distance([1], [0]) == 0.0
        assert C.mean_dependency_distance([], []) == 0.0

    def test_run_on_conll_frame(self, conll_frame: pd.DataFrame) -> None:
        frame = C.run(conll_frame).unwrap().to_frame()
        # doc 1 sent 1: heads [2,3,0,5,3,3] over ids 1..6 -> (1+1+0+1+2+3)/5 = 1.6
        s1 = frame.loc[(frame["Document ID"] == "1") & (frame["Sentence ID"] == 1)].iloc[0]
        assert float(s1["Mean Dependency Distance"]) == pytest.approx(8 / 5)
        assert int(s1["Tokens"]) == 6
        # doc 1 sent 2: positions 1..3, heads [2,0,8]; head 8 is out of
        # range (corrupt) and excluded -> mean over |1-2| only = 1.0
        result = C.run(conll_frame)
        assert any(d.code == "COMPLEXITY_BAD_HEAD" for d in result.diagnostics)
        s2 = frame.loc[(frame["Document ID"] == "1") & (frame["Sentence ID"] == 2)].iloc[0]
        assert float(s2["Mean Dependency Distance"]) == pytest.approx(1.0)

    def test_depth_and_subordination(self, conll_frame: pd.DataFrame) -> None:
        frame = C.run(conll_frame).unwrap().to_frame()
        s1 = frame.loc[(frame["Document ID"] == "1") & (frame["Sentence ID"] == 1)].iloc[0]
        # longest head chain: 4->5->3->0 (to), 2 edges
        assert int(s1["Depth"]) == 2
        assert int(s1["Subordinate Clauses"]) == 0

    def test_subordinate_counts_advcl(self) -> None:
        frame = pd.DataFrame(
            [
                [1, "He", "he", "PRP", "O", 2, "nsubj", 1, 1, "1", "d"],
                [2, "left", "leave", "VBD", "O", 0, "root", 2, 1, "1", "d"],
                [3, "crying", "cry", "VBG", "O", 2, "advcl", 3, 1, "1", "d"],
            ],
            columns=[
                "ID",
                "Form",
                "Lemma",
                "POS",
                "NER",
                "Head",
                "DepRel",
                "Record ID",
                "Sentence ID",
                "Document ID",
                "Document",
            ],
        )
        out = C.run(frame).unwrap().to_frame()
        assert int(out.iloc[0]["Subordinate Clauses"]) == 1

    def test_empty_frame(self) -> None:
        from conftest import CANONICAL_MINIMAL

        empty = pd.DataFrame(columns=CANONICAL_MINIMAL)
        assert C.run(empty).unwrap().to_frame().empty


class TestYngveFrazier:
    def test_tiny_tree_hand_derived(self) -> None:
        # S children [NP, VP] are word-bracket leaves: NP=0+2-1-0=1, VP=0+2-1-1=0
        assert C.sum_yngve(TINY_TREE).unwrap() == pytest.approx(1.0)
        assert C.mean_yngve(TINY_TREE).unwrap() == pytest.approx(0.5)
        # Frazier: first child NP-leaf=0+1.5, VP-leaf reset to 0 -> sum 1.5
        assert C.sum_frazier(TINY_TREE).unwrap() == pytest.approx(1.5)

    @needs_oracle
    def test_matches_legacy_on_rich_tree(self) -> None:
        tree_mod, node_mod = load_legacy_nodes()
        for tree_string in (TINY_TREE, RICH_TREE):
            legacy_node = node_mod.Node(tree_mod.make_tree(tree_string))
            legacy_node.calY()
            legacy_node.calF()
            assert C.sum_yngve(tree_string).unwrap() == pytest.approx(legacy_node.sumY())
            assert C.sum_frazier(tree_string).unwrap() == pytest.approx(legacy_node.sumF())

    def test_empty_tree_string_scores_zero(self) -> None:
        assert C.sum_yngve("").unwrap() == 0.0
        assert C.sum_frazier("").unwrap() == 0.0
        assert C.mean_yngve("").unwrap() == 0.0

    def test_tiny_tree_result_shape(self) -> None:
        assert C.sum_yngve(TINY_TREE).unwrap() == pytest.approx(1.0)
        assert C.mean_yngve(TINY_TREE).unwrap() == pytest.approx(0.5)
        assert C.sum_frazier(TINY_TREE).unwrap() == pytest.approx(1.5)


class TestC67Validation:
    """C6-7: strict int conversion, dependency health, tree validation."""

    def test_bad_head_types_rejected(self) -> None:
        from conftest import CANONICAL_MINIMAL

        base = [
            [1, "He", "he", "PRP", "O", 2, "nsubj", 1, 1, "1", "d"],
            [2, "left", "leave", "VBD", "O", 0, "root", 2, 1, "1", "d"],
            [3, "today", "today", "NN", "O", 2, "tmod", 3, 1, "1", "d"],
        ]
        columns = CANONICAL_MINIMAL
        for bad in ["soon", 2.5, True, float("nan"), float("inf")]:
            rows = [row[:] for row in base]
            rows[1][5] = bad  # Head column position in canonical layout
            frame = pd.DataFrame(rows, columns=columns)
            result = C.run(frame)
            assert any(d.code == "COMPLEXITY_BAD_INT" for d in result.diagnostics), bad

    def test_dependency_health(self) -> None:
        from conftest import CANONICAL_MINIMAL

        columns = CANONICAL_MINIMAL
        # self-head: token 2 points to itself, no root
        frame = pd.DataFrame(
            [
                [1, "a", "a", "NN", "O", 2, "dep", 1, 1, "1", "d"],
                [2, "b", "b", "NN", "O", 2, "dep", 2, 1, "1", "d"],
            ],
            columns=columns,
        )
        result = C.run(frame)
        codes = [d.code for d in result.diagnostics]
        assert "COMPLEXITY_SELF_HEAD" in codes
        assert "COMPLEXITY_NO_ROOT" in codes

    def test_multi_root_warns(self) -> None:
        from conftest import CANONICAL_MINIMAL

        frame = pd.DataFrame(
            [
                [1, "a", "a", "NN", "O", 0, "root", 1, 1, "1", "d"],
                [2, "b", "b", "NN", "O", 0, "root", 2, 1, "1", "d"],
            ],
            columns=CANONICAL_MINIMAL,
        )
        assert any(d.code == "COMPLEXITY_MULTI_ROOT" for d in C.run(frame).diagnostics)

    def test_non_dense_ids_warned(self) -> None:
        from conftest import CANONICAL_MINIMAL

        frame = pd.DataFrame(
            [
                [1, "a", "a", "NN", "O", 2, "det", 1, 1, "1", "d"],
                [7, "b", "b", "NN", "O", 0, "root", 2, 1, "1", "d"],
            ],
            columns=CANONICAL_MINIMAL,
        )
        result = C.run(frame)
        assert any(d.code == "COMPLEXITY_NON_DENSE_IDS" for d in result.diagnostics)
        # Distances use the actual IDs; the invalid head 2 is excluded rather
        # than silently fabricated from row positions.
        assert float(result.unwrap().to_frame().iloc[0]["Mean Dependency Distance"]) == pytest.approx(0.0)

    def test_malformed_trees_get_diagnostics_not_zeroes(self) -> None:
        for bad in ["(S (NP x", "S (NP x))", "() ()"]:
            result = C.sum_yngve(bad)
            assert result.value is None
            assert any(d.code == "COMPLEXITY_BAD_TREE" for d in result.diagnostics), bad


class TestCli:
    @pytest.mark.model_integration
    @pytest.mark.skipif(not has_spacy_model(), reason="needs spaCy model")
    def test_cli_parses_and_writes(self, tmp_path: Path) -> None:
        from tools.sentence_complexity import main

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text("The president went to Italy. He stayed.", encoding="utf-8")
        out = tmp_path / "out"
        assert main([str(corpus), str(out), "--parser", "spacy"]) == 0
        run_dir = next(out.iterdir())
        assert (run_dir / "sentence_complexity.csv").is_file()
