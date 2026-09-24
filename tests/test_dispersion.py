"""CAP-STATS-13 — lexical dispersion.

The measures are pinned to hand-computed reference cases, including the
unequal-part case that is the whole reason Gries' DP is preferred to the
older measures.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.analysis.dispersion import PART_MODES, _gries_dp, _juilland_d, dispersion
from core.conll.schema import Col


def _table(documents: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for document_id, text in enumerate(documents, start=1):
        for token_id, token in enumerate(text.split(), start=1):
            rows.append(
                {
                    Col.ID.value: token_id,
                    Col.FORM.value: token,
                    Col.LEMMA.value: token.lower(),
                    Col.POS.value: "NOUN",
                    Col.NER.value: "",
                    Col.HEAD.value: 0,
                    Col.DEPREL.value: "root",
                    Col.DEPS.value: "",
                    Col.CLAUSE_TAG.value: "",
                    Col.RECORD_ID.value: len(rows) + 1,
                    Col.SENTENCE_ID.value: 1,
                    Col.DOCUMENT_ID.value: document_id,
                    Col.DOCUMENT.value: f"d{document_id}.txt",
                }
            )
    return pd.DataFrame(rows)


class TestFormulas:
    def test_perfectly_even_word_has_dp_zero_and_d_one(self) -> None:
        counts, sizes = [5, 5, 5, 5], [100, 100, 100, 100]
        deviation, _normalised = _gries_dp(counts, sizes, sum(counts), sum(sizes))
        assert deviation == pytest.approx(0.0)
        assert _juilland_d(counts, sizes) == pytest.approx(1.0)

    def test_fully_concentrated_word_hits_the_dp_ceiling(self) -> None:
        """All 20 occurrences in one of four equal parts.

        DP = 0.5 * (|1 - 0.25| + 3 * |0 - 0.25|) = 0.75, which is also the
        theoretical maximum for this corpus, so DP norm is exactly 1.
        """
        counts, sizes = [20, 0, 0, 0], [100, 100, 100, 100]
        deviation, normalised = _gries_dp(counts, sizes, sum(counts), sum(sizes))
        assert deviation == pytest.approx(0.75)
        assert normalised == pytest.approx(1.0)
        assert _juilland_d(counts, sizes) == pytest.approx(0.0)

    def test_dp_is_zero_when_a_word_tracks_unequal_part_sizes(self) -> None:
        """The reason DP is preferred: even means proportional, not equal.

        A word occurring 10/20/30/40 times in parts of 100/200/300/400 tokens
        is perfectly evenly spread, and a measure that ignored part size would
        wrongly call it clumped.
        """
        counts, sizes = [10, 20, 30, 40], [100, 200, 300, 400]
        deviation, _normalised = _gries_dp(counts, sizes, sum(counts), sum(sizes))
        assert deviation == pytest.approx(0.0)
        assert _juilland_d(counts, sizes) == pytest.approx(1.0)

    def test_juilland_d_never_goes_negative(self) -> None:
        """D is bounded at 0; a negative would be floating-point error."""
        for parts in (2, 3, 5, 10):
            counts = [7] + [0] * (parts - 1)
            assert _juilland_d(counts, [50] * parts) >= 0.0


class TestFrameOutput:
    def test_spread_word_beats_clumped_word_of_equal_frequency(self) -> None:
        """The headline claim: equal frequency, opposite dispersion."""
        documents = [
            "spread alpha beta gamma",
            "spread delta epsilon zeta",
            "spread eta theta iota",
            "clump clump clump kappa",
        ]
        frame = dispersion(_table(documents), min_count=3).unwrap()
        spread = frame[frame["Term"] == "spread"].iloc[0]
        clump = frame[frame["Term"] == "clump"].iloc[0]
        assert spread["Frequency"] == clump["Frequency"] == 3
        assert spread["Range"] == 3
        assert clump["Range"] == 1
        assert spread["Gries DP"] < clump["Gries DP"]
        assert spread["Juilland D"] > clump["Juilland D"]
        assert spread["Adjusted Frequency"] > clump["Adjusted Frequency"]

    def test_range_counts_parts_containing_the_term(self) -> None:
        frame = dispersion(_table(["alpha beta", "alpha gamma", "delta delta"]), min_count=1).unwrap()
        assert frame[frame["Term"] == "alpha"]["Range"].iloc[0] == 2
        assert frame[frame["Term"] == "delta"]["Range"].iloc[0] == 1

    def test_chunk_mode_measures_dispersion_inside_one_document(self) -> None:
        """Document mode cannot work on a single text; chunk mode can."""
        single = _table([" ".join(["early"] * 20 + ["late"] * 20)])
        by_document = dispersion(single, parts="document", min_count=5)
        assert by_document.unwrap().empty
        assert any(d.code == "DISP_TOO_FEW_PARTS" for d in by_document.diagnostics)

        by_chunk = dispersion(single, parts="chunk", chunks=4, min_count=5).unwrap()
        assert not by_chunk.empty
        assert by_chunk[by_chunk["Term"] == "early"]["Range"].iloc[0] == 2

    def test_chunk_sizes_differ_by_at_most_one_token(self) -> None:
        """The remainder is spread, not dumped into the last chunk."""
        from core.analysis.dispersion import _parts_by_chunk

        frame = _table([" ".join(f"word{chr(ord('a') + i % 26)}" for i in range(23))])
        parts = _parts_by_chunk(frame, Col.LEMMA.value, 1, 5)
        sizes = [len(part) for part in parts]
        assert sum(sizes) == 23
        assert max(sizes) - min(sizes) <= 1


class TestFiltering:
    def test_min_count_drops_rare_terms(self) -> None:
        frame = dispersion(_table(["alpha alpha", "alpha beta"]), min_count=2).unwrap()
        assert set(frame["Term"]) == {"alpha"}

    def test_min_length_and_non_alphabetic_filters(self) -> None:
        frame = dispersion(_table(["a 12 alpha", "a 12 alpha"]), min_count=1, min_length=2).unwrap()
        terms = set(frame["Term"])
        assert "a" not in terms
        assert "12" not in terms
        assert "alpha" in terms

    def test_top_n_truncates_and_reports(self) -> None:
        documents = [" ".join(f"word{chr(ord('a') + i)} " * 2 for i in range(10))] * 3
        result = dispersion(_table(documents), min_count=1, top_n=3)
        assert result.ok
        assert len(result.unwrap()) == 3
        assert any(d.code == "DISP_TRUNCATED" for d in result.diagnostics)


class TestDeterminism:
    def test_repeated_runs_are_byte_identical(self) -> None:
        frame = _table(["alpha beta gamma", "alpha beta delta", "alpha epsilon"])
        first = dispersion(frame, min_count=1).unwrap()
        for _ in range(5):
            assert dispersion(frame, min_count=1).unwrap().equals(first)

    def test_sorted_by_frequency_then_term(self) -> None:
        frame = dispersion(_table(["alpha beta gamma", "alpha beta delta"]), min_count=1).unwrap()
        expected = frame.sort_values(["Frequency", "Term"], ascending=[False, True], kind="stable").reset_index(
            drop=True
        )
        assert frame.equals(expected)


class TestContracts:
    @pytest.mark.parametrize(
        ("kwargs", "code"),
        [
            ({"parts": "sideways"}, "DISP_BAD_PARTS"),
            ({"chunks": 1}, "DISP_BAD_CHUNKS"),
            ({"min_count": 0}, "DISP_BAD_MIN_COUNT"),
            ({"min_length": 0}, "DISP_BAD_MIN_LENGTH"),
            ({"top_n": 0}, "DISP_BAD_TOP_N"),
            ({"field": Col.POS}, "DISP_BAD_FIELD"),
        ],
    )
    def test_bad_arguments_fail_with_a_named_diagnostic(self, kwargs: dict[str, object], code: str) -> None:
        result = dispersion(_table(["alpha", "beta"]), **kwargs)  # type: ignore[arg-type]
        assert not result.ok
        assert [d.code for d in result.errors] == [code]

    def test_no_surviving_term_warns_rather_than_failing(self) -> None:
        result = dispersion(_table(["alpha", "beta"]), min_count=99)
        assert result.ok
        assert result.unwrap().empty
        assert any(d.code == "DISP_NO_TERMS" for d in result.diagnostics)

    def test_empty_frame_returns_the_full_schema(self) -> None:
        result = dispersion(_table(["alpha"]).iloc[0:0])
        assert result.ok
        assert "Gries DP" in result.unwrap().columns

    def test_every_part_mode_is_reachable(self) -> None:
        frame = _table(["alpha beta alpha", "alpha gamma alpha"])
        for mode in PART_MODES:
            assert dispersion(frame, parts=mode, chunks=2, min_count=1).ok


class TestRegistryWiring:
    def test_registered_outputs_match_the_cli(self) -> None:
        from core.profiler.registry import get_tool

        spec = get_tool("dispersion")
        assert spec is not None
        assert spec.outputs == ("dispersion.csv", "dispersion_plot.html")
        assert spec.capability_ids == ("CAP-STATS-13",)

    def test_adapter_is_wired_for_the_profiler(self) -> None:
        from core.profiler.executor import ADAPTERS

        assert "dispersion" in ADAPTERS
