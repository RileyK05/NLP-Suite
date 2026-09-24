"""Lexical dispersion plot — geometry, provenance and knowable degradation."""

from __future__ import annotations

import builtins
from collections.abc import Iterator
import contextlib

import pandas as pd
import pytest

from core.conll.schema import Col
from core.viz.dispersion_plot import MAX_TERMS, dispersion_plot, occurrence_offsets

DOCUMENTS = ["alpha beta alpha gamma", "delta alpha epsilon", "zeta alpha alpha eta"]


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


@contextlib.contextmanager
def _without_plotly() -> Iterator[None]:
    """Make ``import plotly`` fail, as on a machine without the extra."""
    real_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("plotly"):
            raise ImportError("plotly blocked for test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    builtins.__import__ = blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        builtins.__import__ = real_import


class TestOffsets:
    def test_offsets_are_corpus_wide_token_positions(self) -> None:
        """The x-axis runs across the whole corpus, not per document."""
        points = occurrence_offsets(_table(DOCUMENTS), ["alpha"]).unwrap()
        # alpha is token 0 and 2 of doc 1, token 5 (=4+1) of doc 2, 8 and 9 of doc 3.
        assert points["Offset"].tolist() == [0, 2, 5, 8, 9]

    def test_each_occurrence_keeps_its_document_provenance(self) -> None:
        points = occurrence_offsets(_table(DOCUMENTS), ["alpha"]).unwrap()
        assert points["Document"].tolist() == ["d1.txt", "d1.txt", "d2.txt", "d3.txt", "d3.txt"]

    def test_matching_is_casefolded(self) -> None:
        points = occurrence_offsets(_table(DOCUMENTS), ["ALPHA"]).unwrap()
        assert len(points) == 5

    def test_absent_term_warns_but_succeeds(self) -> None:
        result = occurrence_offsets(_table(DOCUMENTS), ["nowhere"])
        assert result.ok
        assert result.unwrap().empty
        assert any(d.code == "DISPPLOT_NO_MATCHES" for d in result.diagnostics)

    @pytest.mark.parametrize(
        ("terms", "field", "code"),
        [
            ([], Col.LEMMA, "DISPPLOT_NO_TERMS"),
            (["x"] * (MAX_TERMS + 1), Col.LEMMA, "DISPPLOT_TOO_MANY_TERMS"),
            (["alpha"], Col.POS, "DISPPLOT_BAD_FIELD"),
        ],
    )
    def test_bad_arguments_fail_with_a_named_diagnostic(self, terms: list[str], field: Col, code: str) -> None:
        result = occurrence_offsets(_table(DOCUMENTS), terms, field=field)
        assert not result.ok
        assert [d.code for d in result.errors] == [code]


class TestRendering:
    def test_plotly_path_produces_an_interactive_chart(self) -> None:
        pytest.importorskip("plotly")
        result = dispersion_plot(_table(DOCUMENTS), ["alpha", "beta"])
        assert result.ok
        html = result.unwrap()
        assert "plotly" in html.lower()
        assert not any(d.code == "DISPPLOT_SVG_FALLBACK" for d in result.diagnostics)

    def test_without_plotly_it_degrades_to_a_real_svg_plot(self) -> None:
        """Knowable degradation: a plot, not a table, and it says so."""
        with _without_plotly():
            result = dispersion_plot(_table(DOCUMENTS), ["alpha", "beta"])
        assert result.ok
        html = result.unwrap()
        assert "<svg" in html
        assert "plotly" not in html.lower()
        assert any(d.code == "DISPPLOT_SVG_FALLBACK" for d in result.diagnostics)

    def test_svg_draws_one_tick_per_occurrence(self) -> None:
        with _without_plotly():
            html = dispersion_plot(_table(DOCUMENTS), ["alpha", "beta"]).unwrap()
        # alpha occurs 5 times, beta once.
        assert html.count('stroke="#2b6cb0"') == 6

    def test_svg_marks_document_boundaries_except_the_first(self) -> None:
        with _without_plotly():
            html = dispersion_plot(_table(DOCUMENTS), ["alpha"]).unwrap()
        assert html.count("stroke-dasharray") == len(DOCUMENTS) - 1

    def test_a_term_with_no_occurrences_still_gets_a_lane(self) -> None:
        """An empty row answers 'where does this word appear?' with 'nowhere'."""
        with _without_plotly():
            html = dispersion_plot(_table(DOCUMENTS), ["alpha", "nowhere"]).unwrap()
        assert "nowhere (0)" in html

    def test_duplicate_terms_collapse_to_one_lane(self) -> None:
        with _without_plotly():
            html = dispersion_plot(_table(DOCUMENTS), ["alpha", "ALPHA", "alpha"]).unwrap()
        assert html.count("alpha (5)") == 1

    def test_document_names_are_html_escaped(self) -> None:
        """A document name is untrusted text and reaches the SVG."""
        frame = _table(DOCUMENTS)
        frame[Col.DOCUMENT.value] = "<script>x</script>.txt"
        with _without_plotly():
            html = dispersion_plot(frame, ["alpha"]).unwrap()
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_title_is_html_escaped(self) -> None:
        with _without_plotly():
            html = dispersion_plot(_table(DOCUMENTS), ["alpha"], title="<img src=x>").unwrap()
        assert "<img src=x>" not in html
        assert "&lt;img" in html

    def test_render_is_deterministic(self) -> None:
        with _without_plotly():
            first = dispersion_plot(_table(DOCUMENTS), ["alpha", "beta"]).unwrap()
            for _ in range(3):
                assert dispersion_plot(_table(DOCUMENTS), ["alpha", "beta"]).unwrap() == first


class TestTickThinning:
    """One SVG element per occurrence produced a 17 MB file for one lane."""

    @staticmethod
    def _repeated(term: str, count: int, *, stride: int = 1) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for index in range(count * stride):
            token = term if index % stride == 0 else "filler"
            rows.append(
                {
                    Col.ID.value: 1,
                    Col.FORM.value: token,
                    Col.LEMMA.value: token,
                    Col.POS.value: "NOUN",
                    Col.NER.value: "",
                    Col.HEAD.value: 0,
                    Col.DEPREL.value: "root",
                    Col.DEPS.value: "",
                    Col.CLAUSE_TAG.value: "",
                    Col.RECORD_ID.value: index,
                    Col.SENTENCE_ID.value: 1,
                    Col.DOCUMENT_ID.value: 1,
                    Col.DOCUMENT.value: "a.txt",
                }
            )
        return pd.DataFrame(rows)

    def test_ticks_are_capped_regardless_of_corpus_size(self) -> None:
        from core.viz.dispersion_plot import MAX_TICKS_PER_TERM

        with _without_plotly():
            small = dispersion_plot(self._repeated("the", 20_000), ["the"]).unwrap()
            large = dispersion_plot(self._repeated("the", 60_000), ["the"]).unwrap()
        assert small.count('stroke="#2b6cb0"') == MAX_TICKS_PER_TERM
        assert large.count('stroke="#2b6cb0"') == MAX_TICKS_PER_TERM
        # Bounded output, not merely smaller output.
        assert len(large) < 1_000_000

    def test_the_label_keeps_the_true_count(self) -> None:
        """A thinned plot must not also under-report."""
        with _without_plotly():
            html = dispersion_plot(self._repeated("the", 20_000), ["the"]).unwrap()
        assert "the (20000)" in html

    def test_thinning_is_reported(self) -> None:
        with _without_plotly():
            result = dispersion_plot(self._repeated("the", 20_000), ["the"])
        assert any(d.code == "DISPPLOT_THINNED" for d in result.diagnostics)

    def test_no_thinning_diagnostic_when_nothing_is_thinned(self) -> None:
        with _without_plotly():
            result = dispersion_plot(self._repeated("the", 50), ["the"])
        assert not any(d.code == "DISPPLOT_THINNED" for d in result.diagnostics)

    def test_thinning_preserves_the_shape_of_the_distribution(self) -> None:
        """A word confined to the opening must still look confined."""
        import re

        rows = self._repeated("the", 10_000)
        tail = self._repeated("filler", 90_000)
        frame = pd.concat([rows, tail], ignore_index=True)
        with _without_plotly():
            html = dispersion_plot(frame, ["the"]).unwrap()
        positions = [
            float(match)
            for match in re.findall(r'<line x1="([\d.]+)" y1="[\d.]+" x2="[\d.]+" y2="[\d.]+" stroke="#2b6cb0"', html)
        ]
        assert positions
        # Gutter is 130px and the plot area 750px; the word occupies the first
        # tenth of the corpus, so every tick must land in the left tenth.
        assert max(positions) < 130 + 750 * 0.12

    def test_thinning_is_deterministic(self) -> None:
        frame = self._repeated("the", 20_000)
        with _without_plotly():
            first = dispersion_plot(frame, ["the"]).unwrap()
            assert dispersion_plot(frame, ["the"]).unwrap() == first
