"""FR-8.3 child (C2) — declarative form builder contract tests.

Registry specs must render to widget descriptors without Streamlit
(lazy UI import), covering every param type, choices, bounds, and the
None-default/required edges. Fails until app/forms.py lands (C3).
"""

from __future__ import annotations

from core.profiler.registry import TOOL_REGISTRY, get_tool


class TestFormBuilders:
    def test_every_spec_renders(self) -> None:
        from app import forms as forms_mod

        for spec in TOOL_REGISTRY:
            widgets = forms_mod.widgets_for(spec.name)
            assert [w.key for w in widgets] == [p.name for p in spec.params], spec.name

    def test_widget_kinds_cover_types(self) -> None:
        from app import forms as forms_mod

        search = get_tool("search")
        assert search is not None
        kinds = {w.key: w.widget for w in forms_mod.widgets_for("search")}
        assert kinds["mode"] == "selectbox"
        assert kinds["query"] == "text"
        assert kinds["case-sensitive"] == "checkbox"

    def test_numeric_bounds_carried(self) -> None:
        from app import forms as forms_mod

        (alpha,) = [w for w in forms_mod.widgets_for("stats_categorical") if w.key == "alpha"]
        assert alpha.widget == "number"
        assert alpha.minimum == 0.0 and alpha.maximum == 1.0
        assert alpha.default == 0.05

    def test_none_default_allowed(self) -> None:
        from app import forms as forms_mod

        (query,) = [w for w in forms_mod.widgets_for("word2vec_gensim") if w.key == "query"]
        assert query.default is None and query.required is False

    def test_required_path(self) -> None:
        from app import forms as forms_mod

        (wordlist,) = [w for w in forms_mod.widgets_for("spellcheck") if w.key == "wordlist"]
        assert wordlist.widget == "path" and wordlist.required is True

    def test_unknown_tool_fails(self) -> None:
        from app import forms as forms_mod

        try:
            forms_mod.widgets_for("nope")
        except KeyError:
            return
        raise AssertionError("expected KeyError for unknown tool")

    def test_human_labels(self) -> None:
        from app import forms as forms_mod

        labels = {w.key: w.label for w in forms_mod.widgets_for("lda_gensim")}
        assert labels["top-n"] == "Top N"
        assert labels["keep-stopwords"] == "Keep Stopwords"
        assert labels["topics"] == "Topics"
