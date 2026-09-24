"""FR-7 child 2 (C2) — pure profiler plan model contract tests.

Selection validation, parameter checking against the registry specs, and
phase ordering. No execution, no parsing, no I/O: all offline. Fails until
core/profiler/plan.py lands (C3).
"""

from __future__ import annotations

import pytest

from core.profiler import plan as plan_mod


class TestPlanValidation:
    def test_unknown_tool_fails_by_name(self) -> None:
        res = plan_mod.build_plan(["nope"], {})
        assert res.value is None
        assert any(d.code == "PROFILER_UNKNOWN_TOOL" and "nope" in d.message for d in res.diagnostics)

    def test_ineligible_tool_fails(self) -> None:
        res = plan_mod.build_plan(["wordnet"], {})
        assert res.value is None
        assert any(d.code == "PROFILER_NOT_ELIGIBLE" for d in res.diagnostics)

    def test_empty_selection_fails(self) -> None:
        res = plan_mod.build_plan([], {})
        assert res.value is None
        assert any(d.code == "PROFILER_EMPTY_SELECTION" for d in res.diagnostics)

    def test_unknown_param_fails(self) -> None:
        res = plan_mod.build_plan(["readability"], {"readability": {"nope": 1}})
        assert res.value is None
        assert any(d.code == "PROFILER_UNKNOWN_PARAM" for d in res.diagnostics)

    def test_bad_choice_fails(self) -> None:
        res = plan_mod.build_plan(["word2vec_gensim"], {"word2vec_gensim": {"sg": 7}})
        assert res.value is None
        assert any(d.code == "PROFILER_BAD_PARAM" for d in res.diagnostics)

    def test_missing_required_param_fails(self) -> None:
        res = plan_mod.build_plan(["spellcheck"], {})
        assert res.value is None
        assert any(d.code == "PROFILER_MISSING_PARAM" for d in res.diagnostics)

    def test_bad_type_fails(self) -> None:
        res = plan_mod.build_plan(["lda_gensim"], {"lda_gensim": {"topics": "three"}})
        assert res.value is None
        assert any(d.code == "PROFILER_BAD_PARAM" for d in res.diagnostics)

    def test_out_of_bounds_fails(self) -> None:
        res = plan_mod.build_plan(["lda_gensim"], {"lda_gensim": {"topics": 0}})
        assert res.value is None
        assert any(d.code == "PROFILER_BAD_PARAM" for d in res.diagnostics)


class TestPlanShape:
    def test_phase_ordering(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.profiler.registry import ToolSpec

        specs = (
            ToolSpec(
                name="late",
                capability_ids=("CAP-X",),
                packet="test",
                description="phase 3",
                requires_parse=False,
                params=(),
                outputs=("late.csv",),
                version="1",
                execution_phase=3,
            ),
            ToolSpec(
                name="early",
                capability_ids=("CAP-X",),
                packet="test",
                description="phase 1",
                requires_parse=False,
                params=(),
                outputs=("early.csv",),
                version="1",
                execution_phase=1,
            ),
        )
        monkeypatch.setattr(plan_mod, "TOOL_REGISTRY", specs)
        monkeypatch.setattr(plan_mod, "get_tool", lambda name: next((s for s in specs if s.name == name), None))
        res = plan_mod.build_plan(["late", "early"], {})
        assert res.ok, res.diagnostics
        assert [t.name for t in res.unwrap().tools] == ["early", "late"]

    def test_needs_parse(self) -> None:
        assert plan_mod.build_plan(["readability"], {}).unwrap().needs_parse is False
        plan = plan_mod.build_plan(["readability", "sentence_complexity"], {}).unwrap()
        assert plan.needs_parse is True

    def test_defaults_applied(self) -> None:
        plan = plan_mod.build_plan(["lda_gensim"], {}).unwrap()
        assert plan.params["lda_gensim"] == {
            "field": "lemma",
            "topics": 3,
            "top-n": 5,
            "seed": 100,
            "nouns-only": False,
            "keep-stopwords": False,
            "lambda": 0.6,
        }

    def test_explicit_params_kept(self) -> None:
        plan = plan_mod.build_plan(["lda_gensim"], {"lda_gensim": {"topics": 2}}).unwrap()
        assert plan.params["lda_gensim"]["topics"] == 2
        assert plan.params["lda_gensim"]["seed"] == 100

    def test_duplicates_deduped_stably(self) -> None:
        plan = plan_mod.build_plan(["readability", "readability"], {}).unwrap()
        assert [t.name for t in plan.tools] == ["readability"]
