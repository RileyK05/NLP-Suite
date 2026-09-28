"""Corpus comparison: a planted signal must be found, and nothing else claimed (docs/internal/PLAN_0.5.0.md 3.10).

Two small projects -- "Addresses" and "Inaugurals" -- share two speakers; a
third speaker gives only addresses. "budget" is planted in addresses only,
"oath" in inaugurals only, and "border" is used far more in the addresses.
The comparison runs as the app runs it: a job, parsed once, read from two
projects' folders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from desktop_backend.comparison_routes import preview
from desktop_backend.comparisons import ComparisonDefinition, side_documents
from desktop_backend.runner import Runner, run_job
from desktop_backend.store import Workspace

FILLER = (
    "We gather today as one people. Our country is strong and our future is bright. "
    "We will work together for the families who make this nation great. "
)
ADDRESSES = {
    "1950-01-04_ada adams_sotu.txt": "The budget must balance. The border needs care and the border needs order. ",
    "1951-01-08_ada adams_sotu.txt": "Congress will pass the budget. We will secure the border. ",
    "1952-01-09_ada adams_sotu.txt": "A fair budget for every state. The border matters. ",
    "1960-01-07_ben brown_sotu.txt": "The budget is the first duty. We speak of the border and the budget. ",
    "1961-01-12_ben brown_sotu.txt": "This budget invests in schools. The border is safe. ",
    "1970-01-22_cy clark_sotu.txt": "The budget and the border and the budget again. ",
}
INAUGURALS = {
    "1949-01-20_ada adams_ina.txt": "I take this oath with humility. Liberty is our light. ",
    "1953-01-20_ada adams_ina.txt": "The oath binds me to you. Liberty endures. ",
    "1957-01-21_ben brown_ina.txt": "With this oath I promise liberty. ",
    "1961-01-20_ben brown_ina.txt": "My oath is to the people. Liberty and the border. ",
}


def _project(workspace: Workspace, name: str, texts: dict[str, str]) -> str:
    project = workspace.create(name)
    for file_name, text in texts.items():
        workspace.import_document(project["id"], file_name, (text + FILLER * 3).encode("utf-8"))
    return str(project["id"])


@pytest.fixture
def two_projects(tmp_path: Path) -> tuple[Workspace, str, str]:
    workspace = Workspace(tmp_path / "ws")
    return workspace, _project(workspace, "Addresses", ADDRESSES), _project(workspace, "Inaugurals", INAUGURALS)


def _definition(addresses: str, inaugurals: str, **extra: Any) -> ComparisonDefinition:
    return ComparisonDefinition(
        sides=[
            {"name": "State of the Union", "project_id": addresses},
            {"name": "Inaugural", "project_id": inaugurals},
        ],
        **extra,
    )


def _run(
    workspace: Workspace, home: str, definition: ComparisonDefinition, monkeypatch: pytest.MonkeyPatch
) -> dict[str, pd.DataFrame]:
    monkeypatch.setenv("NLP_SUITE_RUN_FIGURES", "0")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    try:
        job = runner.submit_contrast(home, definition, "spacy", name="Test")
        run_job(workspace.root, job["id"])
    finally:
        runner.close()
    [done] = [j for j in workspace.jobs(home) if j["id"] == job["id"]]
    assert done["state"] in ("DONE", "PARTIAL"), done["diagnostics"]
    run = workspace.project_dir(home) / done["run_dir"]
    tables = {path.name: pd.read_csv(path) for path in run.glob("*.csv")}
    tables["_job"] = pd.DataFrame(done["diagnostics"])
    return tables


def test_sides_come_from_their_own_projects_and_carry_their_side(two_projects: tuple[Workspace, str, str]) -> None:
    workspace, addresses, inaugurals = two_projects
    documents = side_documents(workspace, _definition(addresses, inaugurals))
    assert len(documents) == 10
    assert {doc["project_id"] for doc in documents} == {addresses, inaugurals}
    assert {doc["fields"]["Side"] for doc in documents} == {"State of the Union", "Inaugural"}
    assert {doc["fields"]["Project"] for doc in documents} == {"Addresses", "Inaugurals"}
    assert {doc["project_id"] for doc in documents} == {addresses, inaugurals}


def test_carried_over_passages_use_cached_vectors_and_keep_source_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A planted semantic match is found and both locations can be traced back."""
    import numpy as np

    from core.conll.schema import canonical_columns
    from core.contrast.carryover import CarryoverSpec, carried_over
    from core.contrast.sides import align
    from core.io.reader import Corpus, Document
    from core.models.vector_cache import VectorCache
    from core.result import Result

    left = Document(
        1,
        Path("left.txt"),
        "We protect our families today. Astronomers observe distant stars.",
        sha256="left-sha",
        source_id="source-left",
        label="Left speech",
        source_project_id="project-left",
        fields=(("Side", "Before"), ("Speaker", "Ada")),
    )
    right = Document(
        2,
        Path("right.txt"),
        "Families deserve protection today. We will protect families again. Scientists record remote stars.",
        sha256="right-sha",
        source_id="source-right",
        label="Right speech",
        source_project_id="project-right",
        fields=(("Side", "After"), ("Speaker", "Ada")),
    )
    corpus = Corpus((left, right), "fixture-corpus")
    token_rows = []
    for document_id, name, sentences in (
        (
            "1",
            "Left speech",
            (("We", "protect", "our", "families", "today", "."), ("Astronomers", "observe", "distant", "stars", ".")),
        ),
        (
            "2",
            "Right speech",
            (
                ("Families", "deserve", "protection", "today", "."),
                ("We", "will", "protect", "families", "again", "."),
                ("Scientists", "record", "remote", "stars", "."),
            ),
        ),
    ):
        for sentence_id, words in enumerate(sentences, 1):
            for index, word in enumerate(words, 1):
                token_rows.append(
                    {
                        "ID": str(index),
                        "Form": word,
                        "Lemma": word.lower(),
                        "POS": "NOUN",
                        "NER": "O",
                        "Head": "0",
                        "DepRel": "root",
                        "Sentence ID": str(sentence_id),
                        "Document ID": document_id,
                        "Document": name,
                    }
                )
    table = pd.DataFrame(token_rows, columns=canonical_columns())

    class Backend:
        model_name = "fixture-carryover"

        def __init__(self) -> None:
            self.calls = 0

        def embed_texts(self, texts: list[str]) -> np.ndarray:
            self.calls += len(texts)
            return np.asarray(
                [
                    [1.0, 0.0, 0.0]
                    if "famil" in text.lower() and "again" not in text.lower()
                    else [0.99, 0.01, 0.0]
                    if "famil" in text.lower()
                    else [0.0, 1.0, 0.0]
                    for text in texts
                ],
                dtype=float,
            )

    backend = Backend()
    monkeypatch.setattr("core.analysis.doc_embeddings.text_backend", lambda _model: Result.success(backend))
    alignment = align(corpus, ("Before", "After"), "field:Speaker")
    cache = VectorCache(tmp_path / "workspace")
    carry_spec = CarryoverSpec(("Before", "After"), {"family": ["families", "protect"]})
    first = carried_over(table, corpus, alignment, carry_spec, cache=cache)
    assert first.value is not None
    assert len(first.unwrap()) == 1
    assert backend.calls == 3, "unfocused sentences should never be embedded"
    planted = first.unwrap().iloc[0]
    assert planted["Passage A"] == "We protect our families today."
    assert planted["Passage B"] == "Families deserve protection today."
    assert planted["Group"] == "Ada"
    assert planted["Project ID A"] == "project-left"
    assert planted["Document ID A"] == "source-left"
    assert planted["Sentence ID A"] == "1"
    assert planted["Project ID B"] == "project-right"
    assert planted["Document ID B"] == "source-right"
    assert planted["Sentence ID B"] == "1"
    embedded_calls = backend.calls
    from core.contrast.run import ContrastSpec, contrast

    published = contrast(
        table,
        corpus,
        ContrastSpec(
            sides=("Before", "After"),
            alignment="field:Speaker",
            methods=("carryover",),
            focus=carry_spec.focus,
        ),
        lambda _tool, _params: Result.success({}),
        vector_cache=cache,
    )
    assert published.value is not None
    assert "contrast_carryover.csv" in published.unwrap()
    assert len(published.unwrap()["contrast_carryover.csv"]) == 1
    assert not any(item.code == "CONTRAST_NOT_YET" for item in published.diagnostics)
    again = carried_over(table, corpus, alignment, carry_spec, cache=cache)
    assert again.value is not None and len(again.unwrap()) == 1
    assert backend.calls == embedded_calls, "the second comparison must reuse the sentence vectors"


def test_a_document_on_both_sides_is_refused(two_projects: tuple[Workspace, str, str]) -> None:
    workspace, addresses, _inaugurals = two_projects
    twice = ComparisonDefinition(
        sides=[
            {"name": "All", "project_id": addresses},
            {"name": "Adams", "project_id": addresses, "selection": {"fields": {"Speaker": ["Ada Adams"]}}},
        ]
    )
    with pytest.raises(ValueError, match="on both"):
        side_documents(workspace, twice)


def test_carried_over_passages_require_a_focus(two_projects: tuple[Workspace, str, str]) -> None:
    _workspace, addresses, inaugurals = two_projects
    with pytest.raises(ValueError, match="Carried-over passages need focus words"):
        _definition(addresses, inaugurals, methods=["carryover"])


def test_a_missing_project_is_named_by_its_side(two_projects: tuple[Workspace, str, str]) -> None:
    workspace, addresses, _inaugurals = two_projects
    with pytest.raises(ValueError, match="side “Inaugural” is not in this workspace"):
        side_documents(workspace, _definition(addresses, "0" * 32))


def test_the_preview_names_a_detail_with_one_value_and_a_side_it_cannot_read(
    two_projects: tuple[Workspace, str, str],
) -> None:
    workspace, addresses, inaugurals = two_projects
    found = preview(workspace, _definition(addresses, inaugurals))
    sotu, inaugural = found["sides"]
    assert sotu["documents"] == 6 and inaugural["documents"] == 4
    assert sotu["details"]["Kind"] == {"count": 1, "examples": ["sotu"]}
    assert sotu["details"]["Speaker"]["count"] == 3
    [speaker] = [item for item in found["alignments"] if item["name"] == "Speaker"]
    assert speaker["shared_values"] == 2
    assert found["dated"] is True

    broken = preview(workspace, _definition(addresses, "0" * 32))
    assert "Inaugural" in broken["sides"][1]["error"]
    assert broken["alignments"] == [], "nothing lines up with a side that cannot be read"


def test_the_planted_signal_is_found_aligned_by_speaker(
    two_projects: tuple[Workspace, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("spacy")
    workspace, addresses, inaugurals = two_projects
    definition = _definition(
        addresses,
        inaugurals,
        alignment="field:Speaker",
        methods=["measures", "keyness", "focus", "tone"],
        focus={"border": ["border", "borders"]},
    )
    tables = _run(workspace, addresses, definition, monkeypatch)

    notes = " ".join(tables["contrast_notes.csv"]["Note"])
    assert "different collections" in notes  # the setting confound comes first
    assert "Cy Clark" in notes and "left out" in notes  # the speaker only one side has

    words = tables["contrast_keyness.csv"]
    top = {side: set(part.head(5)["Word"]) for side, part in words.groupby("Side")}
    assert "budget" in top["State of the Union"] and "oath" in top["Inaugural"]
    assert (words["Log Ratio"] > 0).all(), "every row reads as 'more often on this side'"
    assert {"Per 10k", "Other per 10k"} <= set(words.columns)

    by_group = tables["contrast_focus_by_group.csv"]
    assert set(by_group["Group"]) == {"Ada Adams", "Ben Brown"}
    for _speaker, rows in by_group.groupby("Group"):
        rates = dict(zip(rows["Side"], rows["border per 10k"], strict=True))
        assert rates["State of the Union"] > rates["Inaugural"]

    measures = tables["contrast_measures.csv"]
    assert set(measures["Group"]) == {"Ada Adams", "Ben Brown"}
    assert "Cy Clark" not in set(measures["Group"])
    tests = tables["contrast_measure_tests.csv"]
    overall = tests[tests["Group"] == "All groups"]
    assert len(overall) >= 4 and overall["Reading"].str.contains("of 2 Speaker groups").all()
    # The direction is said once; the pooled figures follow without repeating it.
    for reading in overall["Reading"]:
        assert reading.count(" than ") == 1, reading
        assert "Over all compared documents: median" in reading, reading

    # The size note counts words as the Compare preview and the Corpus page do.
    sides = tables["contrast_sides.csv"].set_index("Side")
    stored = {doc["id"]: int(doc["words"]) for pid in (addresses, inaugurals) for doc in workspace.documents(pid)}
    by_side = {name: 0 for name in sides.index}
    for doc in side_documents(workspace, definition):
        by_side[doc["fields"]["Side"]] += stored[doc["id"]]
    assert sides["Words"].to_dict() == by_side

    manifest = tables["desktop_inputs.csv"]
    assert {"Side", "Speaker", "Kind", "project_id"} <= set(manifest.columns)


def test_within_one_project_by_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same question without a second project: Kind = sotu against Kind = ina."""
    pytest.importorskip("spacy")
    workspace = Workspace(tmp_path / "ws")
    mixed = _project(workspace, "Mixed", {**ADDRESSES, **INAUGURALS})
    definition = ComparisonDefinition(
        sides=[
            {"name": "Addresses", "project_id": mixed, "selection": {"fields": {"Kind": ["sotu"]}}},
            {"name": "Inaugurals", "project_id": mixed, "selection": {"fields": {"Kind": ["ina"]}}},
        ],
        methods=["keyness"],
    )
    tables = _run(workspace, mixed, definition, monkeypatch)
    notes = " ".join(tables["contrast_notes.csv"]["Note"])
    assert "different collections" not in notes  # one project: no setting confound to warn about
    words = tables["contrast_keyness.csv"]
    assert "oath" in set(words[words["Side"] == "Inaugurals"].head(5)["Word"])


class TestPeriodsOnAnOrderAxis:
    """Shared periods for a book: blocks of chapters, not only decades (plan 1.6)."""

    @staticmethod
    def _doc(index: int, side: str, order: int) -> Any:
        from core.io.reader import Document

        return Document(
            doc_id=index,
            path=Path(f"{side}_{order:02d}.txt"),
            text="text",
            sha256="x",
            date=None,
            fields=(("Side", side), ("Order", str(order))),
        )

    @classmethod
    def _two_books(cls) -> Any:
        from core.io.reader import Corpus

        docs = [
            cls._doc(index, side, order)
            for index, (side, order) in enumerate([(side, order) for side in ("A", "B") for order in range(1, 11)], 1)
        ]
        return Corpus(tuple(docs), "sha")

    def test_two_books_compare_within_blocks_of_chapters(self) -> None:
        from core.contrast.sides import align

        found = align(self._two_books(), ["A", "B"], "period")
        assert len(found.shared) == 4, found.shared  # 20 chapters make four blocks of five
        # Nobody named the order ("Chapter" is the project's label), so a step
        # is a document -- corpus_axis's documented default.
        assert all(label.startswith("Documents ") for label in found.shared), found.shared
        assert found.unplaced == ()
        # Every block holds both books' chapters, so every document is kept.
        assert len(found.groups) == 20

    def test_the_coverage_sentence_says_period_not_decade(self) -> None:
        from core.contrast.sides import align

        found = align(self._two_books(), ["A", "B"], "period")
        sentence = found.coverage("period")
        assert "period" in sentence
        assert "decade" not in sentence

    def test_a_dated_corpus_keeps_the_decade_rule(self) -> None:
        from datetime import date

        from core.contrast.sides import align
        from core.io.reader import Corpus, Document

        docs = tuple(
            Document(
                doc_id=index,
                path=Path(f"{side}_{when.year}.txt"),
                text="text",
                sha256="x",
                date=when,
                fields=(("Side", side),),
            )
            for index, (side, when) in enumerate(
                [("A", date(1930, 1, 1)), ("B", date(1931, 1, 1)), ("A", date(1940, 1, 1)), ("B", date(1941, 1, 1))],
                1,
            )
        )
        found = align(Corpus(docs, "sha"), ["A", "B"], "period")
        assert set(found.shared) == {"1930s", "1940s"}
        assert found.period_noun == "decade"

    def test_a_document_with_no_place_is_left_out_and_said_to_be(self) -> None:
        from core.contrast.sides import align
        from core.io.reader import Corpus, Document

        loose = Document(doc_id=99, path=Path("loose.txt"), text="text", sha256="x", fields=(("Side", "A"),))
        found = align(Corpus((*self._two_books().docs, loose), "sha"), ["A", "B"], "period")
        assert found.unplaced == ("loose.txt",)
        assert "no period" in found.coverage("period")
