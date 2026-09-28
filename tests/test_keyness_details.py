"""Keyness by a document detail: "Kind = sotu against Kind = ina" with no regex (docs/internal/PLAN_0.5.0.md 1.6.5)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.analysis.keyness import detail_groups
from core.io.reader import Corpus, Document
from core.profiler import executor as exec_mod
from core.profiler.plan import build_plan

DETAILS = {
    "1946_truman_sotu.txt": {"Speaker": "Truman", "Kind": "sotu"},
    "1949_truman_ina.txt": {"Speaker": "Truman", "Kind": "ina"},
    "1953_eisenhower_sotu.txt": {"Speaker": "Eisenhower", "Kind": "sotu"},
    "1957_eisenhower_ina.txt": {"Speaker": "Eisenhower", "Kind": "ina"},
    "notes.txt": {},
}


def test_groups_from_a_detail_and_its_values() -> None:
    groups, labels = detail_groups(DETAILS, "kind", "SOTU", "ina").unwrap()
    assert labels == ("kind = SOTU", "kind = ina")
    assert groups == {
        "1946_truman_sotu.txt": "A",
        "1949_truman_ina.txt": "B",
        "1953_eisenhower_sotu.txt": "A",
        "1957_eisenhower_ina.txt": "B",
    }, "a document with neither value (notes.txt) is left out"


def test_an_empty_group_b_is_every_other_document() -> None:
    groups, labels = detail_groups(DETAILS, "Speaker", "Truman").unwrap()
    assert labels == ("Speaker = Truman", "all others")
    assert groups["notes.txt"] == "B" and groups["1953_eisenhower_sotu.txt"] == "B"


def test_refusals_name_what_exists() -> None:
    unknown = detail_groups(DETAILS, "Party", "Democratic")
    assert unknown.value is None and "details here: Kind, Speaker" in unknown.diagnostics[0].message
    wrong = detail_groups(DETAILS, "Kind", "speech")
    assert wrong.value is None and "values: ina, sotu" in wrong.diagnostics[0].message
    empty = detail_groups(DETAILS, "Kind", "")
    assert empty.value is None and empty.diagnostics[0].code == "KEYNESS_NO_GROUP_A"


def _table() -> pd.DataFrame:
    rows = []
    words = {"sotu": ["budget", "congress", "budget"], "ina": ["oath", "liberty", "oath"]}
    record = 0
    for doc_id, (name, fields) in enumerate(DETAILS.items()):
        for sentence, word in enumerate(words.get(fields.get("Kind", ""), ["note"]) * 3, 1):
            record += 1
            rows.append(
                {
                    "ID": 1,
                    "Form": word,
                    "Lemma": word,
                    "POS": "NN",
                    "NER": "O",
                    "Head": 0,
                    "DepRel": "root",
                    "Sentence ID": sentence,
                    "Document ID": str(doc_id),
                    "Document": name,
                    "Deps": "",
                    "Record ID": record,
                    "Clause Tag": "",
                }
            )
    return pd.DataFrame(rows)


def _corpus() -> Corpus:
    docs = tuple(
        Document(doc_id=index, path=Path(name), text="x", fields=tuple(sorted(fields.items())))
        for index, (name, fields) in enumerate(DETAILS.items())
    )
    return Corpus(docs=docs, sha256="test")


def test_the_tool_runs_by_detail_and_its_columns_name_the_groups() -> None:
    plan = build_plan(
        ["keyness"], {"keyness": {"group-field": "Kind", "group-a": "sotu", "group-b": "ina", "top-n": 0}}
    ).unwrap()
    (outcome,) = exec_mod.execute(plan, corpus=_corpus(), table=_table()).outcomes
    assert outcome.ok, outcome.diagnostics
    frame = outcome.frames["keyness.csv"]
    assert {"Freq Kind = sotu", "Freq Kind = ina"} <= set(frame.columns)
    assert "note" not in set(frame["Word"]), "a document with no Kind is in neither group"
    top = frame.sort_values("G2 (log-likelihood)", ascending=False).head(4)
    assert {"budget", "oath"} <= set(top["Word"])


def test_the_tool_says_how_to_name_the_groups_when_given_neither() -> None:
    plan = build_plan(["keyness"], {"keyness": {}}).unwrap()
    (outcome,) = exec_mod.execute(plan, corpus=_corpus(), table=_table()).outcomes
    assert not outcome.ok
    assert outcome.diagnostics[0].code == "KEYNESS_NO_GROUPS"
    assert "group-field" in outcome.diagnostics[0].message


def test_the_regex_path_is_unchanged() -> None:
    plan = build_plan(["keyness"], {"keyness": {"group-pattern": "_sotu"}}).unwrap()
    (outcome,) = exec_mod.execute(plan, table=_table()).outcomes
    assert outcome.ok, outcome.diagnostics
    assert any(str(column).endswith("(pattern docs)") for column in outcome.frames["keyness.csv"].columns)
