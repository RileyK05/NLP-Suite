"""C41-C46 - wordcloud, shapes, data, SQL, validation, PC-ACE."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.data import manipulation as dm_mod, sql as sql_mod, validation as val_mod
from core.pcace import core as pcace_mod
from core.viz import shapes as shape_mod, wordcloud_gephi as wc_mod


def test_wordcloud_and_gephi() -> None:
    df = pd.DataFrame({"Word": ["hello", "world", "hello"], "Count": [5, 3, 5]})
    wc = wc_mod.wordcloud_html(df, word_col="Word", weight_col="Count", title="Test", max_words=2)
    assert wc.ok
    html_str = wc.unwrap()
    assert "hello" in html_str
    assert "span" in html_str

    edge_df = pd.DataFrame({"Source": ["A", "B"], "Target": ["B", "C"], "Weight": [1, 2]})
    gex = wc_mod.gephi_gexf(edge_df, source_col="Source", target_col="Target", weight_col="Weight")
    assert gex.ok
    assert "<?xml" in gex.unwrap()
    assert "B" in gex.unwrap()

    assert not wc_mod.wordcloud_html(pd.DataFrame(columns=["Word", "Count"]), word_col="Word", weight_col="Count").ok
    assert not wc_mod.gephi_gexf(pd.DataFrame(columns=["Source", "Target"]), source_col="Source", target_col="Nope").ok


def test_shapes() -> None:
    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "Form": "Hello",
                "Lemma": "hello",
                "POS": "NNP",
                "NER": "O",
                "Head": 2,
                "DepRel": "nsubj",
                "Sentence ID": 1,
                "Document ID": 1,
                "Document": "a.txt",
                "Deps": "",
                "Record ID": 1,
                "Clause Tag": "",
            },
            {
                "ID": 2,
                "Form": "world",
                "Lemma": "world",
                "POS": "NN",
                "NER": "O",
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": 1,
                "Document ID": 1,
                "Document": "a.txt",
                "Deps": "",
                "Record ID": 2,
                "Clause Tag": "",
            },
            {
                "ID": 1,
                "Form": "Run",
                "Lemma": "run",
                "POS": "VB",
                "NER": "O",
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": 2,
                "Document ID": 1,
                "Document": "a.txt",
                "Deps": "",
                "Record ID": 3,
                "Clause Tag": "",
            },
        ]
    )
    res = shape_mod.story_shape(df)
    assert res.ok
    sdf = res.unwrap()
    assert len(sdf) == 2
    assert "Noun Ratio" in sdf.columns

    sank_df = pd.DataFrame({"Source": ["A", "A"], "Target": ["B", "C"], "Value": [2, 3]})
    sank = shape_mod.sankey_html(sank_df, source="Source", target="Target", value="Value", title="Flow")
    assert sank.ok
    assert "A" in sank.unwrap()

    net = shape_mod.network_gexf(sank_df, source="Source", target="Target", weight="Value")
    assert net.ok
    assert "gexf" in net.unwrap().lower()


def test_data_manipulation() -> None:
    a = pd.DataFrame({"Key": [1, 2], "Val": ["a", "b"]})
    b = pd.DataFrame({"Key": [2, 3], "Val2": ["x", "y"]})
    m = dm_mod.merge(a, b, on="Key", how="inner")
    assert m.ok
    assert len(m.unwrap()) == 1

    c = dm_mod.concat([a, a], axis=0)
    assert c.ok
    assert len(c.unwrap()) == 4

    p = pd.DataFrame({"Idx": ["x", "x", "y"], "Col": ["a", "b", "a"], "Val": [1, 2, 3]})
    piv = dm_mod.pivot(p, index="Idx", columns="Col", values="Val", aggfunc="sum")
    assert piv.ok
    melt = dm_mod.melt(piv.unwrap(), id_vars=["Idx"])
    assert melt.ok

    assert not dm_mod.merge(a, b, on="Nope").ok
    assert not dm_mod.concat([], axis=0).ok


def test_sql(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    r1 = sql_mod.execute(db, "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
    assert r1.ok
    r2 = sql_mod.execute(db, "INSERT INTO t (name) VALUES (?)", ("alice",))
    assert r2.ok
    r3 = sql_mod.execute(db, "INSERT INTO t (name) VALUES (?)", ("bob",))
    assert r3.ok
    q = sql_mod.query(db, "SELECT * FROM t WHERE name = ?", ("alice",))
    assert q.ok
    qdf = q.unwrap()
    assert len(qdf) == 1
    assert qdf["name"].iloc[0] == "alice"

    bad = sql_mod.query(db, "DROP TABLE t")
    assert not bad.ok
    bad2 = sql_mod.query(tmp_path / "no.db", "SELECT * FROM t")
    assert not bad2.ok


def test_validation(tmp_path: Path) -> None:
    td = tmp_path / "corpus"
    td.mkdir()
    (td / "a.txt").write_text("hello world", encoding="utf-8")
    (td / "b.txt").write_text("", encoding="utf-8")
    (td / "c.txt").write_text("a", encoding="utf-8")
    res = val_mod.validate(td, min_tokens=2, required_ext=".txt")
    assert res.ok
    vdf = res.unwrap()
    assert len(vdf) == 3
    assert set(vdf["Status"]) == {"OK", "EMPTY", "SHORT"}

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    e2 = val_mod.validate(empty_dir)
    assert e2.ok
    assert e2.unwrap().empty

    bad = val_mod.validate(tmp_path / "nope")
    assert not bad.ok


def test_pcace(tmp_path: Path) -> None:
    g = pcace_mod.grammar_info()
    assert len(g) > 0
    assert "1.1" in g["Code"].tolist()

    p1 = pcace_mod.parse_code("1.1")
    assert p1.ok
    assert p1.unwrap()["category"] == "Government"
    p2 = pcace_mod.parse_code("bad code !")
    assert not p2.ok
    p3 = pcace_mod.parse_code("")
    assert not p3.ok

    db = tmp_path / "pcace.db"
    c1 = pcace_mod.create_db(db, ["1.1", "2.1", "3"])
    assert c1.ok
    assert db.is_file()
    r = pcace_mod.read_db(db)
    assert r.ok
    assert len(r.unwrap()) == 3

    bad = pcace_mod.create_db(db, ["bad code"])
    assert not bad.ok
    empty = pcace_mod.create_db(tmp_path / "e.db", [])
    assert not empty.ok
