"""KWIC concordance + keyness lab + collocation network + movement tracks +
Earth tour KML + BERT topics (the expansion batch)."""

from __future__ import annotations

import pandas as pd

from conftest import HashEmbeddingBackend
from core.analysis.bert_topics import bert_topics
from core.analysis.keyness import keyness
from core.analysis.kwic import concordance
from core.analysis.movement import movement_summary, movement_tracks
from core.analysis.ngrams import collocation_network
from core.conll.schema import Col
from core.gis.mapping import kml, tour_kml


def _frame(docs: dict[str, list[tuple[str, str, str]]] | None = None) -> pd.DataFrame:
    """docs: name -> list of (form, lemma, ner) tokens; sentence = 3 tokens."""
    rows = []
    rec = 1
    for doc_id, (name, tokens) in enumerate((docs or PROBE).items(), start=1):
        for index, (form, lemma, ner) in enumerate(tokens, start=1):
            sid = (index - 1) // 3 + 1
            rows.append(
                {
                    "ID": index,
                    "Form": form,
                    "Lemma": lemma,
                    "POS": "NN",
                    "NER": ner,
                    "Head": 0,
                    "DepRel": "root",
                    "Sentence ID": sid,
                    "Document ID": str(doc_id),
                    "Document": name,
                    "Deps": "",
                    "Record ID": rec,
                    "Clause Tag": "",
                }
            )
            rec += 1
    return pd.DataFrame(rows)


PROBE = {
    "alice.txt": [("Alice", "alice", "PERSON"), ("loved", "love", "O"), ("Paris", "paris", "GPE")] * 3,
    "bob.txt": [("Bob", "bob", "PERSON"), ("hated", "hate", "O"), ("London", "london", "GPE")] * 3,
    "carol.txt": [("Carol", "carol", "PERSON"), ("loved", "love", "O"), ("Paris", "paris", "GPE")] * 3,
    "dora.txt": [("Dora", "dora", "PERSON"), ("hated", "hate", "O"), ("Berlin", "berlin", "GPE")] * 3,
}


class TestKwic:
    def test_literal_match_with_context(self) -> None:
        result = concordance(_frame(), "Paris", window=2)
        assert result.ok
        out = result.unwrap()
        assert len(out) == 6  # alice.txt + carol.txt, 3 sentences each
        first = out.iloc[0]
        assert first["Hit"] == "Paris"
        assert first["Left Context"] == "Alice loved"
        # context stays inside the sentence: sentence-final hit has empty right
        assert first["Right Context"] == ""

    def test_case_insensitive_default(self) -> None:
        result = concordance(_frame(), "paris")
        assert result.ok and len(result.unwrap()) == 6

    def test_case_sensitive_refuses(self) -> None:
        result = concordance(_frame(), "paris", case_sensitive=True)
        assert result.ok
        assert result.unwrap().empty

    def test_regex(self) -> None:
        result = concordance(_frame(), "^[A-Z][a-z]+$", regex=True)
        assert result.ok
        hits = set(result.unwrap()["Hit"])
        assert "Alice" in hits and "Paris" in hits

    def test_bad_regex_fails(self) -> None:
        result = concordance(_frame(), "([", regex=True)
        assert not result.ok
        assert result.diagnostics[0].code == "KWIC_BAD_REGEX"

    def test_max_hits_truncates_with_warning(self) -> None:
        result = concordance(_frame(), "Paris", max_hits=1)
        assert result.ok
        assert len(result.unwrap()) == 1
        assert any(d.code == "KWIC_TRUNCATED" for d in result.diagnostics)

    def test_window_bounds(self) -> None:
        assert not concordance(_frame(), "x", window=0).ok
        assert not concordance(_frame(), "x", window=51).ok

    def test_lemma_field(self) -> None:
        result = concordance(_frame(), "love", field=Col.LEMMA)
        assert result.ok
        assert len(result.unwrap()) == 6  # alice + carol, 3 sentences each

    def test_empty_query(self) -> None:
        assert not concordance(_frame(), "").ok


class TestKeyness:
    def test_pattern_splits_groups_and_scores(self) -> None:
        # alice/carol love Paris; bob/dora hate elsewhere
        result = keyness(_frame(), r"^(alice|carol)", top_n=0)
        assert result.ok
        out = result.unwrap()
        assert {"Word", "Freq Group A (pattern docs)", "Freq Group B (other docs)"} <= set(out.columns)
        love = out[out["Word"] == "love"]
        assert len(love) == 1
        assert love.iloc[0]["Log Ratio"] > 0  # over-represented in group A
        hate = out[out["Word"] == "hate"]
        assert hate.iloc[0]["Log Ratio"] < 0

    def test_top_n_bounds_output(self) -> None:
        result = keyness(_frame(), r"^(alice|carol)", top_n=3)
        assert result.ok
        assert len(result.unwrap()) == 3

    def test_one_group_refused(self) -> None:
        result = keyness(_frame(), r"^(alice|bob|carol|dora)")
        assert not result.ok
        assert result.diagnostics[0].code == "KEYNESS_ONE_GROUP"

    def test_bad_regex(self) -> None:
        assert not keyness(_frame(), "([").ok

    def test_empty_frame(self) -> None:
        assert not keyness(pd.DataFrame(columns=["Form", "Lemma", "Document ID", "Document"]), "x").ok

    def test_ranking_is_reproducible_across_runs(self) -> None:
        """Word counts travel through a set, whose order varies per process.

        Combined with a tie in G2 that made `--top-n` return a different set of
        words on every run. Both halves are pinned here: identical frame, and a
        ranking that only depends on the data.
        """
        docs = {
            "groupA_one.txt": [(w, w, "O") for w in ("alpha", "bravo", "charlie", "delta")] * 5,
            "other_one.txt": [(w, w, "O") for w in ("alpha", "bravo", "charlie", "delta")] * 1,
        }
        frame = _frame(docs)
        runs = [keyness(frame, r"^groupA", top_n=3).unwrap()["Word"].tolist() for _ in range(5)]
        assert all(run == runs[0] for run in runs), runs
        # A tie must resolve alphabetically rather than by dictionary order.
        assert runs[0] == sorted(runs[0])


def _coll_frame() -> pd.DataFrame:
    # two sentences: "red wine good red wine" / "red wine great"
    rows = []
    rec = 1
    for did, sid, toks in ((1, 1, ["red", "wine", "good", "red", "wine"]), (1, 2, ["red", "wine"])):
        for i, tok in enumerate(toks, start=1):
            rows.append(
                {
                    "ID": i,
                    "Form": tok,
                    "Lemma": tok,
                    "POS": "NN",
                    "NER": "O",
                    "Head": 0,
                    "DepRel": "root",
                    "Sentence ID": sid,
                    "Document ID": str(did),
                    "Document": "d.txt",
                    "Deps": "",
                    "Record ID": rec,
                    "Clause Tag": "",
                }
            )
            rec += 1
    return pd.DataFrame(rows)


class TestCollocationNetwork:
    def test_edges_sorted_and_canonical(self) -> None:
        result = collocation_network(_coll_frame(), min_count=2, min_pmi=0.0)
        assert result.ok
        out = result.unwrap()
        assert {"Word 1", "Word 2", "Weight", "PMI"} == set(out.columns)
        for _, row in out.iterrows():
            assert row["Word 1"] <= row["Word 2"]

    def test_min_pmi_filters(self) -> None:
        result = collocation_network(_coll_frame(), min_count=1, min_pmi=100.0)
        assert result.ok and result.unwrap().empty
        assert any(d.code == "NET_NO_EDGES" for d in result.diagnostics)

    def test_top_edges_truncates(self) -> None:
        result = collocation_network(_coll_frame(), min_count=1, top_edges=1)
        assert result.ok
        assert len(result.unwrap()) <= 1

    def test_bad_top_edges(self) -> None:
        assert not collocation_network(_coll_frame(), top_edges=0).ok


class TestMovementTracks:
    def _ner_frame(self) -> pd.DataFrame:
        rows = []
        rec = 1
        for did, sentences in {
            "1": [
                [("Alice", "PERSON"), ("flew", "O"), ("to", "O"), ("Paris", "GPE")],
                [("Bob", "PERSON"), ("stayed", "O"), ("in", "O"), ("Paris", "GPE")],
            ]
        }.items():
            for sid, sent in enumerate(sentences, start=1):
                for i, (form, ner) in enumerate(sent, start=1):
                    rec += 1
                    rows.append(
                        {
                            "ID": i,
                            "Form": form,
                            "Lemma": form.lower(),
                            "POS": "NN",
                            "NER": ner,
                            "Head": 0,
                            "DepRel": "root",
                            "Sentence ID": sid,
                            "Document ID": did,
                            "Document": "a.txt",
                            "Deps": "",
                            "Record ID": rec,
                            "Clause Tag": "",
                        }
                    )
        return pd.DataFrame(rows)

    def test_pairs_with_geocode(self) -> None:
        result = movement_tracks(self._ner_frame(), geocode=True)
        assert result.ok
        out = result.unwrap()
        assert {"Lat", "Lon"} <= set(out.columns)
        alice = out[out["Entity"] == "Alice"]
        assert len(alice) == 1 and alice.iloc[0]["Location"] == "Paris"
        assert abs(float(alice.iloc[0]["Lat"]) - 48.8566) < 0.01

    def test_no_pairs_is_empty_success(self) -> None:
        frame = self._ner_frame()
        frame["NER"] = "O"
        result = movement_tracks(frame)
        assert result.ok
        assert result.unwrap().empty
        assert any(d.code == "NER_NO_MOVEMENT" for d in result.diagnostics)

    def test_summary_counts(self) -> None:
        pairs = movement_tracks(self._ner_frame()).unwrap()
        summary = movement_summary(pairs)
        assert {"Entity", "Location", "Count", "First Sentence", "Documents"} == set(summary.columns)
        alice = summary[summary["Entity"] == "Alice"]
        assert alice.iloc[0]["Count"] == 1

    def test_unknown_place_keeps_row_without_coordinates(self) -> None:
        """An unresolved place keeps its row but gets NO coordinate.

        This used to write 0.0/0.0, which is Null Island in the Gulf of Guinea
        -- a real point. Every unresolved place plotted there.
        """
        rows = self._ner_frame()
        rows.loc[rows["NER"] == "GPE", "Form"] = "Atlantis"
        result = movement_tracks(rows, geocode=True)
        assert result.ok
        out = result.unwrap()
        assert len(out) == 2  # the rows survive
        assert out["Lat"].isna().all()
        assert out["Lon"].isna().all()
        assert not (out["Lat"] == 0.0).any()
        assert any(d.code == "NER_PLACES_UNGEOCODED" for d in result.diagnostics)

    def test_ungeocoded_places_never_reach_a_map(self) -> None:
        """The KML writers must drop the unresolved rows, not place them at 0,0."""
        rows = self._ner_frame()
        rows.loc[rows["NER"] == "GPE", "Form"] = "Atlantis"
        pairs = movement_tracks(rows, geocode=True).unwrap()
        pairs = pairs.rename(columns={"Location": "Place"})
        assert not kml(pairs).ok  # no valid coordinates at all
        assert not tour_kml(pairs).ok

    def test_missing_columns(self) -> None:
        assert not movement_tracks(pd.DataFrame({"A": [1]})).ok


class TestTourKml:
    def _geo(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"Place": ["Paris", "London", "Berlin"], "Lat": [48.86, 51.51, 52.52], "Lon": [2.35, -0.12, 13.40]}
        )

    def test_track_with_gx_namespace(self) -> None:
        result = tour_kml(self._geo(), title="Grand Tour")
        assert result.ok
        out = result.unwrap()
        assert "<gx:Track>" in out
        assert 'xmlns:gx="http://www.google.com/kml/ext/2.2"' in out
        assert out.count("<when>") == 3
        assert "Paris" in out and "Berlin" in out

    def test_grouped_one_track_per_group(self) -> None:
        frame = self._geo()
        frame["Person"] = ["A", "A", "B"]
        result = tour_kml(frame, group_col="Person")
        assert result.ok
        assert result.unwrap().count("<gx:Track>") == 2

    def test_skips_invalid_coords_with_warning(self) -> None:
        frame = self._geo()
        frame.loc[1, "Lat"] = 999.0
        result = tour_kml(frame)
        assert result.ok
        assert (
            "<when>" not in result.unwrap().split("<gx:Track>")[1].split("</gx:Track>")[0].replace("<when>", "<when>")
            or result.unwrap().count("<when>") == 2
        )
        assert any(d.code == "TOUR_SKIPPED_ROWS" for d in result.diagnostics)

    def test_all_invalid_fails(self) -> None:
        frame = pd.DataFrame({"Place": ["x"], "Lat": [999.0], "Lon": [0.0]})
        result = tour_kml(frame)
        assert not result.ok
        assert result.diagnostics[0].code == "TOUR_NO_COORDS"

    def test_empty_and_missing_columns(self) -> None:
        assert not tour_kml(pd.DataFrame(columns=["Lat", "Lon"])).ok
        assert not tour_kml(pd.DataFrame({"Place": ["x"], "Lat": [1.0]})).ok

    def test_bad_group_column(self) -> None:
        assert not tour_kml(self._geo(), group_col="nope").ok

    def test_named_columns_accept_a_movement_table(self) -> None:
        """`ner --movement --geocode` labels places "Location", not "Place".

        The tour writer takes the column names so the two capabilities compose
        without the user renaming a column between them.
        """
        frame = pd.DataFrame(
            {
                "Entity": ["Bob", "Bob"],
                "Location": ["London", "Berlin"],
                "Lat": [51.5074, 52.52],
                "Lon": [-0.1278, 13.405],
            }
        )
        result = tour_kml(frame, name_col="Location", group_col="Entity", title="Movement")
        assert result.ok
        out = result.unwrap()
        assert "<name>London</name>" in out and "<name>Berlin</name>" in out
        assert out.count("<gx:Track>") == 1  # one folder for Bob

    def test_out_of_range_coordinates_are_skipped_with_a_warning(self) -> None:
        frame = pd.DataFrame({"Place": ["Real", "Bogus"], "Lat": [48.86, 999.0], "Lon": [2.35, 2.35]})
        result = tour_kml(frame)
        assert result.ok
        assert "Bogus" not in result.unwrap()
        assert any(d.code == "TOUR_SKIPPED_ROWS" for d in result.diagnostics)

    def test_kml_and_tour_agree_on_what_is_a_coordinate(self) -> None:
        """`kml` used to accept an out-of-range latitude that `tour_kml` dropped."""
        frame = pd.DataFrame({"Place": ["Real", "Bogus"], "Lat": [48.86, 999.0], "Lon": [2.35, 2.35]})
        placemarks = kml(frame)
        assert placemarks.ok
        assert "Bogus" not in placemarks.unwrap()
        assert any(d.code == "KML_SKIPPED_ROWS" for d in placemarks.diagnostics)


def _bert_frame() -> pd.DataFrame:
    rows = []
    rec = 0
    for did, (theme, n) in {
        "1": ("space rocket launch", 8),
        "2": ("kitchen garden harvest", 8),
        "3": ("space rocket launch", 7),
        "4": ("kitchen garden harvest", 7),
    }.items():
        for sid in range(1, n + 1):
            for word in theme.split():
                rec += 1
                rows.append(
                    {
                        "ID": rec,
                        "Form": word,
                        "Lemma": word.lower(),
                        "POS": "NOUN",
                        "NER": "O",
                        "Head": 0,
                        "DepRel": "root",
                        "Sentence ID": sid,
                        "Document ID": did,
                        "Document": f"d{did}.txt",
                        "Deps": "",
                        "Record ID": rec,
                        "Clause Tag": "",
                    }
                )
    return pd.DataFrame(rows)


class TestBertTopics:
    def test_separates_themes(self) -> None:
        result = bert_topics(_bert_frame(), n_topics=2, backend=HashEmbeddingBackend())
        assert result.ok
        topics, documents = result.unwrap()
        words_by_topic = topics.groupby("Topic")["Word"].apply(set).to_dict()
        assert len(words_by_topic) == 2
        assert documents["Topic"].nunique() == 2
        # the two same-theme docs share a cluster
        by_doc = dict(zip(documents["Document"], documents["Topic"], strict=True))
        assert by_doc["d1.txt"] == by_doc["d3.txt"]
        assert by_doc["d2.txt"] == by_doc["d4.txt"]

    def test_deterministic_for_fixed_seed(self) -> None:
        a = bert_topics(_bert_frame(), n_topics=2, backend=HashEmbeddingBackend(), seed=7).unwrap()
        b = bert_topics(_bert_frame(), n_topics=2, backend=HashEmbeddingBackend(), seed=7).unwrap()
        assert a[0].equals(b[0]) and a[1].equals(b[1])

    def test_explicit_k(self) -> None:
        _, documents = bert_topics(_bert_frame(), n_topics=2, backend=HashEmbeddingBackend()).unwrap()
        assert set(documents["Topic"]) == {0, 1}

    def test_too_small(self) -> None:
        frame = _bert_frame()
        result = bert_topics(frame[frame["Document ID"] == "1"], n_topics=2, backend=HashEmbeddingBackend())
        assert not result.ok
        assert result.diagnostics[0].code == "BERTOPIC_TOO_SMALL"

    def test_k_needs_two(self) -> None:
        assert not bert_topics(_bert_frame(), n_topics=1, backend=HashEmbeddingBackend()).ok

    def test_empty_frame(self) -> None:
        cols = list(_bert_frame().columns)
        assert not bert_topics(pd.DataFrame(columns=cols), backend=HashEmbeddingBackend()).ok
