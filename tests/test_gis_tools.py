"""HW4 GIS tools: location extraction, Nominatim, pin maps, the SVO map.

Offline by construction: the Nominatim client is driven by injected request
functions (the tests/test_online_clients.py contract), the pin map renders
without touching the network, and the SVO handoff is pure DataFrame join
work. No test here reaches a socket or writes a file.
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest

from core.analysis.location_extract import extract_locations, locations_by_document
from core.analysis.svo_map import map_svo, svo_geo_summary
from core.gis.nominatim import NominatimGeocoder, _default_request
from core.gis.pins import pinmap_html
from tools import geocode as geocode_cli, gis_map as gis_map_cli, svo_map as svo_map_cli

CANONICAL = [
    "ID",
    "Form",
    "Lemma",
    "POS",
    "NER",
    "Head",
    "DepRel",
    "Sentence ID",
    "Document ID",
    "Document",
]

NOMINATIM_OK = [{"lat": "48.8566", "lon": "2.3522", "display_name": "Paris, France"}]


def _place_frame() -> pd.DataFrame:
    """Two documents with multi-word and single-word place spans.

    doc 1 says "New York" twice and "Paris" once; doc 2 says "Paris" and
    "Rome". Mentions land as New York=2, Paris=2, Rome=1.
    """
    tokens = [
        ("New", "B-GPE", 1, "1", "doc_a.txt"),
        ("York", "I-GPE", 1, "1", "doc_a.txt"),
        ("is", "O", 1, "1", "doc_a.txt"),
        ("nice", "O", 1, "1", "doc_a.txt"),
        (".", "O", 1, "1", "doc_a.txt"),
        ("New", "B-GPE", 2, "1", "doc_a.txt"),
        ("York", "I-GPE", 2, "1", "doc_a.txt"),
        ("and", "O", 2, "1", "doc_a.txt"),
        ("Paris", "S-GPE", 2, "1", "doc_a.txt"),
        (".", "O", 2, "1", "doc_a.txt"),
        ("Paris", "S-GPE", 1, "2", "doc_b.txt"),
        ("and", "O", 1, "2", "doc_b.txt"),
        ("Rome", "B-LOC", 1, "2", "doc_b.txt"),
        (".", "O", 1, "2", "doc_b.txt"),
    ]
    rows = [
        {
            "ID": index,
            "Form": form,
            "Lemma": form.lower(),
            "POS": "NNP" if ner != "O" else "NN",
            "NER": ner,
            "Head": 0,
            "DepRel": "root",
            "Sentence ID": sent,
            "Document ID": doc_id,
            "Document": doc,
        }
        for index, (form, ner, sent, doc_id, doc) in enumerate(tokens, start=1)
    ]
    return pd.DataFrame(rows, columns=CANONICAL)


class TestExtractLocations:
    def test_columns_and_multiword_spans(self) -> None:
        result = extract_locations(_place_frame())
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert list(frame.columns) == ["Place", "Mentions", "Documents", "First document"]
        assert frame["Place"].tolist() == ["New York", "Paris", "Rome"]
        row = frame.iloc[0]
        assert row["Place"] == "New York"
        assert int(row["Mentions"]) == 2
        assert int(row["Documents"]) == 1
        assert row["First document"] == "doc_a.txt"

    def test_ties_break_by_first_appearance(self) -> None:
        # New York and Paris both have 2 mentions; New York is named first.
        frame = extract_locations(_place_frame()).unwrap()
        assert frame["Place"].tolist() == ["New York", "Paris", "Rome"]
        paris = frame.iloc[1]
        assert int(paris["Documents"]) == 2
        assert paris["First document"] == "doc_a.txt"

    def test_min_count_drops_rare_places_with_an_info(self) -> None:
        result = extract_locations(_place_frame(), min_count=2)
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert frame["Place"].tolist() == ["New York", "Paris"]
        codes = [(d.code, d.severity.value) for d in result.diagnostics]
        assert ("LOCS_MIN_COUNT_DROPPED", "INFO") in codes
        dropped = next(d for d in result.diagnostics if d.code == "LOCS_MIN_COUNT_DROPPED")
        assert dropped.context["count"] == 1

    def test_empty_frame_is_empty_success(self) -> None:
        result = extract_locations(_place_frame().iloc[0:0])
        assert result.ok
        frame = result.unwrap()
        assert frame.empty
        assert list(frame.columns) == ["Place", "Mentions", "Documents", "First document"]

    def test_missing_columns_fail(self) -> None:
        result = extract_locations(_place_frame().drop(columns=["NER"]))
        assert not result.ok
        assert result.diagnostics[0].code == "CONLL_MISSING_COLUMN"

    def test_non_location_entities_are_not_places(self) -> None:
        frame = _place_frame()
        frame.loc[frame["NER"] == "S-GPE", "NER"] = "S-PERSON"
        result = extract_locations(frame)
        assert result.ok
        assert result.unwrap()["Place"].tolist() == ["New York", "Rome"]


class TestLocationsByDocument:
    def test_columns_and_per_document_counts(self) -> None:
        result = locations_by_document(_place_frame())
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert list(frame.columns) == ["Document", "Document ID", "Place", "Mentions"]
        assert len(frame) == 4
        first = frame.iloc[0]
        assert first["Document"] == "doc_a.txt"
        assert first["Place"] == "New York"
        assert int(first["Mentions"]) == 2

    def test_min_count_matches_the_aggregate_view(self) -> None:
        frame = locations_by_document(_place_frame(), min_count=2).unwrap()
        assert set(frame["Place"]) == {"New York", "Paris"}
        assert "Rome" not in set(frame["Place"])

    def test_empty_frame_is_empty_success(self) -> None:
        result = locations_by_document(_place_frame().iloc[0:0])
        assert result.ok
        assert list(result.unwrap().columns) == ["Document", "Document ID", "Place", "Mentions"]


class TestNominatimGeocoder:
    def test_parses_coordinates_and_sends_the_suite_user_agent(self) -> None:
        seen: list[tuple[str, dict[str, str]]] = []

        def fake(url: str, headers: object, body: object, timeout: float) -> object:
            seen.append((url, dict(headers)))  # type: ignore[arg-type]
            return list(NOMINATIM_OK)

        result = NominatimGeocoder(request_fn=fake).geocode("Paris")
        assert result.ok, result.diagnostics
        assert result.unwrap() == (48.8566, 2.3522)
        url, headers = seen[0]
        assert "q=Paris" in url
        assert "format=json" in url
        assert "limit=1" in url
        assert "NLP-Suite" in headers.get("User-Agent", "")

    def test_cache_avoids_second_request(self) -> None:
        calls: list[str] = []

        def fake(url: str, headers: object, body: object, timeout: float) -> object:
            calls.append(url)
            return list(NOMINATIM_OK)

        client = NominatimGeocoder(request_fn=fake)
        assert client.geocode("Paris").ok
        assert client.geocode("paris").ok
        assert len(calls) == 1
        assert client.cache_size == 1

    def test_empty_results_is_unknown(self) -> None:
        client = NominatimGeocoder(request_fn=lambda *a: [])
        result = client.geocode("Xyzzy Nowhere")
        assert not result.ok
        assert result.diagnostics[0].code == "GEOCODE_UNKNOWN"

    def test_transport_failure_is_unavailable_with_the_offline_fix(self) -> None:
        def boom(url: str, headers: object, body: object, timeout: float) -> object:
            raise OSError("no route")

        result = NominatimGeocoder(request_fn=boom).geocode("Paris")
        assert not result.ok
        diag = result.diagnostics[0]
        assert diag.code == "GEOCODE_UNAVAILABLE"
        assert "--provider offline" in diag.context.get("fix", "")

    def test_bad_response_maps(self) -> None:
        client = NominatimGeocoder(request_fn=lambda *a: {"not": "a list"})
        result = client.geocode("Paris")
        assert not result.ok
        assert result.diagnostics[0].code == "GEOCODE_BAD_RESPONSE"

    def test_empty_place_rejected(self) -> None:
        result = NominatimGeocoder(request_fn=lambda *a: list(NOMINATIM_OK)).geocode("   ")
        assert not result.ok
        assert result.diagnostics[0].code == "GEOCODE_EMPTY"

    def test_geocode_many_matches_the_offline_contract(self) -> None:
        def fake(url: str, headers: object, body: object, timeout: float) -> object:
            return list(NOMINATIM_OK) if "q=Paris" in url else []

        result = NominatimGeocoder(request_fn=fake).geocode_many(["Paris", "Atlantis"])
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert list(frame.columns) == ["Place", "Lat", "Lon", "Status"]
        assert frame["Status"].tolist() == ["OK", "UNKNOWN"]
        assert [d.code for d in result.diagnostics] == ["GEOCODE_UNKNOWN"]
        assert result.diagnostics[0].severity.value == "INFO"

    def test_geocode_many_marks_network_failures_unavailable(self) -> None:
        def boom(url: str, headers: object, body: object, timeout: float) -> object:
            raise OSError("no route")

        result = NominatimGeocoder(request_fn=boom).geocode_many(["Paris"])
        assert result.value is not None
        assert result.has_errors
        assert result.unwrap()["Status"].tolist() == ["UNAVAILABLE"]

    def test_geocode_many_rejects_an_empty_list(self) -> None:
        result = NominatimGeocoder(request_fn=lambda *a: list(NOMINATIM_OK)).geocode_many([])
        assert not result.ok
        assert result.diagnostics[0].code == "GEOCODE_EMPTY_LIST"

    def test_default_request_refuses_any_other_host(self) -> None:
        # Raised before any socket is opened: the allow-list is the first check.
        with pytest.raises(ValueError, match="refusing non-Nominatim host"):
            _default_request("https://evil.example/search?q=Paris", {}, None, 1.0)


class TestPinmapHtml:
    def test_empty_frame_is_a_failure(self) -> None:
        result = pinmap_html(pd.DataFrame(columns=["Lat", "Lon", "Place"]))
        assert not result.ok
        assert result.diagnostics[0].code == "PINS_EMPTY"

    def test_missing_columns_fail(self) -> None:
        result = pinmap_html(pd.DataFrame({"Place": ["Paris"]}))
        assert not result.ok
        assert result.diagnostics[0].code == "PINS_MISSING_COLUMN"

    def test_renders_the_place_names(self) -> None:
        frame = pd.DataFrame({"Lat": [48.8566], "Lon": [2.3522], "Place": ["Paris"]})
        result = pinmap_html(frame, title="HW4")
        assert result.ok, result.diagnostics
        assert "Paris" in result.unwrap()

    def test_out_of_range_rows_are_skipped_with_a_warning_count(self) -> None:
        frame = pd.DataFrame({"Lat": [48.8566, 999.0], "Lon": [2.3522, 0.0], "Place": ["Paris", "Nowhere"]})
        result = pinmap_html(frame)
        assert result.ok, result.diagnostics
        warnings = [d for d in result.diagnostics if d.code == "PINS_SKIPPED_ROWS"]
        assert warnings and warnings[0].context["count"] == 1
        assert "Nowhere" not in result.unwrap()

    def test_weight_drives_pin_radius_in_the_svg_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
        frame = pd.DataFrame(
            {"Lat": [48.8566, 51.5074], "Lon": [2.3522, -0.1278], "Place": ["Paris", "London"], "Rows": [1, 10]}
        )
        result = pinmap_html(frame, weight_col="Rows")
        assert result.ok, result.diagnostics
        html = result.unwrap()
        assert any(d.code == "PINS_PLOTLY_UNAVAILABLE" for d in result.diagnostics)
        assert html.count("<circle") == 2
        assert "r='10.0'" in html
        assert "r='3.7'" in html

    def test_all_rows_invalid_is_a_failure(self) -> None:
        frame = pd.DataFrame({"Lat": [float("nan")], "Lon": [0.0], "Place": ["Nowhere"]})
        result = pinmap_html(frame)
        assert not result.ok
        assert result.diagnostics[0].code == "PINS_NO_COORDS"


def _svo_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Subject": "he",
                "Verb": "visited",
                "Object": "museum",
                "Sentence ID": 1,
                "Document ID": "1",
                "Document": "a.txt",
            },
            {
                "Subject": "she",
                "Verb": "left",
                "Object": "city",
                "Sentence ID": 2,
                "Document ID": "1",
                "Document": "a.txt",
            },
            {
                "Subject": "they",
                "Verb": "founded",
                "Object": "bank",
                "Sentence ID": 1,
                "Document ID": "2",
                "Document": "b.txt",
            },
            {
                "Subject": "it",
                "Verb": "rained",
                "Object": "hard",
                "Sentence ID": 1,
                "Document ID": "3",
                "Document": "c.txt",
            },
        ],
        columns=["Subject", "Verb", "Object", "Sentence ID", "Document ID", "Document"],
    )


def _locations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Document": "a.txt", "Document ID": "1", "Place": "Paris", "Mentions": 2},
            {"Document": "a.txt", "Document ID": "1", "Place": "London", "Mentions": 1},
            {"Document": "b.txt", "Document ID": "2", "Place": "Berlin", "Mentions": 1},
            {"Document": "c.txt", "Document ID": "3", "Place": "Atlantis", "Mentions": 1},
        ],
        columns=["Document", "Document ID", "Place", "Mentions"],
    )


def _geocoded() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Place": "Paris", "Lat": 48.8566, "Lon": 2.3522, "Status": "OK"},
            {"Place": "London", "Lat": 0.0, "Lon": 0.0, "Status": "UNKNOWN"},
            {"Place": "Berlin", "Lat": 52.5200, "Lon": 13.4050, "Status": "OK"},
            {"Place": "Atlantis", "Lat": 0.0, "Lon": 0.0, "Status": "UNKNOWN"},
        ],
        columns=["Place", "Lat", "Lon", "Status"],
    )


class TestMapSvo:
    def test_columns_and_document_join(self) -> None:
        result = map_svo(_svo_rows(), _locations(), geocoded=_geocoded())
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert list(frame.columns) == [
            "Subject",
            "Verb",
            "Object",
            "Place",
            "Lat",
            "Lon",
            "Document",
            "Document ID",
            "Sentence ID",
        ]
        assert len(frame) == 3
        assert frame["Place"].tolist() == ["Paris", "Paris", "Berlin"]
        assert frame["Verb"].tolist() == ["visited", "left", "founded"]

    def test_unmatched_triples_are_dropped_with_a_warning_count(self) -> None:
        result = map_svo(_svo_rows(), _locations(), geocoded=_geocoded())
        warnings = [d for d in result.diagnostics if d.code == "SVO_MAP_UNMATCHED"]
        assert warnings and warnings[0].context["count"] == 1
        assert "rained" not in set(result.unwrap()["Verb"])

    def test_one_row_per_svo_and_place_pair(self) -> None:
        # A document about two geocoded places maps each triple twice: the SVO
        # map is a map of events, not of documents.
        geocoded = pd.DataFrame(
            [
                {"Place": "Paris", "Lat": 48.8566, "Lon": 2.3522, "Status": "OK"},
                {"Place": "London", "Lat": 51.5074, "Lon": -0.1278, "Status": "OK"},
                {"Place": "Berlin", "Lat": 52.5200, "Lon": 13.4050, "Status": "OK"},
                {"Place": "Atlantis", "Lat": 0.0, "Lon": 0.0, "Status": "UNKNOWN"},
            ],
            columns=["Place", "Lat", "Lon", "Status"],
        )
        result = map_svo(_svo_rows(), _locations(), geocoded=geocoded)
        frame = result.unwrap()
        assert len(frame) == 5
        pairs = sorted(zip(frame["Verb"], frame["Place"], strict=True))
        assert pairs == [
            ("founded", "Berlin"),
            ("left", "London"),
            ("left", "Paris"),
            ("visited", "London"),
            ("visited", "Paris"),
        ]

    def test_unknown_places_never_claim_null_island(self) -> None:
        frame = map_svo(_svo_rows(), _locations(), geocoded=_geocoded()).unwrap()
        assert set(zip(frame["Lat"], frame["Lon"], strict=True)) == {(48.8566, 2.3522), (52.5200, 13.4050)}

    def test_missing_columns_fail(self) -> None:
        result = map_svo(_svo_rows().drop(columns=["Verb"]), _locations(), geocoded=_geocoded())
        assert not result.ok
        assert result.diagnostics[0].code == "SVO_MAP_MISSING_COLUMN"

    def test_empty_svo_rows_is_empty_success(self) -> None:
        result = map_svo(_svo_rows().iloc[0:0], _locations(), geocoded=_geocoded())
        assert result.ok
        assert result.unwrap().empty


class TestSvoGeoSummary:
    def test_columns_counts_and_rank(self) -> None:
        mapped = map_svo(_svo_rows(), _locations(), geocoded=_geocoded()).unwrap()
        result = svo_geo_summary(mapped)
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert list(frame.columns) == ["Place", "Lat", "Lon", "SVO rows", "Documents"]
        assert frame["Place"].tolist() == ["Paris", "Berlin"]
        paris = frame.iloc[0]
        assert int(paris["SVO rows"]) == 2
        assert int(paris["Documents"]) == 1

    def test_empty_summary_is_empty_success(self) -> None:
        result = svo_geo_summary(pd.DataFrame(columns=["Place", "Lat", "Lon", "Document ID"]))
        assert result.ok
        assert list(result.unwrap().columns) == ["Place", "Lat", "Lon", "SVO rows", "Documents"]

    def test_missing_columns_fail(self) -> None:
        result = svo_geo_summary(pd.DataFrame({"Place": ["Paris"]}))
        assert not result.ok
        assert result.diagnostics[0].code == "SVO_MAP_MISSING_COLUMN"


class TestCliArguments:
    def test_geocode_defaults(self) -> None:
        args = geocode_cli._parse_args(["corpus", "out"])
        assert args.provider == "offline"
        assert args.limit == 200
        assert args.min_count == 1

    def test_gis_map_defaults(self) -> None:
        args = gis_map_cli._parse_args(["in.csv", "out"])
        assert args.lat_col == "Lat"
        assert args.lon_col == "Lon"
        assert args.name_col == "Place"
        assert args.weight_col == ""
        assert args.group_col == ""
        assert args.title == ""

    def test_svo_map_defaults(self) -> None:
        args = svo_map_cli._parse_args(["corpus", "out"])
        assert args.provider == "offline"
        assert args.limit == 200

    def test_google_without_a_key_fails_with_the_nominatim_fix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_MAPS_KEY", raising=False)
        result = geocode_cli.geocode_places("google", ["Paris"])
        assert not result.ok
        diag = result.diagnostics[0]
        assert diag.code == "GEOCODE_NO_KEY"
        assert "--provider nominatim" in diag.context.get("fix", "")

    def test_empty_place_list_is_an_empty_table(self) -> None:
        result = geocode_cli.geocode_places("offline", [])
        assert result.ok
        assert list(result.unwrap().columns) == ["Place", "Lat", "Lon", "Status"]
