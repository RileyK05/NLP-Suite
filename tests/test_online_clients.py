"""FR-6.7/FR-6.8 — online geocoder + DBpedia client contract tests.

Offline: injected request functions (protocol parsing, cache behavior,
bounds, key handling, failure codes). No test touches the network.
"""

from __future__ import annotations

import pytest

from core.gis.online import GoogleGeocoder, api_key_from_env
from core.kg.dbpedia import SpotlightClient

GOOGLE_OK = {
    "status": "OK",
    "results": [{"geometry": {"location": {"lat": 48.8566, "lng": 2.3522}}}],
}
SPOTLIGHT_OK = {
    "Resources": [
        {"@surfaceForm": "Paris", "@URI": "http://dbpedia.org/resource/Paris", "@types": "DBpedia:City"},
    ]
}


class TestGoogleGeocoder:
    def test_parses_coordinates(self) -> None:
        calls: list[str] = []

        def fake(url: str, headers: object, body: object, timeout: float) -> object:
            calls.append(url)
            return dict(GOOGLE_OK)

        res = GoogleGeocoder("key", request_fn=fake).geocode("Paris")
        assert res.ok, res.diagnostics
        assert res.unwrap() == (48.8566, 2.3522)
        assert "address=Paris" in calls[0]

    def test_cache_avoids_second_request(self) -> None:
        calls: list[str] = []

        def fake(url: str, headers: object, body: object, timeout: float) -> object:
            calls.append(url)
            return dict(GOOGLE_OK)

        client = GoogleGeocoder("key", request_fn=fake)
        assert client.geocode("Paris").ok
        assert client.geocode("paris").ok  # case-insensitive hit
        assert len(calls) == 1
        assert client.cache_size == 1

    def test_zero_results_is_unknown(self) -> None:
        client = GoogleGeocoder("key", request_fn=lambda *a: {"status": "ZERO_RESULTS", "results": []})
        res = client.geocode("Xyzzy Nowhere")
        assert not res.ok
        assert res.diagnostics[0].code == "GEOCODE_UNKNOWN"

    def test_api_error_maps(self) -> None:
        client = GoogleGeocoder("key", request_fn=lambda *a: {"status": "REQUEST_DENIED", "error_message": "bad key"})
        res = client.geocode("Paris")
        assert not res.ok
        assert res.diagnostics[0].code == "GEOCODE_FAILED"

    def test_transport_failure_maps(self) -> None:
        def boom(url: str, headers: object, body: object, timeout: float) -> object:
            raise OSError("no route")

        res = GoogleGeocoder("key", request_fn=boom).geocode("Paris")
        assert not res.ok
        assert res.diagnostics[0].code == "GEOCODE_FAILED"

    def test_empty_place_rejected(self) -> None:
        res = GoogleGeocoder("key", request_fn=lambda *a: dict(GOOGLE_OK)).geocode("   ")
        assert not res.ok
        assert res.diagnostics[0].code == "GEOCODE_EMPTY"

    def test_blank_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="API key"):
            GoogleGeocoder("  ")

    def test_limit_bounds_many(self) -> None:
        client = GoogleGeocoder("key", request_fn=lambda *a: dict(GOOGLE_OK))
        res = client.geocode_many(["Paris", "Rome"], limit=1)
        assert not res.ok
        assert res.diagnostics[0].code == "GEOCODE_OVER_LIMIT"
        ok = client.geocode_many(["Paris", "Rome"], limit=2)
        assert ok.ok and len(ok.unwrap()) == 2

    def test_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_MAPS_KEY", "abc")
        assert api_key_from_env() == "abc"
        monkeypatch.delenv("GOOGLE_MAPS_KEY")
        assert api_key_from_env() is None


class TestSpotlight:
    def test_parses_entities(self) -> None:
        client = SpotlightClient(request_fn=lambda *a: dict(SPOTLIGHT_OK))
        res = client.annotate("Paris is lovely.")
        assert res.ok, res.diagnostics
        assert res.unwrap() == [
            {"surface": "Paris", "uri": "http://dbpedia.org/resource/Paris", "types": "DBpedia:City"}
        ]

    def test_empty_text_rejected(self) -> None:
        res = SpotlightClient(request_fn=lambda *a: dict(SPOTLIGHT_OK)).annotate("  ")
        assert not res.ok
        assert res.diagnostics[0].code == "SPOTLIGHT_EMPTY"

    def test_transport_failure_maps(self) -> None:
        def boom(url: str, headers: object, body: object, timeout: float) -> object:
            raise OSError("no route")

        res = SpotlightClient(request_fn=boom).annotate("Paris")
        assert not res.ok
        assert res.diagnostics[0].code == "SPOTLIGHT_FAILED"

    def test_bad_response_maps(self) -> None:
        res = SpotlightClient(request_fn=lambda *a: [1, 2]).annotate("Paris")
        assert not res.ok
        assert res.diagnostics[0].code == "SPOTLIGHT_BAD_RESPONSE"

    def test_bad_confidence_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            SpotlightClient(confidence=2.0)
