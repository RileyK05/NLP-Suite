"""Nominatim geocoding (FR-6.9) — the OpenStreetMap search client.

Shaped exactly like :mod:`core.gis.online`: a small cached client over one
allow-listed host, an injectable request function so tests never touch the
network, and every failure a ``Diagnostic`` rather than a traceback.

Nominatim needs no API key, which is why there is deliberately no
``api_key_from_env`` here and why it is the provider to prefer for coursework:
a run must not fail on a missing billing account, so ``--provider nominatim``
needs only the network. The usage policy asks for an identifying User-Agent;
it is constructed here in code (never from argv) so no copied command line can
supplant the client's identity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import urllib.parse
import urllib.request

import pandas as pd

from core.result import Diagnostic, Result

__all__ = ["NominatimGeocoder"]

_ENDPOINT = "https://nominatim.openstreetmap.org/search"
_TIMEOUT = 15.0

#: nominatim.openstreetmap.org only.
_ALLOWED_HOSTS = frozenset({"nominatim.openstreetmap.org"})

# Nominatim's usage policy requires an identifying User-Agent. Built here, not
# from argv: the policy identity must not be caller-suppliable.
_USER_AGENT = "NLP-Suite-NG/1.0 (NLP Suite coursework geocoder)"

RequestFn = Callable[[str, Mapping[str, str], bytes | None, float], object]

_GEOCODE_COLUMNS = ["Place", "Lat", "Lon", "Status"]


def _default_request(url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> object:
    host = urllib.parse.urlsplit(url).hostname or ""
    if host not in _ALLOWED_HOSTS:
        raise ValueError(f"refusing non-Nominatim host {host!r}")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="GET")  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _parse_response(payload: object, cleaned: str) -> Result[tuple[float, float]]:
    """Map one Nominatim search payload to coordinates (first hit)."""
    if not isinstance(payload, list):
        return Result.failure(Diagnostic.error("GEOCODE_BAD_RESPONSE", "geocode endpoint returned non-JSON"))
    if not payload:
        return Result.failure(Diagnostic.error("GEOCODE_UNKNOWN", f"{cleaned!r} not found by Nominatim", place=cleaned))
    first = payload[0]
    if not isinstance(first, dict):
        return Result.failure(
            Diagnostic.error("GEOCODE_BAD_RESPONSE", "cannot read geocode geometry: first result is not an object")
        )
    try:
        return Result.success((float(first["lat"]), float(first["lon"])))
    except (KeyError, TypeError, ValueError) as exc:
        return Result.failure(Diagnostic.error("GEOCODE_BAD_RESPONSE", f"cannot read geocode geometry: {exc}"))


class NominatimGeocoder:
    """Bounded, cached Nominatim client (no API key, one allow-listed host)."""

    def __init__(
        self,
        *,
        request_fn: RequestFn | None = None,
        timeout: float = _TIMEOUT,
    ) -> None:
        self._request = request_fn or _default_request
        self._timeout = timeout
        self._cache: dict[str, tuple[float, float]] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def geocode(self, place: str) -> Result[tuple[float, float]]:
        """Lat/lon for one place (cached)."""
        cleaned = (place or "").strip()
        if not cleaned:
            return Result.failure(Diagnostic.error("GEOCODE_EMPTY", "place must be non-empty"))
        hit = self._cache.get(cleaned.lower())
        if hit is not None:
            return Result.success(hit)
        query = urllib.parse.urlencode({"format": "json", "limit": 1, "q": cleaned})
        try:
            payload = self._request(f"{_ENDPOINT}?{query}", {"User-Agent": _USER_AGENT}, None, self._timeout)
        except Exception as exc:
            return Result.failure(
                Diagnostic.error(
                    "GEOCODE_UNAVAILABLE",
                    f"online geocode request failed: {exc}",
                    fix="check the network connection or use --provider offline",
                )
            )
        parsed = _parse_response(payload, cleaned)
        if parsed.value is None:
            return parsed
        self._cache[cleaned.lower()] = parsed.unwrap()
        return parsed

    def geocode_many(self, places: Sequence[str]) -> Result[pd.DataFrame]:
        """Batch geocode to the ``core.gis.geocode.geocode_many`` contract.

        Callers can swap providers on this shape. An unknown place is an INFO
        row (Status "UNKNOWN"); a network failure is an ERROR diagnostic over
        a row marked "UNAVAILABLE" — a batch of names is expected to contain
        unresolvable ones, but a dead connection is not a row outcome and must
        never read as "unknown place".
        """
        if not places:
            return Result.failure(Diagnostic.error("GEOCODE_EMPTY_LIST", "no places to geocode"))
        rows: list[dict[str, object]] = []
        diags: list[Diagnostic] = []
        for place in places:
            resolved = self.geocode(place)
            if resolved.ok:
                lat, lon = resolved.unwrap()
                rows.append({"Place": place, "Lat": lat, "Lon": lon, "Status": "OK"})
                continue
            rows.append({"Place": place, "Lat": 0.0, "Lon": 0.0, "Status": "UNKNOWN"})
            diag = (
                resolved.diagnostics[0]
                if resolved.diagnostics
                else Diagnostic.error("GEOCODE_UNAVAILABLE", "geocode failed")
            )
            if diag.code == "GEOCODE_UNKNOWN":
                diags.append(Diagnostic.info("GEOCODE_UNKNOWN", diag.message, place=place))
            else:
                rows[-1]["Status"] = "UNAVAILABLE"
                diags.append(diag)
        return Result.success(pd.DataFrame(rows, columns=_GEOCODE_COLUMNS), *diags)
