"""Online geocoding (FR-6.7) — Google Geocoding API client.

Cached (in-memory per client) and bounded (per-request timeout, explicit
result cap): the legacy GUI fired unbounded requests. The API key comes
from ``GOOGLE_MAPS_KEY`` — never from argv, logs, or envelopes. HTTP
goes through an injectable request function so tests never touch the
network; failures map to diagnostics, never tracebacks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import os
import urllib.parse
import urllib.request

from core.result import Diagnostic, Result

__all__ = ["GoogleGeocoder", "api_key_from_env"]

_KEY_ENV_VAR = "GOOGLE_MAPS_KEY"
_ENDPOINT = "https://maps.googleapis.com/maps/api/geocode/json"
_TIMEOUT = 15.0

#: googleapis hosts only.
_ALLOWED_HOSTS = frozenset({"maps.googleapis.com"})

RequestFn = Callable[[str, Mapping[str, str], bytes | None, float], object]


def api_key_from_env() -> str | None:
    """API key from the environment, or None when unset/blank."""
    key = os.environ.get(_KEY_ENV_VAR, "").strip()
    return key or None


def _default_request(url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> object:
    host = urllib.parse.urlsplit(url).hostname or ""
    if host not in _ALLOWED_HOSTS:
        raise ValueError(f"refusing non-Google host {host!r}")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="GET")  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _parse_response(payload: object, cleaned: str) -> Result[tuple[float, float]]:
    """Map one Google Geocoding payload to coordinates."""
    if not isinstance(payload, dict):
        return Result.failure(Diagnostic.error("GEOCODE_BAD_RESPONSE", "geocode endpoint returned non-JSON"))
    status = str(payload.get("status", ""))
    if status == "ZERO_RESULTS":
        return Result.failure(Diagnostic.error("GEOCODE_UNKNOWN", f"{cleaned!r} not found by Google", place=cleaned))
    if status != "OK":
        message = str(payload.get("error_message", status))
        return Result.failure(Diagnostic.error("GEOCODE_FAILED", f"Google geocode error: {message}"))
    try:
        results = payload["results"]
        if not isinstance(results, list) or not results or not isinstance(results[0], dict):
            raise ValueError("no results[0] object")
        location = results[0]["geometry"]["location"]
        return Result.success((float(location["lat"]), float(location["lng"])))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return Result.failure(Diagnostic.error("GEOCODE_BAD_RESPONSE", f"cannot read geocode geometry: {exc}"))


class GoogleGeocoder:
    """Bounded, cached Google Geocoding client."""

    def __init__(
        self,
        api_key: str,
        *,
        request_fn: RequestFn | None = None,
        timeout: float = _TIMEOUT,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("GoogleGeocoder needs a non-empty API key (set GOOGLE_MAPS_KEY)")
        self._key = api_key
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
        query = urllib.parse.urlencode({"address": cleaned, "key": self._key})
        try:
            payload = self._request(f"{_ENDPOINT}?{query}", {}, None, self._timeout)
        except Exception as exc:
            return Result.failure(Diagnostic.error("GEOCODE_FAILED", f"online geocode request failed: {exc}"))
        parsed = _parse_response(payload, cleaned)
        if parsed.value is None:
            return parsed
        self._cache[cleaned.lower()] = parsed.unwrap()
        return parsed

    def geocode_many(self, places: Sequence[str], *, limit: int = 100) -> Result[list[tuple[str, tuple[float, float]]]]:
        """Geocode up to ``limit`` places (the bound); cache hits are free."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            return Result.failure(Diagnostic.error("GEOCODE_BAD_LIMIT", f"limit must be a positive int, got {limit!r}"))
        if len(places) > limit:
            return Result.failure(
                Diagnostic.error("GEOCODE_OVER_LIMIT", f"{len(places)} places exceed the limit of {limit}", limit=limit)
            )
        resolved: list[tuple[str, tuple[float, float]]] = []
        for place in places:
            result = self.geocode(place)
            if result.value is None:
                return Result.failure(*result.diagnostics)
            resolved.append((str(place).strip(), result.unwrap()))
        return Result.success(resolved)
