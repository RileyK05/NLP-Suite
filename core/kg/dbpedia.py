"""DBpedia client (FR-6.8) — live entity annotation via Spotlight.

The offline stub in ``core.analysis.knowledge_graph`` stays the default;
this client is the network route: DBpedia Spotlight over HTTPS with an
injectable request function (tests never touch the network), explicit
confidence bound, and a timeout. Failures map to diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import urllib.parse
import urllib.request

from core.result import Diagnostic, Result

__all__ = ["SpotlightClient"]

_ENDPOINT = "https://api.dbpedia-spotlight.org/en/annotate"
_TIMEOUT = 20.0

_ALLOWED_HOSTS = frozenset({"api.dbpedia-spotlight.org"})

RequestFn = Callable[[str, Mapping[str, str], bytes | None, float], object]


def _default_request(url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> object:
    host = urllib.parse.urlsplit(url).hostname or ""
    if host not in _ALLOWED_HOSTS:
        raise ValueError(f"refusing non-Spotlight host {host!r}")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


class SpotlightClient:
    """DBpedia Spotlight annotation client."""

    def __init__(
        self,
        *,
        endpoint: str = _ENDPOINT,
        confidence: float = 0.5,
        request_fn: RequestFn | None = None,
        timeout: float = _TIMEOUT,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence!r}")
        self._endpoint = endpoint
        self._confidence = confidence
        self._request = request_fn or _default_request
        self._timeout = timeout

    def annotate(self, text: str) -> Result[list[dict[str, object]]]:
        """Entities for *text*: [{surface, uri, types}] (possibly empty)."""
        if not (text or "").strip():
            return Result.failure(Diagnostic.error("SPOTLIGHT_EMPTY", "text must be non-empty"))
        body = urllib.parse.urlencode({"text": text, "confidence": str(self._confidence)}).encode("utf-8")
        try:
            payload = self._request(self._endpoint, {"Accept": "application/json"}, body, self._timeout)
        except Exception as exc:
            return Result.failure(Diagnostic.error("SPOTLIGHT_FAILED", f"Spotlight request failed: {exc}"))
        if not isinstance(payload, dict):
            return Result.failure(Diagnostic.error("SPOTLIGHT_BAD_RESPONSE", "Spotlight returned non-JSON"))
        resources = payload.get("Resources", [])
        if not isinstance(resources, list):
            return Result.failure(Diagnostic.error("SPOTLIGHT_BAD_RESPONSE", "Spotlight returned no Resources list"))
        entities: list[dict[str, object]] = []
        for resource in resources:
            if not isinstance(resource, dict):
                continue
            entities.append(
                {
                    "surface": str(resource.get("@surfaceForm", "")),
                    "uri": str(resource.get("@URI", "")),
                    "types": str(resource.get("@types", "")),
                }
            )
        return Result.success(entities)
