"""NLPConfig — one frozen, validated configuration object.

ARCHITECTURE.md invariants enforced here:

* **R6** — no global mutable state. Config is constructed once and passed
  explicitly; there is nothing else for a tool to read, so "configuration
  existed; obedience was optional" is no longer expressible.

The legacy suite's language support check was ``if True``. Here an unsupported
(backend, language) pair raises at construction — loudly, at config time,
before any model is touched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

__all__ = [
    "BACKEND_LANGUAGES",
    "ConfigError",
    "NLPConfig",
    "ParserBackend",
    "languages_for",
    "supports_language",
]

ParserBackend = Literal["stanza", "spacy", "corenlp"]


class ConfigError(ValueError):
    """Raised when an NLPConfig is constructed with an impossible combination.

    A config error is a programming error, not a runtime outcome, so it raises
    rather than returning a ``Result``.
    """


# Languages we have confirmed models for, per backend. This is deliberately a
# verified set rather than each parser's theoretical maximum: claiming support
# we cannot back is exactly how the legacy suite ended up with `if True`.
# Stanza has a full integration probe only for English; add another language
# here only after its model/resources are exercised in the same gate.
BACKEND_LANGUAGES: Final[Mapping[str, frozenset[str]]] = {
    "stanza": frozenset({"en"}),
    "spacy": frozenset(
        {
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "nl",
            "ru",
            "zh",
            "ja",
            "pl",
            "ro",
            "ca",
            "hr",
            "da",
            "fi",
            "el",
            "uk",
        }
    ),
    # CoreNLP ships full pipelines for a small set; everything else degrades.
    "corenlp": frozenset({"en", "es", "fr", "de", "zh", "ar", "ru", "it"}),
}


def _normalize_language(language: str) -> str:
    """Normalize a language to a bare lower-case ISO-639-1 code.

    Accepts ``en``, ``EN``, ``en_US`` and a few legacy English names the old
    suite used, so config files written against it still load.
    """
    cleaned = language.strip().lower().replace("_", "-")
    if "-" in cleaned:
        cleaned = cleaned.split("-", 1)[0]
    legacy_names = {
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "russian": "ru",
        "chinese": "zh",
        "japanese": "ja",
        "arabic": "ar",
    }
    return legacy_names.get(cleaned, cleaned)


def languages_for(backend: str) -> frozenset[str]:
    """The languages a backend has confirmed models for. Unknown backend -> empty."""
    return BACKEND_LANGUAGES.get(backend, frozenset())


def supports_language(backend: str, language: str) -> bool:
    """The real language gate. No backend, or no languages, means no support."""
    return _normalize_language(language) in languages_for(backend)


@dataclass(frozen=True, slots=True)
class NLPConfig:
    """The only configuration object in the system.

    Passed explicitly to every tool. Validated at construction so that an
    impossible configuration fails before a single document is read.
    """

    parser: ParserBackend = "stanza"
    language: str = "en"
    encoding: str = "utf-8"
    max_sentence_length: int = 1000
    chart_package: Literal["plotly"] = "plotly"
    seed: int = 0
    output_root: Path = Path("out")

    def __post_init__(self) -> None:
        object.__setattr__(self, "language", _normalize_language(self.language))

        if self.parser not in BACKEND_LANGUAGES:
            raise ConfigError(f"unknown parser backend {self.parser!r}; expected one of {sorted(BACKEND_LANGUAGES)}")
        if not self.language:
            raise ConfigError("language must be a non-empty string")
        if not supports_language(self.parser, self.language):
            supported = sorted(languages_for(self.parser))
            raise ConfigError(
                f"parser {self.parser!r} has no confirmed model for language {self.language!r}; supported: {supported}"
            )
        if self.max_sentence_length <= 0:
            raise ConfigError(f"max_sentence_length must be positive, got {self.max_sentence_length}")
        if not self.encoding:
            raise ConfigError("encoding must be a non-empty string")

    @property
    def parser_language_key(self) -> tuple[str, str]:
        """The identity used to cache a pipeline — see core.pipelines.cache."""
        return (self.parser, self.language)
