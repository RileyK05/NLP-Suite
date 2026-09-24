"""NLPConfig contracts (ARCHITECTURE.md R6, section 2.6).

The anti-regression target: the legacy language check was effectively
``if True`` — every backend accepted every language and failed at model-load
time instead. These tests pin the gate to something real.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from core.config import (
    BACKEND_LANGUAGES,
    ConfigError,
    NLPConfig,
    languages_for,
    supports_language,
)


class TestLanguageGate:
    def test_supported_pair_is_accepted(self) -> None:
        assert supports_language("stanza", "en")
        assert supports_language("spacy", "de")

    def test_unsupported_language_is_rejected(self) -> None:
        """The legacy `if True` check admitted this."""
        assert not supports_language("stanza", "xx")

    def test_unknown_backend_has_no_languages(self) -> None:
        assert languages_for("nonexistent") == frozenset()
        assert not supports_language("nonexistent", "en")

    def test_every_backend_has_a_non_empty_language_set(self) -> None:
        for backend, languages in BACKEND_LANGUAGES.items():
            assert languages, f"{backend} claims no languages"

    def test_gate_is_case_and_region_insensitive(self) -> None:
        assert supports_language("stanza", "EN")
        assert supports_language("stanza", "en_US")


class TestConstruction:
    def test_defaults_are_valid(self) -> None:
        cfg = NLPConfig()
        assert cfg.parser == "stanza"
        assert cfg.language == "en"
        assert cfg.encoding == "utf-8"
        assert cfg.seed == 0

    def test_language_is_normalized(self) -> None:
        assert NLPConfig(parser="stanza", language="English").language == "en"
        assert NLPConfig(parser="stanza", language=" en_US ").language == "en"

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ConfigError, match="unknown parser backend"):
            NLPConfig(parser="nonexistent")  # type: ignore[arg-type]

    def test_unsupported_language_raises_with_the_supported_set(self) -> None:
        with pytest.raises(ConfigError, match="no confirmed model"):
            NLPConfig(parser="corenlp", language="ja")

    def test_empty_language_raises(self) -> None:
        with pytest.raises(ConfigError, match="non-empty"):
            NLPConfig(language="   ")

    def test_non_positive_sentence_length_raises(self) -> None:
        with pytest.raises(ConfigError, match="must be positive"):
            NLPConfig(max_sentence_length=0)

    def test_empty_encoding_raises(self) -> None:
        with pytest.raises(ConfigError, match="encoding"):
            NLPConfig(encoding="")


class TestImmutability:
    def test_is_frozen(self) -> None:
        cfg = NLPConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.language = "de"  # type: ignore[misc]

    def test_dataclass_replace_produces_a_new_config(self) -> None:
        cfg = NLPConfig()
        other = replace(cfg, parser="spacy", language="de")
        assert cfg.language == "en"
        assert other.language == "de"

    def test_is_hashable(self) -> None:
        """A frozen config can key a cache; the legacy global config could not."""
        assert len({NLPConfig(), NLPConfig()}) == 1


class TestPipelineKey:
    def test_parser_language_key(self) -> None:
        assert NLPConfig(parser="spacy", language="de").parser_language_key == ("spacy", "de")

    def test_distinct_keys_do_not_collide(self) -> None:
        keys = {
            NLPConfig(parser="stanza", language="en").parser_language_key,
            NLPConfig(parser="spacy", language="en").parser_language_key,
        }
        assert len(keys) == 2


class TestOutputRoot:
    def test_output_root_is_a_path(self) -> None:
        assert isinstance(NLPConfig().output_root, Path)

    def test_output_root_is_configurable(self) -> None:
        cfg = NLPConfig(output_root=Path("runs"))
        assert cfg.output_root == Path("runs")
