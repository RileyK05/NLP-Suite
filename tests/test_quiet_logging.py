r"""Third-party logging stays out of the suite's output.

The suite reports problems one way: a Diagnostic on a Result (R7). A library
that logs its own errors bypasses that, and the output lands wherever it
likes. Stanza, asked for a model whose files are incomplete, logged

    2026-09-17 13:16:58 ERROR: Cannot load model from ...\ner\ontonotes.pt

to stderr and *then* raised. The raise is what the suite reports on, so the
line was a duplicate -- and because it was emitted during the build, it landed
above everything the tool went on to print. ``nlp-doctor`` opened with a raw
ERROR, directly above its own report explaining that exact situation.
"""

from __future__ import annotations

import logging

import pytest

from core.pipelines.quiet import CapturedLogs, captured_logs


class TestCapture:
    def test_nothing_reaches_the_handlers_during_the_block(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG), captured_logs("noisy_library"):
            logging.getLogger("noisy_library").error("Cannot load model from somewhere")
        assert "Cannot load model" not in caplog.text

    def test_what_was_held_back_is_handed_to_the_caller(self) -> None:
        with captured_logs("noisy_library") as logged:
            logging.getLogger("noisy_library").error("Cannot load model from %s", "a/b.pt")
        assert logged.messages() == ["Cannot load model from a/b.pt"]

    def test_repeats_collapse(self) -> None:
        """A library retrying a load logs the same failure several times."""
        with captured_logs("noisy_library") as logged:
            for _ in range(3):
                logging.getLogger("noisy_library").error("same failure")
        assert logged.messages() == ["same failure"]

    def test_quiet_records_are_kept_out_of_the_summary(self) -> None:
        with captured_logs("noisy_library") as logged:
            logging.getLogger("noisy_library").info("loading")
            logging.getLogger("noisy_library").error("broke")
        assert logged.messages() == ["broke"]
        assert len(logged.records) == 2, "the records are all kept; only the summary filters"

    def test_other_loggers_are_untouched(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG), captured_logs("noisy_library"):
            logging.getLogger("someone_else").error("this must still be heard")
        assert "this must still be heard" in caplog.text


class TestRestoration:
    def test_the_logger_works_again_afterwards(self, caplog: pytest.LogCaptureFixture) -> None:
        with captured_logs("noisy_library"):
            pass
        with caplog.at_level(logging.DEBUG):
            logging.getLogger("noisy_library").error("after")
        assert "after" in caplog.text

    def test_restored_even_when_the_block_raises(self, caplog: pytest.LogCaptureFixture) -> None:
        """The normal path: the wrapped call is expected to fail."""
        logger = logging.getLogger("noisy_library")
        before = (list(logger.handlers), logger.level, logger.propagate)
        with pytest.raises(RuntimeError), captured_logs("noisy_library"):
            logger.error("on the way down")
            raise RuntimeError("as the library does")
        assert (list(logger.handlers), logger.level, logger.propagate) == before
        with caplog.at_level(logging.DEBUG):
            logger.error("after")
        assert "after" in caplog.text

    def test_a_broken_format_string_does_not_become_an_exception(self) -> None:
        """The library's bug must not crash the diagnostic being assembled."""
        with captured_logs("noisy_library") as logged:
            # Deliberately malformed, which is the point of the test.
            logging.getLogger("noisy_library").error("needs %s and %s", "only-one")  # noqa: PLE1206
        assert logged.messages() == []
        assert len(logged.records) == 1


class TestTheStanzaBuilderFoldsItIn:
    def test_a_log_line_naming_the_same_file_is_not_repeated(self) -> None:
        from core.pipelines.stanza_backend import _held_back

        with captured_logs("x") as logged:
            logging.getLogger("x").error("Cannot load model from C:/cache/en/ner/ontonotes.pt")
        exc = FileNotFoundError("Could not find model file C:/cache/en/ner/ontonotes.pt, although there are others")
        assert _held_back(logged, exc) == ""

    def test_a_log_line_with_new_detail_is_kept(self) -> None:
        from core.pipelines.stanza_backend import _held_back

        with captured_logs("x") as logged:
            logging.getLogger("x").error("resource index is corrupt")
        assert _held_back(logged, FileNotFoundError("missing")) == " (resource index is corrupt)"

    def test_nothing_logged_adds_nothing(self) -> None:
        from core.pipelines.stanza_backend import _held_back

        assert _held_back(CapturedLogs(), FileNotFoundError("missing")) == ""
