"""Keeping a third-party library's logging out of the suite's output.

The suite has one channel for things that went wrong: a :class:`Diagnostic`
attached to a :class:`Result` (R7). A library that logs its own errors breaks
that, because its output does not go through the channel and arrives wherever
it likes.

Stanza is the case that forced this. Asked to load a model whose files are
incomplete, it logs::

    2026-09-17 13:16:58 ERROR: Cannot load model from ...\\ner\\ontonotes.pt

to stderr and *then* raises. The suite catches the exception and turns it into
a ``PIPELINE_MODEL_MISSING`` diagnostic carrying the same information plus the
fix command -- so the logged line is a duplicate. Worse, it is emitted during
the import-and-build, which means it lands *before* whatever the tool was
about to print. Running ``nlp-doctor`` on a machine with a half-downloaded
model opened with a raw ERROR line, above the report written to explain
exactly that situation.

So during a build that the suite is prepared to handle, the library's logging
is captured rather than emitted, and handed back so the caller can fold
anything genuinely new into its own diagnostic. Nothing is discarded silently
and nothing is suppressed permanently: the logger's level, handlers and
propagation are restored on the way out, including when the body raises.

This is not a general "hide the warnings" switch. It is only for a call whose
failure is already converted into a diagnostic; anywhere else, losing a
library's log line would lose information the suite is not replacing.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging

__all__ = ["CapturedLogs", "captured_logs"]


def _render(record: logging.LogRecord) -> str:
    """A record's message, or "" if the library's own formatting is broken.

    ``getMessage`` interpolates the library's args into its format string and
    can raise on a mismatch. That is the library's bug, and re-raising it here
    would turn a diagnostic that was being assembled into a crash.
    """
    try:
        return record.getMessage()
    except (TypeError, ValueError, KeyError, IndexError):
        return ""


@dataclass(slots=True)
class CapturedLogs:
    """Log records held back during a build, in the order they were emitted."""

    records: list[logging.LogRecord] = field(default_factory=list)

    def messages(self, minimum: int = logging.ERROR) -> list[str]:
        """Messages at or above *minimum*, de-duplicated, order preserved.

        De-duplicated because a library retrying a load logs the same failure
        several times, and a diagnostic that repeats itself reads like several
        separate problems.
        """
        seen: set[str] = set()
        out: list[str] = []
        for record in self.records:
            if record.levelno < minimum:
                continue
            message = _render(record)
            if message and message not in seen:
                seen.add(message)
                out.append(message)
        return out


class _Collector(logging.Handler):
    def __init__(self, into: CapturedLogs) -> None:
        super().__init__(level=logging.NOTSET)
        self._into = into

    def emit(self, record: logging.LogRecord) -> None:
        self._into.records.append(record)


@contextmanager
def captured_logs(*logger_names: str) -> Iterator[CapturedLogs]:
    """Hold back *logger_names*' output for the duration of the block.

    Restores each logger's handlers, level and ``propagate`` flag afterwards,
    including on an exception -- which is the normal path here, since the point
    is to wrap a call that is expected to fail.
    """
    captured = CapturedLogs()
    collector = _Collector(captured)
    saved: list[tuple[logging.Logger, list[logging.Handler], int, bool]] = []

    for name in logger_names:
        logger = logging.getLogger(name)
        saved.append((logger, list(logger.handlers), logger.level, logger.propagate))
        logger.handlers = [collector]
        # Propagation off, or the root logger's handlers print it anyway --
        # which is exactly how the line reached the console in the first place.
        logger.propagate = False
        # DEBUG, not NOTSET: NOTSET delegates to the parent's effective level,
        # which is the root logger's WARNING by default, so a record below that
        # would be discarded before reaching the collector. Holding a line back
        # must never mean losing it -- the caller decides what is worth
        # repeating, and it cannot decide about a record it never received.
        logger.setLevel(logging.DEBUG)
    try:
        yield captured
    finally:
        for logger, handlers, level, propagate in saved:
            logger.handlers = handlers
            logger.setLevel(level)
            logger.propagate = propagate
