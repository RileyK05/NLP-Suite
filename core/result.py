"""Result[T] and Diagnostic — the one failure channel for the entire system.

ARCHITECTURE.md invariants enforced here:

* **R4** — errors travel as ``Result`` + ``Diagnostic``. Nothing below ``app/``
  raises to a UI, and no library code uses ``print`` as an error channel.
* **R7** — one return shape per function. An operation that can fail returns
  ``Result[T]``. It never returns ``Optional[T]`` where ``None`` means
  "error", "nothing to do", and "empty result" interchangeably.

Partial success is a first-class shape: a 100-document corpus with 3 bad
documents returns a 97-document value plus 3 ERROR diagnostics, rather than
aborting the run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Self

__all__ = ["Diagnostic", "Result", "Severity"]


class Severity(Enum):
    """How bad a diagnostic is. ERROR is the only severity that fails a Result."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Diagnostic:
    """A single, structured, serializable thing that went wrong (or was noted).

    ``code`` is a stable uppercase identifier (``EMPTY_DOC``, ``UNPARSEABLE_DATE``)
    so tests and the viewer can key on it rather than on prose. ``context``
    carries the coordinates of the event — ``doc_id``, ``sentence_id``, ``path``.
    """

    severity: Severity
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("Diagnostic.code must be a non-empty string")
        if not isinstance(self.severity, Severity):
            raise TypeError(f"severity must be a Severity, got {type(self.severity).__name__}")

    @classmethod
    def error(cls, code: str, message: str, **context: Any) -> Self:
        return cls(Severity.ERROR, code, message, dict(context))

    @classmethod
    def warning(cls, code: str, message: str, **context: Any) -> Self:
        return cls(Severity.WARNING, code, message, dict(context))

    @classmethod
    def info(cls, code: str, message: str, **context: Any) -> Self:
        return cls(Severity.INFO, code, message, dict(context))

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-serializable view of this diagnostic."""
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "context": dict(self.context),
        }

    def __str__(self) -> str:
        if not self.context:
            return f"[{self.severity.value}] {self.code}: {self.message}"
        # Coordinates the user can act on (path, doc_id, fix command) read
        # better inline; machinery like the full "tried" list reads as noise.
        parts = [
            f"{key}={value!r}"
            for key, value in self.context.items()
            if key not in ("tried", "attempted", "columns", "missing")
        ]
        where = f" ({', '.join(parts)})" if parts else ""
        return f"[{self.severity.value}] {self.code}: {self.message}{where}"


@dataclass(frozen=True)
class Result[T]:
    """The return type of every failable operation in the system.

    ``ok`` is true only when a value is present **and** no ERROR diagnostic was
    recorded — so warnings never masquerade as success, and a value produced
    alongside a real failure never masquerades as clean.

    >>> Result.success(42).ok
    True
    >>> Result[int].failure(Diagnostic.error("BOOM", "it broke")).ok
    False
    """

    value: T | None = None
    diagnostics: tuple[Diagnostic, ...] = ()

    @classmethod
    def success(cls, value: T, *diagnostics: Diagnostic) -> Self:
        """A succeeded result. Any diagnostics attached here are non-fatal."""
        return cls(value, tuple(diagnostics))

    @classmethod
    def failure(cls, *diagnostics: Diagnostic) -> Self:
        """A failed result — no value, at least one diagnostic explaining why."""
        if not diagnostics:
            raise ValueError("a failure Result needs at least one Diagnostic")
        return cls(None, tuple(diagnostics))

    @property
    def ok(self) -> bool:
        return self.value is not None and not self.has_errors

    @property
    def has_errors(self) -> bool:
        return any(d.severity is Severity.ERROR for d in self.diagnostics)

    @property
    def is_partial(self) -> bool:
        """A value exists but at least one ERROR diagnostic was recorded.

        The four states a failable operation reports:
        complete success (ok), partial-with-warnings (value + no errors +
        warnings), partial-with-errors (value + errors), complete failure
        (no value). Callers MUST branch on ok/is_partial, never on
        ``value is None`` alone — a value next to ERROR diagnostics is not
        a success.
        """
        return self.value is not None and self.has_errors

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity is Severity.WARNING)

    @property
    def infos(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity is Severity.INFO)

    def unwrap(self) -> T:
        """Return the value or raise. For tests and for callers that have
        already checked ``ok`` — never as a substitute for handling failure."""
        if self.value is None:
            raise ValueError(f"unwrap() on a failed Result: {[str(d) for d in self.diagnostics]}")
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value if self.value is not None else default

    def with_diagnostics(self, *diagnostics: Diagnostic) -> Result[T]:
        """A copy carrying additional diagnostics, preserving value and order."""
        return Result(self.value, self.diagnostics + tuple(diagnostics))

    def map[U](self, func: Callable[[T], U]) -> Result[U]:
        """Apply ``func`` to the value if present, carrying diagnostics through."""
        if self.value is None:
            return Result[U](None, self.diagnostics)
        return Result[U](func(self.value), self.diagnostics)
