"""Optional-dependency imports that tell "missing" apart from "broken".

``importlib.util.find_spec`` and a bare ``except ImportError`` see only a
package that is *absent*. A package that is present but raises while importing
is invisible to both: a version gate like gensim's "this version of pandas is
not supported", a Windows DLL that fails to load, a numpy ABI mismatch. All of
those are "installed" to a presence check, so the tool is offered and the
failure surfaces mid-run as a traceback with no fix.

:func:`import_optional` performs the real import and reports which case it is,
so a caller can print "not installed, run this" for one and "installed but
broken, here is why" for the other::

    try:
        LdaModel = import_optional("gensim.models.ldamodel", attr="LdaModel")
    except OptionalDependencyError as exc:
        detail = "not installed" if exc.missing else f"installed but broken ({exc.reason})"

It is import-free itself, so importing this module never pulls an optional
dependency.
"""

from __future__ import annotations

import importlib
from typing import Any
import warnings

__all__ = ["OptionalDependencyError", "import_optional"]


class OptionalDependencyError(ImportError):
    """An optional dependency is absent, or installed but will not import.

    Subclasses :class:`ImportError` so an existing ``except ImportError`` still
    catches a *broken* install (which a plain ``except ImportError`` around the
    import would have missed); ``missing`` and :attr:`reason` let a caller tell
    the two apart and print the right fix.
    """

    def __init__(self, name: str, *, missing: bool, reason: str = "") -> None:
        self.missing = missing
        self.reason = reason
        # name= goes through ImportError itself: it owns ``name`` and resets it
        # to None when constructed without one.
        super().__init__(f"{name} {self.problem}", name=name)

    @property
    def problem(self) -> str:
        """A phrase to place after the package name in a diagnostic."""
        if self.missing:
            return "is not installed"
        return f"is installed but will not import ({self.reason})"


def import_optional(name: str, *, attr: str | None = None, suppress_warnings: bool = False) -> Any:
    """Import *name* (and optionally *attr* from it), or raise :class:`OptionalDependencyError`.

    ``missing`` is True only when the package is genuinely absent; any other
    exception during import or attribute lookup is reported as broken, with its
    first line as the reason. ``suppress_warnings`` quiets warnings a package
    emits on import, for a probe that only wants to know whether the import
    works (the doctor runs under ``filterwarnings = error`` and must not fail on
    a deprecation).
    """
    try:
        if suppress_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                module = importlib.import_module(name)
        else:
            module = importlib.import_module(name)
        return module if attr is None else getattr(module, attr)
    except ImportError as exc:
        # Genuinely absent only if the requested module itself or a parent package is missing.
        if (
            isinstance(exc, ModuleNotFoundError)
            and exc.name is not None
            and (name == exc.name or name.startswith(exc.name + "."))
        ):
            raise OptionalDependencyError(name, missing=True, reason=str(exc)) from exc
        first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        raise OptionalDependencyError(name, missing=False, reason=first) from exc
    except Exception as exc:  # anything else is a present-but-broken install
        first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        raise OptionalDependencyError(name, missing=False, reason=first) from exc
