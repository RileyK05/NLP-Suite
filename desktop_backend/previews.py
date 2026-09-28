"""Short-lived, artifact-scoped capabilities for the browser-only viewer."""

from collections import OrderedDict
from dataclasses import dataclass
import secrets
import threading
import time

PREVIEW_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src data: blob:; font-src data:; connect-src 'none'; frame-src 'none'; "
    "worker-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; "
    "sandbox allow-scripts; frame-ancestors 'self'"
)
MAX_PREVIEW_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ArtifactTarget:
    """A run artifact: which project's job, and which artifact index."""

    project: str
    job: str
    index: int


@dataclass(frozen=True, slots=True)
class KernelTarget:
    """One file of a notebook's kernel: which project, notebook and file."""

    project: str
    notebook: str
    name: str


Target = ArtifactTarget | KernelTarget


class PreviewTickets:
    """No corpus text cached. Tickets expire and never authorize other API calls.

    Two kinds of target share one namespace so a frame URL has one shape:
    an artifact (``project, job, index``) and a notebook kernel file
    (``"kernel", project, name``), told apart by their first element. A
    kernel-file ticket lets the Scripts page frame an interactive chart the
    kernel wrote without handing the frame the workspace token.
    """

    def __init__(self, ttl: float = 300, capacity: int = 128) -> None:
        self.ttl = ttl
        self.capacity = capacity
        self.entries: OrderedDict[str, tuple[float, Target]] = OrderedDict()
        self.lock = threading.Lock()

    def _store(self, target: Target) -> str:
        with self.lock:
            ticket = secrets.token_urlsafe(32)
            self.entries[ticket] = (time.monotonic() + self.ttl, target)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
            return ticket

    def issue(self, project: str, job: str, index: int) -> str:
        return self._store(ArtifactTarget(project, job, index))

    def issue_kernel(self, project: str, notebook: str, name: str) -> str:
        """A ticket for one file of a notebook kernel."""
        return self._store(KernelTarget(project, notebook, name))

    def resolve(self, ticket: str) -> Target:
        with self.lock:
            entry = self.entries.get(ticket)
            if entry is None or entry[0] <= time.monotonic():
                self.entries.pop(ticket, None)
                raise KeyError("Preview expired. Close and reopen the preview.")
            return entry[1]
