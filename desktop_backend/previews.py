"""Short-lived, artifact-scoped capabilities for the browser-only viewer."""

from collections import OrderedDict
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


class PreviewTickets:
    """No corpus text cached. Tickets expire and never authorize other API calls."""

    def __init__(self, ttl: float = 300, capacity: int = 128) -> None:
        self.ttl = ttl
        self.capacity = capacity
        self.entries: OrderedDict[str, tuple[float, tuple[str, str, int]]] = OrderedDict()
        self.lock = threading.Lock()

    def issue(self, project: str, job: str, index: int) -> str:
        with self.lock:
            ticket = secrets.token_urlsafe(32)
            self.entries[ticket] = (time.monotonic() + self.ttl, (project, job, index))
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
            return ticket

    def resolve(self, ticket: str) -> tuple[str, str, int]:
        with self.lock:
            entry = self.entries.get(ticket)
            if entry is None or entry[0] <= time.monotonic():
                self.entries.pop(ticket, None)
                raise KeyError("Preview expired. Close and reopen the preview.")
            return entry[1]
