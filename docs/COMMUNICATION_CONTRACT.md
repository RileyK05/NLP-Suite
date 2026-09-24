# Communication Contract

The standard for every conversation between the desktop frontend
(`desktop/src`) and the Python engine (`desktop_backend`). It exists because
one bug — a topic model that hung — turned out to be two systemic failures at
this boundary, and both were invisible to ordinary testing:

1. **Frontend:** analyses were *implied* by parameter changes. A debounced
   effect asked on the user's behalf, and when a parameter changed while a
   request was in flight the superseded effect and the component disagreed
   about who owned the "busy" flag. The bench said "Working…" forever with
   nothing running (`desktop/src/ExplicitRun.test.tsx` is the autopsy).
2. **Backend:** a tool adapter imported `gensim` *inside its function body*.
   That first import happened in a request thread, where the Windows loader
   lock wedged the whole server at 0% CPU (`tests/test_live_analysis.py` is
   the autopsy).

Rules below are enforced by tests, not by vigilance. Where a rule has an
enforcing test it is named.

## R-C1 — Explicit run

No analysis request is ever sent without a user action (a Run button, a job
submit, a publish). Editing a parameter is free and silent. There is no
debounce, because there is nothing to debounce: changing settings never
schedules work.

*Why:* implicit asks are what turned a parameter edit into a wedged bench.
Iteration — change a setting, look, change it back — must cost zero requests.
*Enforced by:* `desktop/src/ExplicitRun.test.tsx` (no request on param change
alone; Run is the only trigger).

## R-C2 — Request identity

Every request that can race another carries a client-generated `request_id`.
The response echoes it. An answer whose `request_id` is not the current one is
dropped silently — it answered a question nobody is asking.

*Why:* answers arrive out of order; without identity a slow answer for an old
setting overwrites the fast answer for the current one.
*Enforced by:* `tests/test_contract_parity.py` (field present in both
envelopes), `desktop/src/live.test.ts`.

## R-C3 — Single flight, explicit retry

One live analysis at a time. A request out longer than
`REQUEST_TIMEOUT_MS` is *lost*: the slot is released and the UI offers a
Retry button. Nothing retries silently, and nothing leaves "Working…" with
nothing in flight.

*Why:* a dropped fetch used to hold the slot forever. Silent retry hid
whether a request was ever made.
*Enforced by:* `desktop/src/ExplicitRun.test.tsx`, `desktop/src/live.test.ts`
(`requestIsLost`).

## R-C4 — Warm before analyse

`POST /live/warm` must have produced a ready snapshot before
`POST /live/analyse` runs. `analyse` never triggers first-time heavy imports
or parses. A `snapshot_id` mismatch is a 409 (`StaleSnapshot`), not a silent
answer to the wrong corpus.

*Why:* the parse is the slow step and is paid once; also, warm-up is where
heavy imports belong (see R-C5).
*Enforced by:* `desktop_backend/live.py` (`Session.resolved`), live API tests.

## R-C5 — No lazy third-party imports on the request path

Anything a tool adapter may import for the first time must be imported by
`desktop_backend.server::preload_engine` before the socket opens. Function
body imports of third-party packages in adapter-reachable modules must be
listed in `LAZY_BACKENDS` — or the test fails.

*Why:* the first import of a native-extension package inside a request thread
holds the Windows loader lock while the main thread needs it to grow the
worker pool. The server stops answering with no error and no timeout. The
check must walk *function bodies*, not module tops: `core.analysis.lda` was
warm while `gensim` was cold, and every shallow check passed.
*Enforced by:* `tests/test_no_lazy_backend_imports.py` (AST walk of every
adapter-reachable module + preload probe in a clean interpreter).

## R-C6 — Typed envelopes, one source of truth

Request and response shapes live in `desktop_backend/schemas.py` (Pydantic).
`desktop/src/contract.ts` is *generated* from them by
`scripts/gen_contract_ts.py` and checked in. The frontend may not redefine
these shapes by hand.

*Why:* the desktop and the engine have drifted silently before (tool names,
parameter labels, chart kinds). Drift must fail a test, not a user.
*Enforced by:* `tests/test_contract_parity.py` (regenerate + diff; JSON
schemas match field-for-field).

## R-C7 — Diagnostics are the error channel

Failures return `ok: false` with `diagnostics: [{severity, code, message}]`.
Never a hang, never a bare 500 without a code, never an empty success.
Partial success (value + ERROR diagnostics) is a first-class shape (R7 in
`ARCHITECTURE.md`).

*Enforced by:* `core/result.py` contract tests; endpoint tests assert codes.

## R-C8 — Abandoning a question is explicit

`DELETE /api/projects/{id}/live/analyse/{request_id}` says "nobody is waiting
for this any more" (the R-C3 follow-up). The client sends it when a request is
declared lost and when the bench unmounts mid-flight; the server then

- never starts computing a question cancelled before it began, and
- discards the answer of one cancelled while it ran,

answering `ok: false` with a single `LIVE_CANCELLED` diagnostic (so R-C7
still holds: one shape to render, never a hang). A cancellation is honoured
once — the id is not poisoned.

This is deliberately **"discard the answer", not "stop the CPU"**: gensim and
sklearn fits are not interruptible mid-call, and a contract that promised
otherwise would be a lie the reader could measure as a machine still heating
up after they walked away.

*Enforced by:* `tests/test_live_analysis.py::test_an_abandoned_live_question_is_cancelled_not_answered`;
`desktop/src/ExplicitRun.test.tsx` holds the client half (the abort is sent
on loss and on unmount).

## Envelope summary

```jsonc
// POST /api/projects/{id}/live/analyse
// request
{ "request_id": "3f2b…", "tool": "lda_gensim", "params": {"topics": 10},
  "snapshot_id": "9c1e…" }
// response (200; 409 StaleSnapshot on snapshot mismatch)
{ "request_id": "3f2b…", "tool": "lda_gensim", "ok": true,
  "elapsed_ms": 12384, "snapshot_id": "9c1e…",
  "table": {...} | null, "path": "topics.csv",
  "headline": "…", "observations": [], "cautions": [],
  "recommended_charts": [], "diagnostics": [] }

// DELETE /api/projects/{id}/live/analyse/{request_id}   (R-C8)
{ "ok": true, "request_id": "3f2b…", "cancelled": true }
```

`desktop_backend/schemas.py` is the definition; the block above is a
courtesy, not the source.

## What this contract is not

- It does not govern CLI tools (`tools/*.py`) — those speak argv and the
  artifact envelope (`docs/ARCHITECTURE.md` §2.3).
- It does not govern Streamlit (`app/`) beyond R-C5/R-C7.
- Job submission (`POST /jobs`) follows R-C1/R-C7 but is asynchronous by
  design and carries identity as `job_id`, not `request_id`.
