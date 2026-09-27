# Security and privacy review (FR-9.6)

## Threat posture

This suite reads analyst-owned corpora and writes run directories. It has
no accounts, no network services, and no telemetry. The risks that remain:

1. **Command execution** — only `core/jobs.py` (the analyst's chosen tool
   CLI), `core/pipelines/srl_backend.py` (the configured worker
   interpreter), and `core/analysis/mallet.py` (the configured binary)
   spawn processes: argv lists, `shell=False`, no string ever reaches a
   shell. Enforced by `tests/test_security.py` plus ruff `S603`.
2. **Network** — only `core/pipelines/corenlp_backend.py` speaks HTTP, to
   the analyst's own CoreNLP server URL, after `_checked_url` rejects
   non-`http(s)` schemes. Enforced by `tests/test_security.py` plus ruff
   `S310`. Model downloads (spaCy/Stanza/transformers/NLTK) run inside
   those libraries on explicit analyst commands, never implicitly.
3. **Dynamic code** — no `eval`/`exec` anywhere except
   `desktop_backend/kernel.py`, which runs the code a researcher writes in a
   notebook (see "Notebook code" below). The allowlist names that one module
   and a test fails if a second is added (`test_only_one_module_may_run_code`).
4. **Path custody** — the writer allowlist (R3, `tests/test_write_custody.py`)
   keeps runs inside the output root; corpus files are read, never modified
   in place (filenames tool previews before renaming).
5. **Secrets** — nothing to hold: no API keys, no online geocoder keys in
   the default path (online geocoding is out of scope until FR-6.7, keys
   from env when it lands).

## Notebook code (the Scripts page)

A notebook runs Python its author wrote, or pasted from a chatbot. That code
has **the same rights as the app**: it can read and write your files and
reach the network. Python cannot be sandboxed from inside Python — blocking
`open` or `socket` is undone by one import — so the suite does not pretend
to. What it does promise:

- **A crash or an endless loop costs the notebook, not the app.** Each open
  notebook runs in its own process (a *kernel*: the engine started with
  `--script-kernel`). Stop ends that process; the app and its data are
  untouched. A kernel also exits when the app does, even if the app is
  killed, because it stops when its input pipe closes.
- **A kernel cannot drive the app.** It is never given the access token the
  interface uses (`test_the_kernel_is_never_handed_the_access_token`), so it
  cannot create, change or delete projects, runs or documents through the API.
- **Results enter a project only through the library.** A kernel's working
  folder is scratch space deleted when it stops; tables and figures are kept
  by `nlp.show`, `nlp.save` and `nlp.figure`, and a file a cell writes into
  its working folder is reported as thrown away. (A cell that names a path
  elsewhere on disk can still write there; see the first sentence.)
- **Code from outside is marked until it is read.** A cell pasted from a
  chatbot, or a notebook imported from a file, carries a banner, and lines
  worth reading are pointed out (`core/script/lint.py`: imports that reach
  files, programs or the network; `open`, `eval`, `exec`; files written
  outside the library). This is a reading aid, not a defence. *Run all* stops
  above a pasted cell nobody has marked as read, and *Run and save* refuses
  to record a run until every pasted cell has been.
- **No AI is built in.** The chatbot guide is a document the researcher
  copies into a chatbot of their choosing; the app makes no network call to
  any model. With "Include my corpus's details" on, the guide lists document
  names, dates, counts and column names — never document text.

Treat a notebook from someone else like a script from someone else: read it
before running it.

## Privacy

Corpora may be sensitive. Guarantees: parsing and analysis are fully local
except the explicitly remote backends (CoreNLP server, knowledge-graph
clients when FR-6.8 lands — both analyst-configured endpoints); run
envelopes record file names and hashes, never file contents; nothing is
uploaded, phoned home, or cached outside the output root and the
libraries' own model caches.

## Process

`tests/test_security.py` parses every shipped module on every run. Adding a
new subprocess/network/dynamic-code use must extend its allowlist with the
control — the review is continuous, not a one-time document.
