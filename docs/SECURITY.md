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
3. **Dynamic code** — no `eval`/`exec` anywhere (tested).
4. **Path custody** — the writer allowlist (R3, `tests/test_write_custody.py`)
   keeps runs inside the output root; corpus files are read, never modified
   in place (filenames tool previews before renaming).
5. **Secrets** — nothing to hold: no API keys, no online geocoder keys in
   the default path (online geocoding is out of scope until FR-6.7, keys
   from env when it lands).

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
