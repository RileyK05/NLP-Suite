# License and attribution review packet (FR-0.3 — decided)

> **Decision, 2026-09-16 (owner): NLP Suite is released under the MIT
> License.** `LICENSE` holds the terms, `pyproject.toml` carries
> `license = "MIT"` with `license-files = ["LICENSE"]`, and the desktop
> installer ships the notice at the head of its dependency notices
> (`/api/notices`), as MIT requires. The rest of this document remains the
> evidence behind that decision and the record for the research assets,
> whose terms are unchanged by it.

This document was the complete input for that judgment. It states what
ships, what does not, and which questions remain open for third-party
assets. `tests/test_licenses.py` enforces the no-vendoring rule on every run.

## Relationship to the legacy suite

NLP Suite 1.6.38 (Roberto Franzosi, Emory University) is licensed under the
GNU General Public License. This rebuild is independent: no legacy source is
copied in, and the legacy tree is used read-only as a behavioral reference.
The MIT grant therefore covers only original code in this repository. The
scripts that *execute* legacy modules for comparison are development-only
and are not distributed.

## What ships in this repo

Verified against the built distributions on 2026-09-16: the sdist and wheel
contain `core/`, `tools/`, `app/`, `desktop_backend/` and `README.md` and
nothing else. `scripts/`, `docs/`, `desktop/`, `tests/` and the owner's
coursework material are excluded by `MANIFEST.in`, and `tests/test_distribution.py`
enforces that boundary.

- Original code (this suite), no third-party source copied in.
- `tests/fixtures/`: hand-built mini lexicons/corpora (a few rows each,
  invented values) — no oracle bytes.
- `corpus/`: NOT shipped. It is gitignored and absent from both the sdist and
  the installer. Local copies are POTUS State of the Union addresses (public
  domain, U.S. federal government work), kept for the owner's own research.
- `assets/`: empty (`.gitkeep` only). The asset registry pins checksums
  for user-supplied copies; nothing licensed lives here.

## What never ships (user-supplied or download-on-demand)

| Asset | License posture | Source |
|---|---|---|
| VADER lexicon | MIT (package) — notice kept via the package dep | `vaderSentiment` |
| ANEW norms | Research use only (Bradley & Lang 1999) — NOT redistributable | user copy |
| SentiWordNet 3.0 | CC BY-SA 3.0 — attribution + share-alike | download |
| Hedonometer labMT | Research use, attribution required | user copy |
| Brysbaert concreteness | Springer BRM 2014 — redistribution undecided | user copy |
| Iconicity ratings | Research use — redistribution unclear | user copy |
| NRC lexicon | Citation + permission for redistribution; nrclex bundle auto-detected | package/user |
| WordNet 3.0 | Princeton license (redistribution with license text) | download |
| VerbNet / FrameNet | Open (NLTK data) | download |
| Symbolic/actor typology CSVs | Suite-curated, recorded as redistributable | user copy of legacy lists |
| Parser models (spaCy/Stanza) | Own model licenses | download |
| Transformer models (BERT etc.) | Own model licenses (often Apache-2.0) | download |
| Google Geocoding / DBpedia | API terms; keys from env, never logged | network |

## Decisions owed (owner)

### Desktop packaging update (2026-09-06)

The expanded desktop build now deliberately bundles installed Python runtime
dependencies, spaCy's English model, and optionally WordNet data. These are
generated build products under the gitignored `desktop/src-tauri/binaries/`,
not vendored source assets. The source no-vendoring guard excludes that generated
directory; it does not certify binary redistribution compliance.

`scripts/desktop_notices.py` gathers package license metadata and installed
notice files for the bundle's Credits & licenses view. WordNet's own license
is retained inside its bundled `corpora/wordnet.zip`. This generated inventory
is not a replacement for final distribution review. Restricted research
lexicons (including ANEW, NRC, concreteness and iconicity) remain user-supplied.

No license has been assigned to the original code without the owner's answer.
The decisions below remain outstanding.

1. **Brysbaert/iconicity redistribution** — may the suite ever bundle
   these, or stay user-supplied forever? (Recommendation: stay
   user-supplied; the checksums already pin the oracle copies.)
2. **Suite license** — no `LICENSE` file exists yet. Recommendation:
   Apache-2.0 or MIT for code, with `corpus/` public-domain noted and
   all rows above referenced from `pyproject.toml` metadata.
3. **Attribution page** — whether the app gallery needs a permanent
   credits surface (Brysbaert, ANEW, NRC, WordNet, VADER, spaCy/Stanza,
   Hugging Face) beyond this document.

Until decided: FR-0.3 stays open; the no-vendoring test guarantees the
status quo cannot silently change.
