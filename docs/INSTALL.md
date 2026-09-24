# Clean-machine install (FR-9.4)

Verified on Windows 2026-09-16: a built wheel installed into an empty venv,
then `nlp-suite --list`, `nlp-suite readability` and `nlp-doctor` all run
green from a directory outside the source tree (`tests/test_install.py`
guards the entry points). macOS steps are the same commands; a macOS
verification run is still owed (ledger note).

These commands install from a source checkout. The project is not published
to PyPI, so `pip install nlp-suite-ng` by name does not resolve — clone
first, and keep the `.` in the commands below.

## Windows (PowerShell)

```powershell
git clone https://github.com/RileyK05/NLP-Suite
cd NLP-Suite
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install ".[app,spacy]"
python -m spacy download en_core_web_sm
nlp-doctor en
```

## macOS (zsh)

```zsh
git clone https://github.com/RileyK05/NLP-Suite
cd NLP-Suite
python3.12 -m venv .venv
source .venv/bin/activate
pip install ".[app,spacy]"
python -m spacy download en_core_web_sm
nlp-doctor en
```

## What each extra pulls

| Extra | For |
|---|---|
| `stanza` | default parser backend |
| `spacy` | alternate parser backend |
| `converters` | pdf/docx/rtf intake |
| `sentiment` | VADER scoring |
| `topics` | Gensim LDA/Word2Vec |
| `wordnet` | NLTK WordNet/VerbNet/FrameNet data access |
| `embeddings` | transformer vectors + BERT extractive |
| `plotly`, `app` | charts + Streamlit gallery |
| `plotly-image` | static PNG/SVG chart export (kaleido) |
| `excel` | native Excel chart export (openpyxl; bar/line/pie/scatter/radar/bubble) |
| `desktop` | local FastAPI/uvicorn engine for the Tauri/React desktop preview |
| `wordcloud` | raster wordclouds (image masks, shape masks, per-group colors, PNG) |
| `figures` | publication figures (matplotlib + seaborn): each figure's PNG/SVG/PDF, and a finished run's `figures/` |
| `all` | everything above |

The desktop's **Settings & backups** page lists every one of these by name,
with what it is for and whether it was found, so the question "why can this
installation not save a PNG?" has an answer on screen rather than only in this
table.

## Parser models: you need one, not both

Almost every analysis needs a parse, and a parse needs a model. Installing the
package does not download one. The quickest path to a working suite is spaCy's
small English model, a single command and a few tens of megabytes:

```
python -m spacy download en_core_web_sm
```

The configured default backend is Stanza. If its model is missing or only
partly downloaded — the usual result of an interrupted `stanza.download` — a
run **does not fail** while another installed backend has a working model. It
parses with that one instead and says so: on the console, in the run's
diagnostics, and in `result.json`, whose `params.parser` records the backend
that actually ran rather than the one that was asked for.

Backends tokenize and tag differently, so a fallback can change results. Two
ways to control it:

* `nlp-doctor` names every damaged model and the exact command to repair it.
* `--strict-parser` on any tool fails the run instead of substituting, for when
  a specific backend is the requirement.

Optional lexicon/model/data assets are never downloaded by install;
`nlp-suite models OUT` and `nlp-suite assets --root assets` report what is
missing with the exact fix. The separate Windows desktop preview now has a
model-light PyInstaller/Tauri packaging recipe; see [DESKTOP.md](DESKTOP.md).
Its installer is not a full parser/model distribution or a legacy parity signoff.
