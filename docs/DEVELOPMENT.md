# Developer guide

[← Back to downloads and user instructions](../README.md)

These instructions are for source development and CLI/Streamlit use, **not**
for people installing the desktop app. For the desktop development launcher,
see [Desktop setup](DESKTOP.md).

## Quickstart

**New desktop preview:** a local Tauri/React application with drag-and-drop text
imports, persistent projects, background analysis, and downloadable results.
See [Desktop setup and testing](DESKTOP.md). Streamlit and the CLI remain available.
For installer handoff and native Windows/Mac/Linux builds, see the
[release guide](DESKTOP_RELEASE.md). Installed-app users do not need the
Python developer setup below.

```bash
# 1. Environment (Python 3.12; no other version is tested)
pip install -e ".[stanza,plotly,app,dev]"

# 2. One parser model (hard requirement — the suite refuses to
#    silently degrade to a tokenizer without one). Stanza is the
#    default backend; spaCy is the alternate (`--parser spacy`).
python -c "import stanza; stanza.download('en')"

# 3. Check everything is actually runnable
nlp-doctor            # or: python -m tools.doctor

# 4. Run a tool
python -m tools.conll_wordlist <corpus-dir> out/
python -m tools.profiler <corpus-dir> out/     # the whole batch

# 5. View the runs
streamlit run app/Home.py
```

**Failure policy.** The suite fails loudly rather than degrading silently:
a missing parser model stops the run with the exact fix command (and the
doctor report) instead of producing hollow numbers under a green exit code.
Faithful fallbacks that reproduce the same data (e.g. the HTML-table chart
renderer) attach a WARNING diagnostic so the envelope always tells the truth.
Run `nlp-doctor` first — it tells you what is missing and how to install it.

## Status

Early build. **52 of 52 scaffold chunks done** (see
`docs/CHUNK_LEDGER.md`). This does not mean full legacy parity: see
`docs/LEGACY_PARITY.md` for the current comparison and
`docs/FULL_REPLACEMENT_PLAN.md` for the remaining implementation and review
roadmap.

## How work is planned

`docs/BUILD_PLAN.md` is the completed 52-chunk scaffold build order and
`docs/CHUNK_LEDGER.md` records that work. New feature-parity work is defined in
`docs/FULL_REPLACEMENT_PLAN.md`. Each roadmap item is delivered through the
repeated-prompt chunks in `docs/IMPLEMENTATION_CHUNKING_PROTOCOL.md`, followed
by independent review before it can be verified. Live status belongs in
`docs/REPLACEMENT_LEDGER.md`.

## Layout

| Path | Purpose |
|---|---|
| `core/` | Headless library: `result`, `config`, `io`, `conll`, `pipelines`, `artifacts`, `analysis` |
| `tools/` | One thin CLI per tool; `tools/doctor.py` checks the environment |
| `app/` | Viewer (Streamlit) |
| `tests/` | `fixtures/` (inputs) + `golden/` (captured expected outputs) |
| `assets/` | Lexicons and other data files, loaded once |
| `out/` | Run directories; gitignored |
| `docs/ARCHITECTURE.md` | The intended design |
| `docs/FULL_REPLACEMENT_PLAN.md` | Remaining parity, product, packaging, and review roadmap |
| `docs/REPLACEMENT_LEDGER.md` | Live implementation and independent-review status |
| `docs/IMPLEMENTATION_CHUNKING_PROTOCOL.md` | Repeated-prompt execution and correction cycle |

## Third-party logging

The suite reports problems one way: a `Diagnostic` on a `Result`. A library
that logs its own errors bypasses that channel, and its output arrives wherever
it likes — usually ahead of whatever the tool was about to print, because the
library runs first.

Stanza is the case that forced the rule. Asked for a model whose files are
incomplete it logs `ERROR: Cannot load model from ...` to stderr and *then*
raises, so `nlp-doctor` opened with a raw ERROR line directly above its own
report explaining that exact situation.

Wrap such a call in `core.pipelines.quiet.captured_logs(...)`. It holds the
library's records back and hands them to you, so anything the exception does
not already say can be folded into the diagnostic (see `_held_back` in
`core/pipelines/stanza_backend.py`). It restores the logger's handlers, level
and propagation on the way out, including when the body raises — which is the
normal path, since the point is to wrap a call expected to fail.

Use it **only** where the failure is already converted into a diagnostic.
Anywhere else, holding back a library's log line loses information the suite is
not replacing.

## Naming a document in a result table

Every table that names a document uses `core.io.reader.display_names`, never
`doc.path` and never a bare `doc.path.name`. It returns the shortest label that
is still unique across the corpus:

| Corpus | Labels |
|---|---|
| `study/alpha.txt`, `study/beta.txt` | `alpha.txt`, `beta.txt` |
| `2020/report.txt`, `2021/report.txt` | `2020/report.txt`, `2021/report.txt` |
| the same file listed twice | `report.txt (#1)`, `report.txt (#2)` |

Both halves matter. An absolute path is unreadable in a CSV cell, and it
publishes the machine's directory layout into output that gets shared or handed
in. A bare basename is shorter but collapses `2020/report.txt` and
`2021/report.txt` into one label, silently merging two documents in any table
grouped or charted by name.

Diagnostics that name a document use the same labels. `tests/test_document_labels.py`
runs the corpus-level tools against absolute paths and fails if any cell in a
document column contains a filesystem path.

The one deliberate exception is `core/file_ops/classifier.py`, where the file
path *is* the subject of the table; it carries `path` and `filename` side by
side.

## Looking at a result, and publishing one

The desktop draws charts in two places, and the difference matters.

**Explore** (`desktop/src/Explore.tsx`) is the front door: a sidebar page that
asks which table and then hands it to the workbench. Its list comes from
`GET /api/projects/{id}/tables`, which walks every finished run in the project
and offers the CSVs that are **on disk now** — an envelope records what a run
published, which is not the same claim. Input manifests are listed but flagged,
so the page opens on a result instead of on the record of what was read.

The workbench existed for a while with no such page, reachable only through
Runs & results, then a finished job, then the right artifact. It was built,
tested and served, and it was reported as missing, which it effectively was.
`tests/test_desktop_navigation.py` now reads `App.tsx` and fails if any page in
the `Page` union has no way to reach it.

**The workbench** (`desktop/src/Workbench.tsx`) is interactive. It redraws from
the table already in the browser, so changing measure, chart kind or grouping
is instant, and clicking a bar filters the rows beneath it. Its geometry comes
from `desktop/src/chartLayout.ts`, which is pure — no React, no DOM — and is
where the numbers a chart claims are tested.

It draws **the page on screen**, at most 500 rows, and says so under every
chart. It publishes nothing: a run directory carries inputs, hashes, settings
and diagnostics, and an interactive view has none of that.

**Publishing** hands the same settings to `table_charts`, which reads the whole
artifact and records a run. `publishParams` does the translation and has to
respect the engine's rules exactly, because a parameter the engine refuses
fails *after* the job has been queued and run.

### Live analysis

An analysis costs two very different things, and the desktop used to charge for
both on every run. Measured on twenty State of the Union speeches (113,000
words):

| | |
|---|---|
| Parse (spaCy) | **30 s** |
| `readability`, `corpus_statistics` | ~170 ms |
| `collocations`, `tfidf`, `kwic` | 420–510 ms |
| `ngrams`, `sentiment_vader_anew` | 1.5–3.2 s |

The parse is roughly 180× an analysis, so changing an n-gram's `n` from 2 to 3
cost half a minute of reparsing to compute something that takes a sixth of a
second. At that price nobody changes a parameter, which is the opposite of
exploring.

`desktop_backend/live.py` pays for the parse once. `Bench.warm` parses a chosen
set of documents and keeps the token table; `Bench.analyse` runs any analysis
over it. The parse is cached as parquet under `<workspace>/annotations/` — 1.7
MB for that corpus, reloading in 750 ms — so closing the app does not cost the
30 s again. `Session` holds one warm corpus per project behind a lock, because
two simultaneous warms of different selections would parse twice and leave the
wrong one in place.

**A live answer is the published answer.** Both go through the same
`execute(plan, corpus, table)`, so the rows on screen are the rows a publish
would write. `test_a_live_answer_matches_the_run_that_publishes_it` checks it
end to end over HTTP against a real run, because a fast preview that disagrees
with the published result is worse than no preview — someone will act on it.

What live analysis does *not* do is publish: no run directory, no provenance,
no artifact. Publishing submits the ordinary job with the same tool, parameters
and selection.

The cache key covers everything that changes a parse — corpus fingerprint, the
backend that actually ran (not the one requested, since `resolve_pipeline`
falls back), language, model, the installed versions of both, and a
`SCHEMA_VERSION` bumped by hand when the token table's columns change. A cached
parse reused after a model upgrade would answer today's question with last
month's annotations and nothing about the file would say so.

On the front end, `live.ts` handles the two things that break a control-driven
request: it settles for `SETTLE_MS` before asking, so dragging a slider is not
one request per pixel, and it stamps each request with a `questionKey` so a
slow answer for an old setting cannot overwrite the current one.

### Saved views

A view is an arrangement written down: which table, which settings, which
drill-down. Saving one runs nothing and publishes nothing — it is a draft over
an immutable result, which is what lets experimenting happen without
overwriting anything.

`desktop_backend/views.py` is the policy boundary (validation, the publication
contract); `Workspace.save_view` and friends are the storage; `desktop/src/
views.ts` is the one place the desktop's `Settings` and the stored
`ViewSettings` are translated into each other. That translation crosses a
language boundary, so `tests/test_view_parity.py` reads `views.ts` rather than
trusting it — a field added on one side alone would otherwise fail only as a
save button that appears not to work.

Each view records its source's SHA-256. Reopening compares it, so a view whose
artifact has been replaced or removed says so instead of charting whatever is
there now. A view whose run is gone keeps its settings and reports a missing
source: the arrangement someone worked out is worth more than the tidiness of
deleting it with its source. Backups carry views (manifest `version: 2`, and
restore still reads `version: 1`), and restore remaps the job IDs, which are
reissued — a view that came from a run the archive does not contain gets an
empty job ID, which matches nothing, rather than a stale one that might match
something unrelated.

### The stylesheet

`desktop/src/styles.css` is one file, and `tests/test_desktop_styles.py` reads
it. Four invariants, each written after finding it broken:

**Every `var(--x)` names something.** A `var()` with no fallback that resolves
to nothing does not fall back to a default — it invalidates the whole
declaration, silently. Four rules asked for `var(--border)`, which had never
been defined, so `.insight-panel` was drawn with no border at all and three
separator rules drew no separator. Conversely, a fallback beside a token that
*is* defined is dead code: `var(--accent, #2563eb)` promised a blue this
palette does not contain.

**Text meets 4.5:1, icons meet 3:1.** 98 rules did not. Secondary text was
drawn in 67 near-identical greens, most around 2.4:1 — readable if you already
knew what it said — and `--muted` itself was 3.89:1, so the token the app
reaches for when it wants quiet text was the one handing out unreadable text.
Those 67 collapsed into `--muted` and `--icon-muted`; colours outside the
green/grey family kept their hue and were darkened individually.

**No colour is another colour spelled differently.** There were 214, dozens of
them a single hex digit apart. Two within a channel-sum distance of 4 are one
colour and two maintainers. The threshold is deliberately tiny — a palette may
hold two genuinely different greens — and the count has a ceiling so that
adding a hundred one-off shades is a decision rather than a drift.

**No rule sets a property its own later copy overrides.** A block appended near
the end of the file had been correcting the small type by redefining `.badge`,
`.run-name small` and 24 other selectors a thousand lines below the originals.
A quarter of the sheet's smallest font sizes were dead text that a maintainer
would read, believe, and change without effect.

Classes are checked in both directions: every class a component renders has a
rule, and the two deliberate exceptions are named in `UNSTYLED_ON_PURPOSE` with
their reason. The 5px lettering in the overview's illustration is exempt from
the type-size floor only for as long as that illustration stays `aria-hidden`,
which is itself a test.

### Where the preview and the engine disagree

The workbench and `core/viz` are two renderers. Where they differ, the
difference is named on screen rather than discovered afterwards.

Today there is one gap: the workbench can order categories largest-first, and
`ChartSpec` has no field for ordering at all — `prepare_chart_data` sorts x
ascending, always. The same categories are published, in a different sequence.
`publication_gaps` returns that sentence, and `/api/chart-contract` ships it to
the desktop as a lookup table, so the desktop applies no rule of its own. This
is the `/api/tools` lesson: the desktop used to keep its own copy of the tool
names, both copies drifted, and nothing failed.

*Which* categories appear does agree, and one change was needed to keep it that
way: `orderCategories` now breaks ranking ties by category name, which is what
chartspec's `top_n` rule does. Without it a tie at the cut was broken by table
order in the preview and alphabetically in the engine — a disagreement about
the data, not just its arrangement.

Closing the gap properly means giving `ChartSpec` an ordering field and
teaching the renderers to honour it. That touches golden files, the CLI flags
and the chart parity tests, so it is its own slice.

Three things are checked across the language boundary by
`tests/test_chart_kind_parity.py`:

| Checked | Why |
|---|---|
| every live kind is in `CHART_KINDS` | otherwise Publish submits a job the engine refuses |
| the palette matches `plotters._OKABE_ITO` | a preview and its published chart should not disagree on sight |
| `publishParams` never sends `--agg` or `--top-n` where the engine refuses them | this is the defect that made five chart kinds fail from the Visualize dialog |

`chartLayout.looksLikeIdentifier` mirrors `core/insight/profile.py`, including
the part that is easy to get wrong: a **numeric** column is only an identifier
if its *name* says so as well, because every value being distinct is normal in
a score column. Judging on uniqueness alone threw out `Flesch Reading Ease`
along with `Document ID`. Text columns are the other way round — near-unique is
enough — and a text identifier is still a perfectly good axis, which is why
only the measure is filtered.

`measureColumns` then leans on the name rather than only on that mirror,
because the engine reaches "do not chart this" by two routes: `IDENTIFIER` when
an ID column's values vary, and `CONSTANT` when they do not. A one-document run
publishes a table where every column holds one value, so `Document ID` took the
second route, came back "not an identifier", and became the measure the live
chart opened on. For a numeric column the engine's identifier rule requires the
name anyway, so the name test agrees with it and survives a table with one row.

## The words the interface shows

A tool's registry `name` is an identifier (`ngram_cooccurrence`) and a
parameter's `name` is a command-line flag (`--max-df-ratio`). Neither goes in
front of a reader. `core/profiler/labels.py` holds the display text once:

| Table | Answers |
|---|---|
| `TOOL_LABELS` | what a tool is called: `tfidf` -> "Distinctive terms (TF-IDF)" |
| `TOOL_DESCRIPTIONS` | the sentence on its card, written for whoever is choosing it rather than whoever wired it up |
| `PARAM_LABELS` | what a setting is called, by flag: `--sg` -> "Training algorithm" |
| `PARAM_LABEL_OVERRIDES` | where a shared flag means something different in one tool |

`/api/tools` resolves these and ships `label` with every tool and every
parameter. **The desktop never derives a name.** It previously kept its own map
of tool names and turned a flag into a label by replacing underscores with
spaces, and both went wrong without anything failing: three tools added to
`CORPUS_TOOLS` later appeared as `collocations`, `tfidf` and `dispersion` in
lower case, three entries named tools the desktop does not publish, and every
setting was labelled with its flag -- "sg", "op", "col x", "no normalize".

`tests/test_labels.py` fails if a tool or parameter has no label, if a label
names something that no longer exists, or if a label still carries command-line
jargon. The same file covers `desktop_backend.environment.COMPONENT_LABELS`,
which names the installed components Settings lists.

Run states and installed components follow the same rule: `JOB_STATES` in
`desktop_backend/runner.py` is the list of states a job can hold, and
`tests/test_job_states.py` fails if the desktop has no word for one of them.

## Environment

Python 3.12 (see `.python-version`). Tooling: `ruff`, `mypy`, `pytest`.
Quality gate — the same checks `.github/workflows/ci.yml` runs, in the same
order. Anything that passes here passes CI; anything missing from here is a
gate you will only discover after pushing.

```bash
python -m ruff check . && python -m ruff format --check . && python -m mypy core tools app desktop_backend && python -m compileall -q core tools app tests desktop_backend scripts && python -m pytest -m "not model_integration" && python scripts/rc_audit.py
```

The frontend is a separate CI job and a separate local command:

```bash
cd desktop && npm ci && npx vitest run && npm run build
```

`pytest` deselects `model_integration` tests (they need optional NLP packages,
models or lexical data). Run them explicitly with
`python -m pytest -m model_integration`; a passing default gate alone does not
verify those integrations.

### Testing against a real corpus

Some tests need documents nobody would write into a fixture. A hand-built frame
contains the words its author typed, so it cannot show a query and a corpus
disagreeing about `U.S.`, a word position counted from the wrong end of a
sentence, or a parser emitting whitespace tokens the source aligner steps over.
Each of those shipped and was found by pointing the tests at real documents.

`conftest.py` provides:

| Fixture | What it gives you |
| --- | --- |
| `real_documents` | The `.txt` files, sorted |
| `real_corpus` | Them as a `Corpus`, with dates read from the filenames |
| `real_snapshot` | That corpus parsed once per session, with its tokenizer |
| `prime_annotations(root, key, cache)` | Puts the shared parse where a test workspace's own `Bench` will find it |

The corpus defaults to `~/Downloads/POTUS State of the Union 1934-2024` (87
State of the Union addresses, ~600,000 tokens). Point `NLP_SUITE_REAL_CORPUS` at
any folder of `.txt` files to use different material; tests skip when neither
exists. The parse is cached under `%TEMP%/nlp-suite-real-parses` between runs --
override with `NLP_SUITE_REAL_CORPUS_CACHE` -- so the first run pays about a
minute and later ones pay a parquet read.

Write these tests so their assertions come from the corpus rather than from
constants about one folder of files: check a phrase the documents demonstrably
contain, compare two code paths against each other, or recompute a coordinate a
second way. `tests/test_real_corpus_questions.py`,
`tests/test_question_workflow.py`, `tests/test_real_corpus_time.py` and
`tests/test_phrase_matching_options.py` are the worked examples.

Two of those show the shape most clearly. `test_real_corpus_time.py` checks
each row's date against the corpus that supplied it rather than against a list
of years, because a join on the wrong key still fills a `Date` column and a
spot-check of the first row still passes. `test_phrase_matching_options.py`
asks the parse which words share a lemma instead of writing down the endings,
because a hand-written list is a second opinion about the corpus and a second
opinion is not a smaller error than a wrong answer.

### Windows: two temp-directory traps

Both produce failures that look like product bugs and are not.

**Stale temp root.** On this development machine a stale, admin-owned
`%TEMP%/pytest-of-moomi` directory breaks pytest's default temp root
(`WinError 5`), and every test taking `tmp_path` errors. The committed config
carries no machine-specific path; on affected machines pass a local basetemp:

```bash
set PYTEST_ADDOPTS=--basetemp=%TEMP%/nlp_suite_ng_pytest_tmp
```

**Path length.** Keep that basetemp *short* and near the drive root. The
portability and desktop suites deliberately build deep trees and long
filenames (one fixture is a 110-character name), so a basetemp nested more
than a few directories deep pushes paths past Windows' 260-character limit and
produces dozens of unrelated `ValueError`/assertion failures. `C:\pt` works;
a path under `AppData/Local/Temp/<long-session-id>/scratchpad` does not.

Whichever basetemp you choose, its **parent must already exist** — pytest will
not create intermediate directories for `--basetemp`.

## Legacy oracle

The old suite is **not** copied into this repo. It is reached at
`../NLP-Suite-1.6.38/` and treated as read-only: read as a specification to
verify against, never as code to trust or modify.
