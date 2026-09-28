# NLP Suite 0.5.0 plan

Written 2026-09-25 against dev `main` @ `4f77ee3` (0.4.0 released). Every file path, function name, count and table below was read or measured at that commit. When something is a guess, it says "guess" or "decide". Nothing here is built yet.

This plan will be built over several passes (several chat sessions). Each part is written so that a session can start at the top of its part with no memory of the previous session and still do the right thing. That is why it spells out steps a reader might normally infer. **Read section 0 before starting any part.**

---

## How to use this document

1. Read **section 0** (the rules and the measured facts). It takes ten minutes and prevents most wrong turns.
2. Find the next unfinished pass in **section 7** (order of work). Each pass names the part and the sub-sections it covers.
3. Build that pass. Stop at its "Done when" line and check every item in it **in the real app** (section 0.2), not only in tests.
4. Write what you built, and anything that differed from this plan, in the **Progress** section at the bottom of this file. Write new ideas into `docs/internal/PANELS_BACKLOG.md` section 4d (a new 4d-9 heading for 0.5.0), not only in chat.
5. Commit and push to dev `main`.

Words used with a fixed meaning in this plan are defined in the **glossary (section 10)**. The ones most likely to confuse:

- **Document field**: a named fact about one document, such as `Speaker = Harry S Truman` or `Chapter = 3`. Section 1 adds these.
- **Axis**: what a corpus's documents are lined up along: **time** (dates), **order** (chapter 1, 2, 3...) or **none**.
- **Side**: one of the two (or more) document sets being compared on the Compare page.
- **Notebook**: a saved smart script, made of cells.
- **Cell**: one box of Python code or Markdown text inside a notebook.
- **Kernel**: the separate process that runs a notebook's Python cells and remembers their variables between cells.
- **The library**: the Python module scripts import (`nlpsuite`, used as `nlp`). It is how a script reaches the suite's tools.

---

## The goal in one paragraph

0.5.0 makes the suite useful for **any** collection of texts, not only a dated series of speeches. It then adds two ways of working that the suite cannot support today:

1. **Smart scripts.** Describe what you want ("immigration vocabulary over time"), and write or generate a short notebook that calls the suite's own tools. The notebook's tables are charted automatically and saved as CSV for your own matplotlib.
2. **Corpus comparison.** Put two collections side by side, for example a president's State of the Union addresses against their debate answers, and see what carries over and what doesn't.

A few fixes ride along (section 5).

## The three features, and why they are built in this order

| # | Feature | What the user sees | Depends on |
|---|---|---|---|
| 1 | **Documents that carry their own details** (section 1) and **books and long texts** (section 2) | The Corpus page shows each document's date, order, speaker, chapter and any other details, detected from file names, imported from a CSV or typed in. Every figure that drew "over time" can draw "along the chapters" instead. A long book can be split into chapters at import. | nothing |
| 2 | **Smart scripts** (section 4) | A new **Scripts** page. Notebooks with Python and Markdown cells, a documented library over every tool, automatic charts, CSV outputs, templates, a guide to paste into any AI chatbot, and export to `.ipynb`. | 1 (scripts read document details), 5.1 (parse cache) |
| 3 | **Corpus comparison** (section 3) | A new **Compare** page. Pick side A and side B (two projects, or two groups inside one), optionally line them up by speaker or year, and get side-by-side measures, keyness, shared and distinct themes, meaning maps, and pairs of passages where the same idea appears on both sides. | 1 (sides are defined by document details), 5.1 (parse cache) |

**Why details come first.** All three requests are the same missing piece seen from three directions:

- *"The suite is built around our dated corpus"* means documents have no facts except a date parsed from the file name.
- *"Compare SOTU with debates, per president"* needs each document to know its speaker and which collection it belongs to.
- *"A script that plots immigration over time"* needs `nlp.corpus().documents` to hand back those facts as columns.

Build the foundation once, in section 1, and the other two stand on it.

**Why scripts come before comparison** (suggested; Riley can swap them): the script library is the programmable surface over the engine. Comparison is easier to build and to test once `nlp.compare(...)` exists as a library call the Compare page also uses. Two pages sharing one engine call is the pattern the codebase already follows: the live bench and published runs both come from `execute()` (see the `desktop_backend/live.py` module docstring).

---

## 0. Ground rules and the measured facts

### 0.1 Rules (the same as always, plus a few for this release)

- Work on dev `main` in `%USERPROFILE%\personal_projects\NLP-Suite-rework\New_NLP_Suite`. Run `git pull` before you start and `git push` when you finish. Committing and pushing to dev main is allowed without asking.
- Never build or push code in the public repo (RileyK05/NLP-Suite). It changes only through the Publish workflow.
- Before each push, run all of:
  - `ruff check .`
  - `python -m mypy core tools app desktop_backend` (at least on changed modules; say which in the commit if not all)
  - `python -m pytest --basetemp C:/t/<short-name>`. Don't add `-q`; the config already sets it. Delete `C:/t/<short-name>` afterwards. The path must be short: Windows MAX_PATH breaks long ones.
  - `cd desktop && npm test && npx tsc --noEmit`
- New runtime Python dependencies are pinned in `desktop/requirements-runtime-py312.txt` **and** added to `scripts/build_desktop_backend.py` if PyInstaller cannot see them. New JS dependencies go in `desktop/package.json` with an exact or caret version, and the licence is checked (MIT/Apache/BSD only).
- **R3 (reads never write).** Imported documents are immutable copies. Splitting a book, cleaning text or editing details never rewrites `projects/<id>/corpus/*.txt`. Derived text is new files, recorded as derived.
- **R8 (envelopes).** Anything a script or comparison produces reaches Past runs, Explore and the panels as an ordinary run directory with a `result.json`, written through `core.io.writer.OutputWriter`. No new "kind of result" that only one page can read.
- **Security test.** `tests/test_security.py` fails the build on `exec`/`eval` anywhere, subprocess imports outside its allowlist, and network imports outside its allowlist. Section 4 needs `exec` (and no network: the app has no built-in AI, see 4.8). It is added **to the allowlist with a written control**, never by routing around the test (for example with `runpy`, `compile()` + `FunctionType`, or `importlib` tricks). The test exists so that a reviewer sees every such use.
- **Don't delete Riley's things.** Never kill Riley's processes, never delete `.server.lock`, never delete `out/desktop-workspace`. Real-app tests run on a **copy**.
- Use the Edit or Write tools, not shell heredocs, for any text containing backslashes (`\n`, `\t`, `\b`). The shell here turns them into real characters. After editing, scan the file for control characters.
- Clean up every server, worker and temp dir you start.

### 0.2 How "done" is judged (Riley's standing preferences, made concrete)

1. **Reachable from the sidebar or it is not built.** A new page is a new `navigation` entry in `desktop/src/App.tsx` (the `navigation` array, around line 133). `tests/test_desktop_navigation.py` checks reachability; extend it.
2. **Run the real app on real data.** Copy `out/desktop-workspace` to a short path (for example `C:/t/ws`), skipping `projects/*/runs/` if space matters. Start `python -m desktop_backend.server --data-dir C:/t/ws` (see `scripts/desktop_preview.py` / `scripts/interactive_stack.py` for the exact flags and the bearer token), drive it with Playwright, and look at the screenshots. Kill the server afterwards.
3. **"Technically correct but tells me nothing" is a bug.** For every new figure, look at it on the real SOTU project (87 speeches) and, for sections 2 and 3, on a real book and a real second corpus. If a figure cannot say anything on that data, it refuses with a sentence saying why and what would help.
4. **One broken example means the whole class is broken.** Each fix in this plan names its class and a **coverage test** that walks every member of the class (every panel, every tool, every document field), not just the example that was noticed.
5. **Prove the cause before claiming the fix.** Reproduce, change one thing, A/B. For performance claims (parse cache, BERT speed), measure before and after on the same machine, interleaved.

### 0.3 What exists today (measured at `4f77ee3`, the facts this plan stands on)

Each fact has where it was read, so a later session can re-check it.

**Documents and dates**

| Fact | Where |
|---|---|
| A project's `documents` table has `id, project_id, name, stored_name, sha256, bytes, words, created, source_name, source_sha256, import_diagnostics, trashed`. **No date, no author, no order, no group, no free-form details.** | `desktop_backend/store.py` `Workspace.__init__` |
| `core.io.reader.Document` has `doc_id, path, text, date, sha256, source_id, label`. **`date` is the only structured fact.** | `core/io/reader.py` |
| A document's date comes only from its **file name**, via `date_from_filename()` (regex for `YYYY-MM-DD`, `YYYYMMDD`, `MM-DD-YYYY`, `YYYY-MM`). There is no way to set a date any other way. | `core/io/reader.py`, used in `desktop_backend/runner.py::run_job`, `desktop_backend/selection.py::_document_date`, `desktop_backend/server.py`, `core/file_ops/filenames.py` |
| A run's document selection can filter **only** by explicit ids and a date range (`date_from`, `date_to`, `include_undated`). | `desktop_backend/selection.py::CorpusSelection` |
| The executor adds `Date` and `Year` columns to every per-document result table (joined on `Document ID`) **if** the corpus is dated. Nothing else about a document is ever joined on. | `core/profiler/executor.py::_dated`, `_with_dates` |
| "Periods" for meaning-over-time are decades (or years if one decade), from dates only. | `core/profiler/executor.py::_periods` |
| `lexicon_series` can facet by `year`, `decade`, `document` or a regex `pattern` over names. | `core/analysis/lexicon_series.py::facet_labels`, `BY_CHOICES` |
| `keyness` splits the corpus into two groups **by a regex over document names** (group A matches, group B is the rest). | `core/analysis/keyness.py` |
| Figure grouping choices are hard-coded: `GROUPINGS = ("none", "year", "decade", "speaker")`. `speaker` is **parsed from the SOTU file-name pattern** `date_speaker_kind.txt`, so it is empty for anything else. | `core/viz/panel_helpers.py` (`GROUPINGS`, `speaker_of`, `group_of`) |

**Figures on an undated corpus (measured with `core.viz.panels.PANELS`)**

- **165 panels exist; 61 require a `Date` or `Year` column** and so refuse on an undated corpus.
- **13 tools have no figure at all on an undated corpus**, because *every* panel they have requires a date: `corpus_statistics`, `gender_annotator`, `gender_guess`, `kwic`, `lexical_diversity`, `narrative`, `ngram_viewer`, `readability`, `sentence_complexity`, `sentiment_swn_hedono`, `shapes`, `text_statistics`.

  Note `narrative` and `shapes`: the **story-shape** tools, the ones books need most, cannot draw for a book.

- Tools with some dated panels: `bert_topics 1/2`, `date_annotator 1/2`, `ner 1/2`, `nominalization 2/3`, `quote_annotator 2/3`, `sentiment_neural_* 3/5` (x4), `sentiment_vader_anew 6/9`, `shape_hc 1/2`, `verb_analysis 3/6`.
- Re-measure with:
  ```
  python -c "from core.viz.panels import PANELS; import collections; a=collections.Counter(p.tool for p in PANELS); d=collections.Counter(p.tool for p in PANELS if {'Date','Year'} & set(p.requires)); [print(t, d[t], a[t]) for t in sorted(a) if d[t]]"
  ```

**Projects, runs and comparison**

- **One project = one corpus.** Nothing reads documents from two projects at once. Jobs, views, questions and runs are all keyed by `project_id`.
- The existing "compare" (`POST /api/projects/{id}/compare`, `core/compare.py`) is a **golden diff of two run directories** (row by row, "did the numbers change"). It is not corpus comparison. **Do not reuse the name `compare` for new modules.** Section 3 uses `contrast`.
- Backups (`desktop_backend/archives.py`) are at **manifest version 4** (v2 added views, v3 questions, v4 trashed items). Any new table means version 5, and restoring v1–v4 archives must keep working.

**Running code**

- The desktop engine is a **frozen PyInstaller build** (`scripts/build_desktop_backend.py`). It ships `core`, `desktop_backend`, numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, plotly, wordcloud, pyarrow, onnxruntime, tokenizers, and (with the parser) spaCy, gensim, nltk and vaderSentiment. It **excludes** `tools`, `app`, `scripts`, `tests`, `IPython`, `torch`, `transformers`, `stanza`, `streamlit` and `pytest`.
  - So: **there is no Jupyter/IPython kernel** in the app, and user code can only use what is bundled. matplotlib, pandas, numpy, seaborn, scipy and sklearn are all there.
- Jobs run in a **warm worker subprocess**: `desktop_backend/runner.py`, `worker_command()` = the engine executable with `--data-dir <root> --worker`, a JSON-lines protocol over stdin/stdout, one job at a time. This is the pattern the script kernel copies (section 4.4).
- `tests/test_security.py::test_no_dynamic_code_execution` bans `eval`/`exec` **everywhere, with no allowlist**. Section 4 must extend the test deliberately (4.4.6).
- **Published jobs never use the parse cache.** `runner.run_job` calls `resolved_pipeline.parse(corpus)` directly. Only the live bench (`desktop_backend/live.py`, `Bench`, `Annotations`) reads and writes `annotations/<key>.parquet`. That explains backlog 4e ("each desktop job showed 'Parsing English documents' for minutes"). Scripts and comparisons parse a lot, so **section 5.1 fixes this first**.

**The reference script (what "smart scripts" must be able to reproduce)**

- `scripts/build_immigration_evidence.py` (295 lines) is the script behind the immigration figure. It:
  1. reads 87 speeches;
  2. counts 15 lexicon terms per speech (per 1,000 alphabetic tokens);
  3. takes equal-weight yearly means (the two 1953 speeches are averaged, not summed);
  4. takes a five-observation centred mean;
  5. plots points, yearly means and the trend line;
  6. pulls seven dated passages with the sentence before and after;
  7. writes CSVs and a JSON summary with source hashes.
- Its outputs still exist in the old script's output folder (`immigration_evidence/`) (`per_speech_rates.csv`, `annual_means.csv`, `immigration_evidence.csv`, `immigration_summary.json`, `immigration_time_series.png`). **They are the A/B target for section 4's acceptance test.**

**Real data available right now** (in `out/desktop-workspace/workspace.sqlite3`, read-only query):

| Project | Documents | Words | Notes |
|---|---|---|---|
| SOTU | 87 | 516,987 | `1934-01-03_franklin d roosevelt_sotu.txt` … 2024 |
| Inaugural speech | 31 | 66,160 | `1901-03-04_william mckinley_ina.txt` … |
| SOTU (second) | 68 | 354,563 | **mixes `_ina` and `_sotu` files**, e.g. `1965-01-20_lyndon b johnson_ina.txt` next to `1966-01-12_..._sotu.txt` |
| Launch check | 1 | 16 | ignore |

- **SOTU vs Inaugural** is a real cross-project comparison with overlapping speakers, available today. Use it as section 3's first real test.
- The mixed 68-document project is the **within-project** version of the same question: `kind = ina` vs `kind = sotu`, detected from the file names.
- There is **no book** in the workspace; section 2.8 says where to get two. Debates appear in this plan only as an **example** of a second collection (Riley isn't building a debates corpus); the real comparison tests use SOTU vs Inaugural (3.10).

**Backlog items this plan absorbs** (from `docs/internal/PANELS_BACKLOG.md`), so they are not built twice:

- 4d-4 **"Named cohorts instead of regex groups"**: becomes document fields (section 1).
- 4d-4 and 4d **"Multi-run panels"**: becomes panels over a comparison's joined tables (section 3.7).
- 4d-3 **"Decoders (small local LLMs), as suggesters only"**: becomes section 4.8: a guide the user pastes into their own chatbot, which writes code the user reads and runs, never findings. No AI inside the app.
- 4d-5 **"Transcript annotations reach the NER tagger"**: becomes section 5.2.
- 4e **"runner may not be reusing the annotation cache"**: confirmed, becomes section 5.1.
- 4d-7 **"Embeddings over time" / "Cache sentence vectors"**: becomes section 5.4. Comparison and scripts both embed the same corpus repeatedly.

---

## 1. Documents that carry their own details (the generalizable pattern)

### 1.1 The idea, stated plainly

Today a document is a file name plus text, and "the suite understands time" means "the suite can read a date out of a file name".

After this section, **every document has a small table of details** (its *fields*). The corpus as a whole has an **axis**, which is how its documents line up. Every tool and figure that used to say "Date" asks the axis instead.

Examples of the same machinery serving different corpora:

| Corpus | Fields per document | Axis | What "over time" figures become |
|---|---|---|---|
| State of the Union (87) | Date (file name), Speaker (file name), Kind = sotu (file name), Party (imported CSV) | **time** | unchanged: over time, by decade, by speaker |
| Mixed SOTU + inaugurals (68) | Date, Speaker, Kind = sotu/ina | **time** | same, plus "group by Kind" everywhere |
| One novel split into 61 chapters | Work = Pride and Prejudice, Chapter = 1..61, Volume = I/II/III, Title | **order** (chapter number) | "across the chapters": sentiment arc, readability by chapter, characters by chapter |
| Five novels, whole files | Work, Author, Year published | **time** (publication year) or **none** | by author, by work |
| Interview transcripts | Participant, Group = control/treatment, Session = 1..4 | **order** (session) or **none** | by group, by session |
| Tweets exported from a CSV | Date, Account, Likes | **time** | over time, by account |
| A folder of essays with no metadata | (none) | **none** | per-document rankings and distributions; figures that need an axis say "add a date or an order on the Corpus page" |

**Rule:** time-series corpora must stay exactly as easy as today, with **zero** extra steps for SOTU-style file names. Everything new is auto-detected or optional.

### 1.2 Vocabulary (use these exact names in code, UI and docs)

- **Field**: a named document detail. Name: 1–40 characters, letters/digits/spaces/`-_`, case-insensitive unique per project, displayed as written. Value: text up to 500 characters; empty means unknown.
- **Built-in fields**: `Date` (ISO `YYYY-MM-DD`, or `YYYY-MM`, or `YYYY`) and `Order` (a whole number, or a decimal such as 1.5). They are fields like any other in storage, but the engine knows what they mean.
- **Field source**: where a value came from, one of `filename` (detected), `csv` (imported), `split` (set when a book was split), `user` (typed in the grid). Shown in the UI as a small tag, because "Date from file name" and "Date typed by you" deserve different trust. When two sources disagree, **precedence is `user` > `csv` > `split` > `filename`**, and the grid shows the overridden value in a tooltip.
- **Axis**: per project, one of:
  - `time`: uses `Date`;
  - `order`: uses `Order`;
  - `none`.

  Default: `time` if at least 2 documents have a Date, else `order` if at least 2 have an Order, else `none`. The Corpus page lets the user choose. A tool run records the axis it used in its envelope params, so a result can be reproduced after the choice changes.
- **Axis noun**: what one step along the axis is called in figure titles and axis labels. `time` gives "Year"/"Date". `order` gives the project's **order label**, a text setting defaulting to "Order", which the book splitter sets to "Chapter" and a user can set to "Session", "Episode" and so on.
- **Periods**: the axis cut into a few buckets for before/after comparisons (meaning over time, topic prevalence).
  - Time: decades, or years when all documents share one decade (today's `_periods` rule, unchanged).
  - Order: the values of a chosen grouping field if one exists (for example `Volume`); otherwise equal-count blocks. Default 4 blocks ("chapters 1–15, 16–30, ..."), never fewer than 2 documents per block.

### 1.3 Storage (`desktop_backend/store.py`)

Add one table. Keep it in the `Workspace.__init__` `executescript` like the others, with the same "add if missing" migration style.

```sql
CREATE TABLE IF NOT EXISTS document_fields (
    project_id  TEXT NOT NULL REFERENCES projects(id),
    document_id TEXT NOT NULL REFERENCES documents(id),
    name        TEXT NOT NULL,      -- as displayed, e.g. "Speaker"
    key         TEXT NOT NULL,      -- casefolded name, for uniqueness and lookup
    value       TEXT NOT NULL,
    source      TEXT NOT NULL,      -- filename | csv | split | user
    updated     TEXT NOT NULL,
    PRIMARY KEY (document_id, key, source)
);
CREATE INDEX IF NOT EXISTS document_fields_project ON document_fields(project_id, key);
```

And one table for per-project settings (axis, order label, file-name template, text-cleaning options from 5.2):

```sql
CREATE TABLE IF NOT EXISTS project_settings (
    project_id TEXT PRIMARY KEY REFERENCES projects(id),
    settings   TEXT NOT NULL DEFAULT '{}',   -- JSON, validated by a pydantic model
    revision   INTEGER NOT NULL DEFAULT 1,
    updated    TEXT NOT NULL
);
```

**Why a separate row per source** rather than one value: so "reset to the file name's date" is a delete of the `user` row, not a guess, and so the grid can show where a value came from. The effective value is computed by precedence (1.2).

New `Workspace` methods, each with a docstring in the file's existing style (say *why*, not only what):

- `fields(project_id) -> dict[document_id, dict[name, FieldValue]]`, where `FieldValue = {value, source, overridden: [{value, source}]}`.
- `field_names(project_id) -> list[{name, count, sources, sample_values}]`, for pickers.
- `set_fields(project_id, rows: list[{document_id, name, value, source}], expected_revision)`, in one transaction. Refuse trashed or archived projects, like `import_document`. An empty value with source `user` means "clear my override". **Bump the project settings revision**, because the corpus's meaning changed and cached glance and comparison results must know (1.9).
- `delete_field(project_id, name)`: removes every source of that field.
- `settings(project_id)` / `save_settings(project_id, body, expected_revision)`.

Also:

- `purge_document` and `purge_project` must delete the new rows (they "intentionally do not cascade"; add the deletes in dependency order next to `views`/`questions`).
- **Limits**: 40 fields per project, 500-character values, 2,000 documents (the existing `MAX_DOCUMENTS`), so at most 80,000 rows. Name them as constants next to `MAX_DOCUMENTS`.

### 1.4 Where fields come from

#### 1.4.1 File-name detection (automatic, zero clicks for SOTU)

New module **`core/io/filename_fields.py`** (pure, no I/O, R1 and R6):

- `detect_template(names: Sequence[str]) -> Template | None`. Finds the common pattern of a set of file names. It must recognise, at minimum:
  - `1934-01-03_franklin d roosevelt_sotu.txt` → `{Date}_{Speaker}_{Kind}`. This is today's `speaker_of` rule in `core/viz/panel_helpers.py`, moved here so there is one implementation (**R10**).
  - `1934_roosevelt.txt` → `{Date}_{Speaker}`
  - `ch01_the_beginning.txt`, `Chapter 1.txt`, `01 - Title.txt` → `{Order}_{Title}`
  - `austen_pride_1813.txt` → `{Author}_{Title}_{Date}`, but only when every name has three underscore parts and the last is a year. Otherwise leave it undetected rather than guess.
  - Names with no common structure → `None`. **Never guess a field from one document**: a template must fit at least 80% of names, and the ones it doesn't fit are listed.
- `apply_template(name, template) -> dict[field, value]`.
- `Template` is a small frozen dataclass: `parts: tuple[str, ...]`, `separator: str`, `fits: int`, `misses: tuple[str, ...]`. It has `describe()` for the UI: "Date, then Speaker, then Kind, separated by _ (fits 87 of 87)".
- Date parsing still goes through `date_from_filename` (one date rule, R10). Order parsing accepts `1`, `01`, `ch1`, `chapter 1`, `I`..`L` (roman numerals, only when the whole part is roman).

When it runs:

- **On import** (`Workspace.import_document`, and the batch import endpoints). After a batch finishes, re-detect over **all** the project's names (one name alone can't reveal a template) and write `filename` rows. This is cheap: regexes over at most 2,000 names.
- **Migration for existing projects**: on server start, any project that has documents but no `filename` rows gets detection run once. Record that in `project_settings.settings.detected_template` so it doesn't re-run every start.
- The user can **edit the template** on the Corpus page (1.8): rename a part ("Kind" → "Genre"), drop a part, or turn detection off. Saving re-applies it.

**Coverage test** (`tests/test_filename_fields.py`): feed every document name in `tests/fixtures/` corpora plus the three real project name lists (copy the names, not the texts, into a fixture file `tests/fixtures/names/real_projects.txt`). Assert:

- SOTU names give Date/Speaker/Kind for 87 of 87, and the mixed project gives Kind ∈ {sotu, ina};
- no template is returned for a random-names list;
- and **`speaker_of` in `panel_helpers` returns the same speaker as the new detector for every name** (so moving the rule changes no figure).

#### 1.4.2 CSV import ("I already have a spreadsheet of this")

- Corpus page → "Import details from a spreadsheet". The user picks a CSV/TSV (also XLSX if `openpyxl` is already bundled; check first, don't add a dependency for this).
- One column must match document names. Match on exact name, then on name without extension, then case-insensitively. Report every unmatched row and every document with no row.
- The other columns become fields with source `csv`. There is a preview step (first 10 rows, detected column types, match counts) before anything is written.
- Engine: `core/io/metadata_table.py::match_metadata(frame, names) -> Result[MatchReport]` (pure), plus endpoint `POST /api/projects/{id}/fields/import` with a `dry_run` flag. The UI calls it twice: preview, then apply.
- Columns named `date`/`year`/`published` map to `Date`; `order`/`chapter`/`number`/`index` map to `Order`. Ask in the preview rather than silently mapping: the preview shows the proposed mapping with a dropdown per column.

#### 1.4.3 Typing them in

The document details grid (1.8) is editable. It is for fixing the three documents whose file names were odd, not for typing 2,000 rows.

#### 1.4.4 Set by the book splitter

Section 2.3 writes `Work`, `Order`, `Chapter`, `Title` and (when found) `Part`/`Volume`/`Book` with source `split`.

### 1.5 The engine side: `Document` and the runner

1. `core/io/reader.py::Document` gains `fields: Mapping[str, str] = field(default_factory=dict)` (use a frozen mapping: `types.MappingProxyType` or a tuple of pairs, so the dataclass stays hashable and frozen), and a property `order: float | None` that reads `fields["Order"]`. **`date` stays** as the parsed date; it is filled from the effective `Date` field, which is the file-name date when nobody overrode it. So every existing `doc.date` reader keeps working unchanged.
2. `corpus_fingerprint` does **not** change (it's about text identity and keys the parse cache; details don't change a parse). Add a separate `details_fingerprint(docs)` (hash of sorted field pairs + axis), used by caches of results that *do* depend on details (glance, comparisons, scripts).
3. `desktop_backend/runner.py`:
   - `Runner.submit` freezes each document's **effective fields** into `request["documents"][i]["fields"]`, and the project's axis/order label into `request["axis"]`. It must be frozen at submission for the same reason documents are: a run records what it read.
   - `run_job` builds `Document(..., date=<from fields>, fields=item.get("fields", {}))`. **Old queued requests without `fields` must still run**: fall back to `date_from_filename(name)` exactly as today.
4. `desktop_backend/selection.py`: `_document_date` reads the effective `Date` field (not the name). `document_metadata` adds `fields`. `CorpusSelection` gains:
   - `fields: dict[str, list[str]] | None`: keep documents whose field value is in the list, ANDed across fields. The value `"(empty)"` selects documents with no value.
   - `order_from`, `order_to`: floats, same semantics as dates, with `include_unordered`.

   The validators follow the existing style: strict, reject unknown field names with a message listing the known ones.
5. **One function for "a document's date"**: every place that calls `date_from_filename(Path(name))` on a *project document* switches to `document_date(fields)` in the new `core/io/document_fields.py`. Grep: `server.py`, `runner.py`, `selection.py`, `scripts/interactive_stack.py`. `date_from_filename` itself stays; it is the detector.

### 1.6 The engine side: the executor and the axis

New module **`core/corpus_axis.py`** (pure):

```python
@dataclass(frozen=True, slots=True)
class Axis:
    kind: Literal["time", "order", "none"]
    noun: str            # "Year" | "Chapter" | "Session" | ...
    positions: dict[str, float]   # doc_id -> position (decimal year, or order value)
    labels: dict[str, str]        # doc_id -> what to print ("1934-01-03", "Chapter 3")
    def periods(self, *, grouping_field: dict[str, str] | None = None, blocks: int = 4) -> dict[str, str]: ...

def axis_of(corpus: Corpus, kind: str | None, noun: str = "") -> Axis: ...
```

- `axis_of` with `kind=None` applies the default rule from 1.2.
- `periods()` for `time` is exactly today's `_periods` (decade, else year, else empty when fewer than 2). Keep the existing `_PERIODS_NEEDED`. For `order` it is the grouping field's values, else equal-count blocks, labelled "Chapters 1–15".

`core/profiler/executor.py` changes:

1. `_with_dates(frames, corpus)` becomes `_with_details(frames, corpus, axis)`. For every per-document frame (same rule as today: has `Document ID`, doesn't already have the column):
   - `Date` and `Year` exactly as today (**golden outputs for dated corpora must not change**; run the golden tests before and after);
   - `Order` when the corpus has any Order values;
   - **one column per field**, named as the field. If a tool's frame already has a column of that name (a field called `Tokens`, say), name it `Doc: Tokens` instead and add a diagnostic once per run (`DETAILS_COLUMN_RENAMED`);
   - `Position` (float) and `Position label` (text), from the axis, so a figure has one column to put on x whatever the axis is. **Only when the axis is not `none`.**

   Column order: after `Document` (as `_dated` does now), then Date, Year, Order, Position, Position label, then fields in name order.
2. `_periods(corpus)` becomes `axis.periods(...)`. The `word2vec_bert` adapter (`_adapt_word2vec_bert`), and any other caller, take the axis from `BatchContext`.
3. `BatchContext` gains `axis: Axis | None`. `execute(...)` gains an `axis=` keyword (default: `axis_of(corpus, None)`). The runner, the live bench (`desktop_backend/live.py`) and glance pass the project's axis.
4. `_adapt_lexicon_series`: `facet_labels` gains `by` values:
   - `order` (each order value, e.g. each chapter);
   - `period` (the axis periods);
   - `field:<Name>` (group by a field's values).

   Keep `year`, `decade`, `document`, `pattern` (existing saved runs use them).
5. `_adapt_keyness`: new params `group-field` and `group-a` / `group-b`. For example: field `Kind`, A = `sotu`, B = `ina`, or B empty to mean "all others". When `group-field` is set, `group-pattern` is ignored. `core/analysis/keyness.py::keyness` gains a `groups: Mapping[str, str] | None` argument (document name → "A"/"B"/excluded) as an alternative to the regex. Keep the regex path. The output column names `Freq Group A (pattern docs)` change to `Freq A: Kind = sotu` when fields are used, so a table says what it compared.
6. **`ToolSpec` params that should gain field support**, found by grepping for `group-pattern`, `by`, `date` in `core/profiler/registry.py`. At minimum: `keyness`, `lexicon_series`, `ngram_viewer`, `stats_groups`, `stats_trends`, `phrase_distribution`. Each change is additive. Update `core/profiler/labels.py` and `desktop/src/toolGuides.json` for each.

### 1.7 The figures: "over time" becomes "along the axis"

This is the **class fix** for "the suite only works for time series". Do it for all 61 panels, not for one.

1. **`core/viz/panel_helpers.py`**:
   - `dated(frame)` stays (used widely), and a new `positioned(frame) -> tuple[pd.Series, AxisInfo]` reads `Position`/`Position label` when present, else falls back to `dated()`. `AxisInfo = (kind, noun, is_time)`.
   - `GROUPINGS` stops being a constant. `groupings(frame) -> tuple[str, ...]` returns `none`, then `year`/`decade` if dated, then `period` if the axis has periods, then **every field column present in the frame**. `group_of(document, when, by)` gains `row` so it can read a field column.
   - `speaker_of` becomes a thin wrapper over `core/io/filename_fields.py`, for names with no `Speaker` field (old run tables). Mark it as the fallback.
2. **Panel `requires`**: introduce a virtual requirement token `AXIS = "@axis"` in `core/viz/panelspec.py`. `panels_for_envelope` / the requirement check treats `@axis` as satisfied when the frame has `Position`, **or** `Date`, **or** `Year`. Then, for each of the 61 panels:
   - replace `DATE` / `"Year"` in `requires` with `AXIS` where the panel only needs *an* ordering (trend lines, rolling medians, arcs, heat strips by period);
   - keep `DATE` where the panel is genuinely about calendar time (for example a "gap in years between speeches" annotation, `_LINE_GAP_YEARS`), and give it an order-axis twin or a refusal that says why;
   - change the builder to read x from `positioned()` and label the axis with the noun. Titles are built from the noun: "Readability over time" becomes `f"Readability {along(axis)}"`, where `along()` returns "over time", "across the chapters", "across the sessions" or "by order". **Every title string that contains "over time", "by decade", "per year" or "Year"** in `core/viz/panels*.py` must go through the helper. Coverage test in step 4.
3. **The `_over_time` family in `core/viz/panels_document_measures.py`** (readability, lexical diversity, corpus/text statistics, sentence complexity, verb analysis, nominalization) is one factory (`_factory`, `_over_time`, `_by_group`). Generalizing the factory fixes about 25 panels at once. `_LINE_GAP_YEARS` (the gap that breaks a trend line) becomes "the gap in axis units that breaks a line": 5 years for time; for order, a missing chapter number breaks the line.
4. **Coverage test** `tests/test_panels_axis.py`. This is the "fix the class" test. For **every** panel in `PANELS` whose `requires` contained `Date` or `Year` at `4f77ee3` (freeze that list of 61 names in the test file, with a comment saying where it came from), build it from its fixture three ways:
   - (a) the existing dated fixture: **the output must be identical to before**. Compare `PanelData` marks and titles against a snapshot taken at the start of the pass;
   - (b) the same fixture with `Date`/`Year` removed and `Order`/`Position` added (1..N, noun "Chapter"): the panel must draw, and its title and x label must say "chapter", not "year" or "time";
   - (c) with no axis at all: the panel either draws something that needs no axis, or refuses with a message containing the words "Corpus page" and ("date" or "order"). A refusal that says only "missing column Date" fails the test.

   Also assert **no tool is left with zero drawable panels** in case (b): the 13 tools listed in 0.3 are named in the assertion message.
5. `core/insight/recommend.py` / `core/insight/profile.py`: a `Position` column gets the role "axis" and is preferred as x the way `Date` is today. Check `ColumnRole`.
6. Desktop: `desktop/src/panelLayout.ts`, `panelTicks.ts` and `chartLayout.ts` format a time axis with year ticks. An order axis needs integer ticks labelled with the noun ("Ch. 5"). Grep for `decimalYear`/`year` in `desktop/src/*.ts`. The **parity test** between Python shapes and TypeScript (`tests/test_panel_shapes.py` and friends) must be extended for the new axis kind.

### 1.8 The UI: the Corpus page

`desktop/src/App.tsx` renders the Corpus page (`page === "corpus"`). Add, in this order on the page:

1. **A corpus shape line** at the top: "87 documents · dated 1934–2024 · lined up by time · 3 details: Speaker, Kind, Party". It is clickable to change the axis (a small menu: Time / Order / None, plus an order-label text box when Order is chosen).
2. **A "Document details" grid** replacing (or next to) the document list. Columns: Name, Date, Order (only if any), then each field. Cells are editable. A source tag appears on hover ("from file name", "from spreadsheet", "set by you"). It sorts by any column. Keep it lightweight: a plain table with inline inputs; **no data-grid library** unless a plain table measures too slow at 2,000 rows (measure first).
3. Buttons above the grid:
   - **"Detect from file names"** opens a small dialog showing the detected template, how many names it fits, the misfits, and editable part names;
   - **"Import from a spreadsheet"** (1.4.2);
   - **"Add a detail"** (a column with empty values);
   - **"Split a long document into chapters"** (section 2), shown only when a document is over about 20,000 words or the corpus has one document.
4. **`desktop/src/CorpusScope.tsx`** (the document selection used before runs) gains field filters: a chip per field with a multi-select of values, and an order range when the axis is order.

API (`desktop_backend/server.py`, a new router file `desktop_backend/fields.py` to keep `server.py` from growing past 1,423 lines):

- `GET /api/projects/{id}/fields` → `{names: [...], documents: {doc_id: {field: {value, source, overridden}}}, settings: {...}, revision}`
- `POST /api/projects/{id}/fields` → set values (body: rows + expected_revision)
- `POST /api/projects/{id}/fields/detect` → `{template, fits, misses, preview: [{name, fields}]}`; with `apply: true` it writes
- `POST /api/projects/{id}/fields/import` → preview or apply a CSV (1.4.2)
- `DELETE /api/projects/{id}/fields/{name}`
- `GET/POST /api/projects/{id}/settings` → axis, order label, text cleaning (5.2)

Contract types in `desktop/src/api.ts`. If `scripts/gen_contract_ts.py` generates `desktop/src/contract.ts` from pydantic models, add the models there and regenerate. Don't hand-write a second copy.

UI tests: `desktop/src/CorpusDetails.test.tsx` (new component `CorpusDetails.tsx`). It covers editing a cell, the source tag, detect preview/apply, import preview with an unmatched row, and the axis switch changing the shape line.

### 1.9 What else must learn about details

- **Glance** (`desktop_backend/glance.py`, `core/insight/glance.py`): the cache key gains `details_fingerprint`. Its summary sentences say "across 61 chapters" rather than "from 1934 to 2024" when the axis is order. Its sentiment step uses the axis (it currently says "over time if dates exist").
- **Live bench** (`desktop_backend/live.py`): `Bench.analyse` passes the axis to `execute`. The warm key doesn't change (details don't change the parse).
- **Saved questions / phrase distribution** (`core/research/phrase.py`, 12 date references): `phrase_time.csv` becomes positioned on the axis. Its panel titles use the noun.
- **Backups** (`desktop_backend/archives.py`): manifest **version 5** adds `document_fields` and `project_settings`. `_validate_*` for both. Restoring v1–v4 runs file-name detection on the restored project (as the migration does). Test: restore each of the existing archive fixtures (grep `tests/` for `archive`) and assert the SOTU fixture gets Speaker/Kind.
- **Exports** (`/jobs/{id}/export`): the input manifest `desktop_inputs.csv` (`INPUT_MANIFEST` in the runner) gains the field columns, so an exported run says what each document was.

### 1.10 Done when (section 1)

- [x] On a copy of the real workspace, opening the SOTU project shows Speaker and Kind detected for 87/87 with no clicks, and axis = time.
- [x] Every golden/snapshot test for dated output passes **unchanged** (no figure of the SOTU corpus changed).
- [x] `tests/test_panels_axis.py` passes: all 61 formerly date-only panels draw on an order axis with chapter wording, and refuse helpfully on no axis.
- [x] In the real app, keyness on the mixed 68-document project can be run as "Kind = sotu vs Kind = ina" without typing a regex, and the result's columns name the groups.
- [x] A spreadsheet of parties (make one by hand for 10 presidents) imports, matches, and "group by Party" appears in readability's group-by menu.
- [x] A backup made before this change restores and gets its details detected.
- [x] Filter by Speaker in the selection before a run works, and the run's `desktop_inputs.csv` lists the fields.

---

## 2. Books and long texts (non-time-series corpora)

### 2.1 The problem

A novel imported today is **one document**. Every per-document figure then has one bar or one point. "Story shape", character arcs and sentiment across the book are the questions people bring a novel to the suite for, and the tools that answer them (`narrative`, `shapes`, `shape_*`) have no drawable figure without dates (0.3).

Two things are missing: **cutting a long text into its natural sections**, and **an axis that isn't time** (section 1).

### 2.2 Two ways to look inside one text (both are needed)

1. **Sections as documents** (the main path). Split the book into chapters at import. Each chapter is a document with `Order`, `Chapter`, `Title` and `Work` fields. Every existing tool then works per chapter, on the order axis, for free.
2. **Position inside a document** (already exists). `phrase_distribution` has `position-bins` (sections per document), and `core/analysis/dispersion.py` measures spread. These answer "where in the text", without splitting. Keep them, and make sure their figures use the word "position" and the order axis's noun consistently. **No new code unless 2.6's review finds a gap.**

### 2.3 The splitter

New pure module **`core/file_ops/sections.py`** (beside the existing `splitter.py`, which only does fixed-size and delimiter splits; reuse its `Result` conventions):

```python
@dataclass(frozen=True, slots=True)
class Section:
    order: int            # 1-based position in the book
    title: str            # "Chapter 3: The Ball", or "Section 3" when no heading
    level: str            # "chapter" | "part" | "book" | "volume" | "act" | "scene" | "section"
    start: int            # character offset in the source text (inclusive)
    end: int              # exclusive
    parents: tuple[tuple[str, str], ...]   # e.g. (("Volume", "II"),)

def detect_sections(text: str, *, rule: str = "auto") -> Result[SectionPlan]: ...
def front_and_back_matter(text: str) -> tuple[int, int]: ...   # Gutenberg header/footer bounds
```

**Rules** (`rule=`), tried in this order by `auto`. Keep the first that yields at least 3 sections of at least 300 words each and no section over 60% of the text:

1. `headings`: lines matching (case-insensitive, whole line, optional trailing punctuation):
   - `CHAPTER <n>` / `Chapter <n>` / `CHAPTER <ROMAN>` / `Chapter the First`, with an optional title after `.`/`:`/`—` or on the next non-empty line;
   - `BOOK <n>`, `PART <n>`, `VOLUME <n>`, `ACT <n>`, `SCENE <n>`: these become `parents` of the chapters that follow;
   - Markdown `#`/`##` headings;
   - a line that is only a roman numeral or a number (common in older novels). Only accept this when there are at least 5 such lines, spaced at least 300 words apart.
2. `blank-gap`: three or more consecutive blank lines, or form feeds.
3. `words`: fixed blocks of N words (default 2,000). **The fallback always works but is labelled "not the author's chapters"** in the section titles ("Words 1–2,000") and in a diagnostic, so nobody reads block 7 as chapter 7.

The user can also choose a rule explicitly, or give their own heading pattern. A literal prefix is the default input; a "use a regular expression" checkbox makes it a regex. Validate the regex with the existing `re.error` handling pattern from `keyness.py`.

**Front and back matter.** Project Gutenberg texts start with a licence header and end with a licence footer (`*** START OF THE PROJECT GUTENBERG EBOOK ... ***` and `*** END OF ...`). `front_and_back_matter` finds them. The preview offers "Leave out the Project Gutenberg header and footer" (checked by default when found). Also find a table of contents: a run of heading lines with no prose between them near the start. Leave it out and say so, because otherwise "Chapter 1" is matched twice and the first match is a two-word section.

**The plan is a preview first.** `SectionPlan` = sections, the rule used, what was left out (front matter N words, ToC N lines), and warnings (a section under 300 words, one over 20,000). **Nothing is written until the user confirms.**

### 2.4 Storing split sections (R3: the original is never changed)

`Workspace.split_document(project_id, document_id, plan) -> list[document]`:

1. For each section, write a new corpus file `projects/<id>/corpus/<stem>__<order:03d>.txt` containing exactly `text[start:end]`, and insert a `documents` row with name `"<Work> – <order:03d> <title>.txt"`. Use `storage_filename` and `portable_key` as `import_document` does.
2. Write fields with source `split`:
   - `Work` = the original document's name without extension (editable later);
   - `Order` = order;
   - `Chapter` = order, or the parsed chapter number when the heading had one;
   - `Title` = heading title;
   - one field per parent level (`Volume`, `Part`, `Book`).
3. Record provenance: add a `derived_from` column to `documents` (nullable document id) and a `derivation` column (JSON: rule, offsets, and the source's sha256). Use the same "add column if missing" migration.
4. **Move the original to Trash** (not purge) by default, with a checkbox "Keep the whole book as a document too". Default off, because a corpus that holds both the book and its chapters counts every word twice. The run selection must never include both silently: if both are present, add a warning diagnostic to runs (`CORPUS_OVERLAPPING_DOCUMENTS`).
5. Set the project axis to `order` and the order label to the level name ("Chapter") **if** the project had no other axis.
6. `purge_document` of the original must be refused while derived sections exist that name it, **or** it must be allowed, with the derived rows keeping their `derivation` JSON (which carries the sha, so provenance survives). **Decide: allow, keep provenance.** Simpler for users. Write the reason in the docstring.

Endpoints:

- `POST /api/projects/{id}/documents/{doc}/sections/preview` (body: rule, pattern, options) → plan
- `POST /api/projects/{id}/documents/{doc}/sections/apply` → the new documents

### 2.5 Transcripts: turns as documents

The same splitter family. **Lower priority**: no current project needs it (debates were only an example), so build it in P5 only after the chapter splitter is done, or move it to backlog 4d-9. Interview, hearing and debate transcripts interleave speakers:

```
MODERATOR: Let's turn to immigration.
TRUMP: We have to secure the border ...
BIDEN: That's not true ...
```

Comparing "a president's debate rhetoric" with their SOTU means **keeping only that president's turns**. `core/file_ops/sections.py` gains `detect_turns(text) -> Result[TurnPlan]`:

- speaker labels as `NAME:` at line start (upper case, 1–4 words), with an optional `(role)`;
- consecutive turns by the same speaker are merged;
- stage directions `(APPLAUSE)`, `[CROSSTALK]`, `(inaudible)` are counted and left out (5.2 uses the same list);
- the preview shows speakers with turn and word counts.

Apply writes **one document per speaker per transcript** (all of a speaker's turns joined, with a blank line between turns) with fields `Speaker`, `Event` (the transcript's name), `Turns`, and `Date` if the transcript name has one. A "one document per turn" option exists for fine-grained work. Default is per speaker, because a turn is often one sentence and per-document measures on one sentence are noise.

UI: the same dialog as the chapter splitter, with a "This is a transcript with speakers" choice.

### 2.6 Tools and figures that matter for books (review, then fix what's missing)

After section 1, walk each of these on the real book (2.8) and apply 0.2 rule 3 ("does it tell me anything?"). The expected state after section 1 is in the right-hand column. If a figure still says nothing useful, fix it or add the missing figure, and record the finding in the Progress section.

| Question a reader brings to a novel | Tool(s) | Expected figure after section 1 |
|---|---|---|
| How does the mood move through the book? | `sentiment_vader_anew`, `sentiment_neural_bert`, `narrative` (emotion arc) | Sentiment across chapters with a rolling median; the arc from `core/narrative/arcs.py` on the order axis |
| What shape is the story? | `shapes`, `shape_svd`, `shape_nmf`, `shape_hc` | Story-shape clusters. **Known issue** (backlog 4d-5): the shape matrix isn't standardized, so component 1 is sentence length. Fix it here, because books are where shapes matter: standardize features before SVD/NMF, and show a note. **Approved by Riley (D7)**; say in the release notes that new runs give different shapes. |
| Who is in which chapter? | `ner` (PERSON), `core/narrative/characters.py` (`character_arcs`, `character_mentions`) | A character x chapter heat strip. **Check it exists** (grep `character` in `core/viz/panels*.py`); if not, add `narrative_characters_by_order` |
| Does the prose change (dialogue, sentence length, vocabulary)? | `readability`, `sentence_complexity`, `lexical_diversity`, `quote_annotator` (share of text in quotes) | Measures across chapters; share of dialogue per chapter |
| What is each chapter about? | `tfidf` (with `max-df-ratio 0.5`, see 5.5), `keyness` (this chapter vs the rest), `lda_gensim` with chapters as documents | Top terms per chapter; topic prevalence across chapters |
| Where does a word appear? | `dispersion`, `kwic`, `ngram_viewer` | Dispersion plot over the whole book's position; rate per chapter |
| Compare two novels | section 3 with `Work` as the side | — |

### 2.7 UI

- Corpus page: "Split a long document into chapters" opens a dialog:
  - left: detected rule, a rule menu, a pattern box, and checkboxes (leave out front matter, leave out table of contents, keep the whole book too, "this is a transcript with speakers");
  - right: a list of sections with order, title, word count and the first 120 characters, with warnings inline;
  - bottom: "Split into 61 chapters".

  It must stay responsive on a 1 MB text; the preview is computed in the engine, not in React.
- Overview and glance: for a book (axis = order and one `Work`), glance's first sentence reads "Pride and Prejudice, 61 chapters, 122,000 words".

### 2.8 Real data for books

- There is no book in the workspace. **Get two public-domain novels from Project Gutenberg** during the pass:
  - *Pride and Prejudice* (61 chapters, 3 volumes, roman-numeral `Chapter I` headings; a good test of parents and roman numerals);
  - *Alice's Adventures in Wonderland* (12 chapters, `CHAPTER I.` + a title on the next line).

  Both are public domain in the US. Project Gutenberg's licence asks that the header and footer and the "Project Gutenberg" trademark be removed if the text is redistributed without them. Our fixtures strip them, so they are plain public-domain text.
- Import them into the copied workspace as a new project "Novels", split both, and run the 2.6 walk. The split and the figures are checked **in the app** (screenshots), not only in tests.
- **Fixtures** (`tests/fixtures/books/`): trim *Alice* to its first 3 chapters with the real header and footer kept (for the front-matter test), plus a small synthetic text with `BOOK I` / `CHAPTER 1..4` / `BOOK II` / `CHAPTER 5..6` for the parent logic, one with roman-only headings, one with no headings (must fall back to words and label it), and one transcript with 3 speakers and stage directions.

### 2.9 Tests

- `tests/test_sections.py`: every fixture gives the expected section count, titles, parents and offsets. The sections tile the kept text exactly: concatenated they equal `text[front:back]` minus the ToC, with **no characters lost or duplicated**. A regex that matches nothing falls back and says so. A heading pattern that matches inside a sentence ("the chapter 3 of his life") is not a heading (whole-line rule).
- `tests/test_split_documents.py` (store level): apply writes N files and N rows with `split` fields, the original goes to Trash, purge of the original keeps the provenance, and a backup/restore round-trips `derived_from`.
- `tests/test_turns.py`: speakers found, stage directions counted, per-speaker documents correct, and the moderator is excluded when the user deselects them.

### 2.10 Done when (section 2)

- [ ] In the real app, *Pride and Prejudice* splits into 61 chapters with Volume I/II/III, the axis becomes "Chapter", and sentiment, readability, story shape and characters each draw a figure whose x axis says "Chapter".
- [ ] *Alice* splits into 12 with its chapter titles.
- [ ] A 3-speaker transcript splits into per-speaker documents with stage directions removed.
- [ ] None of the 13 formerly figure-less tools is figure-less on the Novels project.

---

## 3. Corpus comparison (a new "Compare" page)

### 3.1 What it is for

The research question behind it: *a president's rhetoric on a policy in one setting (State of the Union) against another setting (debates). Did it carry over? Did the framing change? Which ideas appear only in one setting?*

The general form: **two sets of documents (sides), possibly from different projects, compared on the same measures, with the differences made visible and, where possible, the matching passages shown side by side.**

It must also cover the simpler everyday cases:

- SOTU vs inaugurals (two projects);
- `Kind = sotu` vs `Kind = ina` inside one project;
- Book A vs Book B;
- the 1950s vs the 2010s;
- Democrats vs Republicans (a field from a spreadsheet).

### 3.2 Concepts

- **Side**: a name ("State of the Union"), a project, and a selection within it (a `CorpusSelection`: ids, dates, **fields**, from 1.5). A side must have at least 1 document. The same project may appear on both sides with different selections (the within-project case).
- **Comparison**: 2 sides (**3–6 sides** are allowed for measure and keyness methods, see 3.5; start the UI with 2 and let "Add a side" exist but ship it only if time allows), plus an **alignment**, plus the chosen **methods**.
- **Alignment** ("compare like with like"): optional, one of:
  - `none`: side A as a whole vs side B as a whole;
  - `field:<Name>`: compare within each value of a field present on both sides (e.g. `Speaker`). Results are per speaker, and speakers present on only one side are listed and left out;
  - `period`: within each shared period of the axis (decades for a dated corpus;
    blocks of chapters for an ordered one - both sides share the axis).

  Alignment answers "per president, did the SOTU and the debates differ?", which is the kind of question comparison is for. `none` would mostly show that speeches differ from debates, which is true and tells you nothing.
- **Topic focus** (optional): a lexicon (the same `terms` syntax as `lexicon_series`, e.g. `immigration: immigra*, migrant*, border*, asylum, refugee*`) or a semantic query ("immigration and the border"). When given, methods restrict to sentences about the topic (3.5 method 6). That is "rhetoric around those specific policies".

### 3.3 Where comparisons live (a decision, with the reason)

**Decide: a comparison belongs to a "home" project (side A's), and its runs are ordinary jobs in that project with `tool = "contrast"`. Each document in the job request carries its own `project_id`.**

Why:

- Past runs, panels, views, exports, trash and backups all key on project and job. Adding a workspace-level results store means re-implementing all of them (R10).
- A job's `request["documents"]` is already a frozen snapshot of the documents it read, with sha256. Adding `project_id` per document is additive.
- `run_job` reads each document from `workspace.project_dir(document["project_id"]) / "corpus" / stored_name` instead of the job's own project dir, and checks the sha as it does today.

Consequences to handle:

1. **Backups**: a backed-up project's comparison runs keep their artifacts, so the results survive. Re-running needs the other project. The restore marks such runs "reads documents from another project" (a diagnostic on the job), and the Compare page shows "Side B's project is not in this workspace" instead of failing.
2. **Trash and purge**: purging a project that another project's comparison reads is allowed (runs are snapshots). The comparison definition then shows its side as missing.
3. Saved comparison definitions get a table (`comparisons`: id, project_id (home), name, definition JSON, revision, created, updated), following the `questions` table pattern exactly (limits, `expected_revision`, duplicate, delete). Backup v5 includes it.

### 3.4 Engine: `core/contrast/` (new package)

Pure analysis functions over **one joined token table** with a `Side` column. The parse happens outside, as for every tool.

- `core/contrast/sides.py`:
  - `joined_corpus(sides) -> Corpus` with doc ids unique across sides;
  - `side_of: dict[doc_id, side]`;
  - `aligned_groups(...)`.
  - Document labels become `"<side>: <name>"` only when two sides share a name (`display_names` already handles collisions; extend rather than duplicate).
- `core/contrast/measures.py`: per-document measures (readability, lexical diversity, sentence length, sentiment, share of quotes) side by side. For each measure: the distributions per side, median difference, **effect size** (Cliff's delta, plus Mann-Whitney U and p; `core/analysis/stats_groups.py` already has `mann_whitney` and `_CLIFF_BANDS`, so reuse them) and a plain sentence ("Debate answers are shorter-sentenced than SOTU addresses for 9 of 11 presidents; median 14.2 vs 21.8 words; large effect").
- `core/contrast/keyness.py`: A vs B keyness via the generalized `core/analysis/keyness.py` (1.6 step 5, with the `groups` mapping). For more than 2 sides: each side vs the rest. **Always rates per 10k words, log ratio and G2 together**; never G2 alone (backlog 4d-2's "hides effect size").
- `core/contrast/themes.py`: shared and distinct themes.
  1. **Lexicon rates**: `lexicon_series` by side (and by alignment group). "Immigration terms: 3.1 per 10k words in SOTU, 7.4 in debates."
  2. **One topic model over the union** (`lda_gensim` adapter over the joined table), then topic prevalence per side, so both sides use the same topics. **Never fit two models and compare topic numbers**: the numbering is arbitrary (4d-2's table).
  3. Distinctive topics: prevalence ratio per side, with the top documents of each side for that topic.
- `core/contrast/meaning.py`: embeddings. **Needs 5.4 (the vector cache).**
  1. `doc_embeddings` over the union with a `Side` column on `doc_map.csv`: the map is coloured by side.
  2. Side centroids and their distance, **relative** (centred cosine, the 0.4.0 lesson: raw cosine puts everything at 98%).
  3. For each document, its nearest documents on the other side ("the debate closest to this SOTU").
- `core/contrast/carryover.py` answers **"did the rhetoric carry over?"**. The distinctive method; design it carefully:
  1. Sentences of each side, restricted to the topic focus if given (lexicon match or semantic similarity above a threshold).
  2. Embed sentences (Granite by default; `unit=sentence` already exists in `core/analysis/doc_embeddings.py`) through the vector cache.
  3. For each A sentence, its best B match by relative cosine, **within the alignment group** (same speaker).
  4. Classify pairs:
     - **carried** (similarity above a threshold calibrated on the corpus: the 95th percentile of random cross-side pairs, reported alongside);
     - **reworded** (moderate);
     - **only in A** / **only in B** (no match above the floor).
  5. When both sides are dated, **which came first**: a debate line that precedes the SOTU line by months is "said first in debates".
  6. Output `carryover_pairs.csv` (Side A doc, sentence, Side B doc, sentence, similarity, class, dates, lag in days, group) and `carryover_summary.csv` (per group: counts per class, examples).

  **Honesty rules**, written into the panel notes:
  - similarity is about wording and meaning, not stance: "we will secure the border" and "we will not secure the border" are close;
  - the threshold is calibrated per comparison and shown;
  - transcripts carry other speakers unless 2.5's turn split was used, so check it before interpreting.
- `core/contrast/run.py`: `contrast(table, corpus, sides, alignment, methods, focus, params) -> Result[dict[str, DataFrame]]`, the single entry the executor adapter, the Compare page and `nlp.compare()` (4.3) all call.

Register it:

- `ToolSpec(name="contrast", family="corpus_statistics" or a new family "comparison", input_kind="corpus", requires_parse=True, ...)` in `core/profiler/registry.py`;
- an adapter in `core/profiler/executor.py` (`ADAPTERS`, `ADAPTER_NEEDS = {"table","corpus"}`);
- `core/profiler/labels.py`;
- `docs/MIGRATION.md`, `docs/internal/REPLACEMENT_LEDGER.md` (a capability ID that resolves);
- the `tests/test_tool_registry.py` scope gate.

This is the "five-plus edits" list from `docs/internal/PANELS_BACKLOG.md` section 6. Do all of them in the same commit. **Not** in `desktop_backend/catalog.py`'s `CORPUS_TOOLS`: it has its own page, like `phrase_distribution` has its own endpoints (see `QUESTION_PUBLISHER_SPECS` in `runner.py` for how a tool is kept out of the general catalog).

### 3.5 Methods list (what the user ticks), in the order shown

| Method | Cost | Needs | Output tables |
|---|---|---|---|
| Size and style | fast | parse | `contrast_measures.csv`, `contrast_measure_tests.csv` |
| Distinctive words (keyness) | fast | parse | `contrast_keyness.csv` |
| Topic focus rates | fast | parse + focus | `contrast_focus_rates.csv` (per doc, per side, per group) |
| Shared and distinct topics | medium | parse | `contrast_topics.csv`, `contrast_topic_prevalence.csv` |
| Meaning map | slow (embeddings) | Granite | `contrast_map.csv`, `contrast_neighbours.csv`, `contrast_centroids.csv` |
| Carried-over passages | slow (sentence embeddings) | Granite + focus recommended | `carryover_pairs.csv`, `carryover_summary.csv`, `carryover_calibration.csv` |
| Tone | medium | VADER (always) | `contrast_tone.csv` |

The page estimates cost before running (sentences x the measured sentences-per-second from 0.4.0), like the live bench's `model_budget`.

### 3.6 Normalisation and fairness rules (enforced in code, not only in notes)

1. **Rates, never raw counts**, whenever sides differ in size. Every count column has a per-10k-words twin. Figures use the rate.
2. **Unequal sizes warning**: if one side has more than 5x the words of the other, a diagnostic says so. Keyness on a tiny side is dominated by chance.
3. **Genre confound**: when the sides come from different projects, the summary's first note says "these sides differ in setting as well as whatever you meant to compare (spoken debate vs written address). Differences in sentence length or pronouns may be the setting." This is the "correct but uninformative" trap in comparison form.
4. **Alignment coverage**: "7 speakers are on both sides; 4 are only on side A and were left out: ...".
5. **Same parser for both sides**: one parse of the joined corpus; the parse cache still hits per side, because the cache is keyed by corpus fingerprint (see 5.1 for keying by *document*, which makes this work).

### 3.7 Figures (panels over a comparison run)

New `core/viz/panels_contrast.py`, registered in `PANELS`. Each is a normal panel over the run's tables, so it inherits Explore, views and publication:

1. **Side by side distributions**: per measure, violins or strips per side (per group when aligned: small multiples by speaker).
2. **Effect sizes**: a dot plot of Cliff's delta per measure with CIs. "Where do the sides differ most?"
3. **Distinctive words**: the existing keyness volcano (`core/viz/panels_keyness.py`), with side names as labels.
4. **Focus rates by group**: e.g. immigration rate per president, SOTU vs debates as paired dots with a line between them (a dumbbell chart). **This is likely the headline figure for "per speaker, did the two settings differ?".**
5. **Topic prevalence by side**: diverging bars per topic.
6. **Meaning map by side**: the doc map coloured by side, with centroids marked.
7. **Carried-over passages**: a table-first panel (sentence A | sentence B | similarity | class | lag). Each row opens **both passages in context** (reuse `PanelPassages.tsx` / "Read in context"; it currently opens one document, so extend it to two columns). Plus a stacked bar per group of carried/reworded/only-A/only-B.
8. **Over time, both sides** (when both are dated): focus rate on the time axis, one line per side.

**Coverage test**: each contrast panel draws from a fixture comparison (built by running the engine on two small fixture corpora, **not hand-written**, per the backlog's review checklist) and refuses helpfully when its table is absent.

### 3.8 UI: the Compare page

- Sidebar: a new `navigation` entry `{ key: "compare", label: "Compare", icon: GitCompare (lucide), section: "WORKSHOP", hint: "Two collections side by side" }`, in the new sidebar group below Models (see D2). Add `"compare"` to the `Page` type. Extend `tests/test_desktop_navigation.py`.
- Layout, top to bottom:
  1. **Sides**: two cards, each with a name box, a project picker (all projects in the workspace) and a selection editor (the extended `CorpusScope` from 1.8: fields, dates, order). Each shows a summary ("31 documents, 66,160 words, 1901–2021, 18 speakers").
  2. **Line them up by**: none / a field present on both sides (only fields that exist on both are offered, with the shared-value count, e.g. "Speaker: 11 on both sides") / period.
  3. **Focus (optional)**: a lexicon editor (the same component as `lexicon_series`'s `terms`), or a "by meaning" query box.
  4. **Methods**: checkboxes from 3.5, with cost estimates.
  5. **Compare**: submits a job; progress shows in Past runs as usual.
  6. **Results**: tabs per method, each showing its panels. Saved comparisons are listed at the top of the page ("Open", "Duplicate", "Delete").
- `desktop/src/Compare.tsx` + `Compare.test.tsx`. API:
  - `GET/POST /api/projects/{id}/comparisons`
  - `POST /api/projects/{id}/comparisons/{cid}` (update, expected revision)
  - `.../duplicate`, `.../delete`
  - `POST /api/projects/{id}/comparisons/{cid}/run` → job

### 3.9 Runner changes

- `Runner.submit` accepts `tool == "contrast"` with a `sides` body. It validates each side's project exists and isn't trashed, resolves each selection with `resolve_selection` on **that** project's documents, and stamps `project_id` and `side` into each document dict.
- `run_job`: documents are read from their own project dirs. The input manifest gains `Side` and `Project` columns.
- The parse: one `resolved_pipeline.parse(corpus)` over the joined corpus, through the cache (5.1).

### 3.10 Real data for comparison

1. **Now**: SOTU (87) vs Inaugural speech (31), aligned by `Speaker`. The presidents on both sides are roughly FDR to Biden: Riley's 87-speech project spans 1934–2024 and the inaugural one 1901–2021, so check the overlap on the real data. Also Kind = sotu vs Kind = ina inside the mixed 68-document project.
2. **Debates were only an example** of a second collection. Nobody is building a debates corpus, and no code or test depends on one. SOTU vs Inaugural is the real test.
3. Fixture comparisons for tests: two tiny synthetic corpora (six documents each, two shared speakers, a planted distinctive word and a planted carried-over sentence), so tests assert that the planted signal is found and the rest isn't.

### 3.11 Done when (section 3)

- [ ] In the real app: SOTU vs Inaugural aligned by Speaker runs all fast and medium methods. The focus-rate dumbbell for an immigration lexicon draws one row per shared president. Distinctive words on each side are recognisably about the setting (inaugural: "oath", "god"...), and the genre note is shown.
- [ ] Carried-over passages on that comparison (focus: immigration) returns pairs whose "Read in context" opens both speeches at the right sentences.
- [ ] Within-project Kind = sotu vs Kind = ina works without a second project.
- [ ] The planted-signal fixture test passes, and a comparison with an absent side-B project shows the missing-project message instead of failing.

---

## 4. Smart scripts (notebooks on a new "Scripts" page)

### 4.1 What it is, and what it is not

**Riley's description:** say "I want a graph of the growth of immigration in the US over time", write (or have an assistant write) a small program in an editor inside the app that calls the suite's tools, get a CSV and an automatic chart back, and use matplotlib on the CSV for a publication figure. Riley suggested something like `.ipynb`.

**The design, in one line:** a project has notebooks. A notebook is cells of Markdown and Python. Python cells run in a kernel process inside the engine, with a documented library (`import nlpsuite as nlp`) over every tool. Every table a cell shows becomes a CSV result with an automatic chart. Every matplotlib figure is kept. The notebook exports as a real `.ipynb` plus a `data/` folder that runs outside the app.

**What it is not:**

- It is not a general Python IDE: no pip installs, no arbitrary packages beyond those bundled (0.3).
- It is not a place where an AI decides findings. A chatbot the user already has can *write code* from the suite's guide, which the user reads and runs (4.8). The app contains no AI.
- It is not a second results system. Outputs are ordinary runs (R8).

### 4.2 The reference example, rewritten (the design target)

The library is designed backwards from this. **If this notebook cannot be written in about 20 lines and produce the same numbers as `scripts/build_immigration_evidence.py`, the library is wrong.** Section 4.10's acceptance test is exactly this.

```python
# Cell 1 (Markdown)
# Immigration vocabulary across the State of the Union
# Per-1,000-word rate of a fixed lexicon, yearly means, five-observation trend.

# Cell 2
import nlpsuite as nlp

corpus = nlp.corpus()                      # the project's documents, with Date/Speaker/Kind
LEXICON = ["immigration", "immigrant", "immigrants", "migrant", "migrants",
           "refugee", "refugees", "asylum", "border", "borders",
           "deportation", "deportations", "citizenship", "alien", "aliens"]

rates = nlp.term_rates(corpus, {"immigration": LEXICON}, per=1000, match="exact-lowercase")
nlp.show(rates)                            # table + automatic chart: rate by Date

# Cell 3
yearly = rates.groupby("Year", as_index=False)["immigration per 1000"].mean()
yearly["trend"] = yearly["immigration per 1000"].rolling(5, center=True, min_periods=1).mean()
nlp.show(yearly, x="Year", y=["immigration per 1000", "trend"], kind="line")

# Cell 4
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.plot(rates["Year"], rates["immigration per 1000"], "o", alpha=0.5, label="Address")
ax.plot(yearly["Year"], yearly["trend"], "-", label="Five-observation trend")
ax.set(xlabel="Address year", ylabel="Occurrences per 1,000 words", ylim=(0, None))
ax.legend(frameon=False)
nlp.figure(fig, "immigration_time_series")   # kept as PNG + SVG in the run

# Cell 5
passages = nlp.passages(corpus, LEXICON, context=1, match="exact-lowercase")
nlp.show(passages)                           # clickable: opens Read in context
nlp.save(rates, "per_speech_rates")          # CSV in the run, and in the export's data/
```

What this reveals the library needs:

- `term_rates` (whose tokenization rule must be *selectable*, because the reference script counted `[A-Za-z]+` tokens in the raw text, not parser tokens);
- `passages` with neighbouring sentences;
- `show` / `figure` / `save`;
- a corpus whose `documents` frame has Year.

### 4.3 The library: `nlpsuite` (`core/script/` package, imported as `nlpsuite`)

**Location and name.** Code lives in `core/script/`: `api.py` (public functions), `session.py` (per-kernel state), `outputs.py` (show/figure/save collection), `reference.py` (generates the documentation). A top-level shim package `nlpsuite/__init__.py` re-exports `core.script.api`, so scripts read `import nlpsuite as nlp` and exported notebooks read naturally. Add `nlpsuite` to `pyproject.toml` packages and to the PyInstaller build (`--hidden-import nlpsuite` and `--collect-submodules core.script`).

**Design rules** (write them in the module docstring):

1. **Returns pandas DataFrames.** Every result is a DataFrame or a small object with DataFrames on it, never a custom table type. People know pandas, and the reference script already uses it.
2. **Thin over the executor, never a second implementation (R10).** `nlp.run(tool, ...)` builds a `Plan` with `core.profiler.plan.build_plan` and calls `core.profiler.executor.execute`, exactly as the runner does. Convenience functions (`term_rates`, `passages`, `keyness`, ...) are **named wrappers over existing adapters**. Where a wrapper needs behaviour no adapter has (the raw-regex tokenization of the reference script), it goes into `core/analysis/` first, gets tests, and is then wrapped.
3. **Parse once, cached.** `corpus.tokens()` parses through the same cache as the runner and the bench (5.1), so the first cell pays and later cells don't.
4. **Every call is recorded.** `session.calls` logs (function, params, input fingerprint, output shape, seconds). It is written into the run's provenance (4.6), so a published notebook run says which suite tools produced each table.
5. **No filesystem writes except through `save`/`figure`/`show`.** The kernel's working directory is an empty temp dir. Outputs go to an `OutputWriter` at the end of the run (R3). A script that calls `df.to_csv("x.csv")` writes into the temp dir, which is discarded, and a diagnostic says so at the end of the run: "x.csv was written to a scratch folder and discarded; use nlp.save(df, 'x')".
6. **Errors are readable.** Library functions raise `nlp.SuiteError(message, diagnostics)` with the diagnostics' own messages (which are already written for people), not a stack of `Result.unwrap` frames. Cell tracebacks are trimmed to user code plus the library call's message.
7. **Stable names.** Parameters use Python names (`top_n`), mapped to registry names (`top-n`) by replacing `_` with `-`. Unknown parameters raise with "did you mean" (difflib over the tool's `ParamSpec` names).

**Public API (v1).** Exact signatures to implement. Each needs a docstring, because `reference.py` builds the in-app reference and the AI chatbot guide (4.8) from them.

```python
# Corpus
corpus(selection: dict | None = None, *, project: str | None = None) -> Corpus
#   project=None means the notebook's project; a name or id reads another project (for comparisons).
#   selection is CorpusSelection's fields: ids, date_from/date_to, fields={...}, order_from/order_to.
class Corpus:
    documents: pd.DataFrame        # Document ID, Document, Date, Year, Order, <fields...>, Words
    axis: Axis                     # from core/corpus_axis.py
    def filter(self, **fields) -> Corpus               # corpus.filter(Speaker="Harry S Truman", Kind="sotu")
    def where(self, predicate: Callable[[pd.Series], bool]) -> Corpus   # row-wise, on documents
    def text(self, document) -> str
    def sentences(self) -> pd.DataFrame                # Document ID, Sentence ID, Text (+ document columns)
    def tokens(self) -> pd.DataFrame                   # the parsed CoNLL table (cached)
    def __len__(self) -> int

# Running any tool
tools() -> pd.DataFrame                                # name, label, family, needs parse, outputs
describe(tool: str) -> str                             # parameters with types, defaults, choices, help
run(tool: str, corpus: Corpus, **params) -> RunResult
class RunResult:
    tables: dict[str, pd.DataFrame]                    # every output frame, keyed by file name
    table: pd.DataFrame                                # the primary one (first declared output)
    diagnostics: list[dict]
    def __getitem__(self, name) -> pd.DataFrame

# Convenience wrappers (each is run(...) plus shaping; list grows over time)
term_rates(corpus, groups: dict[str, list[str]] | str, *, per=1000, by="document",
           match="lemma" | "form" | "exact-lowercase", within=None) -> pd.DataFrame
passages(corpus, terms, *, context=1, match=..., limit=500) -> pd.DataFrame
keyness(corpus_a, corpus_b=None, *, field=None, a=None, b=None, top_n=200) -> pd.DataFrame
measures(corpus, which=("readability","lexical_diversity","sentence_length")) -> pd.DataFrame
sentiment(corpus, *, model="vader" | "distilbert-sst2", unit="sentence" | "document") -> pd.DataFrame
entities(corpus, *, kinds=("PERSON","GPE","ORG")) -> pd.DataFrame
topics(corpus, *, k=10, seed=42) -> RunResult
similar(corpus, query: str, *, model="granite-embedding-english-r2", unit="sentence", top_n=20) -> pd.DataFrame
compare(side_a: Corpus, side_b: Corpus, *, align=None, focus=None, methods=("measures","keyness")) -> RunResult

# Output
show(df, *, x=None, y=None, kind=None, title=None, name=None) -> None   # table + automatic chart
figure(fig, name: str) -> None                                          # matplotlib -> PNG + SVG
save(df, name: str) -> None                                             # CSV in the run and the export
note(text: str) -> None                                                 # a sentence in the run's readout

class SuiteError(Exception): ...
```

- **`term_rates` `match` modes**: `lemma` (parser lemmas, which is what `lexicon_series` does), `form` (parser tokens), and `exact-lowercase` (`[A-Za-z]+` over raw text, the reference script's rule). That last mode is new analysis code: `core/analysis/lexicon_series.py` gains a raw-text path, with a test that on the SOTU fixture it reproduces `per_speech_rates.csv`'s numbers exactly. **Say in the docstring why three modes exist**: they give different numbers ("immigrants" vs lemma "immigrant"), and a paper must say which one it used.
- **Wildcards** in term lists (`immigra*`) use the same syntax `lexicon_series` already parses (`parse_lexicon`), one rule (R10).

### 4.4 Execution: the script kernel

#### 4.4.1 Why a separate process

- A notebook runs arbitrary user Python. In the **server process** a runaway loop would freeze the app and a crash would take it down. In the **job worker** it would block every analysis queued behind it.
- So: a separate **kernel process per open notebook** (at most 2 alive; the least recently used one is shut down), started from the same frozen executable with a new flag. The pattern is `runner.py`'s worker (`worker_command`, `worker_loop`, the JSON-lines protocol, kill-before-close on Windows).

#### 4.4.2 Protocol (JSON lines over stdin/stdout; stderr goes to a log file)

Parent to kernel:

- `{"op":"exec","id":<n>,"cell":<cell id>,"code":"..."}`
- `{"op":"interrupt"}` (best effort; see below)
- `{"op":"complete","code":"...","cursor":<n>}` (4.7 autocomplete, v2)
- `{"op":"shutdown"}`

Kernel to parent:

- `{"event":"ready","library":"<version>"}`
- `{"event":"stream","id":n,"name":"stdout"|"stderr","text":"..."}` (batched every 100 ms)
- `{"event":"output","id":n,"kind":"table"|"figure"|"text"|"markdown","payload":{...}}`
- `{"event":"done","id":n,"ok":bool,"error":{"type","message","trace":[...]},"seconds":x}`

Details:

- Tables are sent as the first 200 rows plus a full-table token. Full tables and figures are written to the kernel's session dir and fetched by path. **Never push a 50 MB frame through the pipe.**
- **Interrupt**: Python code can't be safely interrupted from outside on Windows. Implement "Stop" as **kill the kernel and restart it**, and say so in the UI: "Stopping resets the notebook's variables. Run the cells again." Don't pretend `KeyboardInterrupt` works.

#### 4.4.3 Entry point

- `desktop_backend/server.py` main already handles `--worker`. Add `--script-kernel --data-dir <root> --project <id> --session <dir>`.
- The kernel imports `core.script.session`, sets the matplotlib backend to Agg, sets cwd to an empty temp dir, then loops.
- `desktop_backend/kernels.py` (new) manages kernel processes: start, stop, least recently used, idle shutdown after 15 minutes, and "is it alive". It mirrors `Runner`'s worker management. **Reuse its helpers** (`_close_pipes`, the kill-first `_stop_worker` logic): factor them into a small shared module rather than copying (R10, R11 "one subprocess wrapper").

#### 4.4.4 Running a cell

- Each cell's code is `compile(code, f"<cell {n}>", "exec")`, then `exec`'d in the notebook's persistent namespace dict. Its **last expression** is displayed if it isn't None (Jupyter's rule: parse with `ast`, split off a trailing `ast.Expr`, `eval` it). A DataFrame result behaves like `nlp.show(result)` without a chart; a figure behaves like `nlp.figure`.
- `print` output streams to the cell.
- **Limits**, as named constants:
  - one cell: 30 minutes of wall clock (a warning at 2 minutes: "still running");
  - output text: 1 MB per cell, truncated with a note;
  - `show`: 100 tables per run;
  - `figure`: 50 per run.

#### 4.4.5 Safety: what can and cannot be promised

**Be honest in the docs and the UI.** A notebook runs with the same permissions as the app. Python can't be sandboxed well from inside Python: blocking `open` or `socket` is trivially bypassed. So the design is protection by process isolation plus clear warnings, not a fake sandbox:

- separate process (a crash or hang can't hurt the app or its data);
- cwd is a scratch dir; outputs only reach the project through the library;
- the kernel is started with an environment that doesn't include the server's bearer token (check `server.py` for where the token lives; **the kernel must not be able to call the API as the user**);
- a **notebook imported from a file**, or pasted in from a chatbot, opens with a banner: "This code came from outside the app. Read it before running: it can do anything a program on your computer can." Its cells don't auto-run. A notebook written in the app doesn't show the banner.

This goes in `docs/SECURITY.md` as a new section.

#### 4.4.6 The security test change (do it openly)

- `tests/test_security.py::test_no_dynamic_code_execution` currently allows no exceptions. Change it to use the same `ALLOWLIST` mechanism, with a new rule `S102-exec`, and add:
  ```python
  "kernel": ("S102-exec S603-subprocess", "runs the user's own notebook cells in a separate kernel process; no network by the kernel itself; cwd is a scratch dir; outputs only via the nlpsuite library (docs/SECURITY.md, Smart scripts)"),
  ```
- `test_allowlist_entries_are_accurate` must check `S102-exec` entries actually contain `exec(`.
- The manager module (`desktop_backend/kernels.py`) also needs the subprocess rule.
- **No other module gets `exec`.**

### 4.5 Storage

- New table `notebooks` (following `questions`): `id, project_id, name, content (JSON), revision, created, updated`, unique `(project_id, name)`, limit 200 per project. `content` is **nbformat 4 JSON**: `{"nbformat":4,"nbformat_minor":5,"metadata":{...},"cells":[...]}`. Store it in the `.ipynb` shape from day one, so export is a copy rather than a conversion. Cell outputs are **not** stored in `notebooks.content` (they are large and belong to runs); the UI keeps the last session's outputs in memory only.
- Metadata: `metadata.nlpsuite = {"library": "<version>", "project": "<name>", "created_by": "app" | "import" | "pasted"}`. `created_by` drives the "came from outside" banner.
- `Workspace` methods like the questions ones: `notebooks`, `notebook`, `save_notebook`, `update_notebook` (`expected_revision`), `duplicate_notebook`, `delete_notebook`. Backup v5 includes notebooks. Restore validates JSON shape and the limits.

### 4.6 Running a whole notebook as a run (so it is a result, R8)

Two modes:

1. **Interactive**: cells run in the kernel as the user clicks. Outputs show inline. **Nothing is published.** This is like the live bench.
2. **"Run and save"**: runs all cells top to bottom in a **fresh** kernel, as a job (`tool = "notebook"`, queued through `Runner` like any job so it respects "one heavy thing at a time"). It publishes a run directory through `OutputWriter` containing:
   - every `show` table as `tables/<name>.csv` (named by `name=` or `cell<N>_<k>`);
   - every `figure` as `figures/<name>.png` + `.svg`;
   - every `save` as `data/<name>.csv`;
   - `notebook.ipynb`: the notebook **with its outputs** (text outputs inline; tables as HTML previews of 20 rows; figures as PNG), so the run is a readable record;
   - `provenance.json`: library version, notebook revision and sha256 of its code, the `session.calls` log, input documents (the runner's usual manifest with fields), Python/pandas/matplotlib versions.

   Every CSV is then an ordinary table in Explore (`Workspace.tables()` lists it automatically), and the panels/recommendations system proposes charts for it.

"Run and save" is what makes a notebook's outputs **citable** ("Figure 3 was produced by notebook *Immigration* revision 4, run on 2026-10-02").

### 4.7 The Scripts page (UI)

- Sidebar: `{ key: "scripts", label: "Scripts", icon: SquareCode (in lucide-react 0.468), section: "WORKSHOP", hint: "Write it yourself, with the suite's tools" }`, first in the new sidebar group below Models, above Compare (see D2).
- **Left column**: the project's notebooks ("New notebook", "Start from a template", "Import .ipynb").
- **Main area**: the notebook.
  - Cells stacked. Each cell has a type toggle (Python/Markdown), a run button, and a status (idle / running Ns / done Ns / error).
  - The toolbar has "Run all", "Stop (resets variables)", "Restart", "Run and save", "Export", "Get help from an AI chatbot".
  - Keys: Shift+Enter runs a cell and moves to the next; Ctrl+Enter runs in place.
  - Markdown cells render with the existing `Markdown.tsx`.
- **Outputs under each cell**:
  - streamed text;
  - tables via `ResultTable.tsx` (first 200 rows, "open the whole table in Explore" after a Run and save);
  - **an automatic chart next to each shown table**: the same recommendation path Explore uses (`core/insight/recommend.py` via the existing chart contract; `ChartCanvas.tsx` draws it). When `show(..., x=, y=, kind=)` specifies a chart, that is used. The chart has "Copy matplotlib code": 10–15 lines of matplotlib that reproduce it from the saved CSV. This is the "CSV for your own matplotlib" requirement, and it doubles as teaching;
  - figures as images;
  - errors as a readable message plus a collapsible trimmed traceback.
- **Right column** (collapsible): **Library reference**, generated from `core/script/reference.py`. It is searchable. Each function shows its signature, docstring and an example, with an "Insert" button. Below: "Your corpus", showing fields, counts, axis and the `documents` columns, so the user knows which names to type.
- **Editor**: **CodeMirror 6** (`@codemirror/state`, `@codemirror/view`, `@codemirror/commands`, `@codemirror/language`, `@codemirror/lang-python`, `@codemirror/lang-markdown`, `@codemirror/autocomplete`; all MIT). It is about 150 KB minified for this set; measure the bundle before and after. Autocomplete v1 is static: `nlp.` completes from the reference JSON, and corpus field names complete inside `filter(`. Kernel-driven completion (`op: complete`) is v2 and can wait.
  - Alternative considered: a plain `<textarea>`. It has no highlighting, no bracket matching and no completion, and it reads as unbuilt.
  - Monaco: far larger and awkward to bundle offline in Tauri. Rejected.
- **Templates** (`core/script/templates/*.ipynb`, shipped as data files; each must run green on the SOTU fixture in a test):
  1. "Word group over time": the immigration notebook of 4.2, generalized with a `LEXICON` cell.
  2. "Compare two groups": two filters, keyness, measures side by side.
  3. "Chapter arc": sentiment across chapters, for books.
  4. "Who is named, when": entities by period.
  5. "Find passages by meaning": `similar()` plus reading them in context.
  6. "Start from a finished run": load a Past run's CSV and chart it. This needs `nlp.load(run, table)`; add it to the API.
- **Export** writes a zip `<name>.zip` with `<name>.ipynb`, `data/*.csv` (every `save`, plus the `show` tables of the last Run and save), `figures/`, and a `README.md` saying how to run it outside the app:
  - **with the suite installed** (`pip install` the repo), the notebook runs as is;
  - **without it**, the export also contains `nlpsuite_offline.py`, a tiny module where `nlp.load("per_speech_rates")` reads `data/per_speech_rates.csv`, and `show`/`figure`/`save` degrade to `display`/`savefig`/`to_csv`. **The matplotlib cells run anywhere; cells that call tools need the suite.** State this plainly in the README.
- **Import** reads a `.ipynb`: nbformat 4, cells only, outputs dropped, `created_by: import`.

### 4.8 The AI chatbot guide: "Paste this into your chatbot" (no AI built into the app)

**Decision (Riley, 2026-09-25):** the app contains **no AI**: no API key, no network call, no local model. Instead the suite writes one **guide**, a document describing every function and tool. People paste it into whatever chat assistant they already use in the browser (ChatGPT, Claude, Gemini...), describe what they want, and paste the code they get back into a notebook cell.

This is how Riley worked before ("I had an LLM write a script"). The difference is that the assistant now gets the suite's real documentation, so its code **calls the suite's functions** instead of reimplementing counting with regexes.

The rule from backlog 4d-3 still holds: an assistant writes **code**, which the user reads and runs. It never produces findings.

#### 4.8.1 What the guide contains

One Markdown document, **"NLP Suite scripting guide for AI assistants"**. It is **generated from the library and the tool registry** (`core/script/reference.py`), never written by hand, so it can't drift from the code. Its sections, in order:

1. **Instructions to the assistant** (written to the chatbot, not to the user):
   - "You are writing Python for a notebook cell inside the NLP Suite."
   - Use only `nlpsuite` (as `nlp`), pandas, numpy, matplotlib, seaborn, scipy and sklearn. Nothing can be pip-installed.
   - **Prefer a library function over reimplementing it.** Never count words with `re.findall` when `nlp.term_rates` does it.
   - Show results with `nlp.show`, keep figures with `nlp.figure`, keep data with `nlp.save`. Never write files.
   - Compare documents of different lengths with rates, not raw counts.
   - State in a comment which counting rule (`match=`) was used and why.
   - Use the field names listed under "The user's corpus"; if what the user wants needs a field that isn't there, **say so instead of guessing a name**.
   - Split long work into a few cells, each with a one-line comment saying what it does.
2. **How notebooks work**: cells, variables persisting between cells, Stop resetting them, what's installed, the scratch folder that is discarded.
3. **The library reference**: every public function in `core/script/api.py`, with its signature, parameters (type, default, choices), what it returns, **the columns of the returned DataFrame**, and one short example. The column lists matter most: an assistant that knows `term_rates` returns `Document`, `Date`, `Year`, `immigration per 1000` writes the next line correctly.
4. **Every tool**, generated from `core/profiler/registry.py` and `core/profiler/labels.py`: name, what it does in one line, parameters with defaults and choices, output tables and their columns. This is how an assistant uses `nlp.run("collocations", ...)` for tools without a convenience wrapper.
5. **Worked examples**: the template notebooks' code (4.7), including the immigration one from 4.2, each with the request it answers ("Graph how often immigration is mentioned over time").
6. **Common mistakes**, each with the wrong and the right version:
   - reimplementing tokenization;
   - raw counts across unequal documents;
   - `to_csv` instead of `nlp.save`;
   - averaging per-document rates by summing;
   - assuming a Date on an undated corpus (check `corpus.axis`).
7. **The user's corpus** (optional, included by default): document count, words, axis, field names with up to 10 sample values each, and the `documents` columns. **No document text, ever.** It can be switched off for a more private copy.

**Size.** It must fit in a free-tier chat box. Measure the generated guide in characters and words, and aim for about 60,000 characters or less. If it's bigger, the dialog offers two versions:

- **Short**: sections 1, 2, 3 and 7, with section 4 as one line per tool;
- **Full**: everything, better attached as a file than pasted.

**Versioning.** The guide's first line says the library version ("For NLP Suite 0.5.0"). When a later release renames a library function, the old name stays for one release as an alias that warns, so code written from an older guide still runs.

#### 4.8.2 Where people get it

- On the Scripts page, a toolbar button **"Get help from an AI chatbot"** opens a dialog with three numbered steps:
  1. **Copy the guide** (a button; the Short/Full choice; "Include my corpus's details" checkbox, on by default), or **"Save as a file"** (for chatbots that accept attachments).
  2. "Paste it into your chatbot, then describe what you want, for example: *Graph how often immigration is mentioned in each speech over time.*"
  3. **"Paste the code you got back"**: a text box, then "Add as a new cell". The cell gets the "came from outside the app" banner (4.4.5) and isn't run automatically.
- **The Learn page** links to the same dialog, so people find it without opening a notebook.
- **Exports** (4.7) include the guide (`GUIDE_FOR_AI.md`), so a notebook shared with a colleague carries the documentation its code was written against.

#### 4.8.3 Checking pasted code (a reading aid, not security)

`core/script/lint.py` looks at a cell's code with `ast` (it never runs it) and shows warnings next to lines worth reading:

- imports outside `nlpsuite`, pandas, numpy, matplotlib, seaborn, scipy, sklearn, math, re, collections, itertools, statistics, datetime, json;
- `open(`, `os.`, `subprocess`, `socket`, `urllib`, `requests`, `eval`, `exec`, `__import__`;
- counting done by hand (`re.findall`, `.split()` followed by `.count`) when a library function exists: "`nlp.term_rates` does this with the suite's tokenizer; see the guide".

This is not security (4.4.5 explains why Python can't be sandboxed from inside Python). It points the reader at lines to read before running.

#### 4.8.4 Engine and API

- `core/script/guide.py::build_guide(*, version: "short" | "full", corpus: CorpusSummary | None) -> str`. It is **deterministic**: sorted everything, no timestamps. The same inputs give the same bytes, so a guide can be diffed between releases.
- `GET /api/script-guide?project=<id>&version=short|full&corpus=1|0` returns text. The UI copies it or saves it with the Tauri dialog plugin (already a dependency: `@tauri-apps/plugin-dialog`).

#### 4.8.5 Not doing (written into backlog 4d-9 with the reason)

- An API key for a built-in assistant, and a local code model on the Models page. The reason: Riley's decision that people will use the chatbot they already have. It keeps the app offline and "private by design", and there is no key storage, no network dependency and no model to ship. Revisit only if users ask for it.


### 4.9 Tests (section 4)

- `tests/test_script_api.py`: every public function on the SOTU fixture. `term_rates(..., match="exact-lowercase")` equals a direct computation. `run("readability", ...)` equals `execute(...)`'s frame (**same rows, the "live equals published" rule**). Unknown params raise with a suggestion. `SuiteError` messages are the diagnostics' messages.
- `tests/test_script_kernel.py` (spawns the real kernel; use a short basetemp; kill in `finally`):
  - state persists across cells;
  - the last expression displays;
  - stdout streams;
  - an exception reports the user's line number;
  - an infinite loop is stopped by kill and restart within 5 s;
  - `to_csv("x.csv")` produces the discard diagnostic;
  - **the kernel's environment has no bearer token**.
- `tests/test_notebook_run.py`: "Run and save" publishes a run with tables, figures, `notebook.ipynb`, `provenance.json`, and the CSVs appear in `Workspace.tables()`.
- `tests/test_notebook_templates.py`: every shipped template runs green on the fixture.
- `tests/test_notebook_export.py`: the zip contains the named files. `nlpsuite_offline.py` makes the matplotlib cells of the immigration template run in a clean subprocess **without `core` on the path** (set `PYTHONPATH` to only the export dir; this is the "runs outside the app" claim, tested).
- `tests/test_script_guide.py`: the guide names every public library function and every catalog tool, contains no document text (even with long document names), and is byte-identical for identical inputs. **Every code example in it runs green on the fixture corpus** (extract the fenced blocks and run them in the kernel), so the documentation cannot promise a call that fails.
- `tests/test_security.py` changes as described in 4.4.6 and 4.8.
- UI: `Scripts.test.tsx`: create, edit, run (mock kernel), output rendering of each kind, template insert, banner on imported notebooks, export button. `live.test.ts`-style unit tests for the output model.

### 4.10 Done when (section 4)

- [ ] **The acceptance test (A/B against the old script):** on the real SOTU project in the running app, the "Word group over time" template with the 15-term lexicon and `match="exact-lowercase"` gives per-speech rates equal (to 1e-9) to the old script's `per_speech_rates.csv` for all 87 rows. Its yearly means match `annual_means.csv`, and its figure is visually the same shape as `immigration_time_series.png`. Screenshot both side by side in the Progress note.
- [ ] That notebook is at most about 25 lines of code.
- [ ] "Run and save" produces a run whose tables open in Explore with a recommended chart.
- [ ] Export, unzip into a temp dir, and run the matplotlib cell with plain Python and no suite on the path: it produces the figure.
- [ ] The guide, pasted into two different browser chatbots with five requests each (for example "immigration over time", "the most negative speeches", "compare Truman and Eisenhower", "chapters with the most dialogue", "people named per decade"), gives code that runs. Record how many of the ten ran unchanged, and fix the guide wherever an assistant went wrong.

---

## 5. Fixes and smaller items (the "couple of fixes", plus the open items from 0.4.0)

### 5.1 Jobs don't use the parse cache (do this first; it underpins sections 3 and 4)

- **Evidence**: `desktop_backend/runner.py::run_job` calls `resolved_pipeline.parse(corpus)` directly. `Annotations` is only used by `desktop_backend/live.py` (lines ~262, 321, 349). So every published job re-parses. Backlog 4e observed minutes of "Parsing English documents" per job.
- **Measure first** (0.2 rule 5): time a readability-plus-parse job on the 87-speech corpus twice in a row, before the change.
- **Fix**: in `run_job`, compute `annotation_key(corpus.sha256, parser_identity(pipeline))` and load it from `Annotations(workspace.root)`. On a miss, parse and store. The live bench and jobs then share one cache.
- **Better keying for comparisons and subsets** (needed by 3.9 and 4.3): cache **per document**, not per corpus. Key = document sha256 plus parser identity. Parse only the missing documents and concatenate, renumbering `Document ID` to the run's ids. Then a selection of 20 of the 87 speeches, a comparison joining two projects, and a notebook's `corpus.filter(...)` all hit the cache.
  - **Check the CoNLL table's per-document independence first**: sentence ids, record ids and `Document ID` must be rewritable per document without re-parsing. `core/conll/normalize.py` may have assumptions; read it.
  - If it's not safe, keep per-corpus keys and accept misses on subsets. Say which in Progress.
- **Cache size**: the parse cache has no limit today (`Annotations.forget()` exists). Add a size cap (default 2 GB, least recently used eviction) and show the size with a "Clear" button in Settings, since per-document entries will multiply.
- **A/B after**: the second run of the same job must skip the parse (stage "Parsing" absent or under 2 s). Record the before and after times.

### 5.2 NER noise: "Speaker", "Boo", "Chamber", "Pell" tagged as people (the named fix)

- **Cause** (backlog 4d-5): transcripts carry annotations like "(Applause.)", "(Laughter.)", "Mr. Speaker", and audience reactions ("Boo!"). The tagger labels them PERSON. Check the exact strings on the real corpus: grep the SOTU texts for `\(` and `\[` bracketed spans and count them.
- **Class**: "text that isn't the author's words reaches every tool", not only NER. It inflates word counts, sentiment (Applause is positive in VADER), keyness and topics.
- **Fix, in two parts:**
  1. **Stage-direction removal at read time**, as a project setting (`project_settings.text_cleaning = {"stage_directions": true}`), **on by default for new projects and offered for existing ones**, with a preview on the Corpus page ("Found 2,314 bracketed stage directions like (Applause.), (Laughter.); leave them out of analyses?").
     - Implemented in `core/io/cleaning.py::strip_stage_directions(text) -> (text, removed_count)`, pure.
     - Pattern: bracketed or parenthesized spans of 1–4 words matching a known list (`applause`, `laughter`, `cheers`, `boos`, `booing`, `inaudible`, `crosstalk`, `music`, `silence`, `pause`, `sic` is **not** included) plus a user-editable extra list.
     - Applied when the runner and bench build `Document.text`; the original file stays untouched (R3). The setting and removed counts are recorded in the envelope params, so a result says it was cleaned.
     - **The parse cache key must include the cleaning setting** (the text differs), so `corpus_fingerprint` is computed on the cleaned text. Check: `run_job` hashes `text_result.unwrap()`; cleaned text changes that hash, which is correct.
  2. **A reviewed stop-entity list for NER**: `ner` gains a param `ignore` (comma-separated). Its panel gains a line "Most frequent PERSON entities that are probably not people: Speaker, Chamber (click to ignore)", computed with a simple rule: PERSON entities whose lowercase form is a common English noun in the bundled wordlist, or which occur in more than 50% of documents. **Suggested, not applied**: the reader decides. The same helper serves `gender_guess` and `quote_annotator`, which inherit the same noise.
- **Coverage test**: on the SOTU fixture, with cleaning on, "Applause" is absent from every tool's output table (walk every corpus tool through `execute`, grep all string cells), and PERSON excludes "Speaker" once ignored.

### 5.3 BERT speed benchmark and a last-layer-only graph (0.4.0 open item 1)

- Write `scripts/bench_models.py`: embed 500 fixed sentences (`scripts/model_parity_sentences.txt` plus the SOTU fixture), interleaved A/B/A/B across precisions, 5 repeats, and report median sentences/s. **Run it when the machine is idle** (close the app and other heavy processes), and record CPU model and core count in Progress.
- Last-layer graph: export a second BERT ONNX graph that outputs only the last hidden state (`scripts/export_models.py`), shipped as `model_last.onnx` next to `model.onnx`. `OnnxTokenBackend` picks it when `layer == -1`.
  - **Only ship it if the benchmark shows a real gain (at least 1.3x). Otherwise record the number and drop it.**
  - A new graph changes the `models-1` release asset, so a `models-2` release is needed. This is a Riley step (publishing).

### 5.4 Cache sentence and document vectors (backlog 4d-7; needed by 3.4 and 4.3)

- `core/models/vector_cache.py`: parquet files under `<data>/vectors/`, keyed by (model id, model sha, unit, document sha256, cleaning setting). `doc_embeddings`, `carryover`, `similar()` and `bert_topics` read through it.
- Cap and "Clear" share Settings with the parse cache (5.1).
- A/B: a second `doc_embeddings` run on the same corpus is at least 5x faster (measure).

### 5.5 Smaller items (each small; do them when their section touches the code)

1. **tf-idf default `max-df-ratio`** (backlog section 5, question 4): 1.0 gives function words, 0.5 is informative. **Approved by Riley (D8): change the default to 0.5** everywhere it is written: `core/profiler/registry.py` (the `max-df-ratio` ParamSpec, around line 1317), `core/profiler/executor.py::_adapt_tfidf` (`_float(params, "max-df-ratio", 1.0)`), `core/analysis/tfidf.py::tfidf` (`max_df_ratio: float = 1.0`) and the CLI flag in `tools/tfidf.py`. Leave glance's own 0.8 (`core/insight/glance.py`) alone. Update the panels' warning to fire only when a run uses 1.0, and say in the release notes that new runs give different key words. Old runs are unchanged (their params are in their envelopes).
2. **Date annotator reads bare numbers as years** ("1500"): only accept a bare 4-digit number as a year when it is in 1000–2100 **and** next to a date word (`in`, `since`, `year`, a month name) or sentence position suggests a date. Coverage test on the SOTU fixture's `date_annotator` output.
3. **Quote attribution takes pronouns as speakers** ("he", "I"): resolve with `coreference` when available, else mark as "(pronoun)". The panel already hides them.
4. **Story-shape standardization** (2.6): standardize features before SVD/NMF (approved, D7). Update the story-shape tests' expected values in the same commit, with the reason in the message.
5. **NER entity timeline has no per-document tokens**, so it can't be a rate (backlog 4d-5). Add `Tokens` in the executor, as was done for names.
6. **3-class sentiment**: still needs a licence check. **Don't pick a model in this plan.** When revisited: list 2–3 candidates with their licence texts quoted from their model cards into `docs/LICENSE_REVIEW.md`, and let Riley choose.
7. **Version string in the sidebar** (`App.tsx`: "Desktop beta · 0.4.0"): check `scripts/bump_version.py` updates it (it did for 0.4.0). No change expected; verify during release.

### 5.6 Deferred to 0.5.x or later (write into backlog 4d-9, don't build now)

- Two embedding models side by side (4d-8), because section 3's multi-run panels make it easier later.
- Axis presets and SemAxis pole expansion (4d-8).
- Gensim change over time with Procrustes (4d-8). Section 1's periods make it possible for books too.
- A noise floor for change (4d-8).
- Built-in AI of any kind (API key or local model); see 4.8.5.
- Kernel-driven autocomplete (4.7).
- More than 2 sides in the Compare UI (3.2) if not done.

---

## 6. Cross-cutting: backups, docs, release notes

- **Backup manifest v5** (`desktop_backend/archives.py`) adds `document_fields`, `project_settings`, `documents.derived_from` / `derivation`, `comparisons` and `notebooks`. One version bump for the release, not one per section: bump to 5 in the first pass that adds a table, and extend v5 in later passes (it's unreleased until 0.5.0 ships). Tests restore v1–v4 fixtures and a v5 round trip.
- **Docs**:
  - `docs/DESKTOP.md`: sections for Document details, Splitting, Compare, Scripts;
  - `docs/SECURITY.md`: smart scripts, pasted code, and why the app has no built-in AI;
  - `docs/GET_STARTED.md`: a "books and other collections" paragraph;
  - `desktop/src/toolGuides.json`: guides for `contrast`;
  - the Learn page: short "How to" entries for the four new things (check how `Learn.tsx` sources its content).
- **Release notes** `docs/releases/0.5.0.md`, in the style of 0.4.0's (plain, user-facing, bold lead-ins). Write it at the end, not the start.
- **`docs/internal/PANELS_BACKLOG.md`**: add section 4d-9 "0.5.0" with the deferred list (5.6) and anything new found during the passes.

---

## 7. Order of work (the passes)

Each pass is sized to fit one long session. It ends with its "Done when" checks in the real app, a Progress entry, and a push. **Don't start a pass before its dependencies are pushed.**

| Pass | What | Sections | Depends on | Size |
|---|---|---|---|---|
| **P1** | Parse cache for jobs (+ per-document keying if safe), vector cache | 5.1, 5.4 | — | Medium. **Do first: it speeds up every later pass's real-app checks.** |
| **P2** | Document details, engine side: tables, detection, CSV import, `Document.fields`, runner/selection, executor `_with_details`, axis module, keyness/lexicon by field, backup v5 | 1.3–1.6, 1.9 | P1 (not strictly, but avoids re-parsing during checks) | Large |
| **P3** | Figures along the axis: the 61-panel generalization, `groupings()`, TypeScript axis ticks, the coverage test | 1.7 | P2 | Large. Mechanical but wide; the coverage test makes it safe |
| **P4** | Document details UI: grid, detect dialog, import dialog, axis switch, CorpusScope field filters | 1.8, 1.10 | P2, P3 | Medium |
| **P5** | Books: splitter, transcripts, split UI, the Novels walk and fixes (including story-shape standardization) | 2 (all), 5.5.4 | P4 | Medium-large |
| **P6** | Stage directions and NER ignore list | 5.2 | P2 (settings table) | Small-medium |
| **P7** | Script library (`core/script`, `nlpsuite` shim), raw-text term rates, reference generator, templates as tests | 4.2, 4.3 | P1, P2 | Large |
| **P8** | Script kernel and security test change, notebooks table, Run and save | 4.4–4.6 | P7 | Medium-large |
| **P9** | Scripts page UI: CodeMirror, outputs, automatic charts, "Copy matplotlib code", reference panel, export/import, the AI chatbot guide; **the A/B acceptance test** | 4.7, 4.8, 4.10 | P8 | Large |
| **P10** | Contrast engine: sides, measures, keyness, themes, meaning, carry-over, registry edits, runner cross-project reads | 3.2–3.6, 3.9 | P1, P2 (P7 for `nlp.compare`) | Large |
| **P11** | Compare page UI and contrast panels; the SOTU vs Inaugural walk | 3.7, 3.8, 3.10, 3.11 | P10 | Large |
| **P13** | Leftovers: BERT benchmark / last-layer graph, 5.5 items, docs, release notes, backlog 4d-9 | 5.3, 5.5, 6 | all | Medium |
| **P14** | Release (Riley's steps): version bump, test build, publish; `models-2` if 5.3 shipped a graph | — | all | Small |

P6 can run in parallel with anything after P2. P10 can start after P2 if scripts are delayed; in that case `nlp.compare` is added to the library in P10 instead.

---

## 8. Decisions for Riley (flagged, not guessed)

| # | Decision | Recommendation | Why it matters |
|---|---|---|---|
| D1 | Build order: scripts before comparison, or comparison first? | Scripts first (P7–P9 before P10–P11) | The library gives comparison a tested engine call; swap them only if comparison is needed sooner |
| D2 | Sidebar order and names: "Scripts" and "Compare", where? | **Decided (Riley, 2026-09-25): their own sidebar group, below Models.** The heading is "WORKSHOP" (a placeholder name, easy to change: it is the `section` string in `navigation`); Scripts first, then Compare. RESEARCH keeps Interactive, Analyze & visualize, Past runs, Models | Keeps the RESEARCH group as it is |
| D3 | When a book is split, should the whole book stay as a document too? | **Decided (Riley, 2026-09-25): no.** The original goes to Trash; a "keep the whole book too" checkbox exists, off by default | Keeping both double-counts every word in corpus-wide results |
| D4 | AI help with scripts: built into the app, or a guide people paste into their own chatbot? | **Decided (Riley, 2026-09-25): the guide only** (4.8). No API key, no network, no local model | Keeps the app offline; people already have a chatbot in the browser |
| D6 | Stage-direction cleaning on by default for **existing** projects? | Offer it with a preview; on by default only for new projects | Turning it on changes existing results' numbers |
| D7 | Story-shape standardization (5.5.4) changes earlier numbers | **Decided (Riley, 2026-09-25): yes**, with a note in the release notes | Components are currently mostly sentence length |
| D8 | tf-idf default `max-df-ratio` 1.0 → 0.5 | **Decided (Riley, 2026-09-25): yes**, with a note in the release notes | The default currently returns function words |

---

## 9. Things that will bite (read before the relevant pass)

- **A dated-output regression hides easily.** Section 1 touches the column layout of every per-document table. Snapshot the SOTU fixture's outputs **before** P2 (all tools through `execute`, written to a temp dir) and diff after. The only allowed differences are **added** columns (`Order` absent, fields added, `Position`, `Position label`). Write that as a test, not a manual check.
- **Column name collisions.** A user field named `Document`, `Date`, `Tokens` or `Sentence ID` would clash with tool columns. Reserve the CoNLL column names (`core/conll/schema.py::Col`) and the executor's (`Date`, `Year`, `Order`, `Position`, `Position label`) as field names: refuse them in `set_fields` with a message.
- **Order values that aren't numbers** ("Prologue", "Epilogue"). The splitter gives prologues order 0 and epilogues N+1, with the title kept. A typed non-number Order is refused with a message.
- **Two documents with the same Order** (two books both with chapter 1): the axis is still valid, but figures must group by `Work` when a `Work` field has more than one value. Otherwise lines zigzag between books. Add this to the coverage test (fixture with two works).
- **The kernel and Windows.** Kill before closing pipes (see `Runner._stop_worker`'s docstring). Kernel processes must die when the server exits: register cleanup in the server's shutdown, and also have the kernel exit when its stdin closes. **Test that no kernel survives a server stop** (0.1 "clean up processes").
- **Matplotlib in a long-lived kernel leaks figures.** `nlp.figure` closes the figure after saving. At the end of each cell, close any figures the user left open, after capturing them as outputs (Jupyter's behaviour).
- **The frozen build.** `nlpsuite`, `core.script`, the templates (data files), CodeMirror (JS bundle) must all be in the build. Extend `scripts/smoke_desktop.py` to start a kernel, run `import nlpsuite as nlp; nlp.tools()`, and shut it down.
- **Cross-project reads and restore.** A comparison run restored without its other project must still open (artifacts are self-contained) but must not offer "Run again" without explaining why.
- **The guide must stay deterministic**: a timestamp, random id or unsorted dict in it makes every release's guide look different in a diff. Build it with sorted inputs and test byte equality.
- **Don't edit a module while a test run imports it** (runner and kernel tests spawn processes that import from disk).
- **pytest temp paths**: `--basetemp C:/t/<short>` and delete afterwards.

---

## 10. Glossary

- **Axis**: how a corpus's documents line up: `time` (by Date), `order` (by Order) or `none`. Per project. Chosen automatically, changeable on the Corpus page.
- **Alignment** (comparison): comparing like with like, within each value of a shared field (Speaker) or each shared period.
- **AI chatbot guide**: the generated document people paste into their own chatbot so it writes notebook code that calls the suite. The app itself contains no AI.
- **Carried over** (comparison): a sentence on side A whose closest sentence on side B is above a similarity threshold calibrated on the comparison itself.
- **Cell**: a Python or Markdown box in a notebook.
- **Details / fields**: named facts about a document (Speaker, Chapter, Party...). Sources: file name, spreadsheet, split, typed.
- **Envelope**: a run's `result.json`, the provenance record every result has (R8).
- **Kernel**: the process that runs a notebook's cells and keeps their variables.
- **Library (`nlpsuite`)**: the Python module scripts use to call the suite's tools.
- **Live vs published**: live = computed for looking (bench, interactive cells), nothing saved; published = a run with an envelope, visible in Past runs.
- **Notebook**: a saved script (cells), stored as nbformat 4 JSON.
- **Order label / axis noun**: what one step along an order axis is called ("Chapter", "Session").
- **Periods**: the axis cut into a few buckets (decades, or blocks of chapters).
- **Side**: one document set in a comparison.
- **Stage directions**: bracketed non-author text in transcripts, like "(Applause.)".

---

## Progress

(Each pass adds an entry: date, pass number, what was built, what differed from this plan and why, measured numbers, real-app screenshots checked, and open questions.)

### 2026-09-25 — P1, P7, P8, P9: Scripts

**Built.**
- **P1:** published runs, the live bench and notebook kernels share one parse cache (`desktop_backend/project_corpus.py`).
- **P7:** the `nlpsuite` library (`core/script/`).
- **P8:** kernels (`desktop_backend/kernel.py`, `kernels.py`), the `notebooks` table, "Run and save", and backups v5.
- **P9:** the Scripts page in a new WORKSHOP sidebar group, CodeMirror cells, and outputs (text, tables with an automatic chart, figures, errors). The page also has "Copy matplotlib code", the library reference with the corpus's columns, templates, `.ipynb` import, the export zip, and the AI chatbot guide dialog (also reachable from Learn). Pasted code is marked until read, and `core/script/lint.py` points at lines worth reading.

**Measured.**
- **Parse cache (P1):** two identical `corpus_statistics` runs over 20 speeches took 66.7 s then 1.1 s (before: 57.6 s, 49.4 s).
- **The acceptance A/B, in the running app** (a copy of the workspace, the 87-speech SOTU project, the "Word group over time" template, then Run and save):
  - `data/rates_per_document.csv` matches `per_speech_rates.csv` for 87/87 speeches: rate difference 0.0, word counts and match counts equal;
  - the yearly means match `annual_means.csv` for all 85 years (difference 0.0);
  - the five-year trend is off by at most 4.4e-16.
  - The template's code is 24 lines over four cells (28 with its two comments and two blank lines). The target was about 25.
  - Run all takes 4 s on the parsed corpus.
- **Guide:** 19,884 characters short, with the corpus section.
- **Bundle:** the app bundle stayed at 530 KB (528 KB before). The Scripts page and CodeMirror load as a separate 442 KB chunk only when the page opens; the plan estimated 150 KB.

**Differed from the plan, and why.**
- **Templates** are Python data (`core/script/templates.py`), not `.ipynb` files. One source serves the page, the guide's worked examples and the tests, and there is nothing extra for the frozen build to ship. "Chapter arc" and "Who is named, when" wait for sections 1 and 2. "Find passages by meaning" waits for the vector cache (5.4).
- **Markdown cells are plain wrapped text in the editor.** `@codemirror/lang-markdown` pulls in the HTML, CSS and JavaScript parsers, and it more than doubled the app bundle.
- **Where the offline stand-in lives:** `nlpsuite_offline.py` is text in `desktop_backend/notebook_export.py`, not a module under `core`. It must not import the suite, the packaged app ships no `.py` sources to copy, and `core` may not contain file-writing code (`test_write_custody`). A test runs the exported figure cell in a clean process and checks that `core` is never imported.
- **Run all and Run and save both stop at pasted code nobody has marked as read.** The plan said only that pasted code is not run automatically; these are the two other ways it could run without a decision.
- **Library tables say which chart they are for** (`_prefer`). On real speeches the recommender's generic advice charted VADER's `Neg` instead of `Compound`, answered "Which Overrepresented in values have the largest Freq A?" for keyness, and summed sentence numbers for a table of passages. A chart spec also carries its reshaping step (`prepare`: melt, count), so "Copy matplotlib code" draws from the CSV as saved. Before, a chart of two melted columns copied code for columns the CSV did not have. The copied code for every chart kind and both reshaping steps is run in real Python by `desktop/src/notebooks.test.ts`.

**Found in the real app and fixed.**
- A figure was requested from the wrong kernel on the render after switching notebooks (a 404).
- Passages-per-year bars were sorted by size, so the years were shuffled.
- A cell run three times labelled its table "rates 3". A kernel's session also kept every shown DataFrame in memory; `Session.start_cell` now clears both between cells.

**Checked.**
- A hard-killed server leaves no kernel behind: the kernel exits when its stdin closes.
- `scripts/smoke_desktop.py` now starts a kernel, imports `nlpsuite`, lists the tools and shows the corpus. It passes against the source engine; the frozen build itself was not rebuilt in this pass.

**Open.**
- 4.10's guide test with two real chatbots needs a person at a browser.
- The figure beside `immigration_time_series.png` (checked 2026-09-25, drawn from the template's own cells over the real corpus): the five-observation trend is the same line point for point (the 1955 bump, the flat 1970s, the 1985 dip, the 2020 peak near 2.8). The old script's figure also draws yearly-mean dots; the template draws one dot per speech. Add a `yearly` scatter to the template's figure cell if the yearly means should be drawn too.
- Backlog 4d-9 lists what comes next.

### 2026-09-26 — document details (section 1, first slice), P10, P11: Compare

**Built.**
- **Document details** (`core/io/filename_fields.py`, `desktop_backend/fields.py`): details are read from file names on every read and never stored, so renaming a file updates them. Details typed in, taken from a CSV or set by a split are stored in `document_fields` and win in that order (typed > CSV > split > file name). Selections can filter on details, and a document's Date detail is its date. Backups are now v6 (details, settings, comparisons).
- **P10:** the `contrast` tool (`core/contrast/`). It runs the size-and-style measures, distinctive words, focus word rates, tone and shared topics over 2 to 6 sides. Sides can come from different projects, and each document is read from its own project. The sides can be lined up by any detail every side has, or by decade. Every run writes its notes first: the setting confound across collections, sides of very unequal size, and which alignment values only one side has.
- **P11:** the Compare page in the WORKSHOP group, with saved comparisons. Each side card shows a live count of documents, words, years and details. Findings appear as sentences above six figures (`core/viz/panels_contrast.py`).

**Measured, in the running app** (a copy of the workspace):
- **SOTU (87) against Inaugural (31), lined up by Speaker, with an immigration focus:** 96 s the first time, then 20–26 s with the parse cached.
  - Inaugural's distinctive words: "we, our, god, oath, freedom, liberty, heart". SOTU's: "year, congress, tonight, program, billion, budget, fiscal".
  - Readings: SOTU has longer words in 13 of 14 Speaker groups (Cliff's delta +0.71). Inaugurals are easier to read in 13 of 14 (−0.56) and more positive in tone in 11 of 14 (−0.37).
- **The same question within the mixed 68-document project** (Kind = sotu against Kind = ina, no alignment): 48 s. Every measure points the same way as the cross-project run.
- **Bundle:** the app bundle went from 530 KB to 549 KB. The Compare page is not a separate chunk.

**Differed from the plan, and why.**
- **The meaning map and carried-over passages** are listed but unavailable. Both need the sentence-vector cache (5.4). 3.11's second done-when item (carryover with "Read in context") is therefore still open.
- **Grouped ranked bars no longer overlay.** They used to overlay, the second series at half opacity. On the per-president focus figure the two colours blended into a third, and a reader could not tell which length belonged to which side. They now sit one under the other from one baseline, in the app and in the publication figure. This applies to every panel with two series per row.
- **Comparisons re-parse the joined corpus once**, because the parse cache is keyed per corpus, not per document (5.1).

**Found in the real app and fixed.**
- A side card said "1 Kind", which gives the count but not the value. A detail with one value is now named ("Kind: sotu").
- The size note counted words with a regex (522,253), while the side card and the Corpus page split on spaces (516,987). The note now uses the app's count.
- Each reading said its direction twice. The "All groups" sentence now gives the direction once, per group, then the pooled figures.
- The Focus words box and the "counted as" menu shared one `<label>`.

**Open.**
- Section 1's other pieces: choosing the axis, the details grid on the Corpus page, detail filters in CorpusScope, and the axis generalization for the 61 panels (P3). Books (section 2) have not been started.
- The focus figure's "Word group" is a text box and should list the counted groups (backlog 4d-10).
- Backlog 4d-10 lists what comes next.

### 2026-09-26 — section 1: the axis, details on the Corpus page, figures along any axis (P2, P3)

**Built.**
- **The axis** (`core/corpus_axis.py`): time, order or none, chosen on the Corpus page or left on "automatic" (time when two or more documents are dated, else order when two or more have an Order, else none). An unnamed step is a "Document". Every run records the axis it used.
- **The details grid** on the Corpus page (`CorpusDetails.tsx`): details read from file names with no clicks, typed values, and a one-line shape of the corpus ("12 documents · document 1–12 · lined up by chapter").
- **Per-document tables carry the details** as columns (`Order`, `Position`, `Position label`, one per detail). A detail whose name a tool column already has becomes `Detail: X` (diagnostic `DETAILS_COLUMN_RENAMED`).
- **Keyness by a detail:** Kind = sotu against Kind = ina with no regex.
- **The 61 figures that needed Date or Year now ask for the axis** (1.7). On a dated corpus they draw exactly what they drew before: all 137 outputs match the previous commit mark for mark (`tests/fixtures/axis/dated_outputs.json`). On a book they draw across the chapters, and with no axis they draw what needs none or say to add a date or an order.
- **Group by any detail** (1.10): the group-by menus of the by-group, model-score and verb-profile figures list the project's details (Party, Kind).
- **Lexicon counts** along each Order value, the axis's periods, or a detail (`by=order|period|field:<name>`). The rate line draws `order` across the chapters. The heatmap reads chapters in order and names the detail.
- **Order axes print "Ch. 5"** in the app and in the publication figure. `PreparedPanel.x_axis` says which kind of axis a figure drew, so a renderer no longer guesses years from the range of the numbers.
- **The run dialog filters by detail values and by an order window** (1.5), beside the date window. The run's reproducibility record says which chapters and details it was narrowed to.
- **The glance** says "across 12 chapters" and "across the chapters it rises". Its cache key covers the details and the axis.
- **tf-idf's default `max-df-ratio` is 0.5** (D8), in the registry, the adapter, the function and the CLI. The panels' warning now fires only for a run that used 1.0. A corpus of one keeps its terms.
- **Import details from a spreadsheet** (1.4.2): `core/io/metadata_table.py`, `POST /fields/import` with `dry_run`, and `DetailImport.tsx` on the Corpus page.
  - It reads CSV and TSV, and XLSX when the Excel component is installed. Every cell is kept as text, so "01" stays "01".
  - A row matches a document by file name (exact, then without the extension, then ignoring case).
  - **A row can also match through a detail the documents already have:** a sheet of one row per president sets Party on every one of that president's speeches.
  - The preview shows which documents the rows reach, which rows reach nothing, and what each column becomes (its own name, Date, Order, an existing detail, or left out). Nothing is stored before Import.
  - Imported values sit under typed ones. Importing a corrected sheet replaces the previous import of the same details.

**Measured, in the running app** (a copy of the workspace):
- Details with no clicks: 87/87, 31/31 and 68/68 documents in the three real projects.
- **Alice as 12 chapter files:** all 11 figures of readability, VADER, keyword in context and n-grams say chapter. The readability and lexicon publication figures print "Ch. 2 … Ch. 12" ticks. A lexicon run counted per chapter shows the Queen appearing only from chapter 8 and the Cat peaking in chapter 6.
- **The mixed 68-document project:**
  - a fresh readability run grouped by Kind gives rows ina and sotu, and grouped by Speaker gives one row per president;
  - a Kind = sotu filter ran 53 speeches, and the run's scope records it;
  - an order window of 2–4 on Alice read 3 chapters.
- **The 1.10 spreadsheet check:** a hand-made sheet of 10 presidents' parties, typed in lower case, plus one president the project lacks, went through the Corpus page. The rows reached 59 of the 68 speeches through Speaker, and the sheet's Lincoln row was reported as matching nothing. The 9 speeches of the president left out of the sheet kept no Party. A fresh readability run grouped by Party gives Democratic, Republican and "(no party)".
- **A backup made before details existed** (version 5, written by the code at 57f45e2) restores with Date, Speaker and Kind on every document.
- **A run filtered to Speaker = Barack Obama** read his 9 speeches. Its `desktop_inputs.csv` lists Date, Speaker, Kind and the imported Party per document. It now also records the order window and the detail filter (`detail_filter`: "Speaker = Barack Obama").

**Differed from the plan, and why.**
- **There is no separate "period" grouping.** On a book, "decade" and "year" are answered with blocks of chapters, with a note saying so. A second name for the same buckets would only have been a way to ask for them twice.
- **Date and Order are not value filters.** Each has its own window in the run dialog.

**Found in the real app and fixed.**
- Grouping an older run by Kind refused with a list of every measure column. It now says the run predates the detail and to run the tool again.
- The small-multiples footnote said "stems are single years" under chapters.
- The lexicon heatmap sorted chapters as text (1, 10, 11, 2) and labelled a detail axis "Field:kind".
- **Importing the same sheet a second time keyed itself on its own party column.** Once Party existed, "Democratic" reached every Democratic speech, and LBJ's speeches took Obama's row. A column whose values repeat is now never used to match through a detail.

**Open.**
- Every section 1 done-when item is met. The detect-template dialog (editing the file-name pattern's parts) is still to build.
- Books (section 2).
- The release notes must say that new tf-idf runs give different key words (D8).
- Backlog 4d-11 lists what comes next.

### 2026-09-26 — P5 books, and pre-release review

Commit `3ba9560` brought the section splitter, transcript turns, stored
derivation/backup handling, Split Book dialog, order-axis book figures, stage
direction cleaning, stop-entity suggestions, vector cache, and the first round
of book-specific tool corrections into dev `main`. The earlier section 2
handoff was a mid-pass snapshot; this entry supersedes its uncommitted status.

The real-app Novels walk in that handoff found 61 *Pride and Prejudice*
chapters and 12 titled *Alice* chapters, with the axis set to Chapter. Its
edition of *Pride and Prejudice* did not contain Volume headings, so the
Volume I/II/III acceptance claim needs a different edition or a separate
parent-heading fixture. Tests cover parent headings and a three-speaker
transcript; the transcript dialog still needs a recorded real-app check.
Ten formerly figure-less tools drew chapter figures in the walk. The original
finding was not a pass for all 13: `gender_annotator` lacked the local NLTK
`names` resource, `sentiment_swn_hedono` is outside the desktop catalog, and
the remaining coverage needs to be counted explicitly. Book prose that calls
documents "speeches" and stored-file hover labels remain in the panels
backlog. Treat section 2.10's unchecked boxes as open until the real-app
checks and screenshots are recorded.

#### Release readiness check before final commits

**Assessment:** The Windows x64 build is a tested beta candidate for maintainer
review, not yet a production sign-off. Source gates, real-corpus jobs, the frozen
engine, and packaged-payload checks pass. Clean-machine installation and
interactive workflows, the unchecked section 2/3/4 acceptance items, and
Mac/Linux platform builds remain open.

- **Source checks:** Ruff lint/format, strict mypy (355 source files), the
  TypeScript and production Vite build, and `scripts/rc_audit.py` pass. The
  complete frontend suite passed: 684 tests, one skipped. The first full
  Python runs exposed a Windows loky core-count warning and figure output
  paths too long for Windows under a repository-local pytest temp base. Both
  are addressed. The full non-model-integration Python gate passed with a
  short temp base, including a second full pass after the source-context fix.
  It also exposed a missing NER `--ignore` CLI flag; that
  now filters timeline, location, and movement outputs as the desktop does.
  Cargo's seven native launcher tests pass, as does `pip check`.
- **Feature acceptance:** sections 2.10, 3.11 and 4.10 remain open at their
  full stated scope. Carried-over passages now work for two focused sides and
  open both sources in context; the meaning map is still unavailable. A full
  87-SOTU versus 31-Inaugural carryover run aligned by Speaker found 13 pairs
  in 93 s. All 26 cited passages resolve to their full stored source context.
  The cosine cutoff still needs editorial calibration before claiming the full
  3.11 journey. The chatbot guide's ten manual prompts remain unrecorded.
- **Release package:** `docs/releases/0.5.0.md` is drafted and the source now
  identifies as 0.5.0. The Windows parser-enabled frozen engine built and
  passed `scripts/smoke_desktop.py` on the three-document sample corpus,
  including import, parse cache, analyses, notebook, exports, backup/restore,
  and shutdown. Generated third-party notices are present. The Windows x64
  NSIS installer built, and `scripts/verify_desktop_payload.py` passed its
  release launcher/runtime checks and full parser-enabled smoke. A local
  handoff with SHA-256 checksum is in `out/distribution-0.5.0/` (Windows setup
  SHA-256 `48cf330abc0b1b5fdf8b0059f5459c3a141f8e929b7890c5ce1ece4e0804ee68`). The installer
  has not been installed and used interactively on a clean machine. Mac and
  Linux builds and supported-platform acceptance in `DESKTOP_RELEASE.md`
  remain open.
- **Scope:** the 0/72 full-replacement ledger tracks broader legacy parity.
  Continue to describe this desktop release as beta; do not use that ledger
  count as a claim that the 0.5.0 features are absent or complete.

#### Real corpus checks in an isolated workspace

- SOTU (87) against Inaugural (31), aligned by Speaker: 14 shared groups;
  all fast/medium Compare methods produced tables and six figures. Cold run
  138.7 s, identical warm run 35 s. The mixed project's Kind = sotu (53)
  against Kind = ina (15) finished in 25 s. The notes reported the collection
  setting difference, large size difference and unmatched speakers.
- The 87-speech Scripts immigration notebook finished in 2 s. All 87 token
  and match counts and all 85 yearly means agreed with the old script's saved
  evidence; largest rolling-trend difference was `4.44e-16`. The generated
  chart has the same trend; the older reference also shows yearly-mean markers.
- A focused carryover run over nine Obama documents finished in 12 s and
  matched one pair at similarity 0.8079. Both source document IDs resolve
  through the preview API, and both previews contain the cited sentences.
- On all 118 SOTU/Inaugural speeches, the same 15-term immigration focus
  found 13 reciprocal pairs at similarity 0.78 or higher across shared
  Speakers in 93 s. All 26 cited source passages were located. Four would
  have missed the 30,000-character preview or its exact text search, so
  Compare now requests a short excerpt from the complete stored document.
  Several `border` and `citizenship` pairs concern geopolitics or civic duty
  rather than immigration; review the focus terms and cutoff before presenting
  these as thematic carryover.
- The shipped three-chapter Alice fixture split into three derived documents
  with the source in Trash. A readability run finished in 3 s, and its table
  and figure used Chapter, Order and `Ch. 1`–`Ch. 3`. Full 12-chapter Alice and
  61-chapter Pride and Prejudice acceptance needs the complete source texts.
- SVO extraction kept the same 430 ordered outputs and diagnostics as dev
  `HEAD` on 10,000 real cached tokens, while falling from 11.381 s to 0.418 s.
