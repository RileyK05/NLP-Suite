# PC-ACE decomposition (FR-6.9, Class L read-only step)

24,618 legacy lines across four GUI-coupled modules
(`DB_PCACE_data_analysis_main.py` 3,071 + `..._util.py` 7,398 +
`DB_PCACE_data_validation_main.py` 1,421 + `corpus_checker_PCACE_data_main.py`
838) must not be rewritten in one prompt. The NG suite already owns the
seed (`core/pcace/core.py`: code parse, SQLite create/read, grammar info;
`core/pcace/analysis.py`: code validation + analysis; `tools/pcace.py`,
`tools/pcace_analysis.py`, `tools/sql.py` CLIs). The children below map
each legacy capability to its NG home. Each child gets its own dossier
and C0–C5 cycle; none may edit another child's files.

## Children

- **FR-6.9a source import** — 18 xlsx PC-ACE tables (`data_Document`,
  `data_Complex`, …) with overlapping ID fields into one SQLite DB.
  Legacy: `import_PCACE_tables`, `create_pkl_file`, `_load_pcace_df`.
  NG home: extend `core/pcace/core.py::create_db` (needs `openpyxl`;
  new `tables` extra or reuse `converters`). Evidence: fixture xlsx trio
  → joined DB, ID-overlap enforced. Class M.
- **FR-6.9b grammar validation** — required-object rules, complex/simplex
  parent/child navigation, rename/remove/merge grammar objects.
  Legacy: `get_required_value`, `toggle_required_value`,
  `rename_grammar_object`, `remove_grammar_object`,
  `merge_grammar_objects`, `get_setup_complex_*`. NG home: new
  `core/pcace/grammar.py` (pure, pandas/SQLite-backed) + tests on a
  fixture DB. No GUI tree widget. Class M.
- **FR-6.9c relational queries** — the SQL query builder (SELECT/GROUP BY
  templates over the joined tables). Legacy: query-builder sections of
  `..._util.py` (~lines 2192–2900). NG home: extend `tools/sql.py` with
  PC-ACE-aware table inventory. Class M.
- **FR-6.9d aggregation families** — frequency/crosstab distributions over
  coded fields (the charts the legacy pops per query). Legacy:
  aggregation + `charts_util` call sites. NG home: reuse `core/viz`
  surfaces over query CSVs; no new chart code. Class S.
- **FR-6.9e corpus cross-check** — coded spans vs. corpus texts
  (`corpus_checker_PCACE_data_main.py`). NG home: new
  `core/pcace/crosscheck.py` over `read_corpus` + the SQLite DB. Class M.
- **FR-6.9f validation UI flow** — the validation main's guided checks as
  CLI + app page (`DB_PCACE_data_validation_main.py`). NG home:
  `tools/pcace.py` extension + app page. Depends on 6.9a–6.9c. Class M.

## Non-goals

Visual-Basic-style Access forms, in-place xlsx editing, and the
`view_grammar` Tkinter tree have no NG equivalent (see MIGRATION.md
deliberately-not-ported). Grammar edits go through `tools/pcace.py`
with preview-first semantics, like `filenames`.
