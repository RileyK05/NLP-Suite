# Fixture probes (`tests/fixtures/probes/`)

Defect probes per legacy `planning/01_fixture_corpus_spec.md`. Every fixture
exists to pin a **named legacy defect** — a fixture without a cited defect is
dead weight. Unlike the representative `mini-corpus/`, these files are minimal
and adversarial.

Deviations from the spec (documented, not silent):

- `pcace_minimal/` ships as CSVs, not xlsx: the new suite has no xlsx import
  path and must not gain an `openpyxl` core dependency for one fixture. The
  fixture covers the code-validation/analysis contract instead of the legacy
  xlsx-reader bugs (which have no new-suite counterpart).
- `lockfile.bin` is a simulation (per the spec itself): the test holds a
  read-only directory and asserts the writer fails fast with a diagnostic.
  The new writer has no retry loop at all, so there is nothing to bound —
  the probe pins that a write failure is a `Result`, never a hang.
- `19_malformed.csv` fails the run with a diagnostic naming the file and
  the parse error, instead of the spec's "skip rows with a count". Skipping
  rows silently changes statistics; per the suite's fail-big policy the
  whole file is rejected loudly. No valid row is silently corrupted either
  way — that is the probed defect.
- Empty CoNLL tables yield empty `Result`s with no crash. The spec asks for
  an additional diagnostic; that is recorded as follow-up (touching ~20
  analysis modules is out of scope for the fixture packet).

## Text fixtures

| File | Probes (legacy defect) | Expected new behavior |
|---|---|---|
| `00_empty.txt` | `Stanza_util` empty-file `break` kills the corpus; unguarded `data[0]` / `df.iloc[0]` in CoNLL utils; `compute_sentence_table` on empty frame | empty doc → EMPTY_DOC diagnostic, siblings still parsed; empty tables → empty results, no crash |
| `01_no_terminal_punct.txt` | final-sentence drop in `sentence_division` / `CoNLL_record_division` (`CoNLL_util.py:434-437, 475-478`) | all sentences emitted, including the last |
| `02_unicode.txt` | `IO_csv_util` NUL-byte rewrite of the input; `file_checker_util` chardet-None crash + `errors='ignore'`; curly/CJK/emoji through parse→CoNLL→CSV | source bytes never rewritten; NUL stripped with `NULL_BYTE_STRIPPED` info; unicode preserved |
| `03_long_sentence.txt` | per-sentence pipeline instantiation; per-sentence O(n²) blowups; sentence-length limit enforcement | one ~10k-word sentence parses with a single cached pipeline instance |
| `04_doc.txt` … `13_doc.txt` | `str(tok_Document_ID)[:-2]` chops IDs ≥ 10 (`CoNLL_table_search_util.py:452`); `'1.0'` sentinel in division | dense IDs incl. 10–13 survive intact everywhere |
| `14_my.report.v2.txt` | `[0:-4]` extension-strip idiom (multi-dot names) | stem stays `my.report.v2` |
| `15_report_1999-05-04.txt`, `16_report_2001-11-30.txt` | `file_classifier_date_util` swapped sep/format args | dates extracted from filenames |
| `17_dialogue.txt` | `lstrip`/`rstrip` swap in `whole_sent` reconstruction; sentence ends after closing quotes | boundaries correct after `."` |
| `18_nonenglish.txt` | hardcoded `['English']`; always-True language-availability check | configured language honored; missing model fails loudly (`PIPELINE_MODEL_MISSING`), never silently |

## CSV fixtures

| File | Probes | Expected new behavior |
|---|---|---|
| `conll_good.csv` | `normalize_to_canonical` round-trip; positional indexing | canonical columns by name; validates |
| `conll_14col.csv` | 14-vs-15-column handling (`CoNLL_util.py:280`) | extra `Date` column preserved |
| `conll_universal.csv` | `universal_to_penn` mapping; analyzer re-reading raw file with wrong tagset | mapped to Penn once, at load |
| `conll_empty.csv` | unguarded `data[0]` / `df.iloc[0]` everywhere | empty results, no crash (diagnostic = follow-up) |
| `19_malformed.csv` | bare-`except` swallowing; `on_bad_lines='skip'` inconsistency | hard failure with diagnostic (see deviation note) |

## Special fixtures

| Fixture | Probes | Expected new behavior |
|---|---|---|
| lock simulation (test-held) | `IO_csv_util` infinite permission-retry loop | fail fast with `WRITER_FAILED`, bounded time |
| `pcace_minimal/` (`setup_Complex.csv`, `setup_Simplex.csv`, `data_codes.csv`) | `DB_PCACE_data_analysis_util` parent/child `IndexError`; global rebind truncating `setup_Complex_lib` | code validation + per-category analysis correct; no global mutation |
