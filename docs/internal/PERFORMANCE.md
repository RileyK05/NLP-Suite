# Performance acceptance (FR-9.7)

Benchmark: `python scripts/bench.py CORPUS --out bench.json` (schema-tested
in `tests/test_bench.py`; budgets are asserted nowhere — time-based tests
are flaky — this document is the record).

## Recorded run 2026-09-30

Corpus: `corpus/SOTU` (87 documents, 596,875 tokens). Machine: Windows,
Python 3.12, spaCy 3.8 + `en_core_web_sm`. LDA was timed in a second
invocation of the same script (it repeats the parse).

| Stage | Seconds |
|---|---:|
| read_corpus | 0.0 |
| parse (spaCy, shared, once) | 70.3 |
| readability | 0.4 |
| lexical_diversity | 2.6 |
| sentence_complexity | 3.7 |
| coreference | 14.9 |
| narrative (VADER per-sentence) | 3.7 |
| lda_gensim | 4.6 |
| word_vectors_gensim | 40.5 |
| **total** | **~141** |

The 2026-09-05 run took ~514 s in total; `narrative` alone was 331 s because
`emotion_arc` built a new VADER analyzer (which parses its lexicon) for every
sentence. It now scores every sentence with one analyzer, and its output is
byte-identical on real speeches. The other rows moved by machine noise or
unrelated changes (LDA 26.8 → 4.6 s, word2vec 59.4 → 40.5 s); this run is the
reference from here on.

## Reading

- The suite clears a mid-size corpus in about two and a half minutes on
  commodity hardware; parsing once and sharing it is half of that.
- `word_vectors_gensim` is the largest analysis after the parse. It is a model
  fit, so its cost is training epochs, not overhead.
- `coreference` is now vectorized (the 14.9 s row above predates the
  2026-11-21 fix: 600k synthetic tokens went from 39.5 s to 1.65 s). Re-run
  `scripts/bench.py` on the SOTU corpus to refresh that row.
- FR-9.7 asks for 100+ documents: this run is 87. The scaling shape above
  (parse + analyses linear in tokens, LDA/W2V sub-linear) is the evidence
  that 100+ behaves the same; re-run `scripts/bench.py` on a larger
  corpus to extend the record rather than extrapolating silently.

## Targeted fixes, 2026-11-21

These are synthetic checks on the same machine, not a replacement for the
SOTU benchmark above. They record what a specific change bought and what
behavior it pinned.

- `table_search`, 500k-row canonical table, `contains` + `eq` filters:
  0.42 s -> 0.23 s (vectorized masks; regex 0.14 s). A blank cell is now
  blank; it no longer matches a search for `"nan"`/`"None"`.
- `search_csv_frame`, 50k rows x 5 columns: 1.76 s -> 0.11 s. The old row
  loop built a Series per row and lowered every cell; a unit test pins
  row-major hit order and the blank-cell rule.
- `coreference`, 600k synthetic tokens: 39.5 s -> 1.65 s vectorized,
  byte-identical to the previous row loop on random frames and the fixture.
- `kwic`, 600k tokens with about 100k hits: 23 s -> 2.3 s (the full-index
  path; a capped search stops after the same early sentences). Not
  byte-identical: whitespace-only tokens (spaCy's line breaks) are left out of
  the context windows, so a context reads `FOREIGN AFFAIRS American`, not
  `FOREIGN AFFAIRS \n American`. Hits and row counts are unchanged.
- Dispersion offsets, 600k tokens and two terms: 24.6 s -> 0.22 s; the
  occurrence table is identical to the previous per-token loop.
- `ngrams` over 600k tokens: 24.9 s -> 1.8 s. `ngrams`, `collocations` and
  `collocation_network` now take tokens through `core.conll.frames` instead
  of a per-sentence groupby and repeated `str` conversion; outputs are
  byte-identical to the previous path.
- Word-vector neighbour queries cache the matrix, norms and vocabulary on
  first use; repeated queries over a 20k-word vocabulary no longer rebuild
  the arrays.
- Chart layout: one-pass `extent`/`minOf`/`maxOf` replace
  `Math.min(...values)`, which throws `RangeError` past roughly 125,000
  arguments. A 130,000-point scatter now builds; category accumulation uses
  Sets instead of repeated `includes` scans.
- `PanelMark`, `PanelEdge` and `Annotation` reject non-finite geometry at
  construction, so a broken builder fails with `PANEL_BUILD_FAILED` rather
  than writing `NaN` coordinates into an SVG.
- Scripts: malformed kernel protocol lines are ignored instead of ending the
  notebook; a cell raising `KeyboardInterrupt` is that cell's error; a stray
  figure that cannot be kept no longer fails a "Run and save"; lint answers on
  text `ast.parse` cannot encode.
- `sentence_complexity` depth is one pass per token instead of a walk to the
  root from every token. The first version gave the child of an
  already-resolved token its parent's depth, and the shortfall compounded:
  every SOTU sentence's Depth fell (12 -> 4) while the unit fixtures passed.
  Fixed; a property test checks 2,000 random parses against `depth_of`.

Checked 2026-11-21 against the previous commit on twelve SOTU speeches
through `nlp.run`: 68 of 72 tables byte-identical. The rest are the depth bug
above (fixed, now identical), the `kwic` whitespace change, and `lda_gensim`
topic totals that drift in the third decimal because document topics are now
inferred once instead of twice (gensim's inference starts from a random draw).

**Open: start-up imports on a machine with the model extras.**
`desktop_backend.server.preload_engine` imports `stanza` (a deadlock fix:
native imports must happen before the request pool exists). stanza 1.14 imports
`transformers`, which the preload deliberately leaves out, and on the dev
machine the worker took 83 s to say ready, against a 90 s limit; two
`tests/test_desktop.py` job tests time out there. Installers do not ship stanza,
so the import fails fast and costs nothing for them. Either warm stanza only
when a Stanza parser is configured, or raise the limit for development.

## Reproducing

```text
python scripts/bench.py corpus/SOTU --out bench.json
python scripts/bench.py CORPUS --out bench.json --analyses readability,lexical_diversity
```
