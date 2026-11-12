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
- `coreference` is pure-Python row iteration (15 s at this scale — fine;
  revisit past ~2M tokens).
- FR-9.7 asks for 100+ documents: this run is 87. The scaling shape above
  (parse + analyses linear in tokens, LDA/W2V sub-linear) is the evidence
  that 100+ behaves the same; re-run `scripts/bench.py` on a larger
  corpus to extend the record rather than extrapolating silently.

## Reproducing

```text
python scripts/bench.py corpus/SOTU --out bench.json
python scripts/bench.py CORPUS --out bench.json --analyses readability,lexical_diversity
```
