# FR-5.5 oracle notes — Word2Vec reference run

Oracle: gensim 4.4.0 `Word2Vec`, recorded 2026-09-03 on
`sentences.csv` (8 sentences, two disjoint vocab families).
Legacy spec: `word2vec_Gensim_util.py:163-169` (train call),
`word2vec_distances_util.py:24-30` (cosine), `:72-219` (distances).

Reference config (fixed everywhere it matters):

```python
Word2Vec(sentences, vector_size=20, window=2, min_count=1, sg=1, seed=7, workers=1, epochs=20)
```

Exact observed similarities (asserted with 1e-3 tolerance):

- sim(cat, kitten) = 0.0953 vs sim(cat, car) = -0.1791 (within > cross)
- sim(engine, wheel) = -0.0728; sim(engine, purr) = -0.0028

Determinism finding: byte-identical vectors across processes, including
across PYTHONHASHSEED=0/99/unset runs — gensim 4.4 builds vocab in corpus
order here, so no hash-seed coupling. Same-seed determinism is asserted;
cross-version weight stability is NOT (BLAS may shift the 4th decimal,
hence the tolerance).

Why the fixture does not pin theme separation strongly: at this corpus
scale similarities are noisy (engine-wheel is negative!). The port pins
what is stable — determinism, structure, counts, ordering, guards — plus
the one robust margin above (within-theme positive vs cross-theme
negative, margin 0.27). Real class corpora are orders of magnitude larger.

Intentional differences vs the legacy (defect-grade, recorded):

- the legacy passed NO seed (nondeterministic every run) — the port
  requires one (`seed`, default 42, `workers=1` fixed) and records it;
- training sentences come from the CoNLL table ((Document ID, Sentence ID)
  groups) instead of a fresh stanza run — same sentences, no re-parse;
- stopword filtering defaults OFF (embeddings learn from function-word
  contexts; removal is a toggle, not the default);
- skip-gram default (`sg=1`, better for small corpora/rare words);
- t-SNE coordinates come from sklearn (core dep), seeded, as data —
  the legacy's interactive plot call is out;
- save/load is gensim-free (npz + json sidecar): distances can run from a
  saved model without the optional extra installed.
