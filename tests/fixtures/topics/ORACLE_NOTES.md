# FR-5.7 oracle notes — Gensim LDA reference run

Oracle: gensim 4.4.0 (`LdaModel`), recorded 2026-09-03 on the
`cats_cars_texts.csv` fixture (6 docs, two disjoint vocab families).
Legacy spec: `topic_modeling_gensim_util.py:517-525` (LDA params),
`:115-132` (dominant topics), `:328-378` (`corpus_size_advice`).

Reference procedure (params = the legacy's, seed fixed):

```python
Dictionary(texts) -> doc2bow
LdaModel(corpus, id2word, num_topics=2, random_state=100, update_every=1,
         chunksize=100, passes=10, alpha='auto', per_word_topics=True)
```

Exact observed output (informative, not asserted byte-for-byte —
LDA weights are backend-version-sensitive; tests assert the stable
properties below):

- topic 0 top-6: car 0.1798, road/vehicle/drive/engine/wheel 0.1398–0.1399
- topic 1 top-6: cat 0.1798, feline/meow/purr/whisker/kitten 0.1398
- dominant: cats docs → topic 1 (0.982–0.985), cars docs → topic 0
  (0.982–0.985); every document distribution sums to 1.0
- perplexity (log): -2.5135; coherence c_v: 0.0257 (low: the corpus is
  tiny and synthetic — recorded as plumbing proof, not a quality bar)
- same seed twice → byte-identical topics (deterministic in this env)

Stable properties the tests pin (survive backend patch versions):

1. theme separation: each topic's top-6 is contained in one vocab family
   and covers it — a permutation-invariant partition check;
2. dominant alignment: each doc's dominant topic is the topic whose top
   words contain that doc's family, with contribution > 0.5;
3. determinism: same seed twice → identical frames;
4. rows of every document distribution sum to 1.0 (within 1e-6);
5. topic ids are 0-based Gensim-native everywhere (the legacy used +1 in
   the keywords CSV but 0-based in the dominant CSV — the port uses one
   rule, documented here).

Intentional fixes vs the legacy (defect-grade, recorded):

- `stop_words.append(['from', 'subject', 're', 'edu', 'use'])` appended one
  LIST element, so those words were never filtered — the port uses a real
  set union (sklearn's 318 English stopwords + those five words);
- no phrase detection in v1 (legacy Phrases min_count=5/threshold=100
  almost never fires on class-size corpora) — follow-up packet, not silent;
- pyLDAvis HTML/browser launch is out (viewer renders the CSV artifacts);
  MALLET stays FR-5.8.


## Intertopic Distance Map and lambda relevance (added with lda_gensim)

HW2 grades the Gensim GUI's two panels: the Intertopic Distance Map and the
effect of the relevance metric lambda. Both are computed natively rather than
by adding pyLDAvis as a dependency:

- placement: multidimensional scaling of the pairwise Jensen-Shannon distance
  between topic-word distributions (pyLDAvis's own metric). Identical topics
  collapse to one point by rule rather than by MDS's random init.
- term relevance: `lambda * log p(word|topic) + (1 - lambda) * log p(word)`
  (Chuang et al.), with `p(word)` from the fitted corpus. The table carries
  Saliency beside Relevance because the GUI shows both readings.

Verification is definitional (tests/test_lda_views.py): hand-checkable
matrices, not byte-pinned layouts. The same-seed test pins determinism.

MALLET's half (lda_mallet) is separate tooling: `--output-doc-topics` is read
into the same dominant-topic shape as the Gensim output so the two backends
can be compared as tables (tests/test_lda_mallet.py, fixtures
`mallet_doc_topics_*.txt`).
