# FR-4.4 oracle notes — where every expected value came from

Oracle: NLTK 3.10.3 `nltk.corpus.wordnet` reading Princeton WordNet 3.0,
downloaded 2026-09-03 to a scratch dir outside the repo
(`$TEMP/opencode/wn_oracle`, never committed, never read by production code).
Legacy spec: `NLP-Suite-1.6.38/src/semantic_aggregation_WordNet_util.py`
(`aggregate_GoingUP`, `disaggregate_GoingDOWN`, `_climb_to_top`,
`_climb_to_target`, `_resolve_anchor_synsets`, `_get_all_hyponyms`).

How each file below was produced: the legacy pure helpers were copied
verbatim into a scratch script (oracle only, not shipped) and run over the
`*_words.csv` inputs. Every row was then hand-checked against the rules the
code implements — first sense wins, category = the synset's own supersense
unless its lexname is `noun.Tops`, unknown words become `Not found`.

Hand-verified facts a reviewer can check without NLTK
(any WordNet browser shows the same):

- `dog`/`cat` first noun senses are `dog.n.01`/`cat.n.01`, both
  `noun.animal` → category `animal`, single-step path. Same for
  `city.n.01` → `location`, `love.n.01` (`noun.feeling`) → `feeling`,
  `table.n.01` (`noun.group`) → `group`.
- `person.n.01` has lexname `noun.Tops`, which is NOT in the legacy top set;
  its whole hypernym chain (`organism` → … → `entity.n.01`, which has no
  hypernyms) is also `noun.Tops` → category `unknown`. This is legacy
  behavior, kept: the new suite must reproduce it, not "fix" it silently.
- `happy` has no noun or verb senses → `Not found` in both POS runs.
- `zxqfrbl` has no senses → `Not found`.
- `' Dog'` (leading padding/case) is stripped + lowered before lookup →
  same row as `dog`. Deduplication happens on the raw input (legacy
  `unique()` runs before cleaning), so `dog` and `' Dog'` yield two
  identical output rows — replicated, not fixed.
- Verbs: `run.v.01`/`chase.v.01` are `verb.motion`; `love.v.01` is
  `verb.emotion`; `give.v.01` is `verb.possession`; `see.v.01` is
  `verb.perception`.
- Anchor `canine` resolves to the FIRST noun sense `canine.n.01` (the tooth),
  which is not an ancestor of `dog.n.01` → `(other) animal`. First-sense
  resolution is the spec even when linguistically surprising; the fixture
  pins it.
- Anchor `canine.n.02` (explicit synset name) IS the direct hypernym of
  `dog.n.01` → `canine`, path `[dog.n.01, canine.n.02]`; anchor `carnivore`
  (first sense `carnivore.n.01`) → path
  `[dog.n.01, canine.n.02, carnivore.n.01]`.
- Anchor `person` on `dog`: no ancestor match → `(other) animal`.
- Anchor `dog.n.01` on `dog`: the synset matches itself → `dog`.
- Unresolved anchor `zxqfrbl`: ignored with a warning diagnostic; the word
  still classifies via the normal top climb (`animal`).
- DOWN(`city`): `city.n.01` has 3 lemmas (`city`, `metropolis`,
  `urban center`) plus one hyponym each for provincial/state/national
  capital — 6 rows total. Rows after the anchor's own lemmas follow BFS
  order with siblings visited in synset-name order (`national` <
  `provincial` < `state`): NLTK returns relation order nondeterministically
  across processes (verified: three runs, three orders), so the new suite
  normalizes visit order and the legacy's run-to-run row shuffle is
  intentionally not reproduced. Definitions, examples, and lemma-count
  frequencies copied verbatim from the oracle run; the empty `Examples`
  cells for the three capitals are real (WordNet 3.0 has no examples for
  them), not fixture damage. DOWN includes the anchor synset's own lemmas
  (legacy `_get_all_hyponyms` seeds the queue with the synset itself).
  Frequency is the synset-level summed lemma count repeated on each of its
  lemma rows (`city` rows carry 112 = 103 + 7 + 2, not the per-lemma
  count) — this matches the legacy computation exactly.

Out of scope for these fixtures (follow-up packets, not this item):
multi-word phrases, non-English gating, the by-sentence CoNLL join.
