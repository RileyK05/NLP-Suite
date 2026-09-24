# FR-4.2/FR-4.3 oracle notes — full sentiment lexicons

Legacy spec: `sentiment_analysis_VADER_util.py:129-140` (package scorer,
per-sentence, ±0.05 labels), `sentiment_analysis_ANEW_util.py:54-58`
(EnglishShortenedANEW.csv), `sentiment_analysis_SentiWordNet_util.py:117-145`
(first sense, NOUN/ADJ/ADV, mean), `sentiment_analysis_hedonometer_util.py:48-53`
(hedonometer.json), `sentiment_analysis_NRC_util.py:5` (nrclex).

## The no-vendoring decision (owner-visible)

`assets/` holds only `.gitkeep`: nothing is vendored in this repo, including
the checksummed style assets. The four production lexicon files therefore
stay **user-supplied registry assets** — no bytes are committed:

- `vader_lexicon.txt` — 7,517 lines, full VADER, MIT (redistributable with
  notice). sha256 of the oracle copy:
  `6ff1180b…dd19b31c` (full: see registry stamp).
- `EnglishShortenedANEW.csv` — 13,916 rows, Bradley & Lang norms, research
  use only, NOT redistributable. Users copy it from their legacy install
  (migration guide) or request it from the authors.
- `hedonometer.json` — 10,222 labMT entries, attribution required.
- `EnglishANEW.csv` — 0 lines (empty in the oracle); ignored.
- SentiWordNet — NLTK `sentiwordnet` corpus (download-on-demand, like the
  WordNet corpus in FR-4.4).
- NRC — JSON word→[emotions] mapping; users supply the file or install
  `nrclex` (its bundled `nrc_en.json` is auto-detected).

Mini fixture files (`vader_mini.txt`, `anew_mini.csv`,
`hedonometer_mini.json`, `nrc_mini.json`) copy a handful of rows each from
the oracle/packaged sources for offline tests — de minimis excerpts with
sources named here, not substitutes for the assets.

## What each expected file came from

Probe docs (`probe_docs.csv`): one sentence each, so sentence scores equal
document scores in the fixtures.

- `vader_expected.csv` — recorded from the `vaderSentiment` package, which
  IS the legacy scorer (legacy line 129). Verified with BOTH the packaged
  lexicon and `lexicon_file=<oracle vader_lexicon.txt>`: identical on all
  four probes (lexicon drift touches other words). Packaged file sha256
  `1ec9c6e9…f6368040` (differs from oracle; behavior pinned by the
  recorded outputs, and the oracle-file path is the legacy-exact option).
- `mixed` labels **positive** (compound 0.1682 > 0.05): the package's
  negation/booster scope rules on "not good very bad" ("not good" alone is
  −0.3412, "very bad" −0.5849). This is package behavior, kept — the old
  stub's simplified window is retired with it.
- `anew_expected.csv` — hand-computed means over the oracle CSV rows
  (calculator-verified in the C1 session; the replay caught two of my own
  addition slips before they became fixtures). Function words (the/is/on/
  not/very) have no ANEW rows, but `day` (6.36/3.62/5.59), `night`
  (6.68/3.57/5.22), `floor` (5.14/3.33/6.39), and `table` do — so the
  oracle replay has MORE hits than the mini-lexicon unit tests (pos 4
  hits, neg 5 hits). The mini file deliberately stays a subset.
- `hedonometer_expected.csv` — hand-computed means over the oracle JSON
  `happs` values (calculator-verified); every probe word has an entry.
- `swn_expected.csv` — recorded from NLTK `sentiwordnet` + `wordnet`
  (first sense, universal NOUN/ADJ/ADV only, mean of pos−neg). True splits,
  not just nets (the C1 net-only probe lost them; the replay corrected the
  record): terrible→awful.s.02 (0.0, 0.625), awful→atrocious.s.02
  (0.0, 0.875), sad→sad.a.01 (0.125, 0.75), wonderful→fantastic.s.02
  (0.75, 0.0). Satellite adjectives resolve through the adjective file.
  Verbs are skipped (legacy filter): `love`/`hate` score nothing.
  Adverbs ARE scored (legacy includes ADV): `very`→very.r.01 (0.25, 0.25),
  so `mixed` has 3 hits. `day.n.01`, `night.n.01`, `table.n.01`,
  `floor.n.01` are net-zero synsets.
- `nrc_expected.csv` — recorded from `nrclex` 4.1.0 (`load_token_list` +
  `affect_frequencies`, fractions over total affect hits). `sad`, `night`,
  `table`, function words are ABSENT from the map (no hit ≠ zero score:
  they contribute no denominator). `neu` matches nothing → all-zero row.

## Ported semantics (recorded, not debated per-packet)

- VADER: per-sentence package scores; document row = mean of its sentence
  components; labels at ±0.05 (legacy lines 133–140).
- SWN verbs skipped (legacy NOUN/ADJ/ADV filter).
- NRC columns in nrclex `EMOTION_ORDER`
  (fear, anger, anticipation, trust, surprise, positive, negative,
  sadness, disgust, joy).
