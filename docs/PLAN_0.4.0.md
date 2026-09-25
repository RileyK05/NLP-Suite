# NLP Suite 0.4.0 plan

Written 2026-09-24 against dev `main` @ `ccb4071`. File paths are real as of that commit.

## The goal in one line

Every model-based tool works right after install. Nobody visits Hugging Face, installs Python packages or sets a path. Small models ship inside the installer, and big ones are one click inside the app.

## Ground rules (same as always)

- Work on dev `main` (New_NLP_Suite). `git pull` before you start and push when you're done.
- Before each push: `ruff check .` · `python -m mypy core tools app desktop_backend` · `python -m pytest` · `cd desktop && npm test && npx tsc --noEmit`.
- Pinned env: `desktop/requirements-runtime-py312.txt`. Any new runtime dependency is added there, pinned, and to the build script.
- Tests don't download real models (CI and the cloud agent can't reach Hugging Face). Tests that need real models get a marker and run on GitHub runners.
- **Tauri/UI work is deferred** (Phase 6). Phases 1–5 are engine-side (Python), and each one is testable and useful without new UI.

---

## Phase 0: Why BERT doesn't work today (context)

The code exists. The frozen app just can't run it.

| Piece | Where | State |
|---|---|---|
| Contextual embeddings seam | `core/analysis/contextual.py`: `EmbeddingBackend` Protocol (`dimension`, `embed(words, contexts)`), `TransformerBackend`, `default_backend()` | Works, but only with `transformers` + `torch` |
| BERT sentiment | `core/analysis/sentiment_neural.py` → `bert_sentences(model="distilbert-base-uncased-finetuned-sst-2-english")`, `pipeline` seam | Same |
| Tools | `tools/bert_extract.py`, `tools/bert_topics.py`, `tools/word2vec_bert.py`, `tools/sentiment_neural_bert.py` (+ `word_sense_induction.py`) | Same |
| Build | `scripts/build_desktop_backend.py` excludes `torch`, `transformers` | Intentional: torch alone is ~700 MB+ |
| Availability | `desktop_backend/environment.py` `availability()` → `"embeddings": ["torch", "transformers"]` → `needs_setup` | So the tools are greyed out |
| Interactive page | `desktop/src/live.ts` `liveTools()` hides anything `needs_setup` | **That's why BERT isn't in the Interactive menu** |
| Catalog | `desktop_backend/catalog.py`: `sentiment_neural_bert`, `word2vec_bert` in `CORPUS_TOOLS`; `bert_extract`, `bert_topics`, `word_sense_induction` in `INTERNAL_TOOLS` | The last three are hidden entirely |

**Fix strategy:** replace torch/transformers at *runtime* with **ONNX Runtime + `tokenizers`** (~40 MB + ~8 MB). Models are exported to ONNX once, offline, and shipped as files. The `EmbeddingBackend` Protocol already exists, so this is a new backend class rather than a rewrite.

---

## Phase 1: ONNX model runtime (the foundation)

### 1a. Dependencies
- Add `onnxruntime` and `tokenizers` to `desktop/requirements-runtime-py312.txt` (pinned) and to a new `models` extra in `pyproject.toml`.
- `scripts/build_desktop_backend.py`: keep `torch`/`transformers` excluded, and add `--collect-binaries onnxruntime` / hidden imports as needed. Verify with `scripts/smoke_desktop.py` on a local build.
- mypy: add `onnxruntime.*` and `tokenizers.*` to the ignore-missing-imports override in `pyproject.toml`.

### 1b. Model registry: `core/models/` (new package)
- `core/models/registry.py`: one `ModelSpec` per model:
  `id`, `display_name`, `kind` (`token_embeddings` | `sentence_embeddings` | `classifier`), `dims`, `pooling` (`cls` | `mean` | `last_token`), `max_tokens`, `query_prompt` (Qwen uses one), `labels` (classifiers), `license`, `size_mb`, `sha256` per file, `bundled: bool`, `url` (download-only models).
- `core/models/locate.py`: where a model's files live:
  1. **Bundled:** inside the frozen engine (`sys._MEIPASS/models/<id>/`) or `models/<id>/` in a source checkout.
  2. **Downloaded:** the user data dir, `<app data>/models/<id>/`, which survives app updates (the updater replaces the app, not user data). Reuse whatever `desktop_backend/paths.py` already uses for app data.
  3. Override: `NLP_SUITE_MODELS` env var (dev/testing only, never mentioned to users).
- Status per model: `ready` | `not_downloaded` | `corrupt` (sha mismatch).

### 1c. Backends: `core/models/onnx_backend.py`
- `OnnxTokenBackend` implements the existing `EmbeddingBackend` Protocol (`dimension`, `embed(words, contexts)`). It needs word-to-wordpiece alignment: use `tokenizers` `Encoding.offsets` / `word_ids()` to average the target word's pieces. That replaces `_piece_text` in `contextual.py`.
  - `word2vec_bert` `_embed_layer` needs hidden layers other than the last. Export the ONNX graph with **all hidden states as outputs** (or at least the last 4) so `layers=` still works.
- `OnnxSentenceBackend`: `embed_texts(texts) -> np.ndarray` with pooling + L2 normalization per `ModelSpec`. Used by Granite and Qwen.
- `OnnxClassifier`: `classify(texts) -> list[(label, score)]`. Used for sentiment.
- Sessions are cached per model id (loading costs seconds). Use `intra_op_num_threads` = physical cores. CPU provider only.
- Batch and truncate to `max_tokens`. Keep the existing "N sentences truncated" diagnostic.

### 1d. Wire into existing code (small diffs)
- `contextual.default_backend()`: prefer ONNX if the model is `ready`, else fall back to `TransformerBackend` if `transformers` is importable (keeps dev installs working), else the existing `CTX_BACKEND_MISSING` diagnostic, **reworded for users** ("This model isn't installed. Open Models to add it"), not the pip instructions.
- `sentiment_neural.bert_sentences()`: its `pipeline` seam → `OnnxClassifier` adapter.
- `_FIX` / `_FIX_TRANSFORMERS` strings in `word2vec_bert.py` and `sentiment_neural.py`: same rewording.
- `desktop_backend/environment.py` `availability()`: `"embeddings"` requires `onnxruntime` + `tokenizers` + the tool's model `ready`. A missing *downloadable* model returns a new state, `needs_model` (with `model_id`), distinct from `needs_setup`, so the UI can later offer a Download button instead of a dead end.

### 1e. Tests
- **Tiny fixture model:** `scripts/make_tiny_onnx.py` builds a random-weight, 2-layer, 32-dim BERT + tokenizer (~1 MB), committed under `tests/fixtures/models/tiny-bert/`. It's made once, on a machine with torch. All unit tests run against it: alignment, pooling, shapes, layers, caching, sha verification, truncation diagnostics.
- Keep the existing hash-backend tests untouched.
- `@pytest.mark.real_models` tests (skipped unless `NLP_SUITE_MODELS` is set) check real outputs: SST-2 says "I love this" is positive, similar sentences have higher cosine, and so on. Run them on a GitHub runner via the `from_dev` CI input.

**Done when:** all four BERT tools run in a frozen build with no torch/transformers present, and the smoke test covers one of them.

---

## Phase 2: Bundled models (pre-installed)

### 2a. Export pipeline: `scripts/export_models.py` (dev-only, runs with torch + optimum)
For each bundled model: download from HF → export ONNX → **dynamic int8 quantization** → **parity check** against the original transformers output (cosine ≥ 0.99 on ~50 sample sentences, and classifier labels match on ≥ 98%) → write `model.onnx`, `tokenizer.json`, `LICENSE`, `NOTICE`, `spec.json` with sha256.

Run it on a GitHub Actions runner (workflow_dispatch, `models.yml`), since the cloud agent can't reach HF. Upload the result as a release asset on NLP-Suite under a separate tag, e.g. `models-1`, so the model files are versioned independently of the app.

### 2b. Which models (all Apache-2.0, redistributable with LICENSE/NOTICE)

| Model | Role | fp32 | int8 (shipped) |
|---|---|---|---|
| `google-bert/bert-base-uncased` | Token embeddings: bert_extract, bert_topics, word2vec_bert, WSI | ~440 MB | ~110 MB |
| `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | Sentiment (pos/neg) | ~260 MB | ~65 MB |
| `ibm-granite/granite-embedding-english-r2` | Sentence/document embeddings (new) | ~600 MB | ~150 MB |

Sizes are estimates. **Measure after export and record them in `spec.json`.**
Correction to what I said earlier: those sizes assume int8 quantization. At fp32 it would be about 1.3 GB, so quantization isn't optional.

**Installer impact:** roughly +370 MB (models) + ~50 MB (runtime). ⚠️ **Decision point:** if that's too heavy, move BERT base to download-only. Granite R2 is itself a modern BERT-family encoder, but the tools are named "BERT" and users expect BERT, so the default is to keep it bundled.

### 2c. Build integration
- `desktop.yml`: a step before PyInstaller downloads the `models-1` asset, verifies sha256 and unpacks it to `models/`. `build_desktop_backend.py` adds `--add-data models:models`.
- `scripts/smoke_desktop.py`: add a check that each bundled model loads and embeds one sentence.
- `.gitignore`: `models/` (never commit real weights; the tiny fixture lives under `tests/`).

**Done when:** a fresh install on a machine with no internet runs BERT sentiment and bert_topics.

---

## Phase 3: One-click downloadable models (Qwen)

### 3a. Qwen3-Embedding-0.6B
- Apache-2.0. ~2.4 GB fp32, **~600 MB int8** (estimate; measure). Exported and parity-checked by the same `export_models.py`, uploaded to the same `models-N` release.
- Pooling: last token. Queries need the instruction prompt (`Instruct: ...\nQuery: `); documents don't. Put this in `ModelSpec.query_prompt`.
- Hosting it on our own GitHub release (not a HF link) means a pinned, sha-verified file that won't move or get gated. The license and notice ship next to it.

### 3b. Download manager: `core/models/download.py`
- `download(model_id, progress_cb, cancel_event)`: streams to `<id>.partial`, **resumes** via HTTP Range, verifies sha256, then atomically renames into place. Idempotent.
- Disk-space check before starting (needs size + 10% free), with a plain-English error if it's short.
- Also `delete(model_id)`, to reclaim space.
- Proxy/offline errors become one friendly sentence, not a traceback.

### 3c. Engine API (in `desktop_backend/server.py`, UI later)
- `GET  /api/models` → list of `{id, name, kind, size_mb, status, bundled, license}`
- `POST /api/models/{id}/download` → runs as a job (reuse `desktop_backend/runner.py` / jobs), progress via the existing job-status polling
- `DELETE /api/models/{id}`
- Tests: fake HTTP server + fixture file. Cover resume, sha mismatch → `corrupt` and retry, cancel, and not enough disk.

**Done when:** the endpoints work via curl against a dev engine, and tools that pick Qwen report `needs_model` until it's downloaded, then work.

---

## Phase 4: New and upgraded tools

### 4a. Sentence/document embeddings tool: `tools/doc_embeddings.py` + `core/analysis/doc_embeddings.py` (new)
- Params: `model` (`granite` default | `qwen` if ready | `bert`), `unit` (`document` | `sentence`).
- Outputs:
  - `Doc/Model/Dim` vectors (parquet)
  - Nearest neighbours: `Doc/Neighbor/Cosine`, top-k per doc
  - 2-D map: `Doc/X/Y/Cluster` via PCA → t-SNE from scikit-learn (already a dependency, so no UMAP dependency). KMeans clusters with k by silhouette over 2–10.
  - Similarity matrix for the heatmap figure
- A **semantic search** param: `query` → rank documents/sentences by cosine (Qwen uses the query prompt).
- `doc_similarity` could gain `method=embeddings` alongside TF-IDF. Optional; only do it if it's cheap.

### 4b. Model choice on the existing BERT tools
- `bert_extract`, `bert_topics`, `word2vec_bert`, `word_sense_induction`: a `model` param. Choices come from the registry filtered by `kind`, so a new model appears automatically.

### 4c. Sentiment
- `sentiment_neural_bert` runs on the bundled DistilBERT SST-2. **Limitation to state in its guide text:** SST-2 is positive/negative only, with no neutral class. The existing `_compound()` mapping turns confidence into a −1..1 score; low-confidence scores near 0 read as "mixed". An optional later addition is a 3-class model (e.g. a RoBERTa sentiment model), but check its license before picking one.

### 4d. Catalog: make BERT visible (this is "add BERT to the Interactive menu")
- `desktop_backend/catalog.py`: move `bert_extract`, `bert_topics`, `word_sense_induction` from `INTERNAL_TOOLS` into `CORPUS_TOOLS`, and add `doc_embeddings`.
- `core/profiler/labels.py`: plain-language descriptions for each.
- `desktop/src/toolGuides.json`: guide entries. There are no homework references in these.
- Once availability says `available`, `liveTools()` in `desktop/src/live.ts` shows them on the **Interactive** page automatically, so no UI code is needed for that. Check `JOB_ONLY`: any tool too slow for live use (bert_topics on a big corpus?) goes there. Measure first.
- Test: a backend test that `availability()` is `available` for all BERT tools when the tiny model is installed, and a `live.test.ts` case that they pass `liveTools()`.

---

## Phase 5: Figures

### 5a. "Corpus at a glance" (engine side now, button later)
- `core/insight/glance.py` (or `core/pipelines/`): one fixed recipe over the parsed corpus. It reuses existing tools, no new analysis:
  1. Size overview: docs, tokens, types, sentences per doc (`corpus_statistics` / `text_statistics`)
  2. Top terms and keyness (`tfidf` / `keyness`)
  3. Readability and lexical diversity distributions (`readability`, `lexical_diversity`)
  4. Sentiment across documents or over time if dates exist (VADER; fast, always available)
  5. Document similarity heatmap (TF-IDF by default; embeddings if Granite is ready)
  6. Named entities, top N (`ner`)
- **Cache** by corpus fingerprint + recipe version (the fingerprint idea already exists in `desktop_backend/live.py` `annotation_key`). Uploading a file never triggers it. Only the button does, and it re-runs only if the corpus changed.
- Endpoint `POST /api/projects/{id}/glance` → job; `GET` → cached result.
- Output: a bundle of panels (reuse `core/viz/panels*.py` / `core/viz/static/bundles.py`) plus a short text summary.

### 5b. Seaborn publication figures: `core/viz/static/seaborn_figures.py` (new)
seaborn 0.13.2 is already pinned. Static PNG + SVG, consistent theme, colorblind-safe palette, fonts that exist in the frozen app.

| Figure | Input | Fed by |
|---|---|---|
| Clustered similarity heatmap (`clustermap`) | doc × doc matrix | doc_similarity, doc_embeddings |
| Ridgeline of a measure by group | long table: group, value | readability, sentiment, sentence_complexity |
| Pair grid of document measures | wide table of per-doc measures | readability + lexical_diversity + text_statistics |
| Violin/box by group | group, value | any per-doc measure |
| Trend with CI band (`lineplot`) | date, value | sentiment over time, lexicon_series |
| Embedding semantic map (scatter + cluster hulls/labels) | Doc/X/Y/Cluster | doc_embeddings |

- `filterwarnings=error` is on, and seaborn/matplotlib emit FutureWarnings. Fix the calls rather than filtering; if one is unavoidable, filter it narrowly in the module, not globally.
- Use the `Agg` backend. Close every figure (`plt.close(fig)`) to avoid memory growth in the long-running engine.
- Tests: each figure renders from a small frame without warnings, and `core/viz/figure_lint.py` passes on it.

---

## Phase 6: Tauri/UI (deferred)

Do this once the engine-side work is on dev main.
1. **Models screen:** a list from `GET /api/models` showing status, size, license and Download/Delete buttons with a progress bar and cancel. Also reachable from a `needs_model` tool card ("This needs Qwen3 Embedding (600 MB). Download").
2. **Corpus at a glance button:** on the corpus/project view. It shows the cached result instantly, with "Refresh" only if the corpus changed.
3. **Model picker** in tool params: already handled if `model` is a choice param rendered by `ParamFields.tsx`. Verify it.
4. **Seaborn figures** in the Visualize dialog / gallery (`VisualizeDialog.tsx`, `VisualizationCatalog.tsx`).
5. UI tests alongside each (the existing `*.test.tsx` pattern).

---

## Phase 7: Release 0.4.0

Prerequisite: the 4 secrets on NLP-Suite (`PUBLISH_TOKEN`, `TAURI_SIGNING_PRIVATE_KEY`, `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`, `TAURI_UPDATER_PUBKEY`).
1. On dev: `python scripts/bump_version.py 0.4.0`, run the checks, push.
2. Test build first: public repo → Actions → Desktop platform builds → `from_dev` ✓. Download and install on Windows + Mac, then test offline BERT, a Qwen download and the glance button.
3. Public repo → Actions → **Publish dev to public** with `release` ✓ → review the draft → Publish.
4. README: note that 0.3.1 is deprecated and has no auto-update, so those users download 0.4.0 once. Updates are automatic after that.
5. Afterwards: ship 0.4.1 as a tiny fix and confirm a 0.4.0 install updates itself. That's the real test of the updater.

---

## Suggested order and sizing

| # | Chunk | Depends on | Rough size |
|---|---|---|---|
| 1 | Phase 1 (runtime, registry, backends, tiny fixture) | — | Biggest; do it first |
| 2 | Phase 4d catalog + availability (BERT visible in Interactive) | 1 | Small |
| 3 | Phase 2 export + bundling | 1 | Medium; needs a runner |
| 4 | Phase 5b seaborn figures | — | Medium, **independent: good parallel task** |
| 5 | Phase 5a glance (engine) | 4 helps | Medium |
| 6 | Phase 4a doc_embeddings | 1, 2 | Medium |
| 7 | Phase 3 downloads + Qwen | 1, 2 | Medium |
| 8 | Phase 6 UI | all | Deferred |
| 9 | Phase 7 release | all + secrets | Small |

## Deferred / out of scope
- **EmbeddingGemma:** gated on HF under the Gemma Terms. Possibly later as bring-your-own.
- GPU acceleration (CPU-only is simpler and fine at these sizes).
- Linux auto-update (.deb isn't updatable by Tauri; Linux users re-download).

## Open decisions (flag them rather than guessing)
1. Should BERT base be bundled or download-only? It's the installer size trade-off.
2. Which tools go in `JOB_ONLY` (too slow for the Interactive page)? Measure bert_topics / word2vec_bert on ~100k words.
3. Should the sentiment model be 2-class or 3-class? If 3-class, a license check is needed first.

---

## Progress (2026-09-24)

Built and tested; the checks named in Ground rules pass (see the commit). What differs from the plan above, and why:

**Phase 1 — runtime.** `core/models/` holds the registry (`registry.py`, ids `bert-base-uncased`, `distilbert-sst2`, `granite-embedding-english-r2`, `qwen3-embedding-0.6b`; old Hugging Face names resolve as aliases, and `core/profiler/plan.py` maps them so saved runs still validate), `locate.py` (override → bundled → `<app data>/models`; the server sets `NLP_SUITE_MODELS_DIR` from `--data-dir`), `onnx_backend.py` (token, sentence and classifier backends, length-batched, cached sessions) and `align.py` (the word-to-pieces rule both ONNX and the PyTorch fallback use). Availability has a new `needs_model` state. The tiny fixture models are in `tests/fixtures/models/` (`scripts/make_tiny_onnx.py`).

Two real fixes came with it: a word is now read at **its own pieces** (the old rule averaged every piece that was a substring of it), and a **lemma is read at its surface form** ("be" at "was"; it used to get the whole sentence). Word2Vec via BERT no longer embeds the corpus twice for a query, and word senses no longer embed it twice either.

**Phase 2 — precision (measured, not estimated).** The plan assumed int8. Dynamic int8 fails every model's parity (BERT's last layer 0.78, Qwen 0.75), because it quantizes activations. `scripts/export_models.py` tries 4-bit weights, 8-bit weights with a 4-bit vocabulary table (`q8e4`), dynamic int8, 8-bit weights and fp16, and keeps the smallest that passes (cosine ≥ 0.99; classifiers ≥ 98% labels and no probability moved > 0.05):

| Model | Kept | Size | Parity |
|---|---|---|---|
| BERT base | q8e4 | 105 MB | min cosine 0.993 (words), 0.996 (sentences) |
| DistilBERT SST-2 | q8 | 95 MB | 98.3% labels, max probability gap 0.027 |
| Granite R2 | q8e4 | 143 MB | 0.9996 |
| Qwen3 Embedding 0.6B | q8e4 | 569 MB | 0.998 documents, 0.996 queries |

Installer impact: **343 MB of models** (the three bundled) plus ONNX Runtime. Qwen downloads from the Models page. The files are the `models-1` release on RileyK05/NLP-Suite, a **prerelease never marked latest** (the updater reads `releases/latest`). `scripts/publish_models.py` uploads; `scripts/fetch_models.py --bundled` is what `desktop.yml` runs before PyInstaller, followed by the real-model tests on each platform.

**Phase 3 — downloads.** `core/models/download.py` resumes (HTTP Range), verifies SHA-256, renames atomically, refuses non-https. Downloads run on a thread in the engine (`desktop_backend/models.py`), not as a project job: a model belongs to no project. Endpoints: `GET /api/models`, `POST /api/models/{id}/download`, `POST /api/models/{id}/cancel`, `DELETE /api/models/{id}`.

**Phase 4.** `doc_embeddings` is a new tool (vectors, pairs in `doc_similarity`'s shape, neighbours, clustered map, semantic search). Its pair **Similarity is relative to the corpus** (cosine after removing the average document): uncentred, Granite scores every pair of State of the Union addresses 98–99%, which is true and tells them apart not at all. Word senses now cluster (seeded 2-means, kept only when the senses separate; function words left out) instead of splitting on the first vector component. The three hidden BERT tools and `doc_embeddings` are in the catalog with guides.

**JOB_ONLY (open decision 2).** Measured on this machine (shared CPU, so rough): BERT-family models read tens of sentences a second. A whole 87-speech corpus is ~26,000 sentences, far past the bench's 180-second window. Rather than hide BERT from Interactive, the live bench refuses a model tool over more than 1,500 sentences (`desktop_backend/live.py::model_budget`) with the way forward (fewer documents, or Analyze & visualize). A few speeches run live.

**Phase 5.** Glance: `core/insight/glance.py` (recipe + summary) and `desktop_backend/glance.py` (one job, one parse, a batch of seven child runs; cached by document contents + recipe version; `stale` when the corpus changes). Figures: most of the plan's seaborn table already existed in `core/viz/static`; new are the **pair grid** and **ridgeline** bundles for every measure tool, and a **clustered meaning map** and **map by decade** for `doc_embeddings`.

**Phase 6 — UI (done, not deferred).** A **Models** tab (fourth in the Research group): each model's status, size, licence, which tools use it, Download / Cancel / Remove with progress. A `needs_model` tool card says "Needs a model" and its dialog links to Models. **Corpus at a glance** is a panel on the Corpus page.

**What the embeddings mean (added after review).** The t-SNE map and the neighbour list said which words are close, never what the closeness is about. `core/analysis/word_meaning.py` reads a run's `vectors.csv` (centred: BERT type vectors average 0.46 cosine before, 0.00 after) and `core/viz/panels_word_meaning.py` draws, for both Word2Vec tools: **groups of words that mean alike** (themes, named by their central words; nouns by default), **between two ideas** (an axis between pole words; on the State of the Union soldier leans toward war and trade toward peace), **a map on two named axes**, and **a word's neighbourhood** (radial: distance is similarity). For BERT on a dated corpus, the same occurrence vectors are averaged per decade (no extra embedding): **meaning over time** (*energy*: strength, spirit in the 1940s; atomic, nuclear in the 1960s; fuel, oil, renewable from the 1970s) and **words whose company changed most** (*welfare*, *power*, *community*). Word senses gets its first figures: **words used in two senses** (Union: State of the Union / Soviet Union; World: World War / World Bank) and **the two senses of a word**, whose bars open the sentences of that sense containing that word. New outputs: a `Word class` column on `vectors.csv`, `Group` on `tsne.csv` (the map is coloured by it), `meaning_over_time.csv`, `meaning_change.csv`, `senses.csv`, `Separation` on `wsi.csv`. The meaning figures come first for these tools; when the first figure refuses a small run, the run page and the live bench open the next.

**Still open**
1. Speed. BERT-family throughput here was 10–60 sentences/s under load; a clean benchmark per precision is owed, and token models could skip computing all 13 hidden states when only the last is read (a second, last-layer-only graph).
2. Sentiment stays 2-class (open decision 3): a 3-class model needs a licence check first.
3. Release steps (Phase 7) are the maintainer's: version bump, test build, publish.
