# Performance Optimizations & Test Redundancy Audit

This document records the comprehensive performance, algorithmic, and test-suite audit of the `New_NLP_Suite` codebase. It details existing inefficiencies, test redundancies, architectural gaps ("what is there and isn't there"), and a prioritized refactoring roadmap.

---

## 1. Codebase Performance & Algorithmic Optimizations

### 1.1 SVO Semantic-Role Extraction: $O(N^3) \to O(N)$ Refactor
- **File:** `core/analysis/clause_svo.py` (lines 161–220; helper `_agent_name` lines 95–115)
- **Current State:**
  To extract Subject-Verb-Object triples and passive agents, `extract_svo`:
  1. Calls `frame.groupby([Col.DOCUMENT_ID, Col.SENTENCE_ID], sort=False)`.
  2. Runs `for _, row in sent.iterrows():` to build an ID lookup dictionary `by_id`.
  3. Runs `for _, tok in sent.iterrows():` across every token in the sentence.
  4. Inside that, runs `for _, dep in sent.iterrows():` to check if `dep.HEAD == tok.ID`.
  5. When checking passive agents (`full_rel in _PASSIVE_AGENT_CASES`), it calls `_agent_name()`, which runs another `for _, child in sent.iterrows():` to find `pobj` children of the preposition.
- **Problem:**
  Every call to `.iterrows()` allocates a full pandas `Series` with index metadata. For a sentence of length $N$, this produces up to 4 nested loops of DataFrame iteration ($O(N^3)$ complexity). Over a corpus with tens of thousands of sentences, millions of short-lived `Series` objects are allocated and garbage-collected.
- **Optimization:**
  - Convert sentence rows to lightweight dataclasses or dictionaries in a single pass: `tokens = sent.to_dict('records')`.
  - Build an adjacency index of dependents mapped by head ID: `children_by_head: dict[int, list[dict]] = defaultdict(list)`.
  - Looking up dependents of token `tid` becomes an $O(1)$ dictionary lookup `children_by_head[tid]` instead of a full sentence iteration.
  - Looking up preposition objects in `_agent_name` becomes an immediate $O(1)$ child check.

---

### 1.2 Table Search: Vectorized String Operations vs. Row-by-Row `.iterrows()`
- **File:** `core/analysis/table_search.py` (lines 58–62, 110–133)
- **Current State:**
  ```python
  # table_search.py lines 113, 126-128
  re.compile(filt.value, flags=flags) # Compiled to validate, then discarded
  ...
  for _, row in frame.iterrows():
      results = [_cell_matches(row[filt.field], filt) for filt in filters]
  ```
  Inside `_cell_matches`:
  ```python
  matched = re.search(needle, str(cell) if cell is not None else "", flags=flags) is not None
  ```
- **Problem:**
  1. The compiled regex pattern from line 113 is thrown away. Inside `_cell_matches`, `re.search` is called with the raw string needle, forcing regex cache lookups for every cell in the table.
  2. The search uses `.iterrows()` to evaluate predicates cell by cell in pure Python. On a 600,000-row CoNLL table, this takes several seconds rather than milliseconds.
- **Optimization:**
  - Preserve the compiled regex pattern object.
  - Replace row-by-row iteration with vectorized pandas Series methods:
    - `eq`: `mask = frame[col] == val`
    - `contains`: `mask = frame[col].str.contains(val, regex=False, case=filt.case_sensitive)`
    - `starts_with`: `mask = frame[col].str.startswith(val)`
    - `regex`: `mask = frame[col].str.contains(compiled_pattern)`
  - Combine column masks using bitwise `&` (AND) or `|` (OR).

---

### 1.3 Lexical Diversity: Vectorized `vocd` Grid Search
- **File:** `core/analysis/lexical_diversity.py` (lines 157–168)
- **Current State:**
  ```python
  grid = np.arange(10.0, 200.0 + 0.25, 0.5)  # 381 candidates
  for d_candidate in grid:
      err = sum((observed[size] - expected_ttr(float(d_candidate), size)) ** 2 for size in sample_sizes)
      if err < best_err: ...
  ```
- **Problem:**
  Iterates $381 \times 16 = 6,096$ steps in scalar Python, calling `math.sqrt` inside `expected_ttr` repeatedly for every step.
- **Optimization:**
  - Compute `expected_ttr` using NumPy 2D array broadcasting:
    ```python
    d = grid[:, np.newaxis]                      # shape (381, 1)
    s = np.array(sample_sizes)[np.newaxis, :]    # shape (1, 16)
    expected = (d / s) * (np.sqrt(1 + 2 * s / d) - 1)
    obs = np.array([observed[sz] for sz in sample_sizes])[np.newaxis, :]
    mse = np.sum((obs - expected) ** 2, axis=1)
    best_d = float(grid[np.argmin(mse)])
    ```
  - Replaces 6,096 Python iterations with a single C-speed NumPy broadcasted array operation.

---

### 1.4 Word Embeddings: Pre-normalized Matrix for `most_similar`
- **File:** `core/analysis/word2vec_bert.py` (lines 90–97)
- **Current State:**
  ```python
  index = self._index(key)
  matrix = np.array(self.vectors, dtype=float)
  norms = np.linalg.norm(matrix, axis=1)
  norms[norms == 0] = 1.0
  cosines = (matrix @ matrix[index]) / (norms * norms[index])
  ```
- **Problem:**
  Every time `most_similar` is queried, it allocates a new NumPy array from `self.vectors` and calculates the Euclidean norm across every vector in the vocabulary.
- **Optimization:**
  - Normalize the vector matrix once during initialization or after training: `self._normalized_matrix = matrix / norms[:, None]`.
  - Any subsequent similarity query becomes a single dot product: `cosines = self._normalized_matrix @ self._normalized_matrix[index]`.

---

### 1.5 Document Similarity: Vectorized Pair Extraction
- **File:** `core/analysis/doc_similarity.py` (lines 140, 180–195)
- **Current State:**
  1. Converts the scikit-learn cosine matrix into a nested Python list of lists:
     `matrix = [[float(sim[i, j]) * 100.0 for j in range(len(entries))] for i in range(len(entries))]`
  2. Iterates over $i$ and $j$ with nested Python `for` loops.
  3. Builds `out = pd.DataFrame(rows, columns=_PAIR_COLUMNS)` (which excludes `_raw_similarity`).
  4. Builds a second `pd.DataFrame(rows)` solely to sort by `["Similarity", "Document ID A", "Document ID B"]` to extract `_raw_similarity`.
  5. Then sorts `out` separately with the exact same sort criteria.
- **Problem:**
  High memory overhead from multiple DataFrame constructions and duplicate sorting operations; unvectorized matrix traversal.
- **Optimization:**
  - Extract upper-triangle indices directly with `i_idx, j_idx = np.triu_indices(len(entries), k=1)`.
  - Vectorize raw score extraction: `raw_scores = sim[i_idx, j_idx] * 100.0`.
  - Construct the DataFrame once with `_raw_similarity`, sort once, extract the raw series, and drop the column.

---

### 1.6 Eradication of Pervasive `.iterrows()` in Visualization Panels
- **Files:**
  - `core/viz/panels_models.py` (lines 82–83, 335, 467, 633)
  - `core/viz/panels_sentiment.py` (lines 156–157, 163, 308)
  - `core/viz/panels_contrast.py` (lines 71, 120, 194, 260, 316)
  - `core/viz/panels_document_measures.py` (lines 466, 686, 706, 779, 916)
  - `core/viz/panels_terms.py` (lines 222, 761, 913, 1044)
  - `core/viz/panels_time_positions.py` (lines 206, 289, 442, 513, 906, 1045, etc.)
  - *(Over 65 total `.iterrows()` call sites across `core/viz/`)*
- **Problem:**
  `.iterrows()` creates a pandas Series for every single row. In कई panels, the same DataFrame is iterated multiple times concurrently:
  - `panels_models.py:82–83`:
    ```python
    children = {n + index: (int(row["Left id"]), int(row["Right id"])) for index, row in merges.iterrows()}
    height = {n + index: float(row["Height"]) for index, row in merges.iterrows()}
    ```
  - `panels_sentiment.py:156–157`:
    ```python
    [_mark_label(row, label_column, identity_columns, names) for _, row in working.iterrows()],
    [" • ".join(str(row[column]) for column in identity_columns) for _, row in working.iterrows()],
    ```
- **Optimization:**
  - Switch from `.iterrows()` to `.itertuples(index=False)` (typically 20x to 100x faster).
  - Combine multiple comprehensions over the same table into a single pass.

---

### 1.7 CoNLL Sentence Grouping: Linear Scan vs. `groupby([Doc_ID, Sent_ID])`
- **Files:**
  - `core/analysis/ngrams.py` (line 47)
  - `core/analysis/word2vec_bert.py` (line 108)
  - `core/analysis/clause_svo.py` (line 161)
- **Problem:**
  Calling `frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False)` creates tens of thousands of small slice DataFrames.
- **Optimization:**
  CoNLL frames are guaranteed to be sorted sequentially by document and sentence. A single linear scan comparing `(doc_id, sent_id) != prev_key` or segmenting by index boundaries partitions the tokens without DataFrame grouping overhead.

---

### 1.8 Pipeline I/O & spaCy Batching
- **File:** `core/pipelines/spacy_backend.py` (lines 74–95)
- **Current State:**
  ```python
  for doc in corpus.docs:
      spacy_doc = nlp(doc.text)
  ```
- **Problem:**
  Processes documents one-by-one. In a large corpus, this incurs substantial function call overhead and prevents spaCy from vectorizing inference across texts.
  Additionally, creating `rows: list[dict[str, object]] = []` creates 600,000 dictionaries with 12 string keys each (7.2M dict keys) before passing them to `pd.DataFrame(rows)`.
- **Optimization:**
  - Use spaCy's native batched pipeline: `nlp.pipe(texts, batch_size=64)`.
  - Populate columnar lists (`columns = {"Form": [], "POS": [], ...}`) directly, avoiding dictionary allocation per token.

---

### 1.9 Eliminating Redundant File Hashing & Disk Reads
- **File:** `desktop_backend/project_corpus.py` (lines 79–85)
- **Current State:**
  ```python
  if hash_file(path) != item["sha256"]:
      raise ValueError(...)
  text = read_text(path)
  ```
  `hash_file` reads the entire file in chunks to compute SHA256. Immediately after, `read_text` re-opens and reads the entire file from disk again.
- **Optimization:**
  Read the raw bytes once: calculate `hashlib.sha256(raw_bytes).hexdigest()`, and if valid, decode directly from the in-memory buffer.
- **File:** `core/analysis/doc_duplicates.py` (line 44)
  `exact_groups` recalculates `hashlib.sha256(doc.text.encode("utf-8")).hexdigest()`, completely ignoring `doc.sha256`, which was already calculated at intake and stored on the `Document` dataclass.

---

### 1.10 Invariant Set Union in Cleaning Loop
- **File:** `core/io/cleaning.py` (lines 123–125)
- **Current State:**
  ```python
  for text in texts:
      vocabulary = KNOWN_STAGE_WORDS | frozenset(word for term in extra for word in _WORD.findall(term.casefold()))
      for match in _SPAN.finditer(text): ...
  ```
- **Optimization:**
  Hoist the calculation of `vocabulary` outside the `for text in texts:` loop so it evaluates once per call rather than once per document.

---

### 1.11 Deduplicating Math in `corpus_statistics` and `lexical_diversity`
- **Files:** `core/analysis/corpus_statistics.py` (lines 16–28) vs. `core/analysis/lexical_diversity.py` (lines 94–105)
- **Problem:**
  Both modules carry verbatim copies of `ttr`, `root_ttr`, and `log_ttr`. Furthermore, `corpus_statistics.py:33` executes `from collections import Counter` inside `_yule_k`, which runs on every document.
- **Optimization:**
  Consolidate statistical token formulas into a shared helper module; move imports to top-level.

---

### 1.12 Profiler Parallel Tool Execution
- **File:** `core/profiler/executor.py` (lines 2320–2383)
- **Current State:**
  The profiler executes all planned tools strictly sequentially in a single thread:
  ```python
  for tool in plan.tools:
      result = adapter(ctx, tool.params)
  ```
- **Problem:**
  The `BatchContext` contains an immutable `table` (CoNLL DataFrame) and `corpus`. Tools like `collocations`, `dispersion`, `tfidf`, `readability`, and `keyness` only read from these inputs and write independent output frames.
- **Optimization:**
  Execute non-dependent, read-only analysis adapters concurrently using `concurrent.futures.ThreadPoolExecutor`.

---

## 2. Test Suite Redundancies & Overlaps

### 2.1 Repetitive `mini-corpus` Parsing Across Test Modules
- **Files Affected:**
  - `tests/test_analysis_21_25.py:26–35`
  - `tests/test_analysis_26_30.py:27–35`
  - `tests/test_analysis_31_35.py:26–35`
  - `tests/test_analysis_36_40.py:26–35`
  - `tests/test_analysis_47_52.py:24–32`
- **Finding:**
  Every file re-implements an identical `_parsed_frame()` function:
  ```python
  def _parsed_frame() -> pd.DataFrame:
      result = read_corpus(FIXTURE)
      corpus = result.unwrap()
      cache = PipelineCache()
      cache.register("spacy", build_spacy_pipeline)
      return cache.get("spacy", "en").unwrap().parse(corpus).unwrap()
  ```
  Every test marked `@pytest.mark.model_integration` that calls `_parsed_frame()` starts with a blank `PipelineCache`, reads `mini-corpus` from disk, and runs spaCy parser inference afresh.
- **Redundant Tests:**
  `test_analysis_21_25.py:44–85` runs `_parsed_frame()` merely to test table column search filters (`op="eq"`, `op="contains"`, `logic="AND"`). This does not test spaCy at all; it can run on a 3-row synthetic table in less than a millisecond.

---

### 2.2 Duplicated Test Coverage Across Suites
1. **Sentiment Analysis:**
   - `tests/test_sentiment.py` tests `sentiment_vader_anew` and `sentiment_swn_hedono` thoroughly with synthetic tables, fake analyzers, and exact expected scores.
   - `tests/test_analysis_26_30.py:125–273` re-tests the exact same functions using `_parsed_frame()` and local synthetic fixtures.
2. **Collocations and N-Grams:**
   - `tests/test_collocations.py` provides exact hand-computed mathematical validations of PMI, T-score, and contingency tables.
   - `tests/test_analysis_31_35.py:173–243` duplicates assertions on `cooccurrence` and `ngrams`.
3. **Document Duplicates & Similarity:**
   - `tests/test_doc_duplicates.py:46–60` tests `fuzzy_pairs`.
   - `tests/test_similarity.py` tests `doc_similarity.find_duplicates` (the exact implementation underlying `fuzzy_pairs`) with identical threshold checks.

---

### 2.3 Mis-tagged Test Markers
- In `tests/test_analysis_26_30.py:117–123`:
  ```python
  @pytest.mark.model_integration
  def test_csv_stats_empty() -> None:
      empty = pd.DataFrame(columns=["A", "B"])
      res = csv_mod.describe(empty)
      assert res.ok
  ```
  This is tagged as `model_integration` despite being a pure unit test with no dependency on NLP models, parsers, or fixtures.

---

### 2.4 Test Concentration in Static Linters
- Out of ~1,350 total tests in the repository:
  - `tests/test_figure_quality.py`: **416 tests**
  - `tests/test_labels.py`: **361 tests**
- These two files account for **57% of all tests**. While fine-grained parametrization is informative when a failure occurs, running hundreds of individual pytest test cases for static dictionary assertions (`assert label[0].isupper()`) introduces significant test-runner setup and reporting overhead.

---

### 2.5 Proliferation of Local CoNLL Builders
- While `conftest.py` provides `build_frame`, `conll_frame`, and `universal_frame`, dozens of test files (`test_analysis_21_25.py`, `test_analysis_26_30.py`, `test_analysis_31_35.py`, `test_collocations.py`, `test_sentiment.py`) maintain duplicate local helper functions (`_table()` or `_frame()`) defining the same 12-column dictionaries.

---

## 3. "What Is There" vs. "What Isn't There"

| Component | What Is There (Existing) | What Is NOT There (Gaps) |
| :--- | :--- | :--- |
| **Pipeline & Parsing** | Sequential single-document parser; `PipelineCache` in memory; fallback encoding chain. | **No batched inference (`nlp.pipe`)**; no streaming text reader for large files; no parallel multi-file intake. |
| **Analysis Modules** | 64 functional analysis tools returning `Result`/`Diagnostic` objects; accurate mathematical algorithms. | **No vectorization for SVO, table search, or vocd**; pervasive row-by-row `.iterrows()` in panels; duplicate math across statistics modules. |
| **Profiler Engine** | Structured `Plan` and `execute` engine; per-tool execution timing; error isolation. | **No concurrent tool execution**: tools execute sequentially in a single thread even when inputs are read-only. |
| **Desktop Backend** | SQLite project database; live benches; process isolation; cancel/interrupt coordination. | **Double disk reads** on project import; unvectorized pairwise matrix operations. |
| **Test Fixtures** | Minimal corpus (`mini-corpus/01.txt`); adversarial probes; ONNX `tiny_models`; synthetic CoNLL frames. | **No session-scoped parsed mini-corpus fixture**; test files repeatedly invoke spaCy from scratch. |
| **Test Coverage** | Extensive coverage of labels (361 tests), figure linting (416 tests), and unit math formulas. | **No performance/benchmark regression tests**; no concurrency stress tests; missing edge cases (circular syntax trees, extreme thresholds, non-ASCII/RTL edge cases). |

---

## 4. Prioritized Refactoring Plan

```mermaid
flowchart TD
    subgraph Phase 1: Test Suite Optimization
        T1["Centralize parsed mini-corpus fixture in conftest.py"]
        T2["De-duplicate overlapping sentiment & collocation tests"]
        T3["Correct mis-tagged markers & unify DataFrame builders"]
    end

    subgraph Phase 2: Algorithmic & Pandas Vectorization
        C1["Refactor SVO extractor (O(N^3) -> O(N))"]
        C2["Vectorize Table Search & vocd grid search"]
        C3["Replace .iterrows() with .itertuples() across viz panels"]
    end

    subgraph Phase 3: Pipeline & I/O Optimization
        P1["Enable spaCy nlp.pipe() batching & columnar allocation"]
        P2["Eliminate double-read in project_corpus.py & doc_duplicates.py"]
        P3["Consolidate corpus_statistics & lexical_diversity helpers"]
    end

    subgraph Phase 4: Profiler Concurrency
        E1["Run independent read-only profiler tools in ThreadPoolExecutor"]
    end

    Phase 1 --> Phase 2
    Phase 2 --> Phase 3
    Phase 3 --> Phase 4
```

### Phase 1: Test Suite Optimization (Immediate Developer Productivity)
1. **Centralize Mini-Corpus Fixture:**
   - Define `@pytest.fixture(scope="session") def parsed_mini_corpus()` in `conftest.py`.
   - Remove local `_parsed_frame()` functions from `test_analysis_*.py`.
2. **De-duplicate Test Suites:**
   - Retire duplicate tests in `test_analysis_26_30.py` and `test_analysis_31_35.py` in favor of canonical tests in `test_sentiment.py` and `test_collocations.py`.
3. **Correct Markers:**
   - Remove `@pytest.mark.model_integration` from pure tabular unit tests (such as `test_csv_stats_empty`).
4. **Standardize CoNLL Builders:**
   - Migrate local test DataFrame builders to `conftest.build_frame()`.

### Phase 2: Algorithmic & Pandas Vectorization (Core Compute Speedup)
1. **SVO Extractor Refactor:**
   - Replace 4-level nested `.iterrows()` loops with dictionary-based dependent indexing in `core/analysis/clause_svo.py`.
2. **Table Search Vectorization:**
   - Replace row-wise iteration with vectorized pandas Series operations and retain pre-compiled regex objects in `core/analysis/table_search.py`.
3. **`vocd` Grid Search Vectorization:**
   - Replace the 6,096 scalar iterations with 2D NumPy array broadcasting in `core/analysis/lexical_diversity.py`.
4. **Eliminate `.iterrows()` in Panels:**
   - Systematically convert 65+ `.iterrows()` calls in `core/viz/panels_*` to `.itertuples(index=False)`.

### Phase 3: Pipeline & I/O Optimization (Intake & Memory Efficiency)
1. **Batched NLP Pipeline:**
   - Convert `spacy_backend.py` to use `nlp.pipe()` with configurable batch sizes.
   - Construct DataFrames from columnar lists rather than allocating millions of token dictionaries.
2. **Eliminate Double Reads:**
   - Read file bytes once, hash, and decode in `desktop_backend/project_corpus.py`.
   - Re-use `doc.sha256` in `core/analysis/doc_duplicates.py`.
3. **Deduplicate Statistical Routines:**
   - Consolidate lexical diversity math between `corpus_statistics.py` and `lexical_diversity.py`.

### Phase 4: Profiler Concurrency (Scaling Execution)
1. **Concurrent Tool Dispatch:**
   - Update `core/profiler/executor.py` to run read-only tools concurrently over the shared in-memory parse table via `concurrent.futures.ThreadPoolExecutor`.
