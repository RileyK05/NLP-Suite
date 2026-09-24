# Visualization implementation status

Live status for the charts/visualization work (FR-6.5). Updated after each
major gate so work can resume after interruptions. Replaces all earlier
(gate-1/gate-2) status notes.

## Senior verification — completed 2026-09-11

The senior independently inspected all seven corrected blocker PNGs: sparse
category values align in both orientations; histogram labels show the measured
variable and Count/Percent; categorical gaps remain gaps; numeric and datetime
heatmaps preserve their coordinates and values. The earlier percent/rate and
long-label examples were also visually inspected.

Independent affected-suite run: **259 passed, 3 skipped, 2 deselected**
(175.14 seconds). Changed-file lint, formatting, six-module mypy, and actual
unified CLI discovery/unknown-tool exit checks passed. The final evidence
generator was subsequently corrected to record published paths; all seven
manifest run directories contain result.json and all seven PNGs exist with
valid signatures. Its lint/format checks passed. Prior evidence runs are retained.

The reviewed chart and CLI scope is accepted. Image export was verified using
the isolated optional dependencies in out/visual_review/deps, not a global
installation. Existing unrelated homework-script lint findings remain outside
this change's scope.

## Second-review release blockers (FIXED — visually verified)

Work order: `docs/visualization-senior-review.md` §"Second visual review"
(plus the senior checkpoint's three completion fixes). All four blockers and
all three checkpoint items are fixed with failing-before regressions in
`tests/test_viz_final_blockers.py` and real PNG evidence under
`out/visual_review/final_blockers/` (manifest carries the ACTUAL rendered
metadata, not claims).

### Blocker 1: histogram/heatmap effective axis labels — FIXED

* Root cause: `_prepare_by_kind` chose the correct kind-specific label but
  `_finalize_selection` reset it with `y_label = spec.y`, and histogram
  preparation never renamed x.
* Histogram now prepares its x axis as the binned VALUE column (`spec.y`;
  the categorical `spec.x` is an ignored selector that may even contain
  missing values) and y as Count/Percent/Share. Verified:
  `histogram_counts` labels are `val | Count`, percent run is `val |
  Percent` (manifest `effective_labels` + figure `xaxis_title`/`yaxis_title`).
* Heatmap y_label names the GROUP (row) axis; the renderer puts the measure
  on the colorbar (`go.Heatmap.colorbar.title`), x/y axis titles name the
  column/group axes. Explicit `--x-label/--y-label` overrides win in all
  cases (tested for histogram and heatmap).

### Blocker 2: genuine sparse-category mislabelling — FIXED

* Root cause: Plotly inserts categories by trace traversal; with traces
  one=(A,C,E) and two=(B,D,F) the positional ticktext A..F displayed the
  value 3 under B.
* Fix: every categorical renderer now pins
  `categoryorder="array"` + `categoryarray` to the tick order (vertical
  bars, horizontal bars, lines). Tests assert the RENDERED mapping
  (trace category -> tick index) against the full figure, not merely
  ticktext; the old gallery's `sparse_groups` (every group at every
  category) was superseded by the genuinely sparse fixture `c=[A,C,E,B,D,F],
  g=[one,one,one,two,two,two]`.
* Evidence: `final_sparse_bar_vertical`/`final_sparse_bar_horizontal` PNGs;
  manifest `rendered_mapping` shows tick_index 0..5 mapping A->1, B->2,
  C->3, D->4, E->5, F->6 in both orientations.

### Blocker 3: numeric heatmap corruption — FIXED

* Root causes: (a) string-sorting the pivot turned x=[1,10,2];
  (b) `px.imshow` assumes regular integer grid coordinates, resampling the
  matrix onto unrelated positions (single column, axis -3..6).
* Fix: dtype-aware ordering (`_heatmap_matrix`: numeric/datetime sorted by
  value, strings by text) + rendering through `graph_objects.Heatmap`, whose
  x/y arrays carry the true coordinates. Missing cells remain gaps (None).
* Evidence (manifest): numeric coords `1, 2, 10` with z
  `[[1,2,10],[11,12,20]]`; datetime heatmap chronological
  (`2024-01-01, 2024-03-01`); categorical `d1/d2` with the (d2,R2) gap; all
  three PNGs under `out/visual_review/final_blockers/`.

### Blocker 4: double JSON emission — FIXED

* `tools/charts.py` emitted twice on publish_failure (once before checking
  the publish Result, once after) and left staging orphaned on
  finalize-failure. Both paths restructured: exactly one JSON object per
  run; failed publication/finalization abandons staging immediately and
  reports publication + original diagnostics with NO artifact paths.
* Regression tests fault-inject `OutputWriter.publish_failure` and
  `finalize` returning `Result.failure` and assert: exactly one JSON object,
  both diagnostic sets present, `run_dir` null, artifacts `[]`, no
  `.staging-*` leftovers on disk.

### Checkpoint fix 1: tick-angle sentinel — FIXED

* `ChartSpec.x_tick_angle` is now `int | None = None` (CLI default None):
  `None` means UNSPECIFIED and picks per-orientation defaults (vertical -30,
  horizontal upright 0); any explicitly supplied angle — including -30 — is
  honored verbatim (`--x-tick-angle -30` renders -30 on horizontal bars,
  tested). Registry flag mirror updated to default None.

### Checkpoint fix 2: missing axis values rejected — FIXED

* New `_check_missing_axis_values` in `chartspec.py`: None/NaN/NaT in the x
  or group column fails with `CHART_BAD_COLUMN` BEFORE astype(str)/groupby
  could silently drop the row (reproducer `c=[A,None], n=[2,8], agg=sum`
  previously dropped 8) or invent a fake `"nan"`/`None` category
  (reproducer `g=[g,None]` previously invented category `None`).
* Measure/denominator missing values keep their per-row
  `CHART_BAD_NUMERIC` diagnostics. A literal string `"nan"` remains a
  legitimate category. A histogram's unused `--x` selector may be missing.
* CLI/read-DataFrame regressions cover None/NaN/NaT through the CLI JSON
  contract (one object, exit 1) plus the allowed histogram-selector case.

### Checkpoint fix 3: dependency floors — FIXED

* `pyproject.toml`: `plotly-image` now requires `kaleido>=1.0` +
  `plotly>=6.1.1` (kaleido v1 warns/breaks below plotly 6.1.1; the code
  calls v1's `calc_fig_sync`); `plotly`/`app` extras raised to
  `plotly>=6.1.1`. All extra coverage unchanged; no installs performed.

## Senior-review corrections (COMPLETE — awaiting senior visual review)

Work order: `docs/visualization-senior-review.md`. All seven sections addressed.

### 1. Axis correctness (§1) — FIXED, verified on real exports

* Horizontal bars: categorical ticks (`tickvals`/`ticktext`) now applied ONLY
  to the categorical y axis; the numeric x axis keeps numeric ticks and the
  measure label. Verified on a real kaleido PNG (`review_horizontal_bar`)
  and by figure inspection (`build_figure` asserts in tests).
* Effective labels win: percent charts label y "Percent", share "Share",
  rates "mentions per 10000 tokens" — from `PreparedChart.y_label`, never the
  raw column name; explicit `--x-label/--y-label` overrides win over those.
* Histogram axes: x = numeric value (no tick substitution), y = Count /
  Percent / Share from preparation.
* Explicit `--x-tick-angle` honored for ANY number of categories (no hidden
  threshold); categorical tick mapping derived from first-appearance order in
  the prepared frame (sparse/interleaved groups aligned across traces);
  numeric-looking strings stay categorical; heatmap pivot sorted
  (rows/groups sorted, columns/x sorted) deterministically.
* Label wrapping is rendering-only (`<br>` in ticktext); data identity never
  mutated (verified in tests).

### 2. Comparison layout (§2) — FIXED

* Grouped bars default to **side-by-side** (`barmode="group"`); stacking is
  the explicit documented opt-in `--bar-mode stack|relative` (bar charts
  only; ChartSpec validates). Verified in the review images.
* Grouped histograms: overlay mode with explicit bin widths and 0.65 opacity.
* Deterministic Okabe-Ito per sorted group name within a chart; the docs no
  longer claim color stability across different category subsets.
* Dead helpers removed (`_tickvals_ticktext` duplicate, `_wrap`,
  `_color_for`, unused `_group_traces_color`, `_FigureBuilder`).

### 3. Numeric failure handling (§3) — FIXED with regression tests

* `_finite_sum` computes totals under `np.errstate(over="ignore")` +
  suppressed RuntimeWarning (warnings-as-errors safe), then `_check_finite`
  turns +inf into `CHART_NONFINITE_AGGREGATE` — never a silent zero.
* Normalization (total/group/category levels) validates the denominator
  totals AND the final transformed values. The one-element-total broadcast
  bug (reindex NaN-fill on non-default indexes) is fixed and regression-tested.
* Rates validate denominator totals and the final rate values; a finite
  denominator × huge multiplier (infinite rate) fails.
* Tests: `test_overflow_total_refused_for_percent`,
  `test_group_totals_overflow_refused`, `test_category_totals_overflow_refused`,
  `test_rate_multiplier_overflow_refused`, `test_non_default_index_normalized_correctly`,
  plus hand-computed normal-sized controls (`25%/75%`, `250 per 10k`).

### 4. Machine-readable results and provenance (§4) — FIXED

* All JSON artifact paths are ABSOLUTE on success AND on failure-with-
  retained-artifacts (resolved after finalize; run_dir resolved).
* `publish_failure` Result is checked; failed publication abandons staging
  and reports the publication error; `keep_artifacts` docstring now states
  the caller's responsibility.
* Renderer warnings (plotly fallback) are preserved in stdout JSON and the
  envelope on success (verified by the no-plotly fallback test path).
* `--encoding invalid` returns one failure JSON + exit 1 (`CHART_BAD_ENCODING`),
  no uncaught LookupError (tested).
* Envelope records: CSV sha256, full effective parameters, `prepared`
  metadata (agg/bin edges/count/selection order), effective axis labels,
  and rate `denominator` columns in `chart_data.csv` for independent audit.
* No INFO diagnostics on ordinary success (provenance lives in params,
  not as spurious diagnostics).

### 5. Unified executable behavior (§5) — FIXED with subprocess tests

* `python -m tools.unified --list --json` now emits the JSON object and exits 0
  (`__main__` guard added; subprocess-verified).
* `_route` distinguishes a missing requested module (`ModuleNotFoundError`
  naming `tools.<name>` → exit 2 "unknown") from a missing dependency inside
  a known tool (any other module name / other ImportError → exit 3 with the
  missing module named). Both tested in-process + subprocess for unknown.
* Installed `nlp-suite` routing intact; docs updated to the actual JSON shape.

### 6. Real export configuration (§6) — FIXED, verified with kaleido 1.4

* The old `pio.kaleido.chrome = ...` assignment was a silent no-op (verified
  against the installed kaleido v1). Replaced with the supported per-call
  override: `kaleido.calc_fig_sync(..., kopts={"path": ...})`.
* `--browser-path` CLI flag (falls back to `BROWSER_PATH`, then kaleido
  auto-detection). An invalid explicit path FAILS with
  `CHART_IMAGE_BROWSER_NOT_FOUND` (verified: `ChromeNotFoundError` mapped to
  the precise diagnostic).
* Real exports via per-command `PYTHONPATH=out/visual_review/deps` (kaleido
  1.4 + choreographer; no global installs):
  - 6 kinds × {png, svg, pdf} = 18/18 successful, `out/visual_review/images/`
    + `out/visual_review/export_evidence.json`
  - 8 senior-review case PNGs, `out/visual_review/review/` + `manifest.json`
    (all png_path entries verified on disk)
* No Chrome install, no global configuration changes.

### 7. Verification and documentation (§7) — DONE

* Corrected the tests that encoded the horizontal-axis bug (both-axes
  category ticks) to assert correct behavior (categorical ticks on y only).
* All changed-file checks pass with explicit exit codes (below).
* Fresh review artifacts: 8 review-case runs (HTML+CSV+PNG each),
  manifest, six-kind × three-format exports, export evidence manifest.
* Status doc updated with actual evidence; visual review not claimed complete
  on test counts alone — senior inspection of the images pending.

## Checks run (final, all explicit exit codes = 0)

* `pytest tests/test_viz_charts.py tests/test_viz_gate2.py
  tests/test_viz_final_blockers.py tests/test_custody.py
  tests/test_write_custody.py tests/test_layering.py
  tests/test_no_import_side_effects.py tests/test_unified.py
  tests/test_tool_registry.py tests/test_registry_new_tools.py
  tests/test_analysis_36_40.py tests/test_analysis_41_46.py
  tests/test_fail_big.py tests/test_result.py`
  with `-p no:cacheprovider --basetemp=out/visual_review/tmp_final` →
  **265 passed, 3 skipped (model-dependent), 2 deselected**
  (model_integration), 31.5 s
* `ruff check` + `ruff format --check` on all changed files → 0
* `mypy` on `core/viz/chartspec.py core/viz/plotters.py
  core/profiler/registry.py tools/charts.py
  scripts/make_final_blocker_evidence.py tests/test_viz_final_blockers.py`
  → 0
* `compileall` on changed files → 0
* Registry `validate_specs(TOOL_REGISTRY) == ()` (x-tick-angle mirror
  updated to the None sentinel); spec-mirror tests pass without exceptions.
* Architecture guards (layering, no-import-side-effects) pass.
* `python -m tools.unified --list --json` subprocess: exit 0, one JSON
  object (module guard intact).

## Review artifact paths

### Second-review blockers (fresh, for senior re-review)

* `out/visual_review/final_blockers/manifest.json` — per-case ACTUAL figure
  metadata: effective axis labels, heatmap x/y coordinates + z matrix +
  colorbar title, bar ticktext/categoryorder/categoryarray/tickangle and the
  per-trace rendered tick mapping (tick_index + value). Every `run_dir` is
  recorded AFTER successful finalization (published name, absolute, never a
  staging dir) and the script self-verifies on every run: each run_dir must
  exist with `result.json`, each png_path must exist — otherwise exit 1.
  Regenerations preserve previous run dirs (inspection history).
* `final_sparse_bar_vertical_png__*` / `final_sparse_bar_horizontal_png__*`
  — genuinely sparse bar PNGs (value 3 sits under C, both orientations;
  horizontal ticks upright)
* `final_histogram_counts_png__*` / `final_histogram_percent_png__*` —
  histogram axis-label PNGs (x=val, y=Count / Percent)
* `final_heatmap_categorical_gaps_png__*`, `final_heatmap_numeric_png__*`,
  `final_heatmap_datetime_png__*` — go.Heatmap PNGs with gaps / numeric
  coordinates 1, 2, 10 / chronological datetime coordinates

### Earlier rounds (still on disk)

* `out/visual_review/review/` — 8 senior-review case runs (HTML + CSV +
  envelope), one `<case>.png` each in `review_<case>_png__*` run dirs
* `out/visual_review/review/manifest.json` — per-case manifest (verified paths)
* `out/visual_review/images/all_kinds_<kind>.{png,svg,pdf}` — six kinds ×
  three formats (real kaleido exports)
* `out/visual_review/export_evidence.json` — export results + invalid-browser
  -path failure evidence
* `out/visual_review/independent/` — the senior's own reproducer runs

## Remaining limitations

* PNG/SVG/PDF rendering fidelity (fonts, margins, tick text) is confirmed
  non-empty and axis-correct programmatically; final visual judgment is the
  senior reviewer's (fresh `final_blockers` PNGs provided for inspection).
* Kaleido is NOT installed globally; real exports use the isolated
  `out/visual_review/deps` via per-command PYTHONPATH. `plotly-image` now
  floors `kaleido>=1.0` + `plotly>=6.1.1` so a real install gets the
  supported pair.
* `KALEIDO_CHROME_PATH` remains accepted only as a legacy alias after
  `BROWSER_PATH` and `--browser-path`.
* HTML fallback without plotly remains a data table (WARNING diagnostic),
  never claimed as an interactive chart.
* Baseline homework-script lint errors (`scripts/*hw1*`, unrelated) are
  out of scope per the review.
