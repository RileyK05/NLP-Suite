# Visualization review: changes required

Independent review of the initial implementation. Existing passing tests are insufficient for acceptance. Fix these issues without changing homework/corpus files or weakening architectural guards.

## 1. Axis correctness (high priority)

Confirmed from real PNG exports and figure inspection:
- Horizontal bars for categories A/B and counts 20/80 put A/B ticks on the numeric x axis and label both axes with the category name. Only the categorical y axis should receive categorical ticks; x must retain numeric ticks and the measure label.
- A percent-normalized bar has prepared label `Percent`, but rendered y label remains the raw column name `Count`. Renderers must consume effective prepared labels, including rate labels and overrides.
- Histograms plot numeric values on x and counts on y but currently label them with the input category and value column names. Fix preparation and rendering together; preserve Count/Percent/Share semantics.
- Honor explicit tick angle even for fewer than nine categories. Ensure category tick mapping remains correct for sparse/interleaved groups, numeric-looking strings, and heatmap pivot ordering. Wrap categorical axes without changing data identity. Inspect long-label figures visually.

## 2. Comparison layout

The gallery titled "Mean value by category" stacks group means by default. Default grouped comparisons to side-by-side bars; offer an explicit documented bar-mode option for stacking. Keep overlapping grouped histograms readable (opacity or suitable explicit mode). Use a consistent group color map for both lines and markers; remove unused helpers and do not claim colors remain stable across different category subsets unless implemented.

## 3. Numeric failure handling (high priority)

Reproduced with `DataFrame({'c':['A','B'],'y':[1e308,1e308],'d':[1.,1.]})`:
- `ChartSpec(kind='bar',x='c',y='y',normalize='percent')` succeeds with two zero percentages because the total overflows.
- Same spec with `agg='sum',denominator_column='d',rate_per=1e308` succeeds with infinite rates.

Check finite denominators and final transformed values, not only per-cell aggregation. Handle overflow as Result diagnostics even under warnings-as-errors. Add hand-computed normal-sized controls and these overflow regression cases.

## 4. Machine-readable results and provenance

- Missing image dependency returns relative artifact paths (`chart.html`, `chart_data.csv`), contrary to the absolute-path contract. Resolve all published paths on both success and failure, and make run_dir absolute.
- Check the Result from publish_failure. Do not advertise staged/deleted artifacts if publication/finalization fails. Include publication errors in diagnostics and safely abandon unpublished runs.
- Successful JSON currently drops renderer warnings. Preserve diagnostics consistently in stdout JSON and envelopes, including fallback warnings.
- `--encoding invalid-encoding` raises uncaught LookupError. Return one failure JSON object and nonzero exit. Test malformed input and relevant argument failures as well.
- PreparedChart.prepared_by metadata is not included in the envelope. Record actual bin edges/count, selection/transformation order, effective delimiter and labels. Retain aggregated denominators in rate data or a companion artifact so rates are independently auditable.

## 5. Unified executable behavior

- `python -m tools.unified --list --json` emits nothing and exits zero because there is no module entry guard. Add the guard and subprocess tests, including unknown-tool exit status.
- _route catches every ModuleNotFoundError as an unknown tool, including dependency failures inside a known tool. Differentiate a missing requested module from its missing dependency. Test both.
- Keep installed `nlp-suite` routing intact and align documentation with actual JSON shape.

## 6. Real export configuration

PNG exports for all six kinds and SVG/PDF smoke exports succeeded using Kaleido 1.4 installed in isolated `out/visual_review/deps`; normal Python installation was not changed. Set PYTHONPATH to that absolute directory for export checks (preserve existing PYTHONPATH if present).

Passing `chrome_path='C:/definitely-does-not-exist/chrome.exe'` also succeeds: `pio.kaleido.chrome` is an ignored module attribute and changes shared global state. Implement a supported per-call browser override or remove the unsupported custom override and document the supported runtime configuration. Invalid explicitly configured browser paths must fail with a useful diagnostic. Official references: https://plotly.com/python/static-image-export/ and https://github.com/plotly/Kaleido . Installed dependency source is also available for inspection. Do not install Chrome or modify global configuration.

## 7. Verification and documentation

Some earlier tests asserted category labels on both horizontal axes, encoding the bug. Replace those expectations with correct behavior, not additional exceptions. Run changed-file lint/format/types and affected tests with explicit exit-code checks. Baseline unrelated homework-script lint errors are out of scope.

Generate fresh synthetic PNG/HTML examples covering all six kinds plus horizontal bars, normalized and rate charts, grouped bars/histograms, sparse groups, and long labels. Save stable review locations or a manifest under out/visual_review. Verify SVG/PDF success and failure too. Update implementation status with actual evidence and unresolved limitations; do not call visual review complete on test counts alone. The senior will inspect final images.

The additive writer byte method is acceptable in principle. Retaining successful artifacts on partial failure is reasonable, but fix its callers and update contradictory docstrings. Do not label changes as already reviewed/approved without evidence.

## Second visual review: release blockers still present

Independent PNG review after the reported 219 passing tests found:

1. Histogram still labels x as `grp` and y as `val`, although it plots values and counts. `_prepare_chart` discards the label returned by `_prepare_by_kind`; `_finalize_selection` resets it. Fix default histogram x to the value column and y to Count/Percent/Share. Heatmap y must name the group axis, not the measure; colorbar names the measure. Preserve explicit overrides consistently.
2. True sparse categories are MISLABELLED. Reproducer: `c=['A','C','E','B','D','F'], g=['one']*3+['two']*3, n=[1,3,5,2,4,6]`, bar x=c,y=n,group=g. Prepared labels are A..F but Plotly inserts categories by trace traversal A,C,E,B,D,F. Positional ticks falsely display the value 3 under B. Explicitly synchronize the Plotly category order with tick mapping across all categorical renderers, including horizontal bars. Test the actual full figure/rendered mapping, not merely ticktext. The current gallery named sparse_groups has every group at every category; replace/add a genuinely sparse fixture.
3. Numeric heatmap is severely broken. Reproducer: c=[1,2,10,1,2,10],g=['one']*3+['two']*3,n=[1,2,10,11,12,20]. The string sort makes heatmap x=[1,10,2]; PNG shows only one column and an axis around -3..6. Sort using dtype-aware ordering, preserve all six values and valid coordinates. Use a rendering method that supports irregular numeric coordinates correctly (e.g. graph_objects Heatmap), not imshow assumptions; do not silently turn numeric axes into unrelated numeric positions. Verify numeric and datetime heatmaps and categorical label alignment, with missing cells remaining gaps.
4. Image-export failure followed by failure-publication failure calls `_emit` twice in tools/charts.py. Fault-inject publish_failure returning Result.failure; require exactly one JSON object with both diagnostics and no paths to abandoned files. Finalize failures must also clean staging immediately and emit one object.

See independent PNG evidence under out/visual_review/independent. Complete only these blockers and sensible tick defaults (horizontal category labels should be upright by default; explicit rotation still respected). Add tests that fail against the current code, regenerate real PNGs, and run the affected suite/lint/types. Avoid another broad rewrite. Existing successfully reviewed percent/rate/horizontal numeric axis behavior must remain intact.
