# Charts — user guide (FR-6.5)

The `charts` tool renders publication-ready charts from an analyst CSV/TSV:
`bar`, `line`, `scatter`, `histogram`, `box`, `heatmap`, plus the
legacy-parity kinds `pie`, `sunburst`, `treemap`, `violin`, `radar`,
`waffle`, and `calendar` (calendar heatmap). It runs offline
(self-contained HTML by default), writes everything through the run envelope,
and records full provenance (input hash, effective parameters, prepared data).

## Quickstart

```bash
# Bar chart of counts by category (input CSV, output root)
python -m tools.charts mentions.csv out/ --kind bar --x category --y n

# Grouped, aggregated, horizontal
python -m tools.charts mentions.csv out/ --kind bar --x category --y n \
    --group sentiment --agg sum --horizontal --title "Mentions by category"

# Line over years (numeric axis stays chronological: 1, 2, 10 — not text order)
python -m tools.charts trend.csv out/ --kind line --x year --y value --agg mean

# Normalization and shares
python -m tools.charts counts.csv out/ --kind bar --x cat --y n \
    --normalize percent                 # whole chart sums to 100
python -m tools.charts counts.csv out/ --kind bar --x cat --y n \
    --normalize percent --scale-by group  # each group sums to 100

# Top categories (ranked by TOTAL across groups; ties break by name ascending)
python -m tools.charts counts.csv out/ --kind bar --x cat --y n --top-n 5

# Rates: mentions per 10,000 tokens (requires --agg sum)
python -m tools.charts mentions.csv out/ --kind bar --x cat --y mentions \
    --agg sum --rate-per 10000 --denominator-column tokens

# Heatmap: x = column axis, --group = row axis, y = cell value
python -m tools.charts matrix.csv out/ --kind heatmap --x doc --y hits \
    --group role --agg sum

# Histogram (common bin edges across groups; explicit bin count)
python -m tools.charts lengths.csv out/ --kind histogram --x doc --y length --bins 12

# Pie: one level of slices (no --group; that is what sunburst is for)
python -m tools.charts counts.csv out/ --kind pie --x cat --y n --agg sum

# Sunburst / treemap: x = outer ring/section, --group = inner child
python -m tools.charts hierarchy.csv out/ --kind sunburst --x topic --y n \
    --group subtopic --agg sum --top-n 10
python -m tools.charts hierarchy.csv out/ --kind treemap --x topic --y n \
    --group subtopic --agg sum

# Violin: full distribution per category (box + density, observations kept)
python -m tools.charts lengths.csv out/ --kind violin --x doc --y length

# Radar: one spoke per category, one polygon per group
python -m tools.charts profiles.csv out/ --kind radar --x metric --y score \
    --group candidate --agg mean

# Waffle: category shares as a 10x10 grid of 100 squares
python -m tools.charts counts.csv out/ --kind waffle --x cat --y n --agg sum

# Calendar heatmap: x = date column, one row per month, weekday columns
python -m tools.charts daily.csv out/ --kind calendar --x date --y count

# TSV / encodings
python -m tools.charts data.tsv out/ --kind bar --x cat --y n --delimiter tab
python -m tools.charts cp1252.csv out/ --kind bar --x cat --y n --encoding cp1252
```

## Semantics (enforced, tested)

| Rule | Behavior |
|---|---|
| Duplicates without `--agg` | FAIL (`CHART_AMBIGUOUS`) — never a silent mean |
| `--agg` on scatter/box/violin | FAIL (`CHART_UNSUPPORTED`) — observations are never collapsed |
| Heatmap without `--group` | FAIL (`CHART_MISSING_GROUP`) — y is the measure, never an axis |
| Sunburst/treemap without `--group` | FAIL at spec construction — the hierarchy needs two levels |
| `--group` on pie | FAIL (`CHART_UNSUPPORTED`) — use sunburst for two levels |
| Duplicate heatmap/calendar/waffle cells without `--agg` | FAIL (`CHART_AMBIGUOUS`) |
| Non-numeric/non-finite y | FAIL (`CHART_BAD_NUMERIC`) — never zero-filled |
| Aggregate overflow (1e308+1e308) | FAIL (`CHART_NONFINITE_AGGREGATE`) |
| Zero/negative rate or share denominators | FAIL (`CHART_ZERO_DENOMINATOR`) |
| Negative values with `--normalize` | FAIL (`CHART_NEGATIVE_VALUE`) |
| Negative slices (pie/sunburst/treemap/waffle) | FAIL (`CHART_NEGATIVE_SLICE`) — slices encode magnitude |
| Calendar x not a date column | FAIL (`CHART_NOT_DATETIME`) |
| `--agg` with histogram | FAIL (`CHART_UNSUPPORTED`) — bins count directly |
| `--top-n` on non-slice kinds | FAIL (`CHART_UNSUPPORTED`) |
| `--horizontal` on non bar | FAIL (`CHART_UNSUPPORTED`) |
| `--rate-per` without `--denominator-column` (or `--agg` != sum) | FAIL |
| `--top-n` then `--normalize` | normalize runs AFTER selection; recorded in envelope |
| Missing None/NaN/NaT in the x or group axis column | FAIL (`CHART_BAD_COLUMN`) — rows are never silently dropped, and NaN is never coerced into a fake `"nan"` category (a literal string `nan` stays a legitimate category; an unused histogram x selector may be missing) |
| Histogram axes | x is labeled with the binned VALUE column (the `--x` column, if given, is only a row selector and may be ignored), y is `Count`/`Percent`/`Share` |
| Heatmap axes | x names the column axis, y names the group (row) axis, the measure names the COLORBAR — `--x-label`/`--y-label` override |

`--top-n` ranks whole categories by total across groups (never per-group mixtures),
ties break by category name ascending. The order is recorded in the prepared
CSV's provenance (`selection: top_n then normalize`).

## Output

Each run writes (through the OutputWriter only):

* `chart.html` — offline self-contained chart (plotly.js inlined; `--cdn` opts out)
* `chart_data.csv` — the prepared/aggregated data (reproducibility)
* `chart.png|svg|pdf` — only when `--format` names it; needs `nlp-suite-ng[plotly-image]` (kaleido + Chrome). If the export fails, the run FAILS (exit 1) but the HTML + CSV are retained in a `-failed` run directory whose envelope declares them and carries the export diagnostic.
* `result.json` — envelope: CSV sha256, effective parameters, artifacts, diagnostics

### Machine-readable runs

`--json` prints exactly ONE JSON object on stdout, on success AND runtime failure:

```json
{
  "ok": true,
  "run_dir": "C:/data/out/charts__2026-09-11T...-5500e3",
  "artifacts": [{"kind": "chart", "path": "C:/.../chart.html"}, {"kind": "table", "path": "C:/.../chart_data.csv"}],
  "diagnostics": []
}
```

All artifact paths are ABSOLUTE on success and on failure-with-retained-
artifacts. Renderer warnings (e.g. the plotly-fallback degradation) are
preserved in `diagnostics` on success too. The envelope records the CSV
sha256, effective parameters, prepared-data metadata (`prepared`: agg/edges/
selection order), effective axis labels, and rate denominators in the
prepared CSV for independent audit.

## Discovery

```bash
nlp-suite --list          # plain listing (registry tools + utility CLIs)
nlp-suite --list --json   # one JSON object: {"tools": [...], "utility_clis": [...]}
```

Routing: `nlp-suite charts <args>` behaves exactly like `python -m tools.charts ...`.
An unknown tool exits 2; a tool whose module exists but fails to IMPORT
(missing optional dependency) exits 3 with an explicit diagnostic — these are
different failures.

## Styling

Okabe-Ito colorblind-safe palette, deterministic per sorted group name within
a chart (colors depend on the groups present in that chart; they are NOT
guaranteed identical across charts with different category subsets).
Wrapped/rotated categorical tick labels (raw values never mutated); numeric
and datetime axes keep raw values. `--x-tick-angle` left unspecified picks a
per-orientation default (vertical charts -30, horizontal charts upright 0);
an explicitly supplied angle — including -30 — is honored verbatim, for any
number of categories. Title + optional subtitle (`--subtitle`, with extra
top margin). Default 900x500 (`--width`/`--height`).

Grouped bars are **side-by-side by default** (`--bar-mode group`); stacking
means is never implied — `--bar-mode stack` or `relative` is the explicit
opt-in (bar charts only). Grouped histograms draw with partial opacity in
overlay mode so both distributions stay readable.

## Gallery

```bash
python scripts/make_chart_gallery.py out/gallery
```

writes six runs (one per kind) from a synthetic dataset for visual review.

## Image export

```bash
pip install -e ".[plotly-image]"      # kaleido>=1.0 + plotly>=6.1.1, Chrome required
python -m tools.charts data.csv out/ --kind bar --x cat --y n --format png
```

kaleido v1 renders via a Chrome/Chromium browser and requires plotly >= 6.1.1
(older plotly breaks static export; the extra pins the compatible pair). The
supported runtime configuration, in override order:

1. `--browser-path <chrome.exe>` — per-call executable override; a
   nonexistent path FAILS the export with `CHART_IMAGE_BROWSER_NOT_FOUND`
   (never a silent fallback to another browser).
2. `BROWSER_PATH` environment variable — kaleido/choreographer's supported
   discovery override.
3. kaleido auto-detection (typical install paths + PATH + Windows registry).

`KALEIDO_CHROME_PATH` (an earlier alias) is still accepted after
`BROWSER_PATH`. No global configuration is modified by the tool itself.

## Limitations

* Image export (png/svg/pdf) requires the optional `[plotly-image]` extra; a
  missing kaleido/Chrome produces a precise failure diagnostic and exit 1,
  never a silent HTML substitute.
* Histogram `--bins` changes the bin count; edges remain common across groups.
* The HTML fallback without plotly is a data table (same numbers) with a
  WARNING diagnostic — never claimed as an interactive chart.