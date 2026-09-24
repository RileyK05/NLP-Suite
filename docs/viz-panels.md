# Panels: tool-specific figures

A **panel** is a figure that belongs to one analysis and knows that analysis's
result shape. It is the second road beside `ChartSpec`, not a replacement for
it.

## When to write a panel instead of a chart

`core/viz/chartspec.py` draws any tidy table: name a kind, map x, y and group
to columns, and one set of rules for aggregation, normalization and rates
applies uniformly. For frequency tables, lexical series and ranked counts
that is the right tool, and a panel would be worse.

Write a panel when the result is **not a table of y by x**:

| Result | Why the generic path fails |
|---|---|
| Topic model | Two matrices (topic × term, document × topic) plus a geometry over the simplex. "Which column is x?" has no honest answer. |
| Embedding comparison | The subject is the *difference between two vector spaces*, not a column in either. |
| Keyness | The natural figure is effect size against strength of evidence. No combination of kind/x/y/agg expresses it, so it gets drawn as a bar chart of G2 — which hides whether a difference is large or merely well measured. |
| Concordance / dispersion | Positions in text, not categories on an axis. |

Panels do **not** extend `CHART_KINDS`. A kind added there inherits
x/y/group/agg validation that would mean nothing, and the generic path's
strict semantics are worth keeping unpolluted.

## The layer

```
core/viz/panelspec.py        types + validation. No renderer, no I/O.
core/viz/panels.py           the registry and the gate. One list. Owned centrally.
core/viz/panels_<tool>.py    one builder per panel. Imports panelspec only.
core/viz/panel_plotters.py   plotly rendering, HTML and image bytes.
tools/panels.py              the CLI, built from the declarations.
desktop_backend/panels.py    run -> prepared panel as JSON for the app.
desktop/src/panelLayout.ts   panel geometry for the app, pure and tested.
desktop/src/PanelCanvas.tsx  the app's own SVG drawing of a prepared panel.
desktop/src/PanelSection.tsx controls built from the declarations, on Past runs.
desktop/src/PanelPassages.tsx a clicked mark, read in the text it came from.
```

The dependency runs one way: builders know `panelspec`; only `panels.py`
knows builders. That is what keeps one list the single answer to "what
panels are there?".

## Adding a panel

1. Write `core/viz/panels_<tool>.py`. Copy `panels_keyness.py` — it is the
   reference implementation and the shortest complete example.
2. Export a `PanelDefinition` naming its tool, shape, required columns,
   parameters and notes.
3. Add one line to `PANELS` in `core/viz/panels.py`.

Nothing else. The CLI, `--list`, the JSON declarations a front end reads, the
parameter validation and the provenance caption all come from the
declaration.

### Two things that are easy to get wrong

**`tool` is the registered tool name, not the analysis module's name.** The
desktop offers a panel by the tool name written in a run's envelope. The three
LDA panels first shipped as `tool="topic_model"` — the live bench's name for
the analysis — while the tool that writes LDA runs is `lda_gensim`, so the
desktop offered them to no one and every test passed.
`TestRegistry.test_every_panel_belongs_to_a_tool_that_exists` now refuses a
name nothing registers.

**A run can write several tables.** `lda_gensim` writes `topics.csv` first,
then the dominant-topic, relevance and intertopic tables. The desktop picks the
first table whose header carries every column in `requires`, and offers a
panel only when such a table exists — so `requires` is also what decides
*whether* a panel is offered. Declare every column the builder reads.

### Build your test fixtures from the engine

Run the real engine function on a small hand-checkable input, rather than
inventing rows. The relevance panel's first fixture used relevance values of
5.2 and 4.0; the engine can never produce a positive relevance (it is a sum of
logs of probabilities), every test passed, and on the real corpus the panel
drew the top-ranked word as the *shortest* bar. A test written from an
impossible fixture agrees with whatever misconception wrote it.
`tests/test_panels_lda_relevance.py` shows the pattern, including a
`TestTheFixtureIsHonest` check that fails if the fixture stops resembling real
output.

### A builder is a pure function

```python
def my_panel(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
```

It receives a frame whose required columns are present, parameters already
defaulted and bounds-checked, and the provenance already assembled. It
imports no renderer, touches no filesystem (R3), and needs no browser — so
what the panel *claims* is testable without plotly, which is the half worth
testing.

### What the registry checks before you run

| Diagnostic | Meaning |
|---|---|
| `PANEL_UNKNOWN` | No panel by that name; the message lists the ones there are. |
| `PANEL_SHAPE_UNIMPLEMENTED` | The shape is declared but no renderer draws it. |
| `PANEL_EMPTY` | The source table has no rows. |
| `PANEL_MISSING_COLUMN` | A column in `requires` is absent; the message names it. |
| `PANEL_UNKNOWN_PARAM` | A misspelled parameter — refused, never ignored. |
| `PANEL_BAD_PARAM` | Wrong type, or outside the declared bounds. |
| `PANEL_PROVENANCE_INCOMPLETE` | Warning: no source artifact named, so the caption cannot say where the numbers came from. |
| `PANEL_BUILD_FAILED` | The builder raised; contained so one broken figure cannot fail a run that also produced good ones. |

## Two rules that are not optional

**Every mark carries its evidence.** `PanelMark.evidence` is a required
field. A drawn thing that cannot say which rows, documents or sentences it
stands for is decoration, and this is the difference between a figure and a
finding.

Evidence is expressed as **filters** — `(column, value)` pairs — never as row
indices. An index does not survive a re-run, a re-sort or the viewer's
500-row page, and a stale index points at the wrong text without ever
looking wrong.

```python
Evidence(
    scope="terms",
    filters=(("Word", word),),
    count=freq_a + freq_b,
    describe=f"{word}: {freq_a} in group A, {freq_b} in group B",
)
```

Write a test that takes each mark's filters **back to the source frame** and
asserts they select exactly the row the mark was built from. A plausible
filter that selects nothing looks identical until someone clicks. See
`TestEvidenceResolves` in `tests/test_panels.py`.

**Terms evidence also says what to read.** Filters say which *row* a mark came
from; they cannot say what to search the documents for. So a builder whose
marks are words sets `phrase` (the text to look up) and `lemma` (the run
counted lemmas, so the lookup must match every inflection — a term counted as
`be` occurs as "is" and "was"). Clicking such a mark offers **Read in
context**, which runs the same phrase search the Interactive page uses and
shows the passages. Only the builder knows the lookup: keyness uses `Word`,
collocations join `Word 1` and `Word 2`, and a collocation counted in a token
*window* gets no phrase at all, because searching "w1 w2" would find only its
adjacent cases and present that undercount as the evidence. Leave `phrase`
empty whenever no honest lookup exists; the app then offers none.

```python
Evidence(
    scope="terms",
    filters=(("Word", word),),
    count=total,
    describe=...,
    phrase=word,
    lemma=counted_on_lemmas(provenance.settings),  # the run's own `field`
)
```

**Every panel carries its provenance, inside the figure.** The caption is a
plotly annotation, not surrounding HTML, because HTML is lost the moment
anyone exports a PNG and pastes it into a document — which is exactly when a
reader needs to know where the numbers came from. The CLI puts the input
file's real sha256 in it: provenance with an empty digest is an
advertisement, not evidence.

## Shapes

`panelspec.PANEL_SHAPES` is the vocabulary; `panel_plotters.IMPLEMENTED_SHAPES`
is what exists. The vocabulary is deliberately larger, so two panels needing
one geometry cannot invent two names for it. A panel declaring a shape
nothing draws is refused up front — never half-served.

| Shape | Geometry | Status |
|---|---|---|
| `scatter_labelled` | Points in a plane, selectively labelled, with reference lines | **implemented** |
| `ranked_bars` | Horizontal bars in a computed order, optional ghost series | **implemented** — topic-term relevance |
| `stream` | Stacked bands over an ordered axis | **implemented** — topic prevalence over time |
| `line_series` | Independent lines, broken where x jumps by more than `line_gap` | **implemented** — culturomics n-gram frequency |
| `ribbon` | Coloured segments along one axis | **implemented** — paragraph topic flow |
| `small_multiples` | A grid of point-and-line plots, one per facet, shared x, each with its own y | **implemented** |
| `heatmap` | A matrix of cells coloured on a declared scale | **implemented** |
| `distribution` | A box per row with every observation drawn over it | **implemented** |
| `network` | Nodes at builder-computed positions, clickable weighted edges | **implemented** |
| `positions` | A tick per mark along its document's row | **implemented** |

Every declared shape is now drawn by both renderers. Adding a shape means a
branch in `panel_plotters.py` plus an entry in `IMPLEMENTED_SHAPES`, a layout
function in `panelLayout.ts` (a case in its `switch (prepared.shape)`), and a
figure in `PanelCanvas.tsx` (a `layout.shape === "…"` branch).
`tests/test_panel_parity.py` reads those two TypeScript places and fails if
either side draws a shape the other does not.

### What each shape reads, and the rules both renderers share

Each rule below is written once in `panel_plotters.py` and once in
`panelLayout.ts`, with the same arithmetic, so the app and the export agree
to the pixel's colour. Each is pinned by a Python test
(`tests/test_panel_shapes.py`), a TypeScript test (`panelLayout.test.ts`) on
the same sample, and a parity test that reads both.

- **Rows and columns.** `heatmap` reads `x` and `y` as indexes into
  `x_categories`/`y_categories`; `distribution` and `positions` read `y` as an
  index into `y_categories`. Row 0 is at the top in both renderers (Plotly's
  axis is reversed to match). Labels longer than 18 characters end in "…".
- **Heatmap colour.** `mark.value` on `COLOR_SCALES[color_scale]`, linear in
  RGB between evenly spaced stops, each channel rounded half up.
  `sequential` spans min to max; `diverging` spans [-m, m] with m the largest
  magnitude, so zero is always the neutral centre. A one-value domain maps to
  the middle stop. A cell with no mark is a gap, not a zero.
- **Distribution box.** Q1, median and Q3 by linear interpolation at
  position (n−1)·p (numpy's default); whiskers to the row's min and max, not
  1.5 IQR, because every point is drawn anyway. Plotly is handed the
  statistics precomputed so its own quartile method never applies. Points are
  jittered vertically by `(frac(i · 0.618…) − 0.5) · 0.5` row units, `i`
  being the point's order among its row's marks: deterministic, and not a
  staircase when the builder sorts by value.
- **Positions.** The x range is at least 0..1, so the whole document shows
  even where nothing falls.
- **Network.** The builder places nodes in [0, 1]², y = 0 at the top; the
  renderers only scale. Node size is area-proportional like the scatter.
  Edge width is 1px + 5px · weight / max weight. `edge_style="elbow"` draws
  source → (source.x, target.y) → target. Edges are clickable and focusable
  like marks (their own evidence opens their own rows); in the export each
  edge hovers at its midpoint (its corner, for an elbow).
- **Small multiples.** Facets in `facets` order, at most three across; one x
  range for every facet, and each facet's own y range.
- **Lines.** `line_series` and `small_multiples` break a line where x jumps by
  more than `line_gap` (default 1, for annual series). A group in
  `points_only` is drawn as points and never joined.
- **Size.** The app draws at the builder's `width`, and at `height − 80`
  (the export's title and caption margins are HTML in the app), so the
  default 500 draws at the app's usual 420 and a builder can ask for a tall
  panel for 87 rows.

## Say what the figure cannot show

`PreparedPanel.notes` travels with the panel into the HTML export. It is
part of the figure's meaning, not a disclaimer. The volcano's notes say that
G2 measures evidence rather than effect size, that Log Ratio is unstable on
rare words, and that a high score is a reason to go and read the concordance
rather than a result on its own.

Where the data itself invites a wrong reading, say so as a diagnostic. The
volcano emits `PANEL_ALL_ABOVE_THRESHOLD` when every plotted word clears the
significance line, because that almost always means the keyness run was
already cut with `--top-n` — so the plot shows the top of the distribution
rather than all of it, and "294 of 294 significant" would read as a finding
about the corpus when it is a fact about the input.

## Using it

```bash
# what exists, and what each one takes
python -m tools.panels --list

# the declarations, for a front end building controls
python -m tools.panels --list --json

# draw one
python -m tools.panels keyness.csv out/ --panel keyness_volcano \
    --set label-top=18 --set min-frequency=30

# and export a static image (needs the [plotly-image] extra)
python -m tools.panels keyness.csv out/ --panel keyness_volcano --format png
```

Each run writes `panel.html`, `panel_data.csv` (the rows the figure was drawn
from), optionally `panel.png`/`.svg`/`.pdf`, and `result.json` — through the
OutputWriter only. An explicitly requested image export that cannot happen
**fails the run** and keeps the HTML and CSV in a failure-marked run
directory, rather than handing back a directory that quietly lacks the thing
someone asked to publish.

## Worked example: why the volcano exists

Keyness over the 87 State of the Union addresses, mid-century (1934–1969)
against the rest, ranked by G2 as the generic bar chart would rank it:

| Word | G2 | Log Ratio |
|---|---|---|
| of | 881.9 | 0.61 |
| you | 654.1 | −2.13 |
| america | 610.3 | −2.32 |
| the | 482.0 | 0.35 |
| shall | 384.2 | **3.53** |

A bar chart of G2 puts `of` and `the` at the top. They are real differences,
precisely measured, and they tell a reader nothing. `shall` — mid-century
presidents used it, modern ones do not — sits fifth.

On the volcano those are three different places on the page, and the
labelling choice makes the point again:

- `--set label-by=evidence` → of, you, america, the, shall, which, war…
- `--set label-by=effect` → terrorist, iraq, afghanistan, kid, folk, senior,
  affordable, qaida, reconversion…

Both orders are honest and they answer different questions, which is why the
panel offers the choice rather than inventing a blend of the two.
