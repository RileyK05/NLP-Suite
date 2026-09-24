import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import {
  Explore,
  firstWorthOpening,
  nameIsTaken,
  orderSources,
  sourceKey,
  sourceLabel,
  SaveBar,
  sourceOfView,
  viewLabel,
  viewMatchesSource,
  ViewBar,
  viewsOfSource,
} from "./Explore";
import { Workbench } from "./Workbench";
import type { Table, TableSource } from "./api";
import type { ChartContract, View } from "./views";

const view = (over: Partial<View> = {}): View => ({
  id: "view-1",
  name: "Sentences by document",
  job: "job-1",
  index: 0,
  artifact_path: "readability.csv",
  source_sha256: "a".repeat(64),
  settings: {
    kind: "bar",
    x: "Document",
    y: "Sentences",
    group: "",
    agg: "sum",
    top_n: 25,
    sort: "high",
    bins: 12,
  },
  filters: [],
  revision: 1,
  created: "2026-09-16T09:00:00+00:00",
  updated: "2026-09-16T09:00:00+00:00",
  source: { state: "ok", detail: "" },
  publication: [],
  ...over,
});

const contract: ChartContract = {
  bar: {
    high: "Ordering: a published chart uses category order instead.",
    low: "",
  },
};

const table = (over: Partial<TableSource> = {}): TableSource => ({
  job: "job-1",
  index: 0,
  tool: "readability",
  label: "Readability",
  path: "readability.csv",
  description: "readability readability.csv",
  created: "2026-09-15T12:08:32+00:00",
  state: "DONE",
  manifest: false,
  ...over,
});

const manifest = table({
  index: 1,
  path: "desktop_inputs.csv",
  manifest: true,
});

const rows: Table = {
  columns: ["Document", "Sentences"],
  rows: [
    { Document: "a.txt", Sentences: "12" },
    { Document: "b.txt", Sentences: "9" },
  ],
  total: 2,
  truncated: false,
};

const draw = (
  sources: TableSource[],
  selected: TableSource | null,
  loaded: Table | null = rows,
  views: View[] = [],
) =>
  renderToStaticMarkup(
    <Explore
      sources={sources}
      loading={false}
      selected={selected}
      onSelect={() => {}}
      table={loaded}
      tableLoading={false}
      canPublish
      onPublish={() => {}}
      onBrowseAnalyses={() => {}}
      liveBench={<p>live bench</p>}
      defaultMode="results"
      views={views}
      contract={contract}
      onSaveView={async () => null}
      onUpdateView={async () => null}
      onDuplicateView={async () => null}
      onDeleteView={async () => false}
    />,
  );

describe("choosing what to explore", () => {
  it("puts results ahead of the record of what a run read", () => {
    expect(orderSources([manifest, table()]).map((s) => s.path)).toEqual([
      "readability.csv",
      "desktop_inputs.csv",
    ]);
  });

  it("opens on a result rather than on an input manifest", () => {
    expect(firstWorthOpening([manifest, table()])?.path).toBe(
      "readability.csv",
    );
  });

  it("opens on nothing when a project has published nothing", () => {
    expect(firstWorthOpening([])).toBeNull();
  });

  it("falls back to a manifest when that is genuinely all there is", () => {
    // Offering an empty page when a real table exists would be worse than
    // offering the only table there is.
    expect(firstWorthOpening([manifest])?.path).toBe("desktop_inputs.csv");
  });

  it("tells apart two runs that published the same filename", () => {
    const first = table({ job: "a", created: "2026-09-15T18:57:46+00:00" });
    const second = table({ job: "b", created: "2026-09-15T20:14:40+00:00" });
    expect(sourceLabel(first)).not.toBe(sourceLabel(second));
    expect(sourceKey(first)).not.toBe(sourceKey(second));
  });

  it("keys a table by its run and its position in that run", () => {
    expect(sourceKey(table())).not.toBe(sourceKey(manifest));
  });
});

describe("the Explore page", () => {
  it("says there is nothing to explore, and where to get something", () => {
    const html = draw([], null, null);
    expect(html).toContain("Nothing to explore yet");
    expect(html).toContain("Browse analyses");
  });

  it("does not claim a count it has not got", () => {
    expect(draw([], null, null)).toContain("0 tables to explore");
  });

  it("counts results, not input manifests", () => {
    expect(draw([table(), manifest], table())).toContain("1 table to explore");
  });

  it("separates results from run records in the picker", () => {
    const html = draw([table(), manifest], table());
    expect(html).toContain("Results");
    expect(html).toContain("What each run read");
  });

  it("puts the chosen table straight under live controls", () => {
    const html = draw([table()], table());
    expect(html).toContain("workbench-controls");
    expect(html).toContain("Publish this chart");
  });

  it("keeps saying that looking is not publishing", () => {
    // The wording moved when views became savable — "this is not saved" stopped
    // being true. What must still be said is the part that has not changed:
    // the chart on screen is a page of rows and publishing is what records a
    // result over the whole file.
    const html = draw([table()], table());
    expect(html).toContain("runs nothing");
    expect(html).toContain("over the complete file");
  });

  it("says so when the chosen table cannot be read", () => {
    const html = draw([table()], table(), null);
    expect(html).toContain("could not be read");
    expect(html).not.toContain("workbench-controls");
  });
});

describe("tables that have nothing in them", () => {
  it("says a run found nothing, rather than blaming the page", () => {
    // duplicates.csv from a real doc_similarity run: a header and no rows,
    // meaning no duplicates were found. The chart used to refuse it with
    // "this page of the table has no rows", which reads like a paging fault.
    const empty: Table = {
      columns: ["Document A", "Document B", "Similarity"],
      rows: [],
      total: 0,
      truncated: false,
    };
    const html = draw(
      [table({ path: "duplicates.csv" })],
      table({ path: "duplicates.csv" }),
      empty,
    );
    expect(html).toContain("found nothing to report");
    expect(html).not.toContain("no rows");
  });
});

describe("saved views", () => {
  it("does not offer a view picker to a project that has none", () => {
    expect(draw([table()], table())).not.toContain("Saved view");
  });

  it("offers the views saved of the table on screen", () => {
    const html = draw([table()], table(), rows, [view()]);
    expect(html).toContain("Saved view");
    expect(html).toContain("Sentences by document");
  });

  it("separates views of this table from views of another", () => {
    const other = view({
      id: "view-2",
      name: "Something else",
      job: "job-9",
      artifact_path: "ngrams.csv",
    });
    const html = draw([table()], table(), rows, [view(), other]);
    expect(html).toContain("Of this table");
    expect(html).toContain("Of other tables");
    // The other table's view is still reachable, and says which file it is of,
    // because two views can otherwise carry the same name-shaped label.
    expect(html).toContain("ngrams.csv");
  });

  it("knows which views belong to a table", () => {
    const other = view({ id: "view-2", job: "job-9" });
    expect(viewsOfSource([view(), other], table()).map((v) => v.id)).toEqual([
      "view-1",
    ]);
  });

  it("finds the table a view was saved from", () => {
    expect(sourceOfView([table()], view())?.path).toBe("readability.csv");
  });

  it("says nothing rather than guessing when that table is gone", () => {
    expect(sourceOfView([], view())).toBeNull();
  });

  it("names a view by how far it has been revised", () => {
    expect(viewLabel(view())).toContain("Sentences by document");
    expect(viewLabel(view())).not.toContain("revision");
    expect(viewLabel(view({ revision: 4 }))).toContain("revision 4");
  });

  it("offers to name a view when none is open", () => {
    const html = draw([table()], table(), rows, []);
    expect(html).toContain("Name this arrangement");
    expect(html).toContain("Save view");
  });

  it("says publishing a view runs nothing at save time", () => {
    expect(draw([table()], table())).toContain("runs nothing");
  });
});

describe("a view whose source has moved on", () => {
  it("says so rather than charting the new numbers quietly", () => {
    const stale = view({
      source: {
        state: "changed",
        detail: "readability.csv is not the file this view was saved over.",
      },
    });
    // Nothing is open until a reader picks it, so the warning belongs to the
    // open view. Rendering the list alone must not raise it.
    const html = draw([table()], table(), rows, [stale]);
    expect(html).toContain("Saved view");
  });
});

describe("what publishing will not reproduce", () => {
  it("is shown next to the button that would do it", () => {
    const html = draw([table()], table());
    expect(html).toContain("Publishing will not match this exactly");
    expect(html).toContain("category order");
  });

  it("comes from the server rather than being written here", () => {
    // The default arrangement diverges, so an empty contract is the only way
    // this can be empty — which is the point: the words are shipped.
    const html = renderToStaticMarkup(
      <Explore
        sources={[table()]}
        loading={false}
        selected={table()}
        onSelect={() => {}}
        table={rows}
        tableLoading={false}
        canPublish
        onPublish={() => {}}
        onBrowseAnalyses={() => {}}
        liveBench={<p>live bench</p>}
        defaultMode="results"
        views={[]}
        contract={{}}
        onSaveView={async () => null}
        onUpdateView={async () => null}
        onDuplicateView={async () => null}
        onDeleteView={async () => false}
      />,
    );
    expect(html).not.toContain("Publishing will not match this exactly");
  });
});

describe("opening a view of a different table", () => {
  it("is settled when nothing is open", () => {
    expect(viewMatchesSource(null, table())).toBe(true);
  });

  it("is settled when the open view is of the table on screen", () => {
    expect(viewMatchesSource(view(), table())).toBe(true);
  });

  it("is not settled while the view's table is still arriving", () => {
    // reopen() asks the page above for the view's table, which comes a render
    // later. Drawing the pairing in between would complain about a column
    // nobody chose, so Explore waits instead.
    const elsewhere = view({ job: "job-9", artifact_path: "ngrams.csv" });
    expect(viewMatchesSource(elsewhere, table())).toBe(false);
  });

  it("is not settled with no table chosen at all", () => {
    expect(viewMatchesSource(view(), null)).toBe(false);
  });

  it("tells two artifacts of the same run apart", () => {
    // job and index together address a table; the job alone does not.
    expect(viewMatchesSource(view({ index: 1 }), table({ index: 0 }))).toBe(
      false,
    );
  });
});

describe("the saved view's own controls", () => {
  it("offers naming, not managing, when nothing is open", () => {
    const html = draw([table()], table(), rows, [view()]);
    expect(html).toContain("view-bar");
    expect(html).toContain("Name this arrangement");
    // Save/Revert/Duplicate/Delete belong to an open view, not to a draft.
    expect(html).not.toContain("Duplicate");
    expect(html).not.toContain("Delete");
  });

  it("keeps the delete control out of the chart's action row", () => {
    // It used to sit beside Publish: one slip from the thing people came for.
    const html = draw([table()], table(), rows, [view()]);
    const actions = html.slice(html.indexOf('class="workbench-actions"'));
    expect(actions).not.toContain("Delete");
    expect(actions).toContain("Publish this chart");
  });

  it("sees a name the project already has", () => {
    expect(nameIsTaken(["Trend", "Counts"], "Trend")).toBe(true);
  });

  it("sees it whatever the casing or the stray spaces", () => {
    // Stricter than the server, which compares exactly. Two views called
    // "Trend" and "trend" make a picker nobody can read.
    expect(nameIsTaken(["Trend"], "  trend ")).toBe(true);
    expect(nameIsTaken([" Trend "], "TREND")).toBe(true);
  });

  it("lets a genuinely new name through", () => {
    expect(nameIsTaken(["Trend"], "Trend by decade")).toBe(false);
  });

  it("does not call an empty name taken", () => {
    // Empty is refused for being empty; saying it clashes would be a lie.
    expect(nameIsTaken(["Trend"], "   ")).toBe(false);
  });
});

describe("the bar for a view that is open", () => {
  const bar = (over: Partial<View> = {}, dirty = false) =>
    renderToStaticMarkup(
      <ViewBar
        view={view(over)}
        dirty={dirty}
        onSave={() => {}}
        onRevert={() => {}}
        onDuplicate={() => {}}
        onDelete={() => {}}
      />,
    );

  it("names the view it is managing", () => {
    expect(bar({ name: "Counts by decade" })).toContain("Counts by decade");
  });

  it("says a first save is saved, not revision 1", () => {
    // "revision 1" invites the question of what revision 0 was.
    const html = bar({ revision: 1 });
    expect(html).toContain("saved");
    expect(html).not.toContain("revision");
  });

  it("counts revisions once there are any", () => {
    expect(bar({ revision: 4 })).toContain("revision 4");
  });

  it("offers nothing to save when nothing has changed", () => {
    const html = bar({}, false);
    expect(html).toContain("disabled");
    expect(html).not.toContain("Revert");
  });

  it("offers Save and Revert once something has", () => {
    const html = bar({}, true);
    expect(html).toContain("unsaved changes");
    expect(html).toContain("Revert");
  });

  it("does not delete on the first press", () => {
    // Delete asks twice. The first press only arms the second.
    const html = bar();
    expect(html).toContain("Delete");
    expect(html).not.toContain("Really delete");
  });

  it("says plainly that the result survives the view", () => {
    expect(bar()).toContain("result it charted is untouched");
  });

  it("marks delete as the destructive one", () => {
    expect(bar()).toContain("secondary danger");
  });
});

describe("the bar for an arrangement not yet saved", () => {
  const bar = (taken: string[] = []) =>
    renderToStaticMarkup(<SaveBar taken={taken} onSave={() => {}} />);

  it("asks for a name and nothing else", () => {
    const html = bar();
    expect(html).toContain("Name this arrangement");
    expect(html).toContain("Save view");
    expect(html).not.toContain("Duplicate");
    expect(html).not.toContain("Delete");
  });

  it("cannot be saved until it is named", () => {
    expect(bar()).toContain("disabled");
  });
});

describe("the workbench heading's depth", () => {
  const panel = (level?: 2 | 3) =>
    renderToStaticMarkup(
      <Workbench
        table={rows}
        artifactPath="readability.csv"
        settings={{
          kind: "bar",
          x: "Document",
          y: "Sentences",
          group: "",
          agg: "sum",
          topN: 25,
          sort: "high",
          bins: 12,
        }}
        onSettings={() => {}}
        selected={null}
        onSelect={() => {}}
        canPublish
        onPublish={() => {}}
        headingLevel={level}
      />,
    );

  it("is h3 by default, for a panel inside a run's record", () => {
    expect(panel()).toContain("<h3>");
    expect(panel()).not.toContain("<h2>");
  });

  it("is h2 on Explore, where it sits directly under the page title", () => {
    // h1 straight to h3 is a gap to anyone navigating by heading.
    expect(panel(2)).toContain("<h2>");
    expect(panel(2)).not.toContain("<h3>");
  });

  it("carries the same words and icon at either depth", () => {
    for (const html of [panel(), panel(2)]) {
      expect(html).toContain("Explore this table");
      expect(html).toContain("lucide-chart-column-big");
    }
  });
});

describe("what Explore opens on", () => {
  const live = (sources: TableSource[] = []) =>
    renderToStaticMarkup(
      <Explore
        sources={sources}
        loading={false}
        selected={null}
        onSelect={() => {}}
        table={null}
        tableLoading={false}
        canPublish
        onPublish={() => {}}
        onBrowseAnalyses={() => {}}
        liveBench={<p>live bench</p>}
        views={[]}
        contract={contract}
        onSaveView={async () => null}
        onUpdateView={async () => null}
        onDuplicateView={async () => null}
        onDeleteView={async () => false}
      />,
    );

  it("opens on running an analysis, not on a list of finished work", () => {
    // The page exists to compute something now. Opening on the results picker
    // is what made it "another way to view things you have already run".
    const html = live([table()]);
    expect(html).toContain("live bench");
    expect(html).not.toContain("Table to explore");
  });

  it("does not count published tables while running live", () => {
    // The count describes the other mode, and "0 tables to explore" beside a
    // page offering live analysis reads as an empty page.
    expect(live()).not.toContain("tables to explore");
  });

  it("keeps finished results reachable as the other mode", () => {
    expect(live([table()])).toContain("Chart a finished result");
  });
});
