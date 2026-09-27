import { describe, expect, it } from "vitest";
import {
  applyEvents,
  endUnfinished,
  matplotlibCode,
  newCell,
  parseIpynb,
  shownChartSettings,
  type CellRun,
  type KernelEvent,
} from "./notebooks";

/**
 * What a cell's output *is*, decided without drawing anything: which kernel
 * event belongs to which cell, when a cell counts as finished, and what a
 * stopped kernel means for the cells still waiting.
 */

const running = (exec: number | null): CellRun => ({
  exec,
  state: "running",
  outputs: [],
  seconds: null,
  error: null,
  started: 0,
});

describe("applyEvents", () => {
  it("puts each event under the cell whose exec it is", () => {
    const runs = { a: running(1), b: running(2) };
    const events: KernelEvent[] = [
      { event: "stream", id: 2, name: "stdout", text: "for b\n" },
      { event: "stream", id: 1, name: "stdout", text: "for a\n" },
    ];
    const next = applyEvents(runs, { 1: "a", 2: "b" }, events);
    expect(next.a.outputs).toEqual([
      { kind: "stream", name: "stdout", text: "for a\n" },
    ]);
    expect(next.b.outputs).toEqual([
      { kind: "stream", name: "stdout", text: "for b\n" },
    ]);
  });

  it("joins printed text into one block, as a terminal shows it", () => {
    const next = applyEvents({ a: running(1) }, { 1: "a" }, [
      { event: "stream", id: 1, name: "stdout", text: "one\n" },
      { event: "stream", id: 1, name: "stdout", text: "two\n" },
      { event: "output", id: 1, kind: "text", text: "42" },
      { event: "stream", id: 1, name: "stdout", text: "three\n" },
    ]);
    expect(next.a.outputs.map((o) => o.kind)).toEqual([
      "stream",
      "text",
      "stream",
    ]);
    expect(next.a.outputs[0]).toMatchObject({ text: "one\ntwo\n" });
  });

  it("finishes a cell with its time, or its error", () => {
    const next = applyEvents(
      { a: running(1), b: running(2) },
      { 1: "a", 2: "b" },
      [
        { event: "done", id: 1, ok: true, seconds: 0.25, error: null },
        {
          event: "done",
          id: 2,
          ok: false,
          seconds: 0.1,
          error: {
            type: "KeyError",
            message: "'Year'",
            trace: ["cell 3, line 2"],
          },
        },
      ],
    );
    expect(next.a).toMatchObject({ state: "done", seconds: 0.25 });
    expect(next.b).toMatchObject({
      state: "error",
      error: { type: "KeyError" },
    });
  });

  it("drops events for a run the page no longer has, rather than attaching them nearby", () => {
    const runs = { a: running(3) };
    // exec 1 belonged to a cell since deleted; exec 2 to an earlier run of "a".
    const next = applyEvents(runs, { 1: "gone", 2: "a", 3: "a" }, [
      { event: "stream", id: 1, name: "stdout", text: "orphan" },
      { event: "stream", id: 2, name: "stdout", text: "stale" },
    ]);
    expect(next).toBe(runs);
  });
});

describe("endUnfinished", () => {
  it("ends the running cell with the reason and forgets the waiting ones", () => {
    const next = endUnfinished(
      {
        done: { ...running(1), state: "done" },
        now: running(2),
        later: { ...running(null), state: "queued" },
      },
      "Stopped.",
    );
    expect(next.done.state).toBe("done");
    expect(next.now).toMatchObject({
      state: "error",
      error: { type: "Stopped", message: "Stopped." },
    });
    expect(next.later).toBeUndefined();
  });
});

describe("shownChartSettings", () => {
  it("keeps the split nlp.show made from several y columns", () => {
    const settings = shownChartSettings({
      kind: "line",
      x: "Year",
      y: "Value",
      group: "Series",
      agg: "mean",
      top_n: 0,
    });
    expect(settings).toMatchObject({
      kind: "line",
      group: "Series",
      agg: "mean",
      sort: "table",
    });
  });

  it("keeps years in order on a bar chart, and ranks anything else", () => {
    const base = { kind: "bar", y: "Passages", group: "", agg: "", top_n: 0 };
    expect(shownChartSettings({ ...base, x: "Year" })?.sort).toBe("table");
    expect(shownChartSettings({ ...base, x: "Date" })?.sort).toBe("table");
    expect(shownChartSettings({ ...base, x: "Word" })?.sort).toBe("high");
  });

  it("refuses a kind the canvas does not draw instead of substituting one", () => {
    expect(
      shownChartSettings({
        kind: "sankey",
        x: "a",
        y: "b",
        group: "",
        agg: "",
        top_n: 0,
      }),
    ).toBeNull();
  });
});

describe("matplotlibCode", () => {
  it("reads the CSV and draws the same chart", () => {
    const code = matplotlibCode(
      {
        kind: "line",
        x: "Year",
        y: "Value",
        group: "Series",
        agg: "mean",
        top_n: 0,
      },
      "yearly.csv",
      "Yearly mean",
    );
    expect(code).toContain('pd.read_csv("yearly.csv")');
    expect(code).toContain(
      'groupby(["Year", "Series"])["Value"].mean().unstack()',
    );
    expect(code).toContain('ax.set_title("Yearly mean")');
    expect(code).toContain('fig.savefig("yearly.png", dpi=200)');
    const lines = code.trim().split("\n").length;
    expect(lines).toBeGreaterThanOrEqual(8);
    expect(lines).toBeLessThanOrEqual(16);
  });

  it("quotes column names safely, whatever they contain", () => {
    const code = matplotlibCode(
      {
        kind: "bar",
        x: 'He said "no"',
        y: "per 1000",
        group: "",
        agg: "",
        top_n: 20,
      },
      "t.csv",
    );
    expect(code).toContain('table.set_index("He said \\"no\\"")["per 1000"]');
    expect(code).toContain(".head(20)");
  });

  it.each(["bar", "line", "scatter", "bubble", "histogram", "box", "heatmap"])(
    "writes something for a %s chart",
    (kind) => {
      const code = matplotlibCode(
        { kind, x: "A", y: "B", group: "", agg: "", top_n: 0 },
        "t.csv",
      );
      expect(code).toContain("plt.subplots");
      expect(code).toContain("savefig");
    },
  );
});

// Node's own modules, loaded untyped: the app is compiled without Node's
// types (adding them would retype every timer in the app), and this is the
// only test that needs a subprocess.
const node = (name: string) => import(/* @vite-ignore */ name);
const { execFileSync } = await node("node:child_process");
const { existsSync, mkdtempSync, rmSync, writeFileSync } =
  await node("node:fs");
const { tmpdir } = await node("node:os");
const { join } = await node("node:path");

/** Python with pandas and matplotlib, if this machine has it (CI's frontend job may not). */
function python(): string | null {
  for (const command of ["python", "python3"]) {
    try {
      execFileSync(command, ["-c", "import pandas, matplotlib"], {
        stdio: "ignore",
      });
      return command;
    } catch {
      // try the next name
    }
  }
  return null;
}
const PYTHON = python();

describe.skipIf(!PYTHON)("the copied matplotlib code runs", () => {
  const csv = [
    "Year,Speaker,Rate,Words",
    "1990,Bush,1.5,3000",
    "1990,Bush,2.5,2800",
    "1991,Bush,3.0,4100",
    "1994,Clinton,0.5,5200",
    "1995,Clinton,1.0,6100",
  ].join("\n");
  const charts = [
    { kind: "line", x: "Year", y: "Rate", group: "", agg: "mean" },
    { kind: "line", x: "Year", y: "Rate", group: "Speaker", agg: "mean" },
    { kind: "bar", x: "Speaker", y: "Rate", group: "", agg: "sum" },
    { kind: "bar", x: "Speaker", y: "Rate", group: "", agg: "" },
    { kind: "scatter", x: "Words", y: "Rate", group: "Speaker", agg: "" },
    { kind: "bubble", x: "Words", y: "Rate", group: "", agg: "" },
    { kind: "histogram", x: "Rate", y: "Rate", group: "", agg: "" },
    { kind: "box", x: "Speaker", y: "Rate", group: "", agg: "" },
    { kind: "heatmap", x: "Year", y: "Rate", group: "Speaker", agg: "mean" },
    // nlp.show(t, x="Year", y=["Rate", "Words"]): the CSV has the columns side by side.
    {
      kind: "line",
      x: "Year",
      y: "Value",
      group: "Series",
      agg: "mean",
      prepare: { melt: { id: "Year", values: ["Rate", "Words"] } },
    },
    // A table of passages: rows counted per year.
    {
      kind: "bar",
      x: "Year",
      y: "Passages",
      group: "",
      agg: "",
      prepare: { count: { by: "Year", name: "Passages" } },
    },
  ];
  it.each(
    charts.map((c) => [
      `${c.kind}${c.group ? " by group" : ""}${c.agg ? ` (${c.agg})` : ""}${"prepare" in c ? ` after ${Object.keys(c.prepare!)[0]}` : ""}`,
      c,
    ]),
  )(
    "%s",
    (_label, chart) => {
      const folder = mkdtempSync(join(tmpdir(), "mpl-"));
      try {
        writeFileSync(join(folder, "t.csv"), csv);
        writeFileSync(
          join(folder, "draw.py"),
          "import matplotlib\nmatplotlib.use('Agg')\n" +
            matplotlibCode({ ...chart, top_n: 0 }, "t.csv", "A title"),
        );
        execFileSync(PYTHON!, ["draw.py"], { cwd: folder, stdio: "pipe" });
        expect(existsSync(join(folder, "t.png"))).toBe(true);
      } finally {
        rmSync(folder, { recursive: true, force: true });
      }
    },
    60_000,
  );
});

describe("cells and files", () => {
  it("marks a pasted cell so the page asks for it to be read", () => {
    expect(newCell("code", "x", "pasted").metadata).toEqual({
      nlpsuite: { origin: "pasted" },
    });
    expect(newCell("markdown").metadata).toEqual({});
    expect(newCell("markdown")).not.toHaveProperty("outputs");
  });

  it("gives new cells distinct ids", () => {
    const ids = new Set(Array.from({ length: 50 }, () => newCell("code").id));
    expect(ids.size).toBe(50);
  });

  it("says why a file is not a notebook", () => {
    expect(() => parseIpynb("not json")).toThrow(/not JSON/);
    expect(() => parseIpynb('{"nbformat": 3, "worksheets": []}')).toThrow(
      /nbformat 4/,
    );
    expect(parseIpynb('{"nbformat": 4, "cells": []}')).toEqual({
      nbformat: 4,
      cells: [],
    });
  });
});
