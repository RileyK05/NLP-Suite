import { type LiveKind, type Settings } from "./chartLayout";

export const CHART_TASKS = [
  { label: "Compare", kinds: ["bar", "heatmap", "radar"] },
  { label: "Track change", kinds: ["line", "calendar"] },
  { label: "Find relationships", kinds: ["scatter", "bubble"] },
  { label: "See distributions", kinds: ["histogram", "ecdf", "box", "violin"] },
  { label: "Show shares", kinds: ["pie", "waffle", "treemap", "sunburst"] },
  { label: "Follow flows", kinds: ["sankey"] },
] as const satisfies readonly { label: string; kinds: readonly LiveKind[] }[];

type Example = {
  title: string;
  rows: Record<string, string>[];
  settings: Pick<Settings, "x" | "y" | "group">;
};

function example(
  title: string,
  x: string,
  y: string,
  group: string,
  records: readonly (readonly [string | number, number, string?])[],
): Example {
  return {
    title,
    settings: { x, y, group },
    rows: records.map(([label, value, series]) => ({
      [x]: String(label),
      [y]: String(value),
      ...(group && series !== undefined ? { [group]: series } : {}),
    })),
  };
}

const terms = example("Theme mentions by genre", "Theme", "Mentions", "Genre", [
  ["Public trust", 46, "Speeches"],
  ["Public trust", 32, "Editorials"],
  ["Climate policy", 36, "Speeches"],
  ["Climate policy", 54, "Editorials"],
  ["Cost of living", 28, "Speeches"],
  ["Cost of living", 42, "Editorials"],
  ["Education", 32, "Speeches"],
  ["Education", 21, "Editorials"],
  ["Public health", 19, "Speeches"],
  ["Public health", 31, "Editorials"],
]);

const totals = example("Share of all theme mentions", "Theme", "Mentions", "", [
  ["Public trust", 78],
  ["Climate policy", 90],
  ["Cost of living", 70],
  ["Education", 53],
  ["Public health", 50],
]);

const documents = example(
  "Vocabulary and document length",
  "Word count",
  "Unique words",
  "Genre",
  [
    [280, 132, "Editorials"],
    [370, 175, "Editorials"],
    [460, 201, "Editorials"],
    [580, 242, "Editorials"],
    [690, 285, "Editorials"],
    [810, 312, "Editorials"],
    [420, 160, "Speeches"],
    [540, 188, "Speeches"],
    [680, 216, "Speeches"],
    [820, 278, "Speeches"],
    [960, 304, "Speeches"],
    [1120, 336, "Speeches"],
  ],
);

const lengths = example(
  "Sentence lengths by genre",
  "Genre",
  "Words per sentence",
  "",
  Object.entries({
    Editorials: [7, 9, 10, 10, 12, 13, 13, 14, 14, 15, 17, 20, 21, 23, 29],
    Interviews: [4, 5, 6, 6, 7, 8, 8, 9, 9, 10, 12, 15, 18, 22, 28],
    Speeches: [6, 9, 11, 14, 15, 15, 17, 18, 19, 21, 22, 23, 26, 30, 36],
  }).flatMap(([genre, values]) =>
    values.map((value) => [genre, value] as const),
  ),
);

export const CHART_EXAMPLES: Record<LiveKind, Example> = {
  bar: terms,
  heatmap: terms,
  sunburst: terms,
  treemap: terms,
  pie: totals,
  waffle: totals,
  scatter: documents,
  bubble: documents,
  box: lengths,
  violin: lengths,
  histogram: {
    ...lengths,
    settings: { x: "", y: "Words per sentence", group: "Genre" },
  },
  ecdf: {
    ...lengths,
    title: "How many sentences are at or below a given length?",
    settings: { x: "", y: "Words per sentence", group: "Genre" },
  },
  line: example(
    "Climate mentions over time",
    "Year",
    "Mentions per 10,000 words",
    "Genre",
    [18, 24, 21, 37, 42, 38, 54, 58].flatMap((value, index) => [
      [2017 + index, value, "Editorials"],
      [2017 + index, [12, 16, 21, 26, 21, 28, 35, 44][index], "Speeches"],
    ]),
  ),
  radar: example("Genre profiles", "Measure", "Score out of 100", "Genre", [
    ["Dialogue", 26, "Editorials"],
    ["Dialogue", 85, "Interviews"],
    ["Evidence", 84, "Editorials"],
    ["Evidence", 57, "Interviews"],
    ["Formality", 72, "Editorials"],
    ["Formality", 35, "Interviews"],
    ["Personal voice", 48, "Editorials"],
    ["Personal voice", 84, "Interviews"],
    ["Vocabulary", 77, "Editorials"],
    ["Vocabulary", 61, "Interviews"],
  ]),
  calendar: example(
    "Articles published per day",
    "Date",
    "Articles",
    "",
    [31, 29, 31, 30, 31, 30].flatMap((days, month) =>
      Array.from({ length: days }, (_, day) => [
        `2024-${String(month + 1).padStart(2, "0")}-${String(day + 1).padStart(2, "0")}`,
        (day * 13 + month * 7) % (4 + month * 2),
      ]),
    ),
  ),
  sankey: example("Theme mentions by genre", "Genre", "Mentions", "Theme", [
    ["Editorials", 54, "Climate policy"],
    ["Editorials", 32, "Public trust"],
    ["Editorials", 21, "Education"],
    ["Interviews", 14, "Climate policy"],
    ["Interviews", 45, "Public trust"],
    ["Interviews", 13, "Education"],
    ["Speeches", 36, "Climate policy"],
    ["Speeches", 46, "Public trust"],
    ["Speeches", 32, "Education"],
  ]),
};
