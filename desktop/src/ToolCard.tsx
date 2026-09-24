import {
  ArrowUpRight,
  BookOpen,
  Braces,
  ChartColumnBig,
  ChevronRight,
  Feather,
  FileText,
  FolderOpen,
  HeartPulse,
  Images,
  Layers,
  MapPinned,
  Table2,
  Tags,
  TrendingUp,
  TriangleAlert,
  Type,
  Waypoints,
  Wrench,
  type LucideIcon,
} from "lucide-react";
import type { Tool } from "./api";

/**
 * What kind of work a tool is: its symbol, its caption, and its shelf.
 *
 * The identity comes from the engine's `family` (the syllabus taxonomy in
 * `core/profiler/registry.py`), not from where the card happens to sit in a
 * list and not from a guess at its input format. The old guess — icon by
 * list position, then by category/input-kind — could not tell "topic
 * modeling" from "word embeddings" from "sentiment", which is exactly the
 * chunking the course grades. `tests/test_tool_families.py` keeps the engine
 * side honest; the tests here keep this mapping complete.
 */

/** Families in the order the course meets them; the gallery follows this. */
export const FAMILY_ORDER = [
  "file_intake",
  "corpus_statistics",
  "words",
  "style",
  "topics",
  "embeddings",
  "parsers_conll",
  "annotators",
  "gis",
  "narrative_svo",
  "sentiment",
  "story_shape",
  "visualization",
  "utilities",
] as const;

const FAMILY_ICONS: Record<string, LucideIcon> = {
  file_intake: FolderOpen,
  corpus_statistics: Table2,
  words: Type,
  style: Feather,
  topics: Layers,
  embeddings: Waypoints,
  parsers_conll: Braces,
  annotators: Tags,
  gis: MapPinned,
  narrative_svo: BookOpen,
  sentiment: HeartPulse,
  story_shape: TrendingUp,
  visualization: ChartColumnBig,
  utilities: Wrench,
};

/** The fallback derivation, for a payload that predates the taxonomy. */
function legacyFamily(tool: Tool): { key: string; caption: string; Icon: LucideIcon } {
  if (tool.category === "visualization")
    return tool.input_kind === "csv"
      ? { key: "picture", caption: "TABLE VISUALIZATION", Icon: ChartColumnBig }
      : { key: "corpus_picture", caption: "CORPUS VISUALIZATION", Icon: Images };
  if (tool.input_kind === "csv")
    return { key: "table", caption: "TABLE STATISTICS", Icon: Table2 };
  return tool.requires_parse
    ? { key: "parse", caption: "PARSER-ASSISTED", Icon: Waypoints }
    : { key: "text", caption: "TEXT EXPLORATION", Icon: FileText };
}

export function toolFamily(tool: Tool): {
  key: string;
  caption: string;
  Icon: LucideIcon;
} {
  const family = tool.family ?? "";
  if (family && FAMILY_ICONS[family]) {
    // A table visualization is still a visualization; the symbol says which
    // kind of picture work it is, the caption says the shelf it lives on.
    const Icon =
      family === "visualization" && tool.input_kind === "csv"
        ? Images
        : FAMILY_ICONS[family];
    return {
      key: family,
      caption: (tool.family_label ?? family).toUpperCase(),
      Icon,
    };
  }
  return legacyFamily(tool);
}

/** One card, used by the studio gallery and the Learn shelf. */
export function ToolCard({
  tool,
  onOpen,
  disabled,
}: {
  tool: Tool;
  onOpen: () => void;
  disabled: boolean;
}) {
  const family = toolFamily(tool);
  const unavailable = tool.availability?.state === "needs_setup";
  return (
    <button className="analysis-card" onClick={onOpen} disabled={disabled}>
      <div className="analysis-card-top">
        <span className={`analysis-symbol family-${family.key}`}>
          <family.Icon size={21} aria-hidden="true" />
        </span>
        {unavailable ? (
          <span className="card-flag" title={tool.availability?.message}>
            <TriangleAlert size={13} aria-hidden="true" /> Needs setup
          </span>
        ) : (
          <ArrowUpRight size={19} />
        )}
      </div>
      <span className="tool-category">{family.caption}</span>
      <h2>{tool.label}</h2>
      <p>{tool.description}</p>
      <div className="analysis-card-bottom">
        <span>
          {tool.requires_parse
            ? "English language model"
            : "Works directly with your input"}
        </span>
        <ChevronRight size={15} />
      </div>
    </button>
  );
}
