import { useEffect, useState } from "react";
import { api } from "./api";
import type { VisualizationInfo } from "./api";

/**
 * Every visualization the suite can draw, badged by where it came from.
 *
 * The distinction is not decoration. Work built on NLP Suite 1.6.38 often
 * requires its specific outputs — the Excel workbooks and the standalone HTML
 * charts especially — so "which of these did the original suite also draw?"
 * is a question with a practical answer, and filtering to exactly that set is
 * the point of this view.
 */
export function VisualizationCatalog({
  items,
}: {
  items?: VisualizationInfo[];
}) {
  const [loaded, setLoaded] = useState<VisualizationInfo[] | null>(
    items ?? null,
  );
  const [filter, setFilter] = useState<"all" | "legacy" | "new">("all");

  useEffect(() => {
    if (items) return;
    let alive = true;
    api<VisualizationInfo[]>("/visualizations")
      .then((value) => {
        if (alive) setLoaded(value);
      })
      .catch(() => {
        if (alive) setLoaded([]);
      });
    return () => {
      alive = false;
    };
  }, [items]);

  if (!loaded || loaded.length === 0) return null;

  const shown = loaded.filter((item) =>
    filter === "all"
      ? true
      : filter === "legacy"
        ? item.is_legacy
        : !item.is_legacy,
  );
  const legacyCount = loaded.filter((item) => item.is_legacy).length;

  return (
    <section className="panel viz-catalog" aria-label="Visualizations">
      <h2>Visualizations</h2>
      <p>
        {legacyCount} of {loaded.length} are drawn by NLP Suite 1.6.38 as well,
        so work that expects the original suite&apos;s output still has it. The
        rest are additions.
      </p>
      <div className="viz-filters" role="group" aria-label="Filter by origin">
        {(["all", "legacy", "new"] as const).map((value) => (
          <button
            key={value}
            type="button"
            className={`secondary ${filter === value ? "selected" : ""}`}
            aria-pressed={filter === value}
            onClick={() => setFilter(value)}
          >
            {value === "all"
              ? "All"
              : value === "legacy"
                ? "From the original suite"
                : "New here"}
          </button>
        ))}
      </div>
      <ul className="viz-list">
        {shown.map((item) => (
          <li key={item.name}>
            <div className="viz-head">
              <strong>{item.title}</strong>
              <span className={`viz-origin viz-origin-${item.origin}`}>
                {item.origin === "legacy"
                  ? "legacy"
                  : item.origin === "legacy-extended"
                    ? "legacy + extended"
                    : "new"}
              </span>
              <code>{item.produces.join(", ")}</code>
            </div>
            <p>{item.summary}</p>
            {item.legacy_module && (
              <p className="muted">
                Original: <code>{item.legacy_module}</code>. {item.legacy_note}
              </p>
            )}
            {item.advantage && (
              <p className="viz-advantage">{item.advantage}</p>
            )}
            <p className="muted">
              Produced by <code>{item.tool}</code>.
            </p>
          </li>
        ))}
      </ul>
    </section>
  );
}
