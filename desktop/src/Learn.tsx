import { VisualizationCatalog } from "./VisualizationCatalog";
import { useState } from "react";
import { BookOpen, Search } from "lucide-react";
import { type Tool } from "./api";
import { toolFamily } from "./ToolCard";
import content from "./toolGuides.json";

export type Guide = {
  what: string;
  how: string;
  question: string;
  interpretation: string;
  limits: string;
};
export const guides: Record<string, Guide> = content;
export function matchingGuides(tools: Tool[], query: string): Tool[] {
  const needle = query.trim().toLowerCase();
  return tools.filter((tool) =>
    `${tool.label} ${tool.name} ${tool.description} ${Object.values(guides[tool.name] || {}).join(" ")}`
      .toLowerCase()
      .includes(needle),
  );
}

export function Learn({
  tools,
  onChoose,
  canRun,
}: {
  tools: Tool[];
  onChoose: (tool: Tool) => void;
  canRun: boolean;
}) {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState("readability");
  const matches = matchingGuides(tools, query);
  const tool = matches.find((item) => item.name === selected) || matches[0];
  const guide = tool && guides[tool.name];
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">LEARN AS YOU EXPLORE</span>
          <h1>Tool field guide</h1>
          <p>
            What each tool measures, how to use it, and what its results cannot
            tell you.
          </p>
        </div>
        <BookOpen size={28} />
      </div>
      <div className="sample-callout">
        <BookOpen size={23} />
        <div>
          <strong>A good first pass for your corpus</strong>
          <p>
            Import your folder, try Readability, then Document similarity.
            Inspect passages before comparing groups or historical periods.
            Language analyses take longer; some tools need a lexicon file that
            you choose before starting.
          </p>
        </div>
      </div>
      <label className="search-box">
        <Search size={16} />
        <input
          aria-label="Search tool guides"
          placeholder="Search a tool, concept, or research question…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </label>
      <p className="muted" role="status">
        {matches.length} of {tools.length} tool guides
      </p>
      <VisualizationCatalog />
      <div className="learn-layout">
        <nav className="learn-index" aria-label="Tool guides">
          {matches.map((item) => (
            <button
              key={item.name}
              className={`nav-link ${tool?.name === item.name ? "selected" : ""}`}
              aria-current={tool?.name === item.name ? "true" : undefined}
              onClick={() => setSelected(item.name)}
            >
              {item.label}
            </button>
          ))}
        </nav>
        {!tool ? (
          <section className="panel learn-article">
            <h2>No matching guides</h2>
            <p>
              Try another tool name or a concept such as vocabulary, dates or
              sentiment.
            </p>
          </section>
        ) : (
          <article
            className="panel learn-article"
            aria-label={`${tool.label} guide`}
          >
            {/* The same caption the gallery card shows, from the same fact.
                This used to be a fourth wording for the same distinction. */}
            <span className="eyebrow">{toolFamily(tool).caption}</span>
            <h2>{tool.label}</h2>
            {guide ? (
              <>
                {(
                  [
                    ["What it is", guide.what],
                    ["How this suite uses it", guide.how],
                    ["A question to explore", guide.question],
                    ["Reading the results", guide.interpretation],
                    ["Limitations & common mistakes", guide.limits],
                  ] as const
                ).map(([heading, text]) => (
                  <section key={heading}>
                    <h3>{heading}</h3>
                    <p>{text}</p>
                  </section>
                ))}
              </>
            ) : (
              <p>
                {tool.description} A detailed guide for this newly registered
                tool has not been written yet.
              </p>
            )}
            <section>
              <h3>Inputs & setup</h3>
              <p>
                {tool.input_kind === "csv"
                  ? "Select an observation/frequency CSV using the analysis form. A project is required, but it does not need corpus documents."
                  : "Import documents into your project. Analysis uses the copied text, not your original files."}{" "}
                {tool.requires_parse
                  ? "An English parser is required. Package availability alone does not guarantee all models or lexicons are installed."
                  : "No unconditional parser requirement; read the mode-specific guidance above."}
              </p>
            </section>
            <section>
              <h3>Settings reference</h3>
              {tool.params.length ? (
                <dl className="learn-parameters">
                  {tool.params.map((param) => (
                    <div key={param.name}>
                      {/* The label is what the run form prints above the
                          input; the flag is what the same setting is called on
                          the command line. Showing both is what makes this a
                          reference rather than a second vocabulary. */}
                      <dt>
                        {param.label}{" "}
                        <code className="learn-flag">--{param.name}</code>{" "}
                        <small>
                          ({param.type}
                          {param.required ? ", required" : ""})
                        </small>
                      </dt>
                      <dd>
                        {param.help}
                        <br />
                        <span className="muted">
                          Default:{" "}
                          {param.default == null
                            ? "not set"
                            : String(param.default)}
                          {param.choices.length
                            ? ` · Choices: ${param.choices.join(", ")}`
                            : ""}
                          {param.minimum != null
                            ? ` · Minimum: ${param.minimum}`
                            : ""}
                          {param.maximum != null
                            ? ` · Maximum: ${param.maximum}`
                            : ""}
                        </span>
                      </dd>
                    </div>
                  ))}
                </dl>
              ) : (
                <p>No additional tool-specific settings.</p>
              )}
              <p className="muted">
                Settings come from the live execution contract. Validation and
                mode/resource restrictions still apply; for significance levels,
                use a value strictly between 0 and 1.
              </p>
            </section>
            <section>
              <h3>Output reference</h3>
              <ul>
                {(tool.outputs || []).map((name) => (
                  <li key={name}>
                    <code>{name}</code>
                  </li>
                ))}
              </ul>
              <p>
                These are declared outputs; selected modes and successful
                execution determine which files appear. Runs & results shows the
                actual artifact list and diagnostics. Export the full run to
                retain parameters and input checksums.
              </p>
            </section>
            <button
              className="primary"
              disabled={!canRun}
              onClick={() => onChoose(tool)}
            >
              Configure this analysis
            </button>
            <p className="muted">
              Opening settings does not start a run. Inspect a few source
              passages before treating automated output as research evidence.
            </p>
          </article>
        )}
      </div>
    </>
  );
}
