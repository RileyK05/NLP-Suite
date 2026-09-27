import { VisualizationCatalog } from "./VisualizationCatalog";
import { useEffect, useState } from "react";
import {
  BookOpen,
  Bot,
  Compass,
  Download,
  FlaskConical,
  FolderOpen,
  GitCompareArrows,
  Layers3,
  Package,
  Search,
  ShieldCheck,
  SquareCode,
  type LucideIcon,
} from "lucide-react";
import { type Tool } from "./api";
import { scriptReference, type ScriptReference } from "./notebooks";
import { toolFamily } from "./ToolCard";
import content from "./toolGuides.json";

export type Guide = {
  what: string;
  how: string;
  formula: string;
  question: string;
  interpretation: string;
  limits: string;
  settings: string;
  example: string;
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

/** The tour at the top of the page: what each part of the workspace is for. */
const FEATURES: { icon: LucideIcon; name: string; how: string }[] = [
  {
    icon: FolderOpen,
    name: "Import a corpus",
    how: "Corpus → add files or a whole folder: plain text, CSV/TSV, HTML, PDF, Word and RTF. A book can be split into its chapters and sections first, so later readings work one chapter at a time.",
  },
  {
    icon: Compass,
    name: "Ask, and see the answer change",
    how: "Interactive runs a tool over what is selected right now and redraws as you adjust the selection or the settings — the quickest way to find out what is worth running over the whole corpus.",
  },
  {
    icon: FlaskConical,
    name: "Run it over the whole corpus",
    how: "Analyze & visualize queues any tool with its settings; jobs keep running if you switch pages. What each setting means is in the field guide further down this page.",
  },
  {
    icon: Layers3,
    name: "Come back to what you produced",
    how: "Past runs holds every finished run: tables, figures and panels, with the parameters and input checksums that made them. Open an artifact, or export the whole run.",
  },
  {
    icon: GitCompareArrows,
    name: "Compare two collections",
    how: "Compare places two projects side by side — counts, rates and rate ratios — so a difference is read as a rate over each corpus's own words, not as a bigger number.",
  },
  {
    icon: SquareCode,
    name: "Write it yourself",
    how: "Scripts opens a notebook whose cells call the suite's own functions. Templates answer common requests on day one; anything else is a few lines (see below).",
  },
  {
    icon: Package,
    name: "Choose the language models",
    how: "Models shows what parsing and embedding models are installed, and lets you add or remove them. A tool says in its guide when it needs one.",
  },
  {
    icon: Download,
    name: "Take results with you",
    how: "Every table downloads as CSV, figures as images or self-contained HTML, and a full run exports as a ZIP with its settings — enough to show how any number was made.",
  },
  {
    icon: ShieldCheck,
    name: "Everything stays on this machine",
    how: "The workspace is local: your files are copied into it, analyses run offline, and nothing is sent anywhere. Settings & backups controls where the workspace lives.",
  },
];

/** The first script: only calls the library really has (tests run the templates). */
const FIRST_SCRIPT = `import nlpsuite as nlp

# The project's corpus, with its document table (names, years, any columns).
corpus = nlp.corpus()

# How often a group of words is used: whole-word counts and rates per 1,000
# words. "exact-lowercase" counts raw forms; "lemma" needs the parser.
rates = nlp.term_rates(
    corpus,
    {"border": ["border", "borders"]},
    per=1000,
    match="exact-lowercase",
)

nlp.show(rates, title="Rate in each document")  # table, with a chart beside it
nlp.save(rates, "rates_per_document")           # kept with the notebook`;

export function Learn({
  tools,
  onChoose,
  canRun,
  onScriptHelp,
}: {
  tools: Tool[];
  onChoose: (tool: Tool) => void;
  canRun: boolean;
  /** Opens the Scripts page with the "Get help from an AI chatbot" dialog. */
  onScriptHelp?: () => void;
}) {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState("readability");
  const [reference, setReference] = useState<ScriptReference | null>(null);
  const [referenceError, setReferenceError] = useState("");
  const matches = matchingGuides(tools, query);
  const tool = matches.find((item) => item.name === selected) || matches[0];
  const guide = tool && guides[tool.name];
  useEffect(() => {
    let alive = true;
    scriptReference()
      .then((found) => alive && setReference(found))
      .catch(
        (caught) =>
          alive &&
          setReferenceError(String((caught as Error).message || caught)),
      );
    return () => {
      alive = false;
    };
  }, []);
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">LEARN AS YOU EXPLORE</span>
          <h1>Learn</h1>
          <p>
            What this workspace is for, how to use it from your own scripts, and
            what each tool measures — including what its results cannot tell
            you.
          </p>
        </div>
        <BookOpen size={28} />
      </div>
      <section
        className="panel learn-tour"
        aria-label="What the workspace does"
      >
        <h2>What this workspace does</h2>
        <p>
          A collection of texts becomes evidence you can point at: counts with
          the rates behind them, figures you can re-open, and settings recorded
          so a number can be defended later. The pages below are the stages of
          that work — this is what each one is for.
        </p>
        <div className="feature-grid">
          {FEATURES.map(({ icon: Icon, name, how }) => (
            <div className="feature-card" key={name}>
              <Icon size={19} />
              <div>
                <strong>{name}</strong>
                <p>{how}</p>
              </div>
            </div>
          ))}
        </div>
      </section>
      <section
        className="panel learn-scripting"
        aria-label="How to use this tool in scripts"
      >
        <h2>How to use this tool in scripts</h2>
        <p>
          Every analysis the app runs is also a Python function. The Scripts
          page opens a notebook whose cells call them — pandas, numpy,
          matplotlib, seaborn, scipy and scikit-learn are there too, and nothing
          else is installed. Variables live on between cells; a cell's last line
          is displayed if it is a value, so a table or a figure shows itself.
        </p>
        <h3>The shape of a script</h3>
        <pre className="example-code">
          <code>{FIRST_SCRIPT}</code>
        </pre>
        <p>
          Any tool also runs by name —{" "}
          <code>nlp.run(&quot;readability&quot;, corpus)</code> — and{" "}
          <code>nlp.tools()</code> lists them,{" "}
          <code>nlp.describe(&quot;tool&quot;)</code> explains one,{" "}
          <code>nlp.figure(fig, &quot;name&quot;)</code> keeps a figure and{" "}
          <code>nlp.note(&quot;text&quot;)</code> writes a remark into the
          results. Keep things with{" "}
          <code>nlp.save(table, &quot;name&quot;)</code> rather than writing
          files: the notebook's folder is scratch space that is thrown away.
        </p>
        <h3>Worked examples</h3>
        {referenceError ? (
          <p className="muted">
            The examples could not be loaded ({referenceError}). They are also
            in Scripts → New notebook, which offers each of them as a template.
          </p>
        ) : !reference ? (
          <p className="muted">Loading the worked examples…</p>
        ) : (
          reference.templates.map((template) => (
            <article className="example" key={template.id}>
              <em>&ldquo;{template.request}&rdquo;</em>
              <p className="muted">
                <strong>{template.name}</strong> — {template.description}
              </p>
              <pre className="example-code">
                <code>
                  {template.cells
                    .filter((cell) => cell.kind === "code")
                    .map((cell) => cell.text)
                    .join("\n\n")}
                </code>
              </pre>
            </article>
          ))
        )}
        {onScriptHelp && (
          <div className="sample-callout">
            <Bot size={23} />
            <div>
              <strong>
                Need an analysis none of these tools do on their own?
              </strong>
              <p>
                Write it on the Scripts page, where the suite's tools are Python
                functions. The app can write a guide for a chatbot you already
                use: hand the guide to the chatbot, and the code it writes will
                call these real functions instead of guessing at them.
              </p>
            </div>
            <button className="secondary" onClick={onScriptHelp}>
              Get help from an AI chatbot
            </button>
          </div>
        )}
      </section>
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
      <h2 className="learn-section-heading">The tool field guide</h2>
      <p className="muted">
        What each tool measures, how to use it, and what its results cannot tell
        you.
      </p>
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
                    ["How it is measured", guide.formula],
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
                <section>
                  <h3>Use it in scripts</h3>
                  <p>
                    On the Scripts page, in a notebook cell (fuller examples are
                    at the top of this page):
                  </p>
                  <pre className="example-code">
                    <code>{guide.example}</code>
                  </pre>
                </section>
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
              {guide && <p>{guide.settings}</p>}
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
                These are the settings the app sends to the engine. Values are
                checked when the run starts; a significance level must be
                strictly between 0 and 1.
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
                What a successful run can produce. Which files actually appear
                depends on the settings you chose — Runs &amp; results lists
                exactly what each run made. Export the run to keep its settings
                and input checksums with the results.
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
              This opens the settings; it does not start a run. Before treating
              any automated output as research evidence, read a few source
              passages yourself.
            </p>
          </article>
        )}
      </div>
    </>
  );
}
