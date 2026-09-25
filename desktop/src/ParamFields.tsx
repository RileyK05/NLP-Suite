import { ResourceField } from "./ResourceField";
import { desktopParams, type Param, type Tool } from "./api";

/**
 * A tool's parameters as controls, from the registry's own description of them.
 *
 * Every label, choice, bound and help string comes from the engine — the
 * desktop invents none of them, for the reason `/api/tools` exists at all: the
 * app used to keep its own copy of the tool vocabulary and both copies drifted
 * without anything failing.
 *
 * This lives apart from the run dialog because it is now used twice. The
 * dialog collects parameters for a job that will be submitted; the live bench
 * collects the same parameters for an analysis that runs as you change them.
 * One implementation, so a parameter cannot be editable in one place and
 * missing in the other.
 */

export type Params = Record<string, unknown>;

const NUMERIC = new Set(["int", "float"]);

/**
 * Parameters worth showing for this tool, with the ones that do not apply
 * removed.
 *
 * `sentiment_vader_anew` takes an ANEW lexicon only when it is asked to run
 * the ANEW half; offering the field the rest of the time invites someone to
 * fill in a path that will be ignored.
 */
export function visibleParams(tool: Tool, params: Params): Param[] {
  // `desktopParams` first: it drops the flags the desktop does not offer at
  // all (spellcheck's --correct writes files; search's csv mode needs a table),
  // and bypassing it would put them back.
  return desktopParams(tool).filter(
    (param) =>
      !(
        tool.name === "sentiment_vader_anew" &&
        param.name === "anew-lexicon" &&
        params.analysis !== "both"
      ),
  );
}

/** The defaults the engine declares, as a starting set of parameters. */
export function defaultParams(tool: Tool): Params {
  const params: Params = {};
  for (const param of desktopParams(tool)) {
    if (param.default !== null && param.default !== undefined) {
      params[param.name] = param.default;
    }
  }
  return params;
}

export function ParamFields({
  tool,
  params,
  onChange,
  onBusyChange,
  disabled,
}: {
  tool: Tool;
  params: Params;
  onChange: (params: Params) => void;
  /** Called while a resource file is uploading, so callers can block Run. */
  onBusyChange?: (uploading: boolean) => void;
  disabled?: boolean;
}) {
  const set = (name: string, value: unknown) =>
    onChange({ ...params, [name]: value });

  return (
    <>
      {visibleParams(tool, params).map((param) => (
        <label className="field-label" key={param.name}>
          {param.label}
          {param.type === "bool" ? (
            <input
              type="checkbox"
              disabled={disabled}
              checked={!!params[param.name]}
              onChange={(event) => set(param.name, event.target.checked)}
            />
          ) : param.type === "path" ? (
            <ResourceField
              onBusyChange={(uploading) => onBusyChange?.(uploading)}
              value={String(params[param.name] ?? "")}
              onChange={(value) => set(param.name, value)}
            />
          ) : param.choices.length ? (
            <select
              disabled={disabled}
              value={String(params[param.name] ?? "")}
              onChange={(event) =>
                set(
                  param.name,
                  NUMERIC.has(param.type)
                    ? Number(event.target.value)
                    : event.target.value,
                )
              }
            >
              {param.choices.map((choice) => (
                <option key={choice} value={choice}>
                  {param.choice_labels?.[String(choice)] ?? choice}
                </option>
              ))}
            </select>
          ) : (
            <input
              type={NUMERIC.has(param.type) ? "number" : "text"}
              step={param.type === "int" ? "1" : "any"}
              required={param.required}
              disabled={disabled}
              min={param.minimum ?? undefined}
              max={param.maximum ?? undefined}
              value={String(params[param.name] ?? "")}
              onChange={(event) =>
                set(
                  param.name,
                  NUMERIC.has(param.type) && event.target.value !== ""
                    ? Number(event.target.value)
                    : event.target.value,
                )
              }
            />
          )}
          <small>{param.help}</small>
        </label>
      ))}
    </>
  );
}
