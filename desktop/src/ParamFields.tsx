import { useEffect, useState } from "react";
import { ResourceField } from "./ResourceField";
import { desktopParams, type Param, type Tool } from "./api";
import { loadDetails, type ProjectDetails } from "./details";

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

/** Parameters that name a document detail, and those that name one of its values. */
const DETAIL_NAME_PARAMS = new Set(["group-field"]);
const DETAIL_VALUE_PARAMS: Record<string, string> = {
  "group-a": "group-field",
  "group-b": "group-field",
};

/** The prefix a parameter value uses to name a detail ("field:Party"): one
 *  spelling with `core/io/document_fields.FIELD_VALUE_PREFIX`. */
export const FIELD_VALUE_PREFIX = "field:";

/**
 * A detail-taking choice's options: the declared choices plus one
 * `field:<name>` per project detail, labelled with the detail's name. So
 * "the axis to count along" can say `field:Party` without this file knowing
 * which details a corpus has. Date is left out: year and decade already
 * count along it.
 */
export function detailChoiceValues(
  param: Pick<Param, "choices" | "choice_labels" | "details">,
  details: Pick<ProjectDetails, "names"> | null,
): { value: string; label: string }[] {
  const options = param.choices.map((choice) => ({
    value: String(choice),
    label: param.choice_labels?.[String(choice)] ?? String(choice),
  }));
  if (!param.details || !details) return options;
  const seen = new Set(options.map((option) => option.value.toLowerCase()));
  for (const item of details.names) {
    const value = `${FIELD_VALUE_PREFIX}${item.name}`;
    if (item.name.toLowerCase() === "date" || seen.has(value.toLowerCase()))
      continue;
    options.push({ value, label: item.name });
  }
  return options;
}

/**
 * What to offer in each detail parameter's box: the project's detail names,
 * and the values of the detail already chosen. Typing a detail that exists
 * beats remembering how it was spelled.
 */
export function detailSuggestions(
  details: Pick<ProjectDetails, "names" | "documents"> | null,
  params: Params,
): Record<string, string[]> {
  if (!details) return {};
  const found: Record<string, string[]> = {};
  const names = details.names
    .map((item) => item.name)
    .filter((name) => name.toLowerCase() !== "date");
  for (const param of DETAIL_NAME_PARAMS) found[param] = names;
  for (const [param, of] of Object.entries(DETAIL_VALUE_PARAMS)) {
    const chosen = String(params[of] ?? "").toLowerCase();
    const values = new Set<string>();
    for (const perDocument of Object.values(details.documents))
      for (const [name, item] of Object.entries(perDocument))
        if (name.toLowerCase() === chosen) values.add(item.value);
    found[param] = [...values].sort((a, b) => a.localeCompare(b));
  }
  return found;
}

export function ParamFields({
  tool,
  params,
  onChange,
  onBusyChange,
  disabled,
  projectId,
}: {
  tool: Tool;
  params: Params;
  onChange: (params: Params) => void;
  /** Called while a resource file is uploading, so callers can block Run. */
  onBusyChange?: (uploading: boolean) => void;
  disabled?: boolean;
  /** The project whose document details the detail parameters can offer. */
  projectId?: string;
}) {
  const set = (name: string, value: unknown) =>
    onChange({ ...params, [name]: value });
  const wantsDetails = desktopParams(tool).some(
    (param) =>
      DETAIL_NAME_PARAMS.has(param.name) ||
      param.name in DETAIL_VALUE_PARAMS ||
      param.details,
  );
  const [details, setDetails] = useState<ProjectDetails | null>(null);
  useEffect(() => {
    if (!projectId || !wantsDetails) return;
    let alive = true;
    loadDetails(projectId)
      .then((found) => alive && setDetails(found))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [projectId, wantsDetails]);
  const suggestions = detailSuggestions(details, params);

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
              {detailChoiceValues(param, details).map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          ) : (
            <>
              <input
                type={NUMERIC.has(param.type) ? "number" : "text"}
                step={param.type === "int" ? "1" : "any"}
                required={param.required}
                disabled={disabled}
                min={param.minimum ?? undefined}
                max={param.maximum ?? undefined}
                list={
                  suggestions[param.name]?.length
                    ? `suggest-${tool.name}-${param.name}`
                    : undefined
                }
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
              {!!suggestions[param.name]?.length && (
                <datalist id={`suggest-${tool.name}-${param.name}`}>
                  {suggestions[param.name].map((value) => (
                    <option key={value} value={value} />
                  ))}
                </datalist>
              )}
            </>
          )}
          <small>{param.help}</small>
        </label>
      ))}
    </>
  );
}
