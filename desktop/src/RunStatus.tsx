import { Check, LoaderCircle, TriangleAlert } from "lucide-react";
import type { Diagnostic } from "./api";

/**
 * What a run's state is called.
 *
 * The engine's states are identifiers. Only DONE had been given a word, so a
 * finished run read "Completed" while its neighbours read "running", "queued"
 * and — next to a warning triangle — "failed", in lower case. Every state is
 * named here, in one register.
 */
export const STATE_LABELS: Record<string, string> = {
  QUEUED: "Queued",
  RUNNING: "Running",
  DONE: "Completed",
  PARTIAL: "Finished with errors",
  FAILED: "Failed",
  CANCELLED: "Cancelled",
  INTERRUPTED: "Interrupted",
};
export const stateLabel = (state: string): string =>
  STATE_LABELS[state] ??
  state.charAt(0) + state.slice(1).toLowerCase().replaceAll("_", " ");

export function Badge({ state }: { state: string }) {
  return (
    <span className={`badge ${state.toLowerCase()}`}>
      {state === "DONE" ? (
        <Check size={12} />
      ) : ["RUNNING", "QUEUED"].includes(state) ? (
        <LoaderCircle size={12} className="spin" />
      ) : (
        <TriangleAlert size={12} />
      )}{" "}
      {stateLabel(state)}
    </span>
  );
}

/**
 * What a run reported, with the things that change a reading shown.
 *
 * Every diagnostic used to go into one collapsed drawer labelled "Run
 * diagnostics (7)", regardless of severity. That is the right home for the
 * routine ones — a skipped empty document, a rounded value — but it also hid
 * the ones that change what the result means. A run that fell back to a
 * different parser because the configured one had no usable model still
 * produces a table; the only sign that it is not the table you asked for is a
 * WARNING that nobody opened the drawer to see.
 *
 * Errors and warnings are shown. Notes stay in the drawer.
 */
export function Diagnostics({ items }: { items: Diagnostic[] }) {
  if (!items.length) return null;
  const loud = items.filter((d) => d.severity !== "INFO");
  const quiet = items.filter((d) => d.severity === "INFO");
  return (
    <>
      {loud.map((d, i) => (
        <div
          key={`${d.code}-${i}`}
          className={`alert ${d.severity === "ERROR" ? "error" : "warning"}`}
          role={d.severity === "ERROR" ? "alert" : "status"}
        >
          <TriangleAlert size={17} aria-hidden="true" />
          <span>
            {d.message} <code className="diagnostic-code">{d.code}</code>
          </span>
        </div>
      ))}
      {quiet.length > 0 && (
        <details className="diagnostics">
          <summary>
            {quiet.length === 1 ? "1 note" : `${quiet.length} notes`} from this
            run
          </summary>
          {quiet.map((d, i) => (
            <p key={`${d.code}-${i}`}>
              <strong>{d.code}</strong>
              <br />
              {d.message}
            </p>
          ))}
        </details>
      )}
    </>
  );
}
