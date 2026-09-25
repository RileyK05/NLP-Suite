import { useCallback, useEffect, useState } from "react";
import { LoaderCircle, RefreshCw, Sparkles, TriangleAlert } from "lucide-react";

import { glanceFigureUrl, glanceStatus, startGlance, type Glance as GlanceAnswer, type GlanceFigure } from "./api";

/** How often a running glance is read back. */
const POLL_MS = 2000;

function Figure({ projectId, figure }: { projectId: string; figure: GlanceFigure }) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    let alive = true;
    let made = "";
    void glanceFigureUrl(projectId, figure.path).then((value) => {
      made = value;
      if (alive) setUrl(value);
      else URL.revokeObjectURL(value);
    });
    return () => {
      alive = false;
      if (made) URL.revokeObjectURL(made);
    };
  }, [projectId, figure.path]);
  return (
    <figure className="glance-figure">
      {url ? <img src={url} alt={figure.label} /> : <div className="glance-figure-wait" aria-hidden="true" />}
      <figcaption>{figure.label}</figcaption>
    </figure>
  );
}

/**
 * Corpus at a glance: seven analyses the suite already has, over one parse
 * of the whole corpus, and the few sentences worth reading first.
 *
 * Nothing starts on its own: importing a document never runs it, only the
 * button does, and a result stays until the documents change (then it says
 * so and offers a refresh). The engine keeps the result as a finished run,
 * so it is also in Past runs.
 */
export function Glance({ projectId, parser, hasDocuments }: { projectId: string; parser: string; hasDocuments: boolean }) {
  const [answer, setAnswer] = useState<GlanceAnswer | null>(null);
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);

  const refresh = useCallback(async () => {
    if (!projectId) return;
    try {
      setAnswer(await glanceStatus(projectId));
      setError("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }, [projectId]);

  useEffect(() => {
    setAnswer(null);
    void refresh();
  }, [refresh]);

  const working = answer?.state === "queued" || answer?.state === "running";
  useEffect(() => {
    if (!working) return;
    const timer = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [working, refresh]);

  const start = async () => {
    setStarting(true);
    try {
      await startGlance(projectId, parser);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setStarting(false);
    }
  };

  const hasResult = !!answer?.summary?.length || !!answer?.figures?.length;
  const failed = answer?.state === "failed" || answer?.state === "interrupted" || answer?.state === "cancelled";
  return (
    <section className="panel glance" aria-label="Corpus at a glance">
      <div className="section-title">
        <h2>
          <Sparkles size={17} aria-hidden="true" /> Corpus at a glance
        </h2>
        {!working && (answer?.state === "none" || answer?.state === "stale" || failed || !answer) && (
          <button className="secondary" disabled={!hasDocuments || starting || !answer} onClick={() => void start()}>
            {answer?.state === "stale" || failed ? <RefreshCw size={15} /> : <Sparkles size={15} />}
            {answer?.state === "stale" ? "Refresh" : failed ? "Try again" : "Take a look"}
          </button>
        )}
      </div>
      {error && (
        <p className="glance-error" role="alert">
          <TriangleAlert size={15} aria-hidden="true" /> {error}
        </p>
      )}
      {answer?.state === "none" && (
        <p className="muted">
          Size, readability, vocabulary, tone, the most alike documents and the names that come up most: seven quick
          analyses over the whole corpus with one parse. Nothing is sent anywhere.
        </p>
      )}
      {working && (
        <p className="muted glance-working" role="status">
          <LoaderCircle size={15} className="spin" aria-hidden="true" /> {answer?.job?.stage || "Working"}…
        </p>
      )}
      {answer?.state === "stale" && (
        <p className="glance-stale" role="status">
          The documents have changed since this glance was taken. Refresh to read the corpus as it is now.
        </p>
      )}
      {failed && !hasResult && (
        <p className="glance-error" role="status">
          The glance did not finish{answer?.job?.stage ? `: ${answer.job.stage}` : ""}.
        </p>
      )}
      {hasResult && (
        <>
          <ul className="glance-summary">
            {(answer?.summary ?? []).map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
          <div className="glance-figures">
            {(answer?.figures ?? []).map((figure) => (
              <Figure key={figure.path} projectId={projectId} figure={figure} />
            ))}
          </div>
        </>
      )}
    </section>
  );
}
