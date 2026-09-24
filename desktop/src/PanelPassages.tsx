import { useState } from "react";
import { BookOpen } from "lucide-react";
import { api, post } from "./api";
import type { PanelEvidence } from "./api";
import type { LiveState } from "./live";
import type { PhraseAnswer } from "./phrase";

/**
 * A panel mark, read in the text it came from.
 *
 * Clicking a mark already filters the table beside the figure, which says
 * which *row* a point stands for. It does not show a single sentence, and a
 * volcano point or a topic term is a reason to go and read -- the panels'
 * own notes say so. This closes that gap: the mark's evidence names the text
 * to look up (`evidence.phrase`, set by the builder, which alone knows that a
 * collocation is two columns and that a windowed pair is not a phrase), and
 * the answer comes from the same phrase search the Interactive page uses, so
 * there is one implementation of "find this in the corpus", not two.
 *
 * Three decisions worth stating:
 *
 * * **Nothing here parses without being asked.** Searching needs the corpus
 *   loaded. If it is not, the reader is told so and offered the load as an
 *   explicit act, because the first parse of a corpus can take a minute and a
 *   click on a word should not silently start one.
 * * **The search is over the documents loaded now.** They may not be the ones
 *   the run read -- documents are added, removed, and the Interactive page can
 *   narrow the loaded set. The answer says how many documents it searched, and
 *   the page says the difference is possible, rather than implying the
 *   passages are the run's own.
 * * **Lemma counts are read back as lemmas.** A term counted as `be` occurs in
 *   the text as "is" and "was"; `evidence.lemma` widens the search to every
 *   form, so the passages match what was counted.
 */
const SHOWN = 8;

type Status =
  | { kind: "idle" }
  | { kind: "busy"; stage: string }
  | { kind: "cold" }
  | { kind: "answered"; answer: PhraseAnswer }
  | { kind: "failed"; message: string };

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function PanelPassages({
  projectId,
  evidence,
}: {
  projectId: string;
  evidence: PanelEvidence;
}) {
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  if (!evidence.phrase) return null;
  const phrase = evidence.phrase;

  const search = async () => {
    setStatus({ kind: "busy", stage: "Finding passages…" });
    try {
      const answer = await post<PhraseAnswer>(
        `/projects/${projectId}/live/phrase`,
        {
          text: phrase,
          match_lemma: Boolean(evidence.lemma),
          evidence_limit: SHOWN,
        },
      );
      setStatus({ kind: "answered", answer });
    } catch (error) {
      setStatus({ kind: "failed", message: message(error) });
    }
  };

  const read = async () => {
    setStatus({ kind: "busy", stage: "Checking the corpus…" });
    try {
      const live = await api<LiveState>(`/projects/${projectId}/live`);
      if (live.state !== "ready") {
        setStatus({ kind: "cold" });
        return;
      }
    } catch (error) {
      setStatus({ kind: "failed", message: message(error) });
      return;
    }
    await search();
  };

  const load = async () => {
    setStatus({
      kind: "busy",
      stage: "Loading the corpus — the first time can take a minute…",
    });
    try {
      const live = await post<LiveState>(
        `/projects/${projectId}/live/warm`,
        {},
      );
      if (live.state !== "ready") {
        setStatus({
          kind: "failed",
          message: live.error || "The corpus did not finish loading.",
        });
        return;
      }
    } catch (error) {
      setStatus({ kind: "failed", message: message(error) });
      return;
    }
    await search();
  };

  return (
    <div className="panel-passages" aria-live="polite">
      {status.kind === "idle" && (
        <button type="button" className="secondary" onClick={() => void read()}>
          <BookOpen size={14} aria-hidden="true" /> Read “{phrase}” in context
        </button>
      )}
      {status.kind === "busy" && <p className="insight-note">{status.stage}</p>}
      {status.kind === "cold" && (
        <p className="insight-note">
          Reading passages needs the corpus loaded, and it is not loaded yet.{" "}
          <button
            type="button"
            className="secondary"
            onClick={() => void load()}
          >
            Load the corpus
          </button>
        </p>
      )}
      {status.kind === "failed" && (
        <p className="insight-note" role="alert">
          Could not read the passages: {status.message}
        </p>
      )}
      {status.kind === "answered" && (
        <Passages answer={status.answer} lemma={Boolean(evidence.lemma)} />
      )}
    </div>
  );
}

function Passages({ answer, lemma }: { answer: PhraseAnswer; lemma: boolean }) {
  const { summary, evidence } = answer;
  if (!evidence.rows.length) {
    return (
      <p className="insight-note">
        No passages found in the {summary.eligible_documents} documents loaded
        now. They may not be the documents this run read.
      </p>
    );
  }
  return (
    <>
      <p className="insight-note">
        {summary.occurrences} occurrence{summary.occurrences === 1 ? "" : "s"}{" "}
        in {summary.matching_documents} of the {summary.eligible_documents}{" "}
        documents loaded now{lemma ? ", counting every form of the word" : ""}.
        The loaded documents may differ from the ones this run read.
      </p>
      <ol className="passage-list">
        {evidence.rows.map((row) => (
          <li key={row.id}>
            <span className="passage-document">
              {row.document}
              {row.date ? ` · ${row.date}` : ""}
            </span>
            <span className="passage-text">
              {row.left} <mark>{row.match}</mark> {row.right}
            </span>
          </li>
        ))}
      </ol>
      {evidence.total > evidence.rows.length && (
        <p className="insight-note">
          Showing {evidence.rows.length} of {evidence.total}. The phrase
          explorer on the Interactive page pages through all of them, by year
          and by document.
        </p>
      )}
    </>
  );
}
