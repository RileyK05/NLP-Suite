import { useCallback, useEffect, useRef, useState } from "react";
import { BookOpen, X } from "lucide-react";
import {
  applySplit,
  previewSplit,
  previewTurns,
  splitButtonWords,
  type SplitOptions,
  type SplitPreview,
  type TurnPreview,
} from "./details";

const errorText = (caught: unknown) =>
  caught instanceof Error ? caught.message : String(caught);

/** A document this long is offered for splitting (a short story is not). */
export const LONG_DOCUMENT_WORDS = 20_000;

const RULE_WORDS: Record<SplitOptions["rule"], string> = {
  auto: "Find the chapters for me",
  headings: "Heading lines (CHAPTER I, Book 2, # Title)",
  "blank-gap": "Wide blank gaps",
  words: "Blocks of words (not the author's chapters)",
  pattern: "Lines starting with…",
};

const DEFAULTS: SplitOptions = {
  rule: "auto",
  pattern: "",
  regex: false,
  block_words: 2000,
  keep_front_matter: false,
  keep_whole: false,
};

type Book = {
  id: string;
  name: string;
  words?: number;
  derived_from?: string | null;
};

/** The documents worth offering: long ones, or the only one a project has. */
export function splitCandidates(documents: Book[]): Book[] {
  if (documents.length === 1) return documents;
  return documents
    .filter((doc) => (doc.words ?? 0) > LONG_DOCUMENT_WORDS)
    .sort((a, b) => (b.words ?? 0) - (a.words ?? 0));
}

/** What the dialog's menu lists: every document not already cut from another,
 *  longest first. A transcript is often short, so it is chosen here. */
export function splittable(documents: Book[]): Book[] {
  return documents
    .filter((doc) => !doc.derived_from)
    .sort((a, b) => (b.words ?? 0) - (a.words ?? 0));
}

/** "Make 2 speaker documents" / "Make 7 turn documents". */
export function turnButtonWords(preview: TurnPreview, perTurn: boolean) {
  const kind = perTurn ? "turn" : "speaker";
  return `Make ${preview.documents} ${kind} document${preview.documents === 1 ? "" : "s"}`;
}

/**
 * "Split a long document into chapters" (docs/internal/PLAN_0.5.0.md 2.7). The engine
 * finds the cuts; this shows them -- every section with its words and
 * opening, and what was left out -- and stores them only on the button that
 * says how many there will be.
 */
export function SplitBook({
  projectId,
  documents,
  onSplit,
  disabled = false,
}: {
  projectId: string;
  documents: Book[];
  onSplit: () => void;
  disabled?: boolean;
}) {
  const offered = splitCandidates(documents).filter((doc) => !doc.derived_from);
  const candidates = splittable(documents);
  const [open, setOpen] = useState(false);
  const [turns, setTurns] = useState<TurnPreview | null>(null);
  const [chosen, setChosen] = useState("");
  const [options, setOptions] = useState<SplitOptions>(DEFAULTS);
  const [preview, setPreview] = useState<SplitPreview | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  const documentId = chosen || offered[0]?.id || candidates[0]?.id || "";

  const ask = useCallback(
    async (wanted: SplitOptions) => {
      if (!documentId) return;
      setBusy(true);
      setError("");
      try {
        if (wanted.transcript) {
          setTurns(await previewTurns(projectId, documentId, wanted));
          setPreview(null);
        } else {
          setPreview(await previewSplit(projectId, documentId, wanted));
          setTurns(null);
        }
      } catch (caught) {
        setPreview(null);
        setTurns(null);
        setError(errorText(caught));
      } finally {
        setBusy(false);
      }
    },
    [projectId, documentId],
  );

  useEffect(() => {
    if (open) void ask(options);
    // Options are sent with each change below; this runs on open and when
    // another document is chosen.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, documentId]);

  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (open && !element.open) element.showModal?.();
    if (!open && element.open) element.close?.();
  }, [open]);

  if (!offered.length) return null;

  const change = (patch: Partial<SplitOptions>) => {
    const next = { ...options, ...patch };
    setOptions(next);
    // A pattern is only worth asking about once there is one.
    if (next.rule !== "pattern" || next.pattern.trim()) void ask(next);
  };

  const close = () => {
    setOpen(false);
    setPreview(null);
    setTurns(null);
    setError("");
  };

  const shown = options.transcript ? turns : preview;
  const split = async () => {
    if (!shown) return;
    setBusy(true);
    try {
      await applySplit(projectId, documentId, {
        ...options,
        expected_sha256: shown.document.sha256,
      });
      close();
      onSplit();
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setBusy(false);
    }
  };

  const hasLicence = !!preview?.left_out.some((item) =>
    item.what.startsWith("Project Gutenberg"),
  );
  const noteFor = (order: number) =>
    preview?.diagnostics.filter((d) => d.order === order) ?? [];
  const keepSpeaker = (name: string, keep: boolean) => {
    const kept = (turns?.speakers ?? [])
      .filter((speaker) => (speaker.name === name ? keep : speaker.kept))
      .map((speaker) => speaker.name);
    change({ speakers: kept });
  };
  const ready = options.transcript
    ? !!turns && turns.documents > 0
    : !!preview && preview.sections.length > 0;

  return (
    <>
      <button
        type="button"
        className="secondary"
        disabled={disabled}
        onClick={() => setOpen(true)}
      >
        <BookOpen size={14} /> Split a long document into chapters
      </button>
      <dialog
        ref={dialog}
        className="split-dialog"
        aria-label="Split a long document into chapters"
        onCancel={(e) => {
          e.preventDefault();
          close();
        }}
      >
        {open && (
          <>
            <div className="dialog-heading">
              <h2>Split a long document into chapters</h2>
              <button
                className="icon-button"
                aria-label="Close dialog"
                onClick={close}
              >
                <X size={20} />
              </button>
            </div>
            {error && (
              <div className="alert error" role="alert">
                <span>{error}</span>
              </div>
            )}
            <div className="split-layout">
              <div className="split-options">
                {candidates.length > 1 && (
                  <label className="field-label">
                    Document
                    <select
                      aria-label="Document to split"
                      value={documentId}
                      onChange={(e) => setChosen(e.target.value)}
                    >
                      {candidates.map((doc) => (
                        <option key={doc.id} value={doc.id}>
                          {doc.name} ({(doc.words ?? 0).toLocaleString()} words)
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                <label className="inline-check">
                  <input
                    type="checkbox"
                    checked={!!options.transcript}
                    onChange={(e) =>
                      change({ transcript: e.target.checked, speakers: null })
                    }
                  />
                  This is a transcript with speakers
                </label>
                {options.transcript && (
                  <label className="inline-check">
                    <input
                      type="checkbox"
                      checked={!!options.per_turn}
                      onChange={(e) => change({ per_turn: e.target.checked })}
                    />
                    One document per turn (not per speaker)
                  </label>
                )}
                {!options.transcript && (
                  <label className="field-label">
                    Cut at
                    <select
                      aria-label="Cut at"
                      value={options.rule}
                      onChange={(e) =>
                        change({ rule: e.target.value as SplitOptions["rule"] })
                      }
                    >
                      {Object.entries(RULE_WORDS).map(([rule, words]) => (
                        <option key={rule} value={rule}>
                          {words}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                {!options.transcript && options.rule === "pattern" && (
                  <>
                    <label className="field-label">
                      A heading line starts with
                      <input
                        aria-label="A heading line starts with"
                        defaultValue={options.pattern}
                        placeholder="Letter, Canto, Stave…"
                        maxLength={200}
                        onBlur={(e) => change({ pattern: e.target.value })}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") e.currentTarget.blur();
                        }}
                      />
                    </label>
                    <label className="inline-check">
                      <input
                        type="checkbox"
                        checked={options.regex}
                        onChange={(e) => change({ regex: e.target.checked })}
                      />
                      Use a regular expression
                    </label>
                  </>
                )}
                {!options.transcript && options.rule === "words" && (
                  <label className="field-label">
                    Words in each block
                    <input
                      type="number"
                      min={100}
                      max={100000}
                      step={100}
                      defaultValue={options.block_words}
                      onBlur={(e) =>
                        change({ block_words: Number(e.target.value) || 2000 })
                      }
                    />
                  </label>
                )}
                {(hasLicence || options.keep_front_matter) && (
                  <label className="inline-check">
                    <input
                      type="checkbox"
                      checked={!options.keep_front_matter}
                      onChange={(e) =>
                        change({ keep_front_matter: !e.target.checked })
                      }
                    />
                    Leave out the Project Gutenberg header and footer
                  </label>
                )}
                <label className="inline-check">
                  <input
                    type="checkbox"
                    checked={options.keep_whole}
                    onChange={(e) => change({ keep_whole: e.target.checked })}
                  />
                  Keep the whole book as a document too
                </label>
                <small className="muted">
                  {options.keep_whole
                    ? "Runs that read both count every word twice, and say so."
                    : "The whole book goes to Trash; restore it from there at any time."}
                </small>
                {preview && preview.left_out.length > 0 && (
                  <div className="split-left-out">
                    <strong>Left out</strong>
                    <ul>
                      {preview.left_out.map((item) => (
                        <li key={item.what}>
                          {item.what}: {item.size.toLocaleString()}{" "}
                          {item.what.includes("lines") ? "" : "words"}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
              <div className="split-sections">
                {busy && !shown && <p className="muted">Reading…</p>}
                {options.transcript && turns && (
                  <>
                    <p className="muted">
                      {turns.speakers.length} speakers
                      {turns.directions.length > 0 &&
                        ` · left out: ${turns.directions
                          .map((d) => `${d.text} ×${d.count}`)
                          .join(", ")}`}
                      {turns.preamble_words > 0 &&
                        ` · ${turns.preamble_words} words before the first speaker`}
                    </p>
                    <ul className="split-list" aria-label="Speakers found">
                      {turns.speakers.map((speaker) => (
                        <li key={speaker.name}>
                          <label className="inline-check split-title">
                            <span>
                              <input
                                type="checkbox"
                                checked={speaker.kept}
                                aria-label={`Keep ${speaker.name}`}
                                onChange={(e) =>
                                  keepSpeaker(speaker.name, e.target.checked)
                                }
                              />{" "}
                              <strong>{speaker.name}</strong>
                            </span>
                            <span className="muted">
                              {speaker.turns} turn
                              {speaker.turns === 1 ? "" : "s"} ·{" "}
                              {speaker.words.toLocaleString()} words
                            </span>
                          </label>
                        </li>
                      ))}
                    </ul>
                  </>
                )}
                {!options.transcript &&
                  preview &&
                  preview.diagnostics
                    .filter((d) => d.order === undefined)
                    .map((d) => (
                      <p key={d.code} className="alert warning">
                        {d.message}
                      </p>
                    ))}
                {!options.transcript && preview && (
                  <ol className="split-list" aria-label="Sections found">
                    {preview.sections.map((section) => (
                      <li key={section.order}>
                        <div className="split-title">
                          <strong>{section.title}</strong>
                          <span className="muted">
                            {section.parents
                              .map(([level, value]) => `${level} ${value}`)
                              .join(" · ")}
                            {section.parents.length ? " · " : ""}
                            {section.words.toLocaleString()} words
                          </span>
                        </div>
                        <p className="split-opening">{section.opening}</p>
                        {noteFor(section.order).map((d) => (
                          <small key={d.code} className="split-note">
                            {d.message}
                          </small>
                        ))}
                      </li>
                    ))}
                  </ol>
                )}
              </div>
            </div>
            <div className="dialog-actions">
              <button
                type="button"
                className="secondary"
                disabled={busy}
                onClick={close}
              >
                Cancel
              </button>
              <button
                type="button"
                className="primary"
                disabled={busy || !ready}
                onClick={() => void split()}
              >
                {options.transcript && turns
                  ? turnButtonWords(turns, !!options.per_turn)
                  : preview
                    ? splitButtonWords(preview)
                    : "Split"}
              </button>
            </div>
          </>
        )}
      </dialog>
    </>
  );
}
