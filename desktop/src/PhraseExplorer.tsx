import { useRef, useState, type FormEvent } from "react";
import {
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Copy,
  FolderOpen,
  RotateCcw,
  Save,
  Search,
  Trash2,
  TriangleAlert,
  Upload,
  X,
} from "lucide-react";
import {
  barWidth,
  comparable,
  differsFromSaved,
  isDraftDifferent,
  percent,
  phraseRate,
  rateComparison,
  refined,
  type PhraseAnswer,
  type PhraseDocumentTrack,
  type PhraseRequest,
  type PhraseTrackPosition,
  type PhraseWorkspaceAnswer,
  type SavedPhraseQuestion,
  type SourcePassage,
} from "./phrase";

const PAGE_SIZE = 50;

export function PhraseExplorer({
  ready,
  onTrack,
  onReadSource,
  savedQuestions,
  onSaveQuestion,
  onOpenQuestion,
  onDuplicateQuestion,
  onDeleteQuestion,
  onPublishQuestion,
  canPublish,
}: {
  /** Whether a corpus is loaded. The shelf works without one; tracking
   *  a new phrase does not. */
  ready: boolean;
  onTrack: (request: PhraseRequest) => Promise<PhraseAnswer | null>;
  /** Open the linked source reader at a verified offset of a loaded document. */
  onReadSource: (
    documentId: string,
    characterStart: number,
    characterEnd: number | null,
    radius?: number,
  ) => Promise<SourcePassage | null>;
  savedQuestions: SavedPhraseQuestion[];
  onSaveQuestion: (
    name: string,
    answer: PhraseAnswer,
    request: PhraseRequest,
    /** The comparison subject's own answer, which carries both its text and
     *  the tokens it resolved to. */
    comparison: PhraseAnswer | null,
    /** The saved question this replaces, or null to keep a new one. */
    replacing: string | null,
    /** The revision *replacing* was opened at, so a record somebody else has
     *  already changed is refused rather than silently overwritten. */
    expectedRevision: number | null,
  ) => Promise<SavedPhraseQuestion | null>;
  onOpenQuestion: (
    question: SavedPhraseQuestion,
  ) => Promise<PhraseWorkspaceAnswer | null>;
  onDuplicateQuestion: (
    question: SavedPhraseQuestion,
  ) => Promise<SavedPhraseQuestion | null>;
  onDeleteQuestion: (question: SavedPhraseQuestion) => Promise<boolean>;
  onPublishQuestion: (question: SavedPhraseQuestion) => void;
  canPublish: boolean;
}) {
  const [text, setText] = useState("");
  const [comparisonText, setComparisonText] = useState("");
  const [caseSensitive, setCaseSensitive] = useState(false);
  // The three widenings. Separate pieces of state rather than one object
  // because each is an independent question about the phrase, and because a
  // reopened saved question sets them one at a time from its specification.
  const [normalize, setNormalize] = useState(false);
  const [matchLemma, setMatchLemma] = useState(false);
  const [matchNominalization, setMatchNominalization] = useState(false);
  const options = {
    case_sensitive: caseSensitive,
    normalize,
    match_lemma: matchLemma,
    match_nominalization: matchNominalization,
  };
  const [positionBins, setPositionBins] = useState(10);
  const [answer, setAnswer] = useState<PhraseAnswer | null>(null);
  const [comparisonAnswer, setComparisonAnswer] = useState<PhraseAnswer | null>(
    null,
  );
  const [loading, setLoading] = useState(false);
  const [lastRequest, setLastRequest] = useState<PhraseRequest | null>(null);
  const [lastComparisonText, setLastComparisonText] = useState<string | null>(
    null,
  );
  const [evidenceSide, setEvidenceSide] = useState<"primary" | "comparison">(
    "primary",
  );
  /**
   * The brushed position range: relative document coordinates, or null when
   * the evidence is not brushed down. Selection only -- the pooled position
   * chart keeps showing the whole corpus until the brush is applied as an
   * explicit filter, which is what keeps "looking closer" from quietly
   * redefining every other panel's denominator.
   */
  const [brush, setBrush] = useState<{
    start: number;
    end: number;
  } | null>(null);
  /** The document whose track is expanded for close reading. */
  const [trackDocument, setTrackDocument] = useState<string | null>(null);
  /**
   * The source reader's contents, for the side currently showing evidence.
   *
   * One passage at a time: the reader is where an occurrence is read in its
   * wider context, and two cursors of prose at once would be two passages
   * competing for the same attention. Switching the evidence side clears it
   * rather than pretending one document serves both subjects.
   */
  const [passage, setPassage] = useState<SourcePassage | null>(null);
  const [passageLoading, setPassageLoading] = useState(false);
  const [passageError, setPassageError] = useState("");
  /**
   * How much context the reader shows. Growing it refetches the same offsets
   * with a larger radius -- it never reruns the parser and never remaratches,
   * because the passage endpoint reads the already-parsed snapshot's text.
   */
  const [passageRadius, setPassageRadius] = useState(100);
  const [passageAnchor, setPassageAnchor] = useState<{
    documentId: string;
    start: number;
    end: number | null;
    side: "primary" | "comparison";
  } | null>(null);
  const [saveName, setSaveName] = useState("");
  /**
   * Two different things, kept apart because conflating them let the interface
   * describe a record it had never loaded.
   *
   * *selectedId* is the row highlighted in the shelf -- what Open, Duplicate,
   * Publish and Delete act on. *opened* is the record whose contents are
   * actually on screen, adopted only after an open succeeds, and it is what
   * Update replaces. Choosing B from the list while looking at A's answer used
   * to label A's answer "Open: B" and aim Update at B.
   */
  const [selectedId, setSelectedId] = useState("");
  const [opened, setOpened] = useState<{
    id: string;
    name: string;
    revision: number;
  } | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [comparisonFailed, setComparisonFailed] = useState(false);

  /**
   * Run one question and adopt its answer.
   *
   * *compared* is the second subject as the caller resolved it, never read
   * from the form here: the two callers below differ precisely in whether the
   * form is allowed to have a say.
   */
  const run = async (request: PhraseRequest, compared: string) => {
    if (!request.text.trim() || loading) return;
    setLoading(true);
    setComparisonFailed(false);
    try {
      const [result, comparison] = await Promise.all([
        onTrack(request),
        compared
          ? onTrack({ ...request, text: compared })
          : Promise.resolve(null),
      ]);
      if (!result) return;
      // Two requests, so two chances for the corpus to change underneath. A
      // comparison that came back describing other documents is not a
      // comparison, however real each half is on its own.
      const agreed = comparable(result, comparison);
      setAnswer(result);
      setComparisonAnswer(agreed);
      // A brush belonged to the question that was on screen. A new subject
      // starts unbrushed rather than carrying a range the old phrase chose.
      if (!request.evidence_position_start && !request.evidence_position_end) {
        setBrush(null);
      } else if (result.evidence.filter.position_start !== null) {
        setBrush({
          start: result.evidence.filter.position_start,
          end: result.evidence.filter.position_end ?? 1,
        });
      }
      // Bound to the snapshot that answered, so every later page or drill-down
      // is refused rather than silently served from a reloaded corpus.
      setLastRequest({ ...request, snapshot_id: result.snapshot_id });
      setLastComparisonText(agreed ? compared : null);
      if (!agreed) {
        setEvidenceSide("primary");
        // A comparison that was asked for and did not arrive must not look
        // like a question that never had one.
        if (compared) setComparisonFailed(true);
      }
    } finally {
      setLoading(false);
    }
  };

  /** Submitting the form: the draft becomes the question, paging restarts. */
  const resolve = () =>
    run(
      {
        text: text.trim(),
        ...options,
        position_bins: positionBins,
        evidence_offset: 0,
        evidence_limit: PAGE_SIZE,
        evidence_year: null,
        evidence_document_id: null,
        evidence_position_start: null,
        evidence_position_end: null,
      },
      comparisonText.trim(),
    );

  /**
   * Looking at different evidence for the question already on screen.
   *
   * Reads `lastRequest`, not the form. Typing a new phrase and then clicking
   * Next used to run the new phrase at the old phrase's page offset and put
   * the result under the old phrase's heading.
   */
  const refine = (window: {
    evidence_offset?: number;
    evidence_year?: number | null;
    evidence_document_id?: string | null;
    evidence_position_start?: number | null;
    evidence_position_end?: number | null;
  }) => {
    if (!lastRequest) return;
    void run(refined(lastRequest, window), lastComparisonText ?? "");
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void resolve();
  };
  // What the form says, against what the answer on screen actually answers.
  const draftDiffers = isDraftDifferent(
    { text, ...options, position_bins: positionBins },
    comparisonText,
    lastRequest,
    lastComparisonText,
  );
  const activeYear = answer?.evidence.filter.year ?? null;
  const activeDocument = answer?.evidence.filter.document_id ?? null;
  const activeBrush =
    answer?.evidence.filter.position_start !== null
      ? {
          start: answer?.evidence.filter.position_start ?? 0,
          end: answer?.evidence.filter.position_end ?? 1,
        }
      : null;
  const timeMax = Math.max(
    0,
    ...(answer?.time.map((row) => row.occurrences_per_10000 ?? 0) ?? []),
    ...(comparisonAnswer?.time.map((row) => row.occurrences_per_10000 ?? 0) ??
      []),
  );
  const positionMax = Math.max(
    0,
    ...(answer?.position.map((row) => row.occurrences_per_10000 ?? 0) ?? []),
    ...(comparisonAnswer?.position.map(
      (row) => row.occurrences_per_10000 ?? 0,
    ) ?? []),
  );
  /** The shelf row under the cursor: what Open, Duplicate, Publish and Delete
   *  act on. */
  const saved =
    savedQuestions.find((question) => question.id === selectedId) ?? null;
  /** The record actually on screen, as the shelf currently reports it. */
  const openRecord =
    savedQuestions.find((question) => question.id === opened?.id) ?? null;
  // Against the saved record, not against the form: draftDiffers goes quiet
  // the moment a changed question is tracked, which is exactly when the open
  // record and the answer have come apart.
  const unsavedChanges = differsFromSaved(
    openRecord,
    lastRequest,
    lastComparisonText,
  );
  const comparisonTimes = new Map(
    comparisonAnswer?.time.map((row) => [row.year, row]) ?? [],
  );
  const comparisonPositions = new Map(
    comparisonAnswer?.position.map((row) => [row.bin, row]) ?? [],
  );
  const comparisonDocuments = new Map(
    comparisonAnswer?.documents.map((row) => [row.document_id, row]) ?? [],
  );

  /**
   * Brushing selects a position range; applying it filters the evidence.
   *
   * Two actions, because "inspect these occurrences" and "make this my
   * evidence" are different intents. Clicking one bin previews its range;
   * the filter is applied immediately for a single bin (the common case of
   * "show me what is in here") but a range built by extending across bins
   * is applied only when the reader asks, keeping the pooled chart global
   * until then.
   */
  const applyBrush = (start: number, end: number) => {
    setBrush({ start, end });
    refine({
      evidence_offset: 0,
      evidence_position_start: start,
      evidence_position_end: end,
    });
  };

  const clearBrush = () => {
    setBrush(null);
    refine({
      evidence_offset: 0,
      evidence_position_start: null,
      evidence_position_end: null,
    });
  };

  /** Whether a bin's range overlaps the brush drawn on the chart. */
  const isBrushed = (start: number, end: number) =>
    brush !== null && start < brush.end && end > brush.start;

  const openSaved = async () => {
    if (!saved || loading) return;
    setLoading(true);
    setConfirmDelete(false);
    const spec = saved.specification;
    const request: PhraseRequest = {
      text: spec.text,
      case_sensitive: spec.case_sensitive,
      // Defaulted, because a question saved before these existed records
      // none of them, and the question it was saved as is the literal one.
      normalize: spec.normalize ?? false,
      match_lemma: spec.match_lemma ?? false,
      match_nominalization: spec.match_nominalization ?? false,
      position_bins: spec.position_bins,
      evidence_offset: 0,
      evidence_limit: PAGE_SIZE,
      evidence_year: spec.evidence_year,
      evidence_document_id: spec.evidence_document_id,
      evidence_position_start: null,
      evidence_position_end: null,
    };
    try {
      const result = await onOpenQuestion(saved);
      if (result) {
        const agreed = comparable(result.primary, result.comparison);
        setText(spec.text);
        setComparisonText(spec.comparison_text ?? "");
        setCaseSensitive(spec.case_sensitive);
        setNormalize(spec.normalize ?? false);
        setMatchLemma(spec.match_lemma ?? false);
        setMatchNominalization(spec.match_nominalization ?? false);
        setPositionBins(spec.position_bins);
        setAnswer(result.primary);
        setComparisonAnswer(agreed);
        // The reopened question's saved brush, if it had one, is what the
        // position chart shows as selected.
        setBrush(
          result.primary.evidence.filter.position_start !== null
            ? {
                start: result.primary.evidence.filter.position_start,
                end: result.primary.evidence.filter.position_end ?? 1,
              }
            : null,
        );
        // Bound to the snapshot that answered, exactly as a tracked question
        // is. Without this the reopened answer's first page or drill-down
        // named no snapshot and could be served from a corpus loaded since.
        setLastRequest({ ...request, snapshot_id: result.primary.snapshot_id });
        setLastComparisonText(agreed ? spec.comparison_text : null);
        setComparisonFailed(Boolean(spec.comparison_text) && !agreed);
        setEvidenceSide("primary");
        // Adopted only now: an open that failed leaves the previous answer on
        // screen, and it would still be that answer that Update replaced.
        setOpened({
          id: saved.id,
          name: saved.name,
          revision: saved.revision,
        });
        setSaveName(saved.name);
      }
    } finally {
      setLoading(false);
    }
  };

  /**
   * Keep this question, as a revision of the open one or as a new record.
   *
   * *replacing* is what makes reopening a saved question, changing it and
   * keeping it a single action. Saving always created a record before, so
   * that ordinary path ended at "a question with that name already exists".
   */
  const keep = async (replacing: { id: string; revision: number } | null) => {
    if (!answer || !lastRequest || !saveName.trim() || loading) return;
    setLoading(true);
    try {
      const stored = await onSaveQuestion(
        saveName.trim(),
        answer,
        lastRequest,
        comparisonAnswer,
        replacing?.id ?? null,
        replacing?.revision ?? null,
      );
      if (stored) {
        setSelectedId(stored.id);
        setOpened({
          id: stored.id,
          name: stored.name,
          revision: stored.revision,
        });
        setSaveName(stored.name);
      }
    } finally {
      setLoading(false);
    }
  };

  const saveCurrent = (event: FormEvent) => {
    event.preventDefault();
    // Enter keeps doing the safe thing: a new record, never an overwrite.
    void keep(null);
  };

  const duplicateSaved = async () => {
    if (!saved || loading) return;
    setLoading(true);
    try {
      const copy = await onDuplicateQuestion(saved);
      // Selected, not opened: nothing of the copy has been loaded, the answer
      // on screen is still whatever was there before, and Update still aims at
      // whatever record that came from. Open the copy to work on it.
      if (copy) setSelectedId(copy.id);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Open the linked source reader at one occurrence.
   *
   * Fetches from the snapshot the evidence names, so the passage is the text
   * the offsets were verified against. Widening the context refetches the
   * same anchor with a larger radius rather than re-asking the question. A
   * superseded fetch (the side switched, another occurrence opened) is
   * dropped rather than overwriting the reader with a passage nobody asked
   * for last.
   */
  const readerRequest = useRef(0);
  const openReader = async (
    documentId: string,
    start: number,
    end: number | null,
    side: "primary" | "comparison" = evidenceSide,
    radius = passageRadius,
  ) => {
    const mine = ++readerRequest.current;
    setPassageAnchor({ documentId, start, end, side });
    setPassage(null);
    setPassageLoading(true);
    setPassageError("");
    try {
      const read = await onReadSource(documentId, start, end, radius);
      if (mine !== readerRequest.current) return;
      if (read) {
        setPassage(read);
      } else {
        setPassageError(
          "That passage could not be opened. The documents may have been reloaded.",
        );
      }
    } finally {
      if (mine === readerRequest.current) setPassageLoading(false);
    }
  };

  /** Wider context, same place in the text. */
  const widenReader = (radius: number) => {
    if (!passageAnchor) return;
    setPassageRadius(radius);
    void openReader(
      passageAnchor.documentId,
      passageAnchor.start,
      passageAnchor.end,
      passageAnchor.side,
      radius,
    );
  };

  const closeReader = () => {
    readerRequest.current += 1;
    setPassage(null);
    setPassageAnchor(null);
    setPassageError("");
  };

  return (
    <section
      className="panel phrase-explorer"
      aria-labelledby="track-phrase-heading"
    >
      <div className="phrase-heading">
        <div>
          <h2 id="track-phrase-heading">Track a phrase</h2>
          <p className="muted">
            Ask where a phrase appears, how its rate changes over time, and
            where it falls inside each text.
          </p>
        </div>
      </div>

      <form className="phrase-query" onSubmit={submit}>
        <label className="field-label phrase-input">
          Phrase
          <input
            value={text}
            maxLength={300}
            placeholder="for example: public health"
            onChange={(event) => setText(event.target.value)}
          />
        </label>
        <label className="field-label phrase-input phrase-compare-input">
          Compare with <span className="optional">optional</span>
          <input
            value={comparisonText}
            maxLength={300}
            placeholder="for example: private health"
            onChange={(event) => setComparisonText(event.target.value)}
          />
        </label>
        <label className="field-label phrase-bins">
          Position detail
          <select
            value={positionBins}
            onChange={(event) => setPositionBins(Number(event.target.value))}
          >
            <option value={5}>5 sections</option>
            <option value={10}>10 sections</option>
            <option value={20}>20 sections</option>
          </select>
        </label>
        <fieldset className="phrase-matching">
          <legend>What counts as this phrase</legend>
          <label className="check-field phrase-case">
            <input
              type="checkbox"
              checked={caseSensitive}
              onChange={(event) => setCaseSensitive(event.target.checked)}
            />
            <span>
              Match capitalization exactly
              <small>Off, “People” and “people” are the same word.</small>
            </span>
          </label>
          <label className="check-field">
            <input
              type="checkbox"
              checked={normalize}
              onChange={(event) => setNormalize(event.target.checked)}
            />
            <span>
              Ignore typography
              <small>
                Curly and straight quotes, the several dashes and accents are
                folded together, so a phrase pasted from a word processor finds
                the same words as one typed here.
              </small>
            </span>
          </label>
          <label className="check-field">
            <input
              type="checkbox"
              checked={matchLemma}
              onChange={(event) => setMatchLemma(event.target.checked)}
            />
            <span>
              Count every inflection
              <small>
                “promise” also counts promised, promises and promising — the
                forms this corpus’s own parse gives the same dictionary word.
              </small>
            </span>
          </label>
          <label className="check-field">
            <input
              type="checkbox"
              checked={matchNominalization}
              onChange={(event) => setMatchNominalization(event.target.checked)}
            />
            <span>
              Count nominalizations
              <small>
                “govern” also counts government and governance, and the other
                way round. Needs WordNet; if it is missing, the answer says so
                rather than quietly counting nothing extra.
              </small>
            </span>
          </label>
        </fieldset>
        <button
          className="primary"
          type="submit"
          disabled={!text.trim() || loading || !ready}
          title={
            ready ? undefined : "Load documents before tracking a new phrase"
          }
        >
          <Search size={15} /> {loading ? "Tracing…" : "Track phrase"}
        </button>
      </form>
      {!ready && (
        <p className="muted">
          Load documents to track a new phrase. A saved question below can be
          opened without loading anything first — it brings its own documents.
        </p>
      )}

      <div className="question-shelf">
        <label className="field-label">
          Saved questions
          <select
            value={selectedId}
            onChange={(event) => {
              setSelectedId(event.target.value);
              setConfirmDelete(false);
            }}
          >
            <option value="">
              {savedQuestions.length
                ? "Choose a saved question…"
                : "No saved questions yet"}
            </option>
            {savedQuestions.map((question) => (
              <option key={question.id} value={question.id}>
                {question.name} · revision {question.revision}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="secondary"
          disabled={!saved || loading}
          onClick={() => void openSaved()}
        >
          <FolderOpen size={14} /> Open
        </button>
        <button
          type="button"
          className="secondary"
          disabled={!saved || loading}
          onClick={() => void duplicateSaved()}
          title="Keep a copy to change without touching the original"
        >
          <Copy size={14} /> Duplicate
        </button>
        <button
          type="button"
          className="secondary"
          disabled={!saved || loading || !canPublish}
          onClick={() => saved && onPublishQuestion(saved)}
          title="Create an immutable run with tables, evidence, inputs and parser provenance"
        >
          <Upload size={14} /> Publish saved
        </button>
        {saved &&
          (confirmDelete ? (
            <>
              <button
                type="button"
                className="secondary danger"
                disabled={loading}
                onClick={() => {
                  void onDeleteQuestion(saved).then((deleted) => {
                    if (deleted) {
                      setSelectedId("");
                      // The answer stays on screen; what is gone is the record
                      // it came from, so there is nothing left to update.
                      if (opened?.id === saved.id) setOpened(null);
                      setConfirmDelete(false);
                    }
                  });
                }}
              >
                <Trash2 size={14} /> Really delete
              </button>
              <button
                type="button"
                className="text-button"
                onClick={() => setConfirmDelete(false)}
              >
                Keep it
              </button>
            </>
          ) : (
            <button
              type="button"
              className="secondary danger"
              onClick={() => setConfirmDelete(true)}
            >
              <Trash2 size={14} /> Delete
            </button>
          ))}
      </div>

      {answer && (
        <>
          {comparisonFailed && (
            <p className="alert warning" role="status">
              <span>
                The comparison phrase could not be answered from the same
                documents, so everything below describes “
                {answer.question.subject.text}” alone.
              </span>
            </p>
          )}
          {draftDiffers && (
            <p className="alert warning" role="status">
              <span>
                The form has been changed. Everything below still answers “
                {answer.question.subject.text}” — track again to ask the new
                question.
              </span>
            </p>
          )}
          <p className="phrase-interpretation">
            Interpreted as{" "}
            {answer.question.subject.tokens
              .map((token) => `“${token}”`)
              .join(" + ")}
            , contiguous within one sentence
            {comparisonAnswer
              ? `; compared with ${comparisonAnswer.question.subject.tokens.map((token) => `“${token}”`).join(" + ")}`
              : ""}
            . Punctuation is preserved and overlapping matches count.
          </p>
          <CountedForms answer={answer} />
          {comparisonAnswer && <CountedForms answer={comparisonAnswer} />}

          <SummaryStrip answer={answer} label={answer.question.subject.text} />
          {comparisonAnswer && (
            <SummaryStrip
              answer={comparisonAnswer}
              label={comparisonAnswer.question.subject.text}
            />
          )}
          {comparisonAnswer && (
            <p className="comparison-reading">
              {rateComparison(
                answer.summary.occurrences_per_10000,
                comparisonAnswer.summary.occurrences_per_10000,
              )}
            </p>
          )}

          <form className="question-save" onSubmit={saveCurrent}>
            <label className="field-label">
              Keep this question
              <input
                value={saveName}
                maxLength={120}
                placeholder="for example: Public health over time"
                onChange={(event) => setSaveName(event.target.value)}
              />
            </label>
            {opened && (
              <button
                className="secondary"
                type="button"
                disabled={!lastRequest || !saveName.trim() || loading}
                onClick={() => void keep(opened)}
                title={`Keep this as the next revision of “${opened.name}”`}
              >
                <Save size={14} /> Update “{opened.name}”
              </button>
            )}
            <button
              className="secondary"
              type="submit"
              disabled={!saveName.trim() || !lastRequest || loading}
            >
              <Save size={14} /> {opened ? "Save as new" : "Save question"}
            </button>
          </form>
          {opened && (
            <p className="muted question-revision">
              Open: “{opened.name}”, revision{" "}
              {openRecord?.revision ?? opened.revision}.{" "}
              {unsavedChanges
                ? "This answer differs from what was saved under that name."
                : "Updating keeps that name and adds a revision; saving as new leaves it untouched."}
            </p>
          )}
          {saved && opened && saved.id !== opened.id && (
            <p className="muted question-revision">
              “{saved.name}” is selected in the list but not open. Open it to
              work on it; Update still replaces “{opened.name}”.
            </p>
          )}

          <div className="phrase-grid">
            <section
              className="phrase-panel"
              aria-labelledby="phrase-time-heading"
            >
              <h3 id="phrase-time-heading">Over time</h3>
              {answer.time.length ? (
                <div className="research-bars">
                  {answer.time.map((row) => {
                    const compared = comparisonTimes.get(row.year);
                    return (
                      <button
                        key={row.year}
                        type="button"
                        className={`research-bar ${activeYear === row.year ? "selected" : ""} ${!row.observed ? "gap" : ""}`}
                        disabled={!row.observed}
                        title={
                          row.observed
                            ? `${row.occurrences} occurrences in ${row.eligible_documents} documents`
                            : "No dated text in this year"
                        }
                        onClick={() =>
                          refine({
                            evidence_offset: 0,
                            evidence_year:
                              activeYear === row.year ? null : row.year,
                          })
                        }
                      >
                        <span className="research-bar-label">{row.year}</span>
                        <span
                          className={
                            compared
                              ? "research-bar-tracks"
                              : "research-bar-track"
                          }
                        >
                          <span
                            className="primary-series"
                            style={{
                              width: barWidth(
                                row.occurrences_per_10000,
                                timeMax,
                              ),
                            }}
                          />
                          {compared && (
                            <span
                              className="comparison-series"
                              style={{
                                width: barWidth(
                                  compared.occurrences_per_10000,
                                  timeMax,
                                ),
                              }}
                            />
                          )}
                        </span>
                        <span>
                          {row.observed
                            ? phraseRate(row.occurrences_per_10000)
                            : "no text"}
                          {compared
                            ? ` / ${phraseRate(compared.occurrences_per_10000)}`
                            : ""}
                        </span>
                      </button>
                    );
                  })}
                </div>
              ) : (
                <p className="muted">
                  No selected document has a date in its filename.
                </p>
              )}
              <small>
                Rate per 10,000 word tokens. Empty years are gaps; observed
                zeroes remain zero.
              </small>
            </section>

            <section
              className="phrase-panel"
              aria-labelledby="phrase-position-heading"
            >
              <h3 id="phrase-position-heading">Within texts</h3>
              {activeBrush && (
                <p className="brush-status" role="status">
                  Evidence is filtered to positions{" "}
                  {positionLabel(activeBrush.start)}–
                  {positionLabel(activeBrush.end)} of each document.
                </p>
              )}
              <div className="research-bars">
                {answer.position.map((row) => {
                  const compared = comparisonPositions.get(row.bin);
                  const binStart = row.start_percent / 100;
                  const binEnd = row.end_percent / 100;
                  return (
                    <button
                      type="button"
                      className={`research-bar ${isBrushed(binStart, binEnd) ? "brushed" : ""}`}
                      key={row.bin}
                      title={
                        `${row.occurrences} occurrences in ${row.contributing_documents} documents` +
                        (isBrushed(binStart, binEnd)
                          ? " — in the brushed selection"
                          : " — brush or click to inspect")
                      }
                      onClick={() => applyBrush(binStart, binEnd)}
                    >
                      <span className="research-bar-label">
                        {row.start_percent.toFixed(0)}–
                        {row.end_percent.toFixed(0)}%
                      </span>
                      <span
                        className={
                          compared
                            ? "research-bar-tracks"
                            : "research-bar-track"
                        }
                      >
                        <span
                          className="primary-series"
                          style={{
                            width: barWidth(
                              row.occurrences_per_10000,
                              positionMax,
                            ),
                          }}
                        />
                        {compared && (
                          <span
                            className="comparison-series"
                            style={{
                              width: barWidth(
                                compared.occurrences_per_10000,
                                positionMax,
                              ),
                            }}
                          />
                        )}
                      </span>
                      <span>
                        {phraseRate(row.occurrences_per_10000)}
                        {compared
                          ? ` / ${phraseRate(compared.occurrences_per_10000)}`
                          : ""}
                      </span>
                    </button>
                  );
                })}
              </div>
              {activeBrush && (
                <button
                  type="button"
                  className="secondary clear-brush"
                  onClick={() => clearBrush()}
                >
                  <RotateCcw size={13} /> Clear position filter
                </button>
              )}
              <small>
                Each occurrence is positioned inside its own document, then
                pooled by word exposure. A position filter selects occurrences
                by where each one starts.
              </small>
            </section>
          </div>

          <section
            className="phrase-panel phrase-tracks"
            aria-labelledby="phrase-tracks-heading"
          >
            <h3 id="phrase-tracks-heading">Within each document</h3>
            <p className="muted">
              One row per document: where each occurrence falls along that
              document's own length. Click a position to read that passage.
            </p>
            <div className="track-list">
              {answer.tracks.map((track) => (
                <DocumentTrack
                  key={track.document_id}
                  track={track}
                  comparison={
                    comparisonAnswer?.tracks.find(
                      (row) => row.document_id === track.document_id,
                    ) ?? null
                  }
                  expanded={trackDocument === track.document_id}
                  onToggle={() =>
                    setTrackDocument(
                      trackDocument === track.document_id
                        ? null
                        : track.document_id,
                    )
                  }
                  onOpenOccurrence={(position) =>
                    position.character_start !== null
                      ? openReader(
                          track.document_id,
                          position.character_start,
                          position.character_end,
                        )
                      : undefined
                  }
                />
              ))}
            </div>
            <small>
              {answer.tracks.filter((track) => track.occurrences === 0).length}{" "}
              of {answer.tracks.length} documents contain none of the tokens;
              their empty tracks keep the denominator visible.
            </small>
          </section>

          <section
            className="phrase-panel phrase-documents"
            aria-labelledby="phrase-documents-heading"
          >
            <h3 id="phrase-documents-heading">Across documents</h3>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Document</th>
                    <th>Date</th>
                    <th>Occurrences · {answer.question.subject.text}</th>
                    <th>Rate · {answer.question.subject.text}</th>
                    {comparisonAnswer && (
                      <th>
                        Occurrences · {comparisonAnswer.question.subject.text}
                      </th>
                    )}
                    {comparisonAnswer && (
                      <th>Rate · {comparisonAnswer.question.subject.text}</th>
                    )}
                    <th>Words</th>
                  </tr>
                </thead>
                <tbody>
                  {answer.documents.map((row) => {
                    const compared = comparisonDocuments.get(row.document_id);
                    return (
                      <tr
                        key={row.document_id}
                        className={
                          activeDocument === row.document_id
                            ? "selected-row"
                            : ""
                        }
                      >
                        <td>
                          <button
                            className="table-link"
                            type="button"
                            onClick={() =>
                              refine({
                                evidence_offset: 0,
                                evidence_document_id:
                                  activeDocument === row.document_id
                                    ? null
                                    : row.document_id,
                              })
                            }
                          >
                            {row.document}
                          </button>
                        </td>
                        <td>{row.date ?? "Undated"}</td>
                        <td>{row.occurrences.toLocaleString()}</td>
                        <td>{phraseRate(row.occurrences_per_10000)}</td>
                        {comparisonAnswer && (
                          <td>
                            {compared?.occurrences.toLocaleString() ?? "0"}
                          </td>
                        )}
                        {comparisonAnswer && (
                          <td>
                            {phraseRate(
                              compared?.occurrences_per_10000 ?? null,
                            )}
                          </td>
                        )}
                        <td>{row.word_tokens.toLocaleString()}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <small>
              Documents with no match remain visible because they belong in the
              denominator.
            </small>
          </section>

          {comparisonAnswer && (
            <div
              className="evidence-subjects"
              role="group"
              aria-label="Phrase evidence to inspect"
            >
              <span className="evidence-subjects-label">Passages for:</span>
              <button
                type="button"
                className={evidenceSide === "primary" ? "selected" : ""}
                onClick={() => {
                  setEvidenceSide("primary");
                  // Each side follows its own cursor; the reader follows the
                  // side being read, so switching sides closes it rather than
                  // leaving prose from the other subject on screen.
                  closeReader();
                }}
              >
                {answer.question.subject.text}
              </button>
              <button
                type="button"
                className={evidenceSide === "comparison" ? "selected" : ""}
                onClick={() => {
                  setEvidenceSide("comparison");
                  closeReader();
                }}
              >
                {comparisonAnswer.question.subject.text}
              </button>
            </div>
          )}
          <Evidence
            answer={
              evidenceSide === "comparison" && comparisonAnswer
                ? comparisonAnswer
                : answer
            }
            loading={loading}
            onPage={(offset) => refine({ evidence_offset: offset })}
            onClear={() => {
              setBrush(null);
              refine({
                evidence_offset: 0,
                evidence_year: null,
                evidence_document_id: null,
                evidence_position_start: null,
                evidence_position_end: null,
              });
            }}
          />
          <SourceReader
            passage={passage}
            anchor={passageAnchor}
            loading={passageLoading}
            error={passageError}
            radius={passageRadius}
            onWiden={widenReader}
            onClose={closeReader}
          />
        </>
      )}
    </section>
  );
}

function Summary({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function SummaryStrip({
  answer,
  label,
}: {
  answer: PhraseAnswer;
  label: string;
}) {
  return (
    <div className="summary-with-subject">
      <strong className="subject-label">{label}</strong>
      <div className="phrase-summary" aria-label={`Summary for ${label}`}>
        <Summary
          label="Occurrences"
          value={answer.summary.occurrences.toLocaleString()}
        />
        <Summary
          label="Per 10,000 words"
          value={phraseRate(answer.summary.occurrences_per_10000)}
        />
        <Summary
          label="Documents using it"
          value={`${answer.summary.matching_documents} of ${answer.summary.eligible_documents}`}
        />
        <Summary
          label="Document coverage"
          value={percent(answer.summary.document_prevalence_percent)}
        />
      </div>
    </div>
  );
}

function Evidence({
  answer,
  loading,
  onPage,
  onClear,
}: {
  answer: PhraseAnswer;
  loading: boolean;
  onPage: (offset: number) => void;
  onClear: () => void;
}) {
  const evidence = answer.evidence;
  const filtered =
    evidence.filter.year !== null || evidence.filter.document_id !== null;
  const first = evidence.filtered_total ? evidence.offset + 1 : 0;
  const last = Math.min(
    evidence.offset + evidence.rows.length,
    evidence.filtered_total,
  );
  return (
    <section
      className="phrase-panel phrase-evidence"
      aria-labelledby="phrase-evidence-heading"
    >
      <div className="evidence-heading">
        <div>
          <h3 id="phrase-evidence-heading">In context</h3>
          <p className="muted">
            Showing {first.toLocaleString()}–{last.toLocaleString()} of{" "}
            {evidence.filtered_total.toLocaleString()}
            {filtered
              ? ` filtered occurrences (${evidence.total.toLocaleString()} overall)`
              : " occurrences"}
            .
          </p>
        </div>
        {filtered && (
          <button
            type="button"
            className="secondary"
            onClick={onClear}
            disabled={loading}
          >
            <X size={14} /> Clear evidence filter
          </button>
        )}
      </div>
      {evidence.rows.length ? (
        <ol className="concordance" start={evidence.offset + 1}>
          {evidence.rows.map((row) => (
            <li key={row.id}>
              <div className="concordance-source">
                <strong>{row.document}</strong>
                <span>
                  {row.date ?? "Undated"} · sentence {row.sentence_id}
                </span>
              </div>
              <p>
                <span>
                  {row.exact_source_highlight ? row.left_source : row.left}{" "}
                </span>
                <mark>
                  {row.exact_source_highlight ? row.match_source : row.match}
                </mark>
                <span>
                  {" "}
                  {row.exact_source_highlight ? row.right_source : row.right}
                </span>
              </p>
              <small
                className={
                  row.exact_source_highlight
                    ? "source-verified"
                    : "source-reconstructed"
                }
              >
                {row.exact_source_highlight
                  ? `Exact source span ${row.character_start}–${row.character_end}`
                  : "Reconstructed from parsed tokens; exact source span unavailable"}
              </small>
            </li>
          ))}
        </ol>
      ) : (
        <p className="muted">
          No occurrences match the current evidence filter.
        </p>
      )}
      <div className="evidence-pages">
        <button
          type="button"
          className="secondary"
          disabled={loading || evidence.offset === 0}
          onClick={() => onPage(Math.max(0, evidence.offset - evidence.limit))}
        >
          <ChevronLeft size={14} /> Previous
        </button>
        <button
          type="button"
          className="secondary"
          disabled={loading || !evidence.truncated}
          onClick={() => onPage(evidence.offset + evidence.limit)}
        >
          Next <ChevronRight size={14} />
        </button>
      </div>
      <small>
        {/* The filtered figures, because they describe the passages above.
            The corpus-wide status called a fully verified page "partial"
            whenever anything anywhere in the corpus was unaligned. */}
        {evidence.source_offsets.filtered_status === "verified"
          ? "Every highlight in this selection is verified against the imported text. Stored offsets use Unicode code points and include browser UTF-16 coordinates."
          : evidence.source_offsets.filtered_status === "partial"
            ? `${evidence.source_offsets.filtered_verified} of ${evidence.source_offsets.filtered_total} occurrences in this selection have verified source offsets; the rest use reconstructed token context.`
            : "The parser tokens could not be aligned exactly to the imported text, so these passages use reconstructed token context."}
        {evidence.source_offsets.filtered_total <
          evidence.source_offsets.total &&
          ` Across the whole corpus, ${evidence.source_offsets.verified} of ${evidence.source_offsets.total} are verified.`}
      </small>
    </section>
  );
}

/** A relative position in words a reader can act on: "about 40% in". */
export function positionLabel(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/**
 * One document's occurrences along its own length.
 *
 * The row always renders, even with zero occurrences: a gap here is evidence
 * about the denominator, and hiding it would turn "this document does not use
 * the phrase" into "this document does not exist". Each point opens the
 * linked source reader at that occurrence's verified offset.
 */
export function DocumentTrack({
  track,
  comparison,
  expanded,
  onToggle,
  onOpenOccurrence,
}: {
  track: PhraseDocumentTrack;
  comparison: PhraseDocumentTrack | null;
  expanded: boolean;
  onToggle: () => void;
  onOpenOccurrence: (position: PhraseTrackPosition) => void;
}) {
  const label = `${track.document} — ${track.occurrences} occurrence${track.occurrences === 1 ? "" : "s"}`;
  return (
    <div className={`document-track ${expanded ? "expanded" : ""}`}>
      <button
        type="button"
        className="track-toggle"
        onClick={onToggle}
        aria-expanded={expanded}
        title={label}
      >
        <span className="track-name">{track.document}</span>
        <span className="track-meta">
          {track.date ?? "Undated"} · {track.occurrences} ·{" "}
          {track.word_tokens.toLocaleString()} words
        </span>
        <span
          className={comparison ? "research-bar-tracks" : "research-bar-track"}
          aria-hidden="true"
        >
          {track.positions.map((position) => (
            <span
              key={position.occurrence_id}
              className="track-point"
              style={{
                left: `${(position.relative_position ?? 0) * 100}%`,
              }}
            />
          ))}
          {comparison?.positions.map((position) => (
            <span
              key={position.occurrence_id}
              className="track-point comparison-point"
              style={{
                left: `${(position.relative_position ?? 0) * 100}%`,
              }}
            />
          ))}
        </span>
      </button>
      {expanded && (
        <ol className="track-positions">
          {track.positions.length ? (
            track.positions.map((position) => (
              <li key={position.occurrence_id}>
                <button
                  type="button"
                  className="table-link"
                  disabled={position.character_start === null}
                  title={
                    position.character_start === null
                      ? "Exact source span unavailable for this occurrence"
                      : "Open this passage in the source reader"
                  }
                  onClick={() => onOpenOccurrence(position)}
                >
                  <BookOpen size={12} /> sentence {position.sentence_id} ·{" "}
                  {positionLabel(position.relative_position ?? 0)}
                  {position.exact_source_highlight
                    ? ""
                    : " · reconstructed context only"}
                </button>
              </li>
            ))
          ) : (
            <li className="muted">No occurrences in this document.</li>
          )}
          {track.positions_truncated && (
            <li className="muted">
              Showing the first {track.positions.length} positions; this
              document has more.
            </li>
          )}
        </ol>
      )}
    </div>
  );
}

/**
 * The linked source reader: one document's imported text around one verified
 * offset, with the matched span marked.
 *
 * Widening grows the same window; it never re-asks the question, so changing
 * context size cannot rerun the parser. Offsets are the imported text's code
 * points -- for converted PDF or DOCX content they index the extracted text,
 * not the original page layout, and the caption says so rather than implying
 * page coordinates.
 */
/**
 * Which words the count is actually made of.
 *
 * A widened question is only trustworthy if the reader can see what it
 * widened to. "174 occurrences of promise" over documents containing 112 is
 * not a number anybody can check, and "count every inflection" is a promise
 * about behaviour that the interface otherwise gives no way to inspect. So
 * the forms are listed, commonest first, per typed token.
 *
 * Silent when the question was literal: repeating the word back under the
 * heading "counting" would be noise on every answer to pay for the few that
 * need it.
 */
export function CountedForms({ answer }: { answer: PhraseAnswer }) {
  const profile = answer.question.matching_profile;
  const widened =
    profile.normalize || profile.match_lemma || profile.match_nominalization;
  const groups = answer.question.subject.counted_forms ?? [];
  const unavailable = profile.unavailable ?? [];
  if (!widened && !unavailable.length) return null;
  const extra =
    groups.reduce((total, group) => total + group.length, 0) > groups.length;
  return (
    <div className="phrase-counted">
      {unavailable.map((reason) => (
        <p className="alert warning" role="status" key={reason}>
          <TriangleAlert size={16} aria-hidden="true" />
          <span>{reason}</span>
        </p>
      ))}
      {extra ? (
        <p className="phrase-counted-list">
          <span className="phrase-counted-label">
            Counting, in “{answer.question.subject.text}”:
          </span>
          {groups.map((group, index) => (
            <span className="phrase-counted-token" key={`${index}-${group[0]}`}>
              {group.map((form) => (
                <code key={form}>{form}</code>
              ))}
            </span>
          ))}
        </p>
      ) : (
        <p className="phrase-counted-list muted">
          These documents use no other form of “{answer.question.subject.text}”,
          so the widened question found the same words the literal one does.
        </p>
      )}
    </div>
  );
}

export function SourceReader({
  passage,
  anchor,
  loading,
  error,
  radius,
  onWiden,
  onClose,
}: {
  passage: SourcePassage | null;
  anchor: {
    documentId: string;
    start: number;
    end: number | null;
    side: "primary" | "comparison";
  } | null;
  loading: boolean;
  error: string;
  radius: number;
  onWiden: (radius: number) => void;
  onClose: () => void;
}) {
  if (!passage && !error && !loading) return null;
  const anchorStart = anchor?.start ?? 0;
  const anchorEnd = anchor?.end ?? null;
  // API offsets count Unicode code points; JS string.slice counts UTF-16 units.
  const characters = passage ? Array.from(passage.text) : [];
  const before = passage
    ? characters.slice(0, anchorStart - passage.character_start).join("")
    : "";
  const marked =
    passage && anchorEnd !== null
      ? characters
          .slice(
            anchorStart - passage.character_start,
            anchorEnd - passage.character_start,
          )
          .join("")
      : "";
  const after =
    passage && anchorEnd !== null
      ? characters.slice(anchorEnd - passage.character_start).join("")
      : passage
        ? characters.slice(anchorStart - passage.character_start).join("")
        : "";
  return (
    <section
      className="phrase-panel phrase-reader"
      aria-labelledby="phrase-reader-heading"
    >
      <div className="evidence-heading">
        <div>
          <h3 id="phrase-reader-heading">
            <BookOpen size={15} /> Source reader
          </h3>
          {passage && (
            <p className="muted">
              {passage.document} · characters {passage.character_start}–
              {passage.character_end} of the imported text
            </p>
          )}
        </div>
        <button
          type="button"
          className="secondary"
          onClick={onClose}
          title="Close the source reader"
        >
          <X size={14} /> Close
        </button>
      </div>
      {error && (
        <p className="alert warning" role="status">
          <span>{error}</span>
        </p>
      )}
      {loading && <p className="muted">Opening the passage…</p>}
      {passage && (
        <>
          <pre className="reader-passage">
            <span>{before}</span>
            {marked && <mark>{marked}</mark>}
            <span>{after}</span>
          </pre>
          <div className="reader-controls">
            <span>Context</span>
            {[200, 1000, 5000].map((value) => (
              <button
                key={value}
                type="button"
                className={radius === value ? "selected" : ""}
                disabled={loading}
                onClick={() => onWiden(value)}
              >
                ±{value.toLocaleString()}
              </button>
            ))}
          </div>
          <small>
            Offsets index the imported text by Unicode code points (with browser
            UTF-16 coordinates alongside). For converted PDF or DOCX documents
            they do not refer to the original page layout.
          </small>
        </>
      )}
    </section>
  );
}
