export type PhraseRequest = {
  text: string;
  /**
   * Which loaded corpus this question is about.
   *
   * Empty only for a first question, where "whatever is loaded" is what the
   * reader meant. Every later request -- a page, a year, a document, a saved
   * question reopened -- names the snapshot its answer came from, so a reload
   * underneath it is refused instead of answered from different documents
   * under the same heading.
   */
  snapshot_id?: string;
  case_sensitive: boolean;
  /**
   * The three widenings, each off by default and each *adding* forms rather
   * than replacing them. They are part of the question, not of how it was
   * read: ticking one changes the answer, so it belongs beside the phrase.
   */
  normalize: boolean;
  match_lemma: boolean;
  match_nominalization: boolean;
  position_bins: number;
  evidence_offset: number;
  evidence_limit: number;
  evidence_year?: number | null;
  evidence_document_id?: string | null;
  /**
   * The position range a brush selected, in relative document coordinates.
   * Absent when the evidence is not brushed down to a range.
   */
  evidence_position_start?: number | null;
  evidence_position_end?: number | null;
};

/**
 * Whether the form has been edited away from the question on screen.
 *
 * The answer below the form was produced by a particular subject and settings.
 * Once the form says something else, the two disagree, and any action that
 * silently used the form -- turning a page, clicking a year -- would answer a
 * question the reader never asked while leaving the old one's heading in place.
 */
export type MatchingOptions = Pick<
  PhraseRequest,
  "case_sensitive" | "normalize" | "match_lemma" | "match_nominalization"
>;

/** The options two questions must agree on to be the same question. */
export const OPTION_KEYS = [
  "case_sensitive",
  "normalize",
  "match_lemma",
  "match_nominalization",
] as const;

/**
 * Whether two readings of the same phrase count the same words.
 *
 * Listed once, and read from that list everywhere, so that adding a fourth
 * widening cannot be half-done: a new option missing from one of these
 * comparisons is an edit the interface does not notice, which means a draft
 * that silently escapes into a page request or an unsaved change the save
 * button says is saved.
 */
export function sameOptions(
  left: Partial<MatchingOptions>,
  right: Partial<MatchingOptions>,
): boolean {
  return OPTION_KEYS.every((key) => !!left[key] === !!right[key]);
}

export function isDraftDifferent(
  draft: Pick<PhraseRequest, "text" | "position_bins"> &
    Partial<MatchingOptions>,
  comparison: string,
  resolved: PhraseRequest | null,
  resolvedComparison: string | null,
): boolean {
  if (!resolved) return false;
  return (
    draft.text.trim() !== resolved.text ||
    !sameOptions(draft, resolved) ||
    draft.position_bins !== resolved.position_bins ||
    comparison.trim() !== (resolvedComparison ?? "")
  );
}

/**
 * The resolved question again, with only its evidence window moved.
 *
 * Drill-down and paging are questions about the *same* subject. Rebuilding
 * them from the form is what let an unsubmitted phrase escape into a page
 * request and replace the answer it was paging through.
 */
export function refined(
  resolved: PhraseRequest,
  window: {
    evidence_offset?: number;
    evidence_year?: number | null;
    evidence_document_id?: string | null;
    evidence_position_start?: number | null;
    evidence_position_end?: number | null;
  },
): PhraseRequest {
  return { ...resolved, ...window };
}

/**
 * The comparison subject, but only if it describes the same documents.
 *
 * The two subjects are asked as two requests. A corpus reloaded between their
 * answers gives two real answers about different documents, and putting them
 * side by side reads as one comparison. Nothing in either answer would say so,
 * so the mismatch is treated as a comparison that did not arrive.
 */
export function comparable(
  primary: PhraseAnswer,
  comparison: PhraseAnswer | null,
): PhraseAnswer | null {
  if (!comparison) return null;
  return comparison.snapshot_id === primary.snapshot_id ? comparison : null;
}

/**
 * Whether the question on screen has moved away from the saved record.
 *
 * Distinct from {@link isDraftDifferent}, which compares the form with the
 * answer below it and goes quiet the moment the new question is tracked. This
 * compares the tracked question with what the saved record says, which is what
 * "unsaved changes" has to mean once a saved question is open.
 */
export function differsFromSaved(
  saved: SavedPhraseQuestion | null,
  resolved: PhraseRequest | null,
  resolvedComparison: string | null,
): boolean {
  if (!saved || !resolved) return false;
  const spec = saved.specification;
  return (
    spec.text !== resolved.text ||
    !sameOptions(spec, resolved) ||
    spec.position_bins !== resolved.position_bins ||
    (spec.comparison_text ?? "") !== (resolvedComparison ?? "") ||
    (spec.evidence_year ?? null) !== (resolved.evidence_year ?? null) ||
    (spec.evidence_document_id ?? null) !==
      (resolved.evidence_document_id ?? null)
  );
}

export type PhraseSummary = {
  occurrences: number;
  word_tokens: number;
  occurrences_per_10000: number | null;
  eligible_documents: number;
  matching_documents: number;
  document_prevalence_percent: number | null;
  dated_documents: number;
  undated_documents: number;
};

export type PhraseTimeRow = {
  year: number;
  observed: boolean;
  occurrences: number | null;
  word_tokens: number | null;
  eligible_documents: number;
  matching_documents: number;
  occurrences_per_10000: number | null;
  document_prevalence_percent: number | null;
};

export type PhrasePositionRow = {
  bin: number;
  start_percent: number;
  end_percent: number;
  occurrences: number;
  word_tokens: number;
  contributing_documents: number;
  occurrences_per_10000: number | null;
};

export type PhraseDocumentRow = {
  document_id: string;
  document: string;
  content_sha256: string;
  date: string | null;
  year: number | null;
  word_tokens: number;
  occurrences: number;
  occurrences_per_10000: number | null;
  contains_phrase: boolean;
};

export type PhraseOccurrence = {
  id: string;
  document_id: string;
  document: string;
  date: string | null;
  sentence_id: string;
  token_start: number;
  token_end: number;
  character_start: number | null;
  character_end: number | null;
  browser_character_start: number | null;
  browser_character_end: number | null;
  character_offset_unit: "unicode_code_point" | null;
  /** Zero-based word offset of the match's first token inside its document. */
  word_start: number | null;
  relative_position: number | null;
  left: string;
  match: string;
  right: string;
  left_source: string | null;
  match_source: string | null;
  right_source: string | null;
  exact_source_highlight: boolean;
};

/** One occurrence laid out along its own document's length. */
export type PhraseTrackPosition = {
  word_start: number;
  relative_position: number | null;
  position_bin: number;
  sentence_id: string;
  occurrence_id: string;
  character_start: number | null;
  character_end: number | null;
  browser_character_start: number | null;
  browser_character_end: number | null;
  exact_source_highlight: boolean;
};

/** A document's occurrences along that document's own length. */
export type PhraseDocumentTrack = {
  document_id: string;
  document: string;
  content_sha256: string;
  date: string | null;
  year: number | null;
  word_tokens: number;
  occurrences: number;
  positions: PhraseTrackPosition[];
  positions_truncated: boolean;
  position_bins: number;
};

/** How a question's text became the tokens that were matched. */
export type MatchingProfile = {
  version: number;
  tokenizer: string;
  resolved_by_snapshot: boolean;
  source: "saved" | "snapshot" | "approximate";
  normalize: boolean;
  match_lemma: boolean;
  match_nominalization: boolean;
  /**
   * An option that was asked for and could not be applied, in the engine's
   * words. Empty is the claim that every option asked for was used -- which
   * is why it is shown: an option that cannot work and says nothing is
   * indistinguishable from one that is switched off.
   */
  unavailable: string[];
};

export type PhraseAnswer = {
  schema_version: number;
  kind: "phrase_distribution";
  snapshot_id: string;
  question: {
    subject: {
      text: string;
      tokens: string[];
      case_sensitive: boolean;
      boundary: "sentence";
      punctuation: "preserve";
      overlapping_matches: boolean;
      /**
       * Per typed token, every form in this corpus being counted under it,
       * commonest first. A literal question lists the token itself. Shown
       * because a reader who widens the question has a right to see which
       * words the number now includes, rather than only that it went up.
       */
      counted_forms: string[][];
    };
    matching_profile: MatchingProfile;
  };
  summary: PhraseSummary;
  time: PhraseTimeRow[];
  position: PhrasePositionRow[];
  documents: PhraseDocumentRow[];
  /** One row per document, in corpus order, including documents with no match. */
  tracks: PhraseDocumentTrack[];
  evidence: {
    rows: PhraseOccurrence[];
    total: number;
    filtered_total: number;
    offset: number;
    limit: number;
    truncated: boolean;
    filter: {
      year: number | null;
      document_id: string | null;
      position_start: number | null;
      position_end: number | null;
    };
    source_offsets: {
      /** Coverage across every occurrence in the corpus. */
      status: "verified" | "partial" | "unavailable";
      unit: "unicode_code_point";
      browser_unit: "utf16_code_unit";
      verified: number;
      total: number;
      /**
       * Coverage across the occurrences this filter selected -- the ones a
       * reader is looking at. The corpus-wide status alone said "partial" over
       * a filtered set whose every passage is verified, which reads as a
       * warning about the passages on screen.
       */
      filtered_status: "verified" | "partial" | "unavailable";
      filtered_verified: number;
      filtered_total: number;
    };
  };
};

export type PhraseWorkspaceAnswer = {
  primary: PhraseAnswer;
  comparison: PhraseAnswer | null;
};

/** A wider window of one document's imported text around a verified offset. */
export type SourcePassage = {
  document_id: string;
  document: string;
  content_sha256: string;
  character_start: number;
  character_end: number;
  browser_character_start: number;
  browser_character_end: number;
  character_offset_unit: "unicode_code_point";
  text: string;
};

export type SavedPhraseQuestion = {
  id: string;
  name: string;
  snapshot_id: string;
  parser: "spacy" | "stanza";
  document_ids: string[];
  specification: {
    schema_version: 1;
    kind: "phrase_distribution";
    text: string;
    comparison_text: string | null;
    case_sensitive: boolean;
    normalize: boolean;
    match_lemma: boolean;
    match_nominalization: boolean;
    position_bins: number;
    evidence_year: number | null;
    evidence_document_id: string | null;
  };
  /**
   * How this question's text was read, saved beside it. Absent rules --
   * version 0, source "unrecorded" -- mean the record predates this, and
   * reopening or publishing resolves the text again with the loaded parser.
   */
  matching: {
    version: number;
    tokenizer: string;
    source: "saved" | "snapshot" | "approximate" | "unrecorded";
    tokens: string[];
    comparison_tokens: string[];
  };
  revision: number;
  created: string;
  updated: string;
};

export function phraseRate(value: number | null): string {
  return value === null
    ? "No text"
    : value.toLocaleString(undefined, { maximumFractionDigits: 1 });
}

export function percent(value: number | null): string {
  return value === null
    ? "—"
    : `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })}%`;
}

export function barWidth(value: number | null, maximum: number): string {
  if (value === null || maximum <= 0) return "0%";
  return `${Math.max(0, Math.min(100, (value / maximum) * 100))}%`;
}

export function rateComparison(
  primary: number | null,
  comparison: number | null,
): string {
  if (primary === null || comparison === null)
    return "There is not enough eligible text to compare rates.";
  const difference = primary - comparison;
  if (difference === 0)
    return "The two overall rates are equal in this corpus.";
  const direction = difference > 0 ? "higher" : "lower";
  const magnitude = Math.abs(difference).toLocaleString(undefined, {
    maximumFractionDigits: 1,
  });
  if (comparison === 0) {
    return `The first phrase is ${magnitude} occurrences per 10,000 words ${direction}; a ratio is undefined because the comparison rate is zero.`;
  }
  const ratio = (primary / comparison).toLocaleString(undefined, {
    maximumFractionDigits: 2,
  });
  return `The first phrase is ${magnitude} occurrences per 10,000 words ${direction} (${ratio}× the comparison rate).`;
}
