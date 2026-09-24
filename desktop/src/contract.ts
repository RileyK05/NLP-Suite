/** GENERATED FILE — do not edit.
 *
 * Rendered from desktop_backend/schemas.py by scripts/gen_contract_ts.py
 * (communication contract R-C6). Change the models, re-run the script,
 * and let tests/test_contract_parity.py confirm the two agree.
 */

export interface DiagnosticModel {
  severity: 'INFO' | 'WARNING' | 'ERROR';
  /** stable machine code, e.g. TOPIC_BAD_K */
  code: string;
  /** plain words for the reader */
  message: string;
}

export interface RecommendedChart {
  kind: string;
  x: string;
  y: string;
  agg: string;
  top_n?: number | null;
  question: string;
  why: string;
}

export interface FigureOffer {
  panel: string;
  title: string;
  question: string;
  shape: string;
}

export interface ReadingModel {
  headline: string;
  observations: string[];
  cautions: string[];
  recommended_charts: RecommendedChart[];
  figures: FigureOffer[];
  /** why this result is read as a table; empty when a figure leads */
  table_first: string;
}

export interface TableModel {
  columns: string[];
  rows: Record<string, string>[];
  total: number;
  truncated: boolean;
  filtered_total?: number | null;
  /** row offset of this page; null when unpaginated */
  offset?: number | null;
}

export interface AnalyseRequest {
  /** client-generated; echoed back */
  request_id: string;
  tool: string;
  params: Record<string, unknown>;
  snapshot_id: string;
}

export interface AnalyseResponse {
  headline: string;
  observations: string[];
  cautions: string[];
  recommended_charts: RecommendedChart[];
  figures: FigureOffer[];
  /** why this result is read as a table; empty when a figure leads */
  table_first: string;
  /** echo of AnalyseRequest.request_id */
  request_id: string;
  tool: string;
  ok: boolean;
  elapsed_ms: number;
  snapshot_id: string;
  path: string;
  table?: TableModel | null;
  panel?: Record<string, unknown> | null;
  panels: Record<string, unknown>[];
  /** cite this to redraw a figure with other settings */
  answer_id: string;
  diagnostics: DiagnosticModel[];
}

export interface LiveSelection {
  documents: number;
  words: number;
  names: string[];
  ids: string[];
}

export interface LiveStateModel {
  state: 'cold' | 'warming' | 'ready' | 'failed';
  stage: string;
  error: string;
  documents: number;
  tokens: number;
  parsed: boolean;
  cached: boolean;
  source: '' | 'parsed' | 'disk' | 'memory';
  parse_ms: number;
  selection?: LiveSelection | null;
  snapshot_id: string;
  parser: '' | 'spacy' | 'stanza';
}

export interface WarmRequest {
  parser: string;
  selection?: Record<string, unknown> | null;
}
