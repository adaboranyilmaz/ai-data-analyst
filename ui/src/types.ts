export type Status = "answered" | "declined" | "clarify" | "withheld";

export interface Band {
  range: [number, number];
  n: number;
  accuracy: number | null;
  interval: [number, number] | null;
  thin: boolean;
}

export interface Confidence {
  stated: number | null;
  calibrated: number | null;
  band: Band | null;
  threshold: number | null;
  withheld: boolean;
  not_calibrated?: boolean;
}

export interface MeterSummary {
  threshold: number;
  held_out: {
    questions: number;
    answered: number;
    accuracy: { estimate: number; low: number; high: number };
    coverage: { estimate: number };
  };
  bands: Band[];
  min_band_n: number;
  calibration: { method: string; slope: number; intercept: number; fitted_on: number; target_accuracy: number };
}

export interface IndexEntry {
  id: string;
  kind: string;
  slot: string;
  question: string;
  db_id: string;
  status: Status;
}

export interface Meta {
  mode: "replay" | "live";
  local_mode: boolean;
  database: string;
  meter: MeterSummary;
  connection: Record<string, unknown> | null;
  guardrail?: { enabled: boolean; available?: boolean };
  suggested: IndexEntry[];
  max_question_chars: number;
}

export interface StepEvent {
  type: "step";
  kind: "model" | "tool" | "submit";
  text: string;
}
export interface SqlEvent {
  type: "sql";
  sql: string;
  outline: Outline | null;
}
export interface RowsEvent {
  type: "rows";
  ok: boolean;
  error: { kind: string; message: string } | null;
  columns: string[];
  rows: unknown[][];
  fetched_rows: number;
  truncated: boolean;
}
export interface CheckLine {
  name: string;
  ok: boolean;
  text: string;
}
export interface AnswerEvent {
  type: "answer";
  status: Status;
  text: string | null;
  sql: string | null;
  declined: boolean;
  decline_reason: string | null;
  clarifying_question: string | null;
  assumptions: string[];
  premise_correction: string | null;
  notice?: string | null;
}
export interface StartEvent {
  type: "start";
  id: string;
  mode: "replay" | "live";
  question: string;
  hint: string | null;
  db_id: string;
  kind: string;
}
export interface ResultTableData {
  columns: string[];
  rows: unknown[][];
  total_rows: number;
  truncated: boolean;
}
export interface Explanation {
  expected: ResultTableData | null;
  expected_sql: string | null;
  comparison: { relation: string; text: string } | null;
  category: string | null;
  category_text: string | null;
  note: string | null;
  hint: string | null;
}
export interface DoneEvent {
  type: "done";
  id: string;
  status: Status;
  evaluation: { correct?: boolean | null; rule?: string; reviewed_success?: boolean | null; note?: string | null } | null;
  explanation?: Explanation | null;
  cost_usd?: number;
}
export interface ErrorEvent {
  type: "error";
  kind: string;
  message: string;
}

export type StreamEvent =
  | StartEvent
  | StepEvent
  | SqlEvent
  | RowsEvent
  | { type: "checks"; checks: CheckLine[] }
  | { type: "statistics"; statistics: Statistics }
  | { type: "chart"; spec: Record<string, unknown>; columns: string[] }
  | AnswerEvent
  | ({ type: "confidence" } & Confidence)
  | DoneEvent
  | ErrorEvent;

// The outline is a list of phrases; each phrase a list of tokens (text, column, operator ...).
export type Token = { t: string; v: string };
export type Phrase = Token[];
export interface Outline {
  kind: string;
  tables?: { name: string }[];
  filters?: Phrase[];
  groups?: Phrase[];
  returns?: (Phrase & { name?: string | null })[] | { name?: string | null }[];
  limit?: number | null;
  [k: string]: unknown;
}

export interface Statistics {
  analysis: string;
  n: number;
  detected: boolean;
  groups?: { label: string; n: number; estimate: number; ci: [number, number]; events?: number }[];
  comparisons?: { group: string; measure: string; estimate: number; ci: [number, number]; p_value: number; method: string }[];
  stratified?: {
    adjusted: { estimate: number; ci: [number, number]; p_value: number };
    crude: { estimate: number; ci: [number, number] };
    change: string;
    strata: string[];
  };
  warnings?: { kind: string; [k: string]: unknown }[];
  [k: string]: unknown;
}

/** One run as the page shows it: what the stream builds up, or a stored evidence record. */
export interface RunView {
  id: string | null;
  mode: "replay" | "live" | null;
  question: string;
  hint: string | null;
  db_id: string | null;
  steps: StepEvent[];
  sql: SqlEvent | null;
  rows: RowsEvent | null;
  checks: CheckLine[];
  statistics: Statistics | null;
  chart: { spec: Record<string, unknown>; columns: string[] } | null;
  answer: AnswerEvent | null;
  confidence: Confidence | null;
  done: DoneEvent | null;
  error: ErrorEvent | null;
  finished: boolean;
}