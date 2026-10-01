import type { Meta, RunView, StreamEvent } from "./types";

export function emptyRun(question = ""): RunView {
  return {
    id: null, mode: null, question, hint: null, db_id: null, steps: [], sql: null, rows: null,
    checks: [], statistics: null, chart: null, answer: null, confidence: null, done: null,
    error: null, finished: false,
  };
}

/** Fold one event into the run the page shows. */
export function reduce(run: RunView, e: StreamEvent): RunView {
  switch (e.type) {
    case "start":
      return { ...run, id: e.id, mode: e.mode, question: e.question, hint: e.hint, db_id: e.db_id };
    case "step":
      return { ...run, steps: [...run.steps, e] };
    case "sql":
      return { ...run, sql: e };
    case "rows":
      return { ...run, rows: e };
    case "checks":
      return { ...run, checks: e.checks };
    case "statistics":
      return { ...run, statistics: e.statistics };
    case "chart":
      return { ...run, chart: { spec: e.spec, columns: e.columns } };
    case "answer":
      return { ...run, answer: e };
    case "confidence": {
      const { type: _t, ...c } = e;
      return { ...run, confidence: c };
    }
    case "done":
      return { ...run, done: e, finished: true };
    case "error":
      return { ...run, error: e, finished: true };
  }
}

/** Parse a Server-Sent Events body incrementally: feed chunks, get whole events back. */
export class SseParser {
  private buffer = "";
  feed(chunk: string): StreamEvent[] {
    this.buffer += chunk.replace(/\r\n/g, "\n");
    const out: StreamEvent[] = [];
    let i: number;
    while ((i = this.buffer.indexOf("\n\n")) >= 0) {
      const block = this.buffer.slice(0, i);
      this.buffer = this.buffer.slice(i + 2);
      const data = block
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).replace(/^ /, ""))
        .join("\n");
      if (data) out.push(JSON.parse(data) as StreamEvent);
    }
    return out;
  }
}

export class AskError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

/** Ask a question (or replay a recorded run by id) and call `onEvent` as the run streams. */
export async function ask(
  body: { question?: string; run_id?: string },
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* keep the status text */
    }
    throw new AskError(detail, res.status);
  }
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const e of parser.feed(decoder.decode(value, { stream: true }))) onEvent(e);
  }
}

export async function getMeta(): Promise<Meta> {
  const res = await fetch("/api/meta");
  if (!res.ok) throw new Error(`meta: ${res.status}`);
  return res.json();
}

/** A stored run's evidence record, rebuilt into the page's view of a run. */
export async function getRun(id: string): Promise<RunView> {
  const res = await fetch(`/runs/${encodeURIComponent(id)}`);
  if (!res.ok) throw new AskError("No such run.", res.status);
  const ev = await res.json();
  let run = emptyRun(ev.question);
  const events: StreamEvent[] = [
    { type: "start", id: ev.id, mode: "replay", question: ev.question, hint: ev.hint, db_id: ev.db_id, kind: ev.kind },
    ...ev.steps,
  ];
  for (const e of events) run = reduce(run, e);
  run = reduce(run, { type: "sql", sql: ev.answer.sql, outline: ev.sql_outline } as StreamEvent);
  if (!ev.answer.sql) run = { ...run, sql: null };
  if (ev.result) run = reduce(run, { type: "rows", ok: ev.result.ok, error: ev.result.error, columns: ev.result.columns, rows: ev.result.rows, fetched_rows: ev.result.fetched_rows, truncated: ev.result.truncated });
  if (ev.checks.length) run = reduce(run, { type: "checks", checks: ev.checks });
  if (ev.statistics) run = reduce(run, { type: "statistics", statistics: ev.statistics });
  if (ev.chart_spec) run = reduce(run, { type: "chart", spec: ev.chart_spec, columns: ev.result?.columns ?? [] });
  run = reduce(run, { type: "answer", status: ev.status, ...ev.answer });
  run = reduce(run, { type: "confidence", ...ev.confidence });
  run = reduce(run, { type: "done", id: ev.id, status: ev.status, evaluation: ev.evaluation, explanation: ev.explanation });
  return run;
}