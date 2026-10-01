import { meterWording, pct, statusLabel } from "./wording";
import type { MeterSummary, RowsEvent, RunView } from "./types";

const show = (v: unknown): string => (v === null || v === undefined ? "∅" : typeof v === "object" ? JSON.stringify(v) : String(v));

/**
 * The result in one plain line, built from the rows and nothing else: the analyst writes its
 * answer before it sees them, so its sentence may not state the number. One row reads
 * "column = value"; more rows point to the table.
 */
export function resultLine(rows: RowsEvent | null): string | null {
  if (!rows || !rows.ok) return null;
  if (rows.rows.length === 0) return "The query returned no rows.";
  if (rows.rows.length === 1 && rows.columns.length <= 6) {
    return rows.columns.map((c, i) => `${c} = ${show(rows.rows[0][i])}`).join("  ·  ");
  }
  return `${rows.fetched_rows}${rows.truncated ? "+" : ""} rows returned: see the table below.`;
}

/** An answer with its evidence as Markdown, for a report or an audit trail. */
export function toMarkdown(run: RunView, meter: MeterSummary, origin = ""): string {
  const out: string[] = [`## ${run.question}`, ""];
  const a = run.answer;
  if (a) {
    out.push(`**Status:** ${statusLabel[a.status] ?? a.status}`, "");
    if (a.decline_reason && a.status === "declined") out.push(a.decline_reason, "");
    if (a.clarifying_question) out.push(`The analyst asks: ${a.clarifying_question}`, "");
    if (a.notice) out.push(`> ${a.notice}`, "");
    if (a.text && a.status !== "declined") out.push(a.text, "");
    if (a.assumptions.length) out.push(`**Assumes:** ${a.assumptions.join("; ")}`, "");
  }
  if (run.confidence) {
    const w = meterWording(run.confidence, meter);
    const r = w.record ? ` (95% interval ${pct(w.record.low)} to ${pct(w.record.high)} on ${w.record.n} held-out questions)` : "";
    out.push(`**Confidence:** ${w.headline}. ${w.detail}${r}`, "");
  }
  if (run.sql) out.push("**Query**", "", "```sql", run.sql.sql, "```", "");
  const line = resultLine(run.rows);
  if (line) out.push(`**Result:** ${line}`, "");
  const why = run.done?.explanation;
  if (why && run.done?.evaluation?.correct === false) {
    out.push("**Scored wrong in the evaluation**", "");
    if (why.note) out.push(why.note, "");
    if (why.comparison) out.push(why.comparison.text, "");
  }
  if (run.checks.length) out.push("**Automatic checks**", "", ...run.checks.map((c) => `- ${c.ok ? "✓" : "✗"} ${c.text}`), "");
  if (run.id) out.push(`Run: ${origin}/#/run/${run.id}`, "");
  return out.join("\n").trimEnd() + "\n";
}