import { describe, expect, it } from "vitest";
import { emptyRun, reduce } from "./api";
import { resultLine, toMarkdown } from "./summary";
import type { MeterSummary, RowsEvent, StreamEvent } from "./types";

const rows = (over: Partial<RowsEvent>): RowsEvent => ({
  type: "rows", ok: true, error: null, columns: ["n"], rows: [[682]], fetched_rows: 1, truncated: false, ...over,
});
const meter: MeterSummary = {
  threshold: 0.8,
  held_out: { questions: 320, answered: 23, accuracy: { estimate: 0.95, low: 0.85, high: 1 }, coverage: { estimate: 0.07 } },
  bands: [], min_band_n: 20,
  calibration: { method: "platt", slope: 6.72, intercept: -4.82, fitted_on: 150, target_accuracy: 0.9 },
};

describe("resultLine", () => {
  it("states a single row as column = value", () => {
    expect(resultLine(rows({}))).toBe("n = 682");
    expect(resultLine(rows({ columns: ["a", "b"], rows: [[1, "x"]] }))).toBe("a = 1  ·  b = x");
  });
  it("says so when there are no rows, and points to the table when there are many", () => {
    expect(resultLine(rows({ rows: [], fetched_rows: 0 }))).toBe("The query returned no rows.");
    expect(resultLine(rows({ rows: [[1], [2]], fetched_rows: 100, truncated: true }))).toBe("100+ rows returned: see the table below.");
  });
  it("is silent for a failed query or no rows event", () => {
    expect(resultLine(rows({ ok: false }))).toBeNull();
    expect(resultLine(null)).toBeNull();
  });
  it("shows an empty value as ∅", () => {
    expect(resultLine(rows({ rows: [[null]] }))).toBe("n = ∅");
  });
});

describe("toMarkdown", () => {
  const events: StreamEvent[] = [
    { type: "start", id: "bench-1", mode: "replay", question: "How many loans?", hint: null, db_id: "financial", kind: "benchmark" },
    { type: "sql", sql: "SELECT COUNT(*) FROM loan", outline: null },
    rows({}),
    { type: "checks", checks: [{ name: "a", ok: true, text: "Rows returned." }, { name: "b", ok: false, text: "A row repeats." }] },
    { type: "answer", status: "answered", text: "There are 682 loans.", sql: "SELECT 1", declined: false, decline_reason: null, clarifying_question: null, assumptions: ["all loans"], premise_correction: null },
    { type: "confidence", stated: 0.9, calibrated: 0.7, band: null, threshold: 0.8, withheld: false },
    { type: "done", id: "bench-1", status: "answered", evaluation: null },
  ];
  const md = toMarkdown(events.reduce(reduce, emptyRun()), meter, "http://localhost:8000");
  it("holds the question, answer, confidence, query, result, checks and link", () => {
    expect(md).toContain("## How many loans?");
    expect(md).toContain("There are 682 loans.");
    expect(md).toContain("**Assumes:** all loans");
    expect(md).toContain("**Confidence:** Calibrated confidence 70%");
    expect(md).toContain("```sql\nSELECT COUNT(*) FROM loan\n```");
    expect(md).toContain("**Result:** n = 682");
    expect(md).toContain("- ✓ Rows returned.");
    expect(md).toContain("- ✗ A row repeats.");
    expect(md).toContain("Run: http://localhost:8000/#/run/bench-1");
  });
  it("ends with one newline", () => {
    expect(md.endsWith("\n") && !md.endsWith("\n\n")).toBe(true);
  });
});

describe("toMarkdown for a wrong answer", () => {
  it("adds the reviewer's note and the computed comparison", () => {
    const run = emptyRun("How many?");
    run.answer = { type: "answer", status: "answered", text: "A number.", sql: null, declined: false, decline_reason: null, clarifying_question: null, assumptions: [], premise_correction: null };
    run.done = {
      type: "done", id: "r", status: "answered", evaluation: { correct: false, rule: "execution accuracy" },
      explanation: { expected: null, expected_sql: null, comparison: { relation: "single_value", text: "It returned 5; the expected answer is 4." }, category: null, category_text: null, note: "The time of day was ignored.", hint: null },
    };
    const md = toMarkdown(run, meter);
    expect(md).toContain("**Scored wrong in the evaluation**");
    expect(md).toContain("The time of day was ignored.");
    expect(md).toContain("It returned 5; the expected answer is 4.");
  });
});