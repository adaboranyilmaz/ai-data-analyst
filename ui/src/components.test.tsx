import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { AnswerCard, Checks, ConfidenceMeter, QuestionPager, ResultTable, WhyWrong } from "./components";
import { emptyRun, reduce } from "./api";
import type { AnswerEvent, IndexEntry, MeterSummary, RowsEvent, RunView, StreamEvent } from "./types";

afterEach(cleanup);

const meter: MeterSummary = {
  threshold: 0.8,
  held_out: { questions: 320, answered: 23, accuracy: { estimate: 0.95, low: 0.85, high: 1 }, coverage: { estimate: 0.07 } },
  bands: [], min_band_n: 20,
  calibration: { method: "platt", slope: 6.72, intercept: -4.82, fitted_on: 150, target_accuracy: 0.9 },
};
const answer = (over: Partial<AnswerEvent>): AnswerEvent => ({
  type: "answer", status: "answered", text: "There are 682 loans.", sql: "SELECT 1", declined: false,
  decline_reason: null, clarifying_question: null, assumptions: [], premise_correction: null, ...over,
});
const rows = (over: Partial<RowsEvent>): RowsEvent => ({
  type: "rows", ok: true, error: null, columns: ["n"], rows: [[682]], fetched_rows: 1, truncated: false, ...over,
});
function runWith(a: AnswerEvent, extra: StreamEvent[] = [], mode: "replay" | "live" = "replay"): RunView {
  const events: StreamEvent[] = [
    { type: "start", id: "r1", mode, question: "How many loans?", hint: null, db_id: "financial", kind: "benchmark" },
    ...extra,
    a,
    { type: "done", id: "r1", status: a.status, evaluation: { correct: true, rule: "execution accuracy" } },
  ];
  return events.reduce(reduce, emptyRun());
}

describe("ResultTable", () => {
  it("shows an empty result as an empty table, not an error", () => {
    render(<ResultTable rows={rows({ rows: [], fetched_rows: 0 })} />);
    expect(screen.getByText(/0 of 0 rows shown/)).toBeTruthy();
  });
  it("says why a query failed", () => {
    render(<ResultTable rows={rows({ ok: false, error: { kind: "timeout", message: "took too long" } })} />);
    expect(screen.getByText(/took too long/)).toBeTruthy();
  });
  it("shows an empty value as ∅ and aligns numeric columns to the right", () => {
    render(<ResultTable rows={rows({ columns: ["name", "n"], rows: [[null, 5]], fetched_rows: 1 })} />);
    expect(screen.getByText("∅")).toBeTruthy();
    expect(screen.getByText("n").className).toBe("num");
    expect(screen.getByText("name").className).toBe("");
  });
  it("marks a result that was cut", () => {
    render(<ResultTable rows={rows({ rows: [[1]], fetched_rows: 100, truncated: true })} />);
    expect(screen.getByText(/1 of 100\+ rows shown/)).toBeTruthy();
  });
});

describe("AnswerCard", () => {
  it("states the result from the rows beside the analyst's sentence", () => {
    const a = answer({});
    render(<AnswerCard a={a} run={runWith(a, [rows({})])} meter={meter} />);
    expect(screen.getByText("n = 682")).toBeTruthy();
    expect(screen.getByText("There are 682 loans.")).toBeTruthy();
  });
  it("keeps a held-back answer's text behind a link, closed", () => {
    const a = answer({ status: "withheld" });
    const { container } = render(<AnswerCard a={a} run={runWith(a, [rows({})])} meter={meter} />);
    const details = container.querySelector("details")!;
    expect(details.open).toBe(false);
    expect(details.textContent).toContain("Show the answer it would have given");
    expect(screen.getByText(/not confident enough/)).toBeTruthy();
    expect(container.querySelector(".resultline")).toBeNull();
  });
  it("shows what is missing for a declined answer, with no result line", () => {
    const a = answer({ status: "declined", declined: true, text: "No data.", decline_reason: "The bank's profit is not recorded." });
    const { container } = render(<AnswerCard a={a} run={runWith(a, [rows({})])} meter={meter} />);
    expect(screen.getByText(/profit is not recorded/)).toBeTruthy();
    expect(container.querySelector(".resultline")).toBeNull();
  });
  it("shows the clarifying question", () => {
    const a = answer({ status: "clarify", clarifying_question: "Which year?" });
    render(<AnswerCard a={a} run={runWith(a)} meter={meter} />);
    expect(screen.getByText(/The analyst asks: Which year\?/)).toBeTruthy();
  });
  it("shows a notice about what could not be done", () => {
    const a = answer({ notice: "The statistical analysis is not available here." });
    render(<AnswerCard a={a} run={runWith(a)} meter={meter} />);
    expect(screen.getByText(/statistical analysis is not available/)).toBeTruthy();
  });
  it("reports the evaluation's verdict for a recorded run only", () => {
    const a = answer({});
    const { unmount } = render(<AnswerCard a={a} run={runWith(a)} meter={meter} />);
    const box = screen.getByText(/In the evaluation this answer was scored:/);
    expect(box.className).toBe("evalbox ok");
    expect(box.textContent).toContain("correct");
    unmount();
    render(<AnswerCard a={a} run={runWith(a, [], "live")} meter={meter} />);
    expect(screen.queryByText(/In the evaluation/)).toBeNull();
  });
  it("offers the Markdown copy once the run has finished", () => {
    const a = answer({});
    render(<AnswerCard a={a} run={runWith(a)} meter={meter} />);
    expect(screen.getByRole("button", { name: /Copy the answer and its evidence as Markdown/ })).toBeTruthy();
  });
});

describe("ConfidenceMeter", () => {
  it("shows the model's own number, labeled as such, when nothing was calibrated", () => {
    render(<ConfidenceMeter c={{ stated: 0.9, calibrated: null, band: null, threshold: null, withheld: false, not_calibrated: true }} meter={meter} />);
    expect(screen.getByText("stated by the model")).toBeTruthy();
    expect(screen.getAllByText(/not been calibrated/).length).toBeGreaterThan(0);
  });
  it("marks the decline threshold on a calibrated gauge", () => {
    const { container } = render(<ConfidenceMeter c={{ stated: 0.9, calibrated: 0.7, band: null, threshold: 0.8, withheld: true }} meter={meter} />);
    expect(container.querySelector(".tick")?.getAttribute("data-label")).toBe("decline threshold 80%");
    expect(container.querySelector(".fill.low")).not.toBeNull();
    expect(container.querySelector("[role=meter]")?.getAttribute("aria-valuenow")).toBe("70");
  });
});

describe("Checks", () => {
  it("marks a failed check", () => {
    const { container } = render(<Checks checks={[{ name: "a", ok: true, text: "Fine." }, { name: "b", ok: false, text: "A row repeats." }]} />);
    expect(container.querySelectorAll("li.bad")).toHaveLength(1);
  });
  it("renders nothing when there are no checks", () => {
    const { container } = render(<Checks checks={[]} />);
    expect(container.firstChild).toBeNull();
  });
});

describe("ConfidenceMeter: the held-out record on the slider", () => {
  const band = { range: [0.7, 0.8] as [number, number], n: 156, accuracy: 0.76, interval: [0.68, 0.82] as [number, number], thin: false };
  const c = { stated: 0.9, calibrated: 0.77, band, threshold: 0.8, withheld: true };
  it("draws the interval as a span on a second slider, not in the sentence", () => {
    const { container } = render(<ConfidenceMeter c={c} meter={meter} />);
    const span = container.querySelector(".record .span") as HTMLElement;
    expect(span.style.left).toBe("68%");
    expect(parseFloat(span.style.width)).toBeCloseTo(14, 5);
    expect((container.querySelector(".record .dot") as HTMLElement).style.left).toBe("76%");
    expect(container.querySelector(".record")?.getAttribute("aria-label")).toContain("95% interval 68% to 82%");
    expect(container.querySelector(".detail")?.textContent).not.toContain("interval");
    expect(screen.getByText(/from 156 held-out questions/)).toBeTruthy();
  });
  it("explains how the calibrated number is made, with this answer's numbers worked through", () => {
    const { container } = render(<ConfidenceMeter c={c} meter={meter} />);
    const how = container.querySelector("details.how")!;
    expect(how.querySelector("summary")?.textContent).toBe("How is this calculated?");
    expect(how.textContent).toContain("Platt scaling");
    expect(how.textContent).toContain("150 calibration questions");
    expect(how.textContent).toContain("6.72 × 0.90 − 4.82");
    expect(how.textContent).toContain("= 77%");
    expect(how.textContent).toContain("320 held-out");
  });
  it("has no record bar and no explanation for an uncalibrated confidence", () => {
    const { container } = render(<ConfidenceMeter c={{ stated: 0.9, calibrated: null, band: null, threshold: null, withheld: false, not_calibrated: true }} meter={meter} />);
    expect(container.querySelector(".record")).toBeNull();
    expect(container.querySelector("details.how")).toBeNull();
  });
  it("fades an answered confidence from light to solid, as a held-back one fades", () => {
    const ok = render(<ConfidenceMeter c={{ ...c, withheld: false, calibrated: 0.85 }} meter={meter} />);
    expect(ok.container.querySelector(".fill.ok")).not.toBeNull();
    cleanup();
    const low = render(<ConfidenceMeter c={c} meter={meter} />);
    expect(low.container.querySelector(".fill.low")).not.toBeNull();
  });
});

describe("QuestionPager", () => {
  const items: IndexEntry[] = Array.from({ length: 12 }, (_, i) => ({
    id: `q${i}`, kind: i < 7 ? "benchmark" : "banking", slot: "", question: `Question ${i}?`, db_id: "financial", status: "answered" as const,
  }));
  it("shows five to a page with a dot for each page, all questions reachable", () => {
    const { container } = render(<QuestionPager items={items} current={null} disabled={false} live={false} onPick={() => {}} />);
    expect(screen.getByText("Page 1 of 3 · 12 questions")).toBeTruthy();
    expect(container.querySelectorAll(".pager-dots button")).toHaveLength(3);
    expect(container.querySelectorAll(".pager-slide")).toHaveLength(3);
    expect(container.querySelectorAll("button.q")).toHaveLength(12);
  });
  it("moves with the arrows and the dots, and the first page has no previous page", () => {
    render(<QuestionPager items={items} current={null} disabled={false} live={false} onPick={() => {}} />);
    expect((screen.getByRole("button", { name: "Previous page" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    expect(screen.getByText(/Page 2 of 3/)).toBeTruthy();
    expect(screen.getByText("Benchmark questions · Banking questions")).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "Go to page 3" }));
    expect(screen.getByText(/Page 3 of 3/)).toBeTruthy();
    expect((screen.getByRole("button", { name: "Next page" }) as HTMLButtonElement).disabled).toBe(true);
  });
  it("picks a question and marks the current one", () => {
    const picked: string[] = [];
    const { container } = render(<QuestionPager items={items} current="q3" disabled={false} live={true} onPick={(q) => picked.push(q.id)} />);
    fireEvent.click(screen.getByText("Question 4?"));
    expect(picked).toEqual(["q4"]);
    expect(container.querySelector("button.q[aria-current=true]")?.textContent).toContain("Question 3?");
    expect(container.textContent).toContain("recorded run");
  });
});
describe("the verdict box and the held-out line", () => {
  it("boxes a wrong verdict in the red tone", () => {
    const a = answer({});
    const run = runWith(a);
    run.done = { ...run.done!, evaluation: { correct: false, rule: "execution accuracy" } };
    render(<AnswerCard a={a} run={run} meter={meter} />);
    const box = screen.getByText(/In the evaluation this answer was scored:/);
    expect(box.className).toBe("evalbox no");
    expect(box.textContent).toContain("wrong");
  });
  it("puts a colon after the held-out line, with the percentage inside its box", () => {
    const band = { range: [0.7, 0.8] as [number, number], n: 156, accuracy: 0.76, interval: [0.68, 0.82] as [number, number], thin: false };
    const { container } = render(<ConfidenceMeter c={{ stated: 0.9, calibrated: 0.77, band, threshold: 0.8, withheld: true }} meter={meter} />);
    const head = container.querySelector(".record-head")!;
    expect(head.textContent).toBe("How often answers like this were right: 76%");
    expect(head.querySelector("b")?.textContent).toBe("76%");
  });
});
describe("WhyWrong", () => {
  const x = {
    expected: { columns: ["count"], rows: [[4941]], total_rows: 1, truncated: false },
    expected_sql: "SELECT COUNT(Id) FROM users WHERE DATE(LastAccessDate) > '2014-09-01'",
    comparison: { relation: "single_value", text: "It returned 5146; the expected answer is 4941." },
    category: "output", category_text: "it returned different columns",
    note: "The expert's query ignores the time of day.", hint: "last accessed after 2014/9/1 refers to LastAccessDate > '2014-09-01'",
  };
  const a = answer({});
  const run = (): RunView => {
    const r = runWith(a, [rows({ columns: ["count"], rows: [[5146]] })]);
    r.done = { ...r.done!, evaluation: { correct: false, rule: "execution accuracy" }, explanation: x };
    return r;
  };
  it("puts the analyst's result beside the expected one, with the reviewer's note and the computed line", () => {
    render(<WhyWrong x={x} run={run()} />);
    expect(screen.getByText("The analyst's result")).toBeTruthy();
    expect(screen.getByText("The expected result")).toBeTruthy();
    expect(screen.getByText("5146")).toBeTruthy();
    expect(screen.getByText("4941")).toBeTruthy();
    expect(screen.getByText(/ignores the time of day/)).toBeTruthy();
    expect(screen.getByText("It returned 5146; the expected answer is 4941.")).toBeTruthy();
  });
  it("keeps the expert's query and the hint behind a link, with a note that the analyst never saw it", () => {
    const { container } = render(<WhyWrong x={x} run={run()} />);
    const d = container.querySelector("details")!;
    expect(d.open).toBe(false);
    expect(d.textContent).toContain("DATE(LastAccessDate)");
    expect(d.textContent).toContain("Hint given to the analyst");
    expect(d.textContent).toContain("never saw the expert's query");
  });
  it("falls back to the automatic classification, labeled as structure only, without a note", () => {
    render(<WhyWrong x={{ ...x, note: null }} run={run()} />);
    expect(screen.getByText(/automatic comparison finds the first difference here: it returned different columns/)).toBeTruthy();
    expect(screen.getByText(/not their intent/)).toBeTruthy();
  });
  it("is shown on a wrong recorded answer and not on a right one", () => {
    const { container, unmount } = render(<AnswerCard a={a} run={run()} meter={meter} />);
    expect(container.querySelector(".why")).not.toBeNull();
    unmount();
    const right = runWith(a);
    right.done = { ...right.done!, explanation: null };
    const r2 = render(<AnswerCard a={a} run={right} meter={meter} />);
    expect(r2.container.querySelector(".why")).toBeNull();
  });
  it("lists a missing expected result as no result", () => {
    render(<WhyWrong x={{ ...x, expected: null }} run={run()} />);
    expect(screen.getByText("No result.")).toBeTruthy();
  });
});