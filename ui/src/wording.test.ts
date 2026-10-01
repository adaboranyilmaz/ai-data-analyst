import { describe, expect, it } from "vitest";
import { SseParser, emptyRun, reduce } from "./api";
import { meterWording, workedExample } from "./wording";
import type { Confidence, MeterSummary, StreamEvent } from "./types";

const meter: MeterSummary = {
  threshold: 0.796,
  held_out: { questions: 320, answered: 23, accuracy: { estimate: 0.9565, low: 0.85, high: 1 }, coverage: { estimate: 0.07 } },
  bands: [],
  min_band_n: 20,
  calibration: { method: "platt", slope: 6.72, intercept: -4.82, fitted_on: 150, target_accuracy: 0.9 },
};
const base: Confidence = { stated: 0.95, calibrated: 0.83, band: null, threshold: 0.796, withheld: false };

describe("meterWording", () => {
  it("quotes the held-out rate and count, and returns the interval as data for the gauge", () => {
    const w = meterWording({ ...base, band: { range: [0.8, 0.9], n: 20, accuracy: 0.95, interval: [0.76, 0.99], thin: false } }, meter);
    expect(w.headline).toBe("Calibrated confidence 83%");
    expect(w.detail).toContain("right 95% of the time");
    expect(w.detail).not.toContain("interval");
    expect(w.record).toEqual({ accuracy: 0.95, low: 0.76, high: 0.99, n: 20, thin: false });
    expect(w.detail).toContain("20 held-out benchmark questions");
    expect(w.detail).toContain("clears the decline threshold");
  });
  it("says so when the band is thin", () => {
    const w = meterWording({ ...base, band: { range: [0.1, 0.2], n: 7, accuracy: 0.14, interval: [0.03, 0.51], thin: true } }, meter);
    expect(w.detail).toContain("few questions");
  });
  it("says there is no record when no band applies", () => {
    expect(meterWording(base, meter).detail).toContain("no record to quote");
  });
  it("explains a withheld answer", () => {
    const w = meterWording({ ...base, calibrated: 0.5, withheld: true }, meter);
    expect(w.detail).toContain("holds this answer back");
    expect(w.detail).toContain("23 of 320");
  });
  it("does not pass a model's own number off as calibrated", () => {
    const w = meterWording({ stated: 0.9, calibrated: null, band: null, threshold: null, withheld: false, not_calibrated: true }, meter);
    expect(w.headline).toBe("Stated confidence 90%");
    expect(w.detail).toContain("not been calibrated");
  });
  it("gives a comparison no confidence figure and points to its intervals", () => {
    const w = meterWording({ stated: null, calibrated: null, band: null, threshold: null, withheld: false, not_calibrated: true }, meter);
    expect(w.headline).toBe("No confidence figure");
    expect(w.detail).toContain("intervals");
  });
  it("has nothing to rate when the analyst declined", () => {
    const w = meterWording({ stated: 0.9, calibrated: null, band: null, threshold: 0.8, withheld: false }, meter);
    expect(w.detail).toContain("declined");
  });
});

describe("SseParser", () => {
  const block = (e: object) => `event: x\ndata: ${JSON.stringify(e)}\n\n`;
  it("returns whole events and keeps a partial one for the next chunk", () => {
    const p = new SseParser();
    const full = block({ type: "step", kind: "tool", text: "a" });
    const second = block({ type: "done", id: "r", status: "answered", evaluation: null });
    expect(p.feed(full + second.slice(0, 20))).toHaveLength(1);
    const rest = p.feed(second.slice(20));
    expect(rest).toHaveLength(1);
    expect(rest[0].type).toBe("done");
  });
  it("copes with CRLF line ends", () => {
    expect(new SseParser().feed(block({ type: "step", kind: "tool", text: "a" }).replace(/\n/g, "\r\n"))).toHaveLength(1);
  });
});

describe("reduce", () => {
  it("builds a run from its events and ends on done or error", () => {
    const events: StreamEvent[] = [
      { type: "start", id: "r1", mode: "replay", question: "q?", hint: null, db_id: "financial", kind: "banking" },
      { type: "step", kind: "model", text: "Read" },
      { type: "done", id: "r1", status: "answered", evaluation: null },
    ];
    const run = events.reduce(reduce, emptyRun());
    expect(run.id).toBe("r1");
    expect(run.steps).toHaveLength(1);
    expect(run.finished).toBe(true);
    expect(reduce(emptyRun(), { type: "error", kind: "x", message: "m" }).finished).toBe(true);
  });
});

describe("workedExample", () => {
  it("applies the fitted curve to a stated confidence", () => {
    const w = workedExample(0.9, meter);
    expect(w.z).toBeCloseTo(6.72 * 0.9 - 4.82, 10);
    expect(w.result).toBeCloseTo(1 / (1 + Math.exp(-w.z)), 10);
    expect(w.result).toBeGreaterThan(0.76);
    expect(w.result).toBeLessThan(0.78);
  });
});