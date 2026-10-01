import type { Confidence, MeterSummary } from "./types";

export const pct = (x: number, digits = 0) => `${(100 * x).toFixed(digits)}%`;

export interface HeldOutRecord {
  accuracy: number;
  low: number;
  high: number;
  n: number;
  thin: boolean;
}

/**
 * What a confidence means, in words, from the held-out record it rests on. The numbers come from
 * the service (the calibration's held-out reliability table); nothing is computed here but the
 * formatting, so the page never states a rate the evaluation did not measure. The interval is
 * returned as data (`record`) for the gauge to draw, not written into the sentence.
 */
export function meterWording(
  c: Confidence,
  meter: MeterSummary,
): { headline: string; detail: string; record: HeldOutRecord | null } {
  if (c.not_calibrated && c.stated == null) {
    return {
      headline: "No confidence figure",
      detail:
        "The analyst's confidence was calibrated on lookups and counts, not on comparisons. For a comparison, the intervals in the statistics are the measure of uncertainty.",
      record: null,
    };
  }
  if (c.not_calibrated) {
    return {
      headline: `Stated confidence ${pct(c.stated as number)}`,
      detail:
        "This is the number the model gave for itself. It has not been calibrated for this kind of question or database, so it does not say how often answers like this are right.",
      record: null,
    };
  }
  if (c.calibrated == null) {
    return { headline: "No confidence", detail: "The analyst declined, so there is no answer to rate.", record: null };
  }
  const bits: string[] = [];
  const b = c.band;
  let record: HeldOutRecord | null = null;
  if (b && b.accuracy != null && b.interval) {
    record = { accuracy: b.accuracy, low: b.interval[0], high: b.interval[1], n: b.n, thin: b.thin };
    bits.push(
      `Answers at about this confidence were right ${pct(b.accuracy)} of the time on ${b.n} held-out benchmark questions.` +
        (b.thin ? " That is few questions, so treat the rate as rough." : ""),
    );
  } else {
    bits.push("No held-out answer reached this confidence, so there is no record to quote.");
  }
  const h = meter.held_out;
  if (c.withheld) {
    bits.push(
      `It is under the decline threshold (${pct(meter.threshold)}), so the analyst holds this answer back. Above the threshold, ${h.answered} of ${h.questions} held-out questions were answered, ${pct(h.accuracy.estimate, 1)} of them correctly.`,
    );
  } else {
    bits.push(
      `It clears the decline threshold (${pct(meter.threshold)}), which ${h.answered} of ${h.questions} held-out questions did; ${pct(h.accuracy.estimate, 1)} of those were right.`,
    );
  }
  return { headline: `Calibrated confidence ${pct(c.calibrated)}`, detail: bits.join(" "), record };
}

/** The calibration worked through for one stated confidence: the curve's input, its sum and result. */
export function workedExample(stated: number, meter: MeterSummary) {
  const { slope, intercept } = meter.calibration;
  const z = slope * stated + intercept;
  return { slope, intercept, z, result: 1 / (1 + Math.exp(-z)) };
}

export const statusLabel: Record<string, string> = {
  answered: "Answered",
  declined: "Declined: the data cannot answer this",
  clarify: "The analyst needs a clarification",
  withheld: "Held back: confidence under the threshold",
};