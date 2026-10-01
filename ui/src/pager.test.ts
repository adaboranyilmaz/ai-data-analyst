import { describe, expect, it } from "vitest";
import { pageOf, pageTitle, paginate } from "./pager";
import type { IndexEntry } from "./types";

const q = (id: string, kind: string): IndexEntry => ({ id, kind, slot: "", question: id, db_id: "d", status: "answered" });
const items = [
  ...Array.from({ length: 10 }, (_, i) => q(`b${i}`, "benchmark")),
  ...Array.from({ length: 6 }, (_, i) => q(`k${i}`, "banking")),
  ...Array.from({ length: 4 }, (_, i) => q(`g${i}`, "guardrail")),
];

describe("pager", () => {
  it("splits twenty questions into four pages of five", () => {
    const pages = paginate(items);
    expect(pages.map((p) => p.length)).toEqual([5, 5, 5, 5]);
    expect(pages.flat().map((x) => x.id)).toEqual(items.map((x) => x.id));
  });
  it("keeps a short last page", () => {
    expect(paginate([1, 2, 3, 4, 5, 6]).map((p) => p.length)).toEqual([5, 1]);
    expect(paginate([])).toEqual([]);
  });
  it("titles a page by the sections it holds", () => {
    const pages = paginate(items);
    expect(pageTitle(pages[0])).toBe("Benchmark questions");
    expect(pageTitle(pages[3])).toBe("Banking questions · Comparisons and causes");
  });
  it("finds the page of a run, or the first", () => {
    const pages = paginate(items);
    expect(pageOf(pages, "k3")).toBe(2);
    expect(pageOf(pages, "g2")).toBe(3);
    expect(pageOf(pages, "nope")).toBe(0);
    expect(pageOf(pages, null)).toBe(0);
  });
});