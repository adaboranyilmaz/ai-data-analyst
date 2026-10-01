import type { IndexEntry } from "./types";

export const PAGE_SIZE = 5;

/** The questions in pages of PAGE_SIZE, in the order given. */
export function paginate<T>(items: T[], size = PAGE_SIZE): T[][] {
  const pages: T[][] = [];
  for (let i = 0; i < items.length; i += size) pages.push(items.slice(i, i + size));
  return pages;
}

export const kindLabel: Record<string, string> = {
  benchmark: "Benchmark",
  banking: "Banking",
  guardrail: "Comparison",
};

const sectionTitle: Record<string, string> = {
  benchmark: "Benchmark questions",
  banking: "Banking questions",
  guardrail: "Comparisons and causes",
};

/** What a page holds, as a heading: its section, or the sections it spans. */
export function pageTitle(page: IndexEntry[]): string {
  const kinds = [...new Set(page.map((q) => q.kind))];
  return kinds.map((k) => sectionTitle[k] ?? k).join(" · ");
}

/** The page an item is on, or 0. */
export function pageOf(pages: IndexEntry[][], id: string | null | undefined): number {
  if (!id) return 0;
  const i = pages.findIndex((p) => p.some((q) => q.id === id));
  return i < 0 ? 0 : i;
}