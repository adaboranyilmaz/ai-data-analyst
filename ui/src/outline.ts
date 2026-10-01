import type { Outline } from "./types";

type Tok = { t: string; v?: string; arg?: Tok[]; args?: Tok[][] };
type Part = { phrase?: Tok[]; unrendered?: boolean; name?: string | null };

const OPS: Record<string, string> = {
  eq: "is", neq: "is not", gt: "is more than", gte: "is at least", lt: "is less than", lte: "is at most",
  and: "and", or: "or", not: "not", is_empty: "is empty", is_not_empty: "is not empty",
  in: "is one of", not_in: "is not one of", like: "matches", not_like: "does not match",
  between: "is between", not_between: "is not between", list_sep: ",", plus: "+", minus: "-",
  times: "×", divided_by: "÷", all_columns: "every column", true: "true", false: "false", null: "empty",
};
const AGG: Record<string, string> = {
  sum: "the total of", avg: "the average of", max: "the highest", min: "the lowest",
  count: "the number of", count_rows: "the number of rows", count_distinct: "the number of different",
};
const FN: Record<string, string> = {
  round: "rounded", round_whole: "rounded", year_of: "the year of", month_of: "the month of",
  day_of: "the day of", whole_years: "whole years between", nullif: "empty if equal", abs: "the size of",
  concat: "joined", substring: "part of", coalesce: "the first available of", case: "depending on a condition",
  case_no_else: "depending on a condition",
};

function fn(t: Tok): string {
  const a = (t.args ?? []).map((x) => phrase(x));
  switch (t.v) {
    case "case": return `(${a[0]} if ${a[1]}, otherwise ${a[2]})`;
    case "case_no_else": return `(${a[0]} if ${a[1]})`;
    case "round": return `${a[0]} rounded to ${a[1]} places`;
    case "round_whole": return `${a[0]} rounded`;
    case "whole_years": return `whole years from ${a[0]} to ${a[1]}`;
    case "coalesce": return `${a[0]}, or ${a[1]} if empty`;
    case "nullif": return `${a[0]}, empty if equal to ${a[1]}`;
    case "concat": return `${a[0]} joined with ${a[1]}`;
    case "substring": return `part of ${a[0]} from ${a[1]}, length ${a[2]}`;
    default: return `${FN[t.v ?? ""] ?? t.v} ${a.join(", ")}`.trim();
  }
}

function phrase(p: Tok[] | undefined): string {
  if (!p) return "";
  const out: string[] = [];
  for (const t of p) {
    if (t.t === "col") out.push(t.v ?? "");
    else if (t.t === "lit") out.push(t.v ?? "");
    else if (t.t === "op") out.push(OPS[t.v ?? ""] ?? (t.v ?? ""));
    else if (t.t === "open") out.push("(");
    else if (t.t === "close") out.push(")");
    else if (t.t === "agg") out.push(`${AGG[t.v ?? ""] ?? t.v} ${phrase(t.arg)}`.trim());
    else if (t.t === "fn") out.push(fn(t));
  }
  return out.join(" ").replace(/\( /g, "(").replace(/ \)/g, ")").replace(/ ,/g, ",");
}

const part = (x: Part): string => (x.unrendered ? "a part this summary cannot put in words (see the SQL)" : phrase(x.phrase));

export interface OutlineLine {
  label: string;
  text: string;
}

/** The query in words: what it reads, keeps, groups, returns and how it sorts. Exact where it speaks. */
export function outlineLines(o: Outline | null): OutlineLine[] {
  if (!o) return [];
  if (o.kind === "unparsed") return [{ label: "Note", text: "This query could not be summarized; read the SQL." }];
  if (o.kind === "combined") return [{ label: "Note", text: "This query combines other queries; read the SQL." }];
  const lines: OutlineLine[] = [];
  const tables = (o.tables ?? []).map((t) => t.name);
  if (tables.length) lines.push({ label: "Reads", text: tables.join(", ") });
  const links = (o.links ?? []) as { table: string | null; keep_unmatched: boolean; on: Part[]; using: string[] }[];
  for (const l of links) {
    const how = l.on.map(part).join(" and ") || l.using.join(", ");
    lines.push({ label: "Links", text: `${l.table ?? "a result"} on ${how}${l.keep_unmatched ? " (keeping rows with no match)" : ""}` });
  }
  const filters = (o.filters ?? []) as Part[];
  if (filters.length) lines.push({ label: "Keeps rows where", text: filters.map(part).join(" and ") });
  const groups = (o.groups ?? []) as Part[];
  if (groups.length) lines.push({ label: "Groups by", text: groups.map(part).join(", ") });
  const groupFilters = (o.group_filters ?? []) as Part[];
  if (groupFilters.length) lines.push({ label: "Keeps groups where", text: groupFilters.map(part).join(" and ") });
  const returns = (o.returns ?? []) as Part[];
  if (returns.length) lines.push({ label: o.distinct ? "Returns different" : "Returns", text: returns.map((r) => (r.name ? `${part(r)} as ${r.name}` : part(r))).join(", ") });
  const order = (o.order ?? []) as (Part & { descending?: boolean })[];
  if (order.length) lines.push({ label: "Sorts by", text: order.map((x) => `${part(x)}${x.descending ? " (largest first)" : ""}`).join(", ") });
  if (o.limit != null) lines.push({ label: "Keeps", text: `the first ${o.limit} rows` });
  return lines;
}