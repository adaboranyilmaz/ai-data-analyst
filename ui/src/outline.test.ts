import { describe, expect, it } from "vitest";
import { outlineLines } from "./outline";

describe("outlineLines", () => {
  it("puts a filter and a return in words", () => {
    const lines = outlineLines({
      kind: "select",
      tables: [{ name: "comments" }],
      filters: [{ phrase: [{ t: "col", v: "score" }, { t: "op", v: "eq" }, { t: "lit", v: "17" }] }],
      returns: [{ phrase: [{ t: "col", v: "text" }], name: null }],
      links: [], groups: [], group_filters: [], order: [], limit: null,
    } as never);
    expect(lines).toEqual([
      { label: "Reads", text: "comments" },
      { label: "Keeps rows where", text: "score is 17" },
      { label: "Returns", text: "text" },
    ]);
  });
  it("is silent about what it cannot express", () => {
    expect(outlineLines({ kind: "unparsed" })[0].text).toContain("read the SQL");
    expect(outlineLines(null)).toEqual([]);
  });
  it("words an aggregate and a limit", () => {
    const lines = outlineLines({
      kind: "select", tables: [{ name: "loan" }], filters: [], groups: [], group_filters: [], links: [], order: [], limit: 5,
      returns: [{ phrase: [{ t: "agg", v: "avg", arg: [] as never }], name: "avg_amount" }],
    } as never);
    expect(lines.some((l) => l.label === "Keeps" && l.text === "the first 5 rows")).toBe(true);
    expect(lines.find((l) => l.label === "Returns")?.text).toContain("avg_amount");
  });
});