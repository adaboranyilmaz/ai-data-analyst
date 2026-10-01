import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const meta = { suggested: [{ id: "bench-1", question: "How many  heroes?" }] };
const stream = [
  { wait_ms: 0, event: { type: "start", id: "bench-1" } },
  { wait_ms: 5, event: { type: "step", text: "looking" } },
  { wait_ms: 5, event: { type: "done", id: "bench-1" } },
];
const files: Record<string, unknown> = {
  "data/meta.json": meta,
  "data/streams/bench-1.json": stream,
};

describe("the static demo reads files, not the service", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.stubEnv("MODE", "static");
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        url in files
          ? new Response(JSON.stringify(files[url]))
          : new Response("{}", { status: 404, statusText: "Not Found" }),
      ),
    );
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("streams a recorded run by id", async () => {
    const { ask } = await import("./api");
    const seen: string[] = [];
    await ask({ run_id: "bench-1" }, (e) => seen.push(e.type));
    expect(seen).toEqual(["start", "step", "done"]);
  });

  it("finds a recorded question word for word, ignoring case and spacing", async () => {
    const { ask } = await import("./api");
    const seen: string[] = [];
    await ask({ question: "how many heroes?" }, (e) => seen.push(e.type));
    expect(seen).toHaveLength(3);
  });

  it("says live questions need the service, as the service does", async () => {
    const { ask } = await import("./api");
    await expect(ask({ question: "something else" }, () => {})).rejects.toMatchObject({
      status: 404,
      message: expect.stringContaining("recorded runs only"),
    });
  });

  it("stops at once when the page abandons a run", async () => {
    const { ask } = await import("./api");
    const controller = new AbortController();
    const seen: string[] = [];
    const run = ask({ run_id: "bench-1" }, (e) => seen.push(e.type), controller.signal);
    controller.abort();
    await expect(run).rejects.toMatchObject({ name: "AbortError" });
    expect(seen.length).toBeLessThan(3);
  });
});
