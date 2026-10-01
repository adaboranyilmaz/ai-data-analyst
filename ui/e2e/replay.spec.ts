import { expect, test } from "@playwright/test";

// Replay mode: the recorded runs, streamed at a high speed. Run ids come from results/demo.
const CORRECT_ABOVE = "bench-782"; // answered, above the decline threshold
const WITHHELD = "bench-726";
const DECLINED = "bank-own-d03";
const GUARDED = "guard-own-f02";

async function pick(page: import("@playwright/test").Page, id: string) {
  const meta = await (await page.request.get("/api/meta")).json();
  const entry = meta.suggested.find((s: { id: string }) => s.id === id);
  await page.getByRole("button", { name: new RegExp(entry.question.slice(0, 40).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")) }).click();
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("mode")).toHaveText("replay of recorded runs");
});

test("lists the recorded questions and does not accept a new one", async ({ page }) => {
  await expect(page.getByRole("region", { name: "Suggested questions" }).locator("button.q")).toHaveCount(20);
  await expect(page.getByLabel("Your question")).toBeDisabled();
  await expect(page.getByRole("button", { name: "Ask", exact: true })).toBeDisabled();
});

test("ask -> stream -> evidence visible -> permalink works", async ({ page }) => {
  await pick(page, CORRECT_ABOVE);
  await expect(page.getByRole("region", { name: "SQL" })).toContainText("SELECT");
  await expect(page.getByRole("region", { name: "Result" }).getByRole("table")).toBeVisible();
  await expect(page.getByRole("region", { name: "Steps" })).toContainText("Submitted an answer");
  await expect(page.getByRole("region", { name: "Checks" })).toContainText("The query returned rows");
  const answer = page.getByRole("region", { name: "Answer" });
  await expect(answer).toContainText("Answered");
  await expect(answer).toContainText("In the evaluation this answer was scored");
  const meter = page.getByRole("region", { name: "Confidence" });
  await expect(meter).toContainText("Calibrated confidence");
  await expect(meter).toContainText("held-out benchmark questions");

  await expect(page).toHaveURL(new RegExp(`#/run/${CORRECT_ABOVE}$`));
  const sql = await page.getByRole("region", { name: "SQL" }).locator("pre").innerText();
  await page.goto("/");
  await page.goto(`/#/run/${CORRECT_ABOVE}`);
  await page.reload();
  await expect(page.getByRole("region", { name: "SQL" }).locator("pre")).toHaveText(sql);
  await expect(page.getByRole("region", { name: "Confidence" })).toBeVisible();
});

test("a run under the decline threshold is held back, with its evidence still shown", async ({ page }) => {
  await pick(page, WITHHELD);
  const answer = page.getByRole("region", { name: "Answer" });
  await expect(answer).toContainText("Held back");
  await expect(page.getByRole("region", { name: "Confidence" })).toContainText("holds this answer back");
  await expect(page.getByRole("region", { name: "SQL" })).toBeVisible();
});

test("a declined run says what is missing and shows no confidence meter", async ({ page }) => {
  await pick(page, DECLINED);
  const answer = page.getByRole("region", { name: "Answer" });
  await expect(answer).toContainText("Declined");
  await expect(answer.locator(".callout").first()).toContainText(/\S{4,}/);
  await expect(page.getByRole("region", { name: "Confidence" })).toHaveCount(0);
});

test("a guarded comparison shows its intervals and the association caveat", async ({ page }) => {
  await pick(page, GUARDED);
  const stats = page.getByRole("region", { name: "Statistics" });
  await expect(stats).toContainText("95% intervals");
  await expect(stats).toContainText("interval");
  await expect(stats).toContainText("not proof of cause");
  await expect(page.getByRole("region", { name: "Confidence" })).toContainText("No confidence figure");
});

test("keyboard: a suggestion can be reached and opened without a mouse", async ({ page }) => {
  const first = page.getByRole("region", { name: "Suggested questions" }).locator("button.q").first();
  await first.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("region", { name: "Answer" })).toContainText(/Answered|Held back|Declined|clarification/);
});

test("the theme can be switched and is remembered", async ({ page }) => {
  await page.getByRole("button", { name: /Switch between light and dark/ }).click();
  const first = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
  expect(["light", "dark"]).toContain(first);
  await page.reload();
  expect(await page.evaluate(() => document.documentElement.getAttribute("data-theme"))).toBe(first);
});

test("an unknown permalink says so", async ({ page }) => {
  await page.goto("/#/run/not-a-run");
  await expect(page.getByRole("alert")).toContainText("No run with this link");
});
test("the result is stated from the rows, and the answer can be copied as Markdown", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await pick(page, CORRECT_ABOVE);
  const answer = page.getByRole("region", { name: "Answer" });
  await expect(answer.locator(".resultline")).toContainText("rows returned");
  await answer.getByRole("button", { name: /Copy the answer and its evidence as Markdown/ }).click();
  await expect(answer.getByRole("button", { name: /Copy the answer/ })).toHaveText("Copied");
  const md = await page.evaluate(() => navigator.clipboard.readText());
  expect(md).toContain("```sql");
  expect(md).toContain(`/#/run/${CORRECT_ABOVE}`);
});

test("a held-back answer's text is behind a link, and no review notes are shown", async ({ page }) => {
  await pick(page, WITHHELD);
  const answer = page.getByRole("region", { name: "Answer" });
  await expect(answer.getByText("Show the answer it would have given")).toBeVisible();
  await pick(page, GUARDED);
  await expect(page.getByRole("region", { name: "Answer" })).not.toContainText("gold query");
});
test("the twenty questions are in four pages of five that can be paged with arrows and dots", async ({ page }) => {
  const region = page.getByRole("region", { name: "Suggested questions" });
  await expect(region).toContainText("Page 1 of 4 · 20 questions");
  await expect(region).toContainText("Benchmark questions");
  await expect(region.getByRole("button", { name: "Previous page" })).toBeDisabled();
  await region.getByRole("button", { name: "Next page" }).click();
  await expect(region).toContainText("Page 2 of 4");
  await region.getByRole("tab", { name: "Go to page 4" }).click();
  await expect(region).toContainText("Page 4 of 4");
  await expect(region).toContainText("Banking questions · Comparisons and causes");
  await expect(region.getByRole("button", { name: "Next page" })).toBeDisabled();
  // a run on the last page opens, and following a permalink to it moves the pager there
  await pick(page, GUARDED);
  await expect(page.getByRole("region", { name: "Statistics" })).toBeVisible();
  await page.goto(`/#/run/${CORRECT_ABOVE}`);
  await page.reload();
  await expect(region).toContainText("Page 1 of 4");
});

test("the confidence meter draws the held-out interval on a slider and explains the calibration", async ({ page }) => {
  await pick(page, CORRECT_ABOVE);
  const meter = page.getByRole("region", { name: "Confidence" });
  await expect(meter.locator(".record .span")).toBeVisible();
  await expect(meter.locator(".record")).toHaveAttribute("aria-label", /95% interval \d+% to \d+%/);
  await expect(meter.locator(".detail")).not.toContainText("interval");
  await meter.getByText("How is this calculated?").click();
  await expect(meter).toContainText("Platt scaling");
  await expect(meter).toContainText("150 calibration questions");
});
test("the confidence gauge's fade flows left to right, in the green and in the red tones", async ({ page }) => {
  for (const [id, cls] of [[CORRECT_ABOVE, "ok"], [WITHHELD, "low"]] as const) {
    await page.goto(`/#/run/${id}`);
    const fill = page.getByRole("region", { name: "Confidence" }).locator(`.gauge .fill.${cls}`);
    await expect(fill).toBeVisible();
    expect(await fill.evaluate((el) => getComputedStyle(el).animationName)).toBe("flow");
    // The pattern is twice the bar's width, so a position of p% puts it p% of a bar-width to the
    // left: a falling percentage carries the fade to the right. Sampled twice, 250 ms apart, the
    // percentage must have fallen by less than half a period (200 points), allowing for one wrap.
    const read = () => fill.evaluate((el) => parseFloat(getComputedStyle(el).backgroundPositionX));
    const a = await read();
    await page.waitForTimeout(250);
    const b = await read();
    const fell = (((a - b) % 200) + 200) % 200;
    expect(fell).toBeGreaterThan(0);
    expect(fell).toBeLessThan(100);  }
});
test("the rows it used come before the steps and the query", async ({ page }) => {
  await pick(page, CORRECT_ABOVE);
  const top = async (name: string) => (await page.getByRole("region", { name }).boundingBox())!.y;
  const rows = await top("Result");
  expect(rows).toBeLessThan(await top("Steps"));
  expect(rows).toBeLessThan(await top("SQL"));
  expect(await top("Answer")).toBeLessThan(rows);
});
test("a wrong answer shows the expected result beside its own, and why it was scored wrong", async ({ page }) => {
  await pick(page, "bench-533");
  const why = page.getByRole("region", { name: "Why this was scored wrong" });
  await expect(why).toBeVisible();
  await expect(why.getByText("The analyst's result")).toBeVisible();
  await expect(why.getByText("The expected result")).toBeVisible();
  await expect(why).toContainText("It returned 5146; the expected answer is 4941.");
  await expect(why).toContainText("time of day");
  await why.getByText("Show the expert's query and the benchmark's hint").click();
  await expect(why).toContainText("DATE(LastAccessDate)");
  // a right answer has no such panel
  await pick(page, CORRECT_ABOVE);
  await expect(page.getByRole("region", { name: "Why this was scored wrong" })).toHaveCount(0);
});

test("the rows it used are aligned to the left, numbers included", async ({ page }) => {
  await pick(page, "bench-829");
  const cell = page.getByRole("region", { name: "Result" }).locator("td").first();
  expect(await cell.evaluate((el) => getComputedStyle(el).textAlign)).toBe("left");
  const head = page.getByRole("region", { name: "Result" }).locator("th").first();
  expect(await head.evaluate((el) => getComputedStyle(el).textAlign)).toBe("left");
});