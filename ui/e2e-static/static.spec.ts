import { expect, test } from "@playwright/test";

// The static demo: the same recorded runs, from files, under a path, with nothing to ask.
const CORRECT_ABOVE = "bench-782";
const WITHHELD = "bench-726";

test("lists the recorded questions, and takes no new one", async ({ page }) => {
  await page.goto("./");
  await expect(page.getByTestId("mode")).toHaveText("replay of recorded runs");
  await expect(page.getByRole("region", { name: "Suggested questions" }).locator("button.q")).toHaveCount(20);
  await expect(page.getByLabel("Your question")).toBeDisabled();
});

test("a recorded run streams with its evidence, and its permalink reloads it", async ({ page }) => {
  const failed: string[] = [];
  page.on("response", (r) => r.status() >= 400 && failed.push(`${r.status()} ${r.url()}`));
  await page.goto("./");
  const meta = await (await page.request.get("data/meta.json")).json();
  const entry = meta.suggested.find((s: { id: string }) => s.id === CORRECT_ABOVE);
  await page.getByRole("button", { name: entry.question.slice(0, 40) }).click();
  await expect(page.getByRole("region", { name: "SQL" })).toContainText("SELECT", { timeout: 20_000 });
  await expect(page.getByRole("region", { name: "Answer" })).toContainText("Answered");
  await expect(page.getByRole("region", { name: "Confidence" })).toContainText("Calibrated confidence");
  await expect(page).toHaveURL(new RegExp(`#/run/${CORRECT_ABOVE}$`));
  const sql = await page.getByRole("region", { name: "SQL" }).locator("pre").innerText();
  await page.goto(`./#/run/${CORRECT_ABOVE}`);
  await page.reload();
  await expect(page.getByRole("region", { name: "SQL" }).locator("pre")).toHaveText(sql);
  expect(failed).toEqual([]);
});

test("a run under the decline threshold is held back", async ({ page }) => {
  await page.goto(`./#/run/${WITHHELD}`);
  await expect(page.getByRole("region", { name: "Answer" })).toContainText("Held back");
});
