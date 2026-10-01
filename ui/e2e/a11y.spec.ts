import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

// Automated accessibility audit (WCAG 2.1 A and AA rules) of the welcome screen and of runs in
// each state, in both themes. It finds what a machine can find (contrast, names, roles, landmarks);
// it does not replace a screen-reader pass.
const PAGES = [
  ["welcome", "/"],
  ["answered run", "/#/run/bench-782"],
  ["held-back run", "/#/run/bench-726"],
  ["declined run", "/#/run/bank-own-d03"],
  ["guarded comparison", "/#/run/guard-own-f02"],
  ["wrong answer with its explanation", "/#/run/bench-533"],
] as const;

for (const scheme of ["light", "dark"] as const) {
  for (const [name, url] of PAGES) {
    test(`${name} has no accessibility violations (${scheme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme });
      await page.goto(url);
      await expect(page.getByTestId("mode")).toBeVisible();
      if (url.includes("run/")) await expect(page.getByRole("region", { name: "Answer" })).toContainText(/Answered|Held back|Declined/);
      await page.waitForTimeout(600); // let the entrance animation finish: contrast is measured on settled colors
      const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
      const summary = results.violations.map((v) => `${v.id} (${v.impact}): ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
      expect(summary).toEqual([]);
    });
  }
}