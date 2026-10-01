// Screenshots of the same recorded run in both tracing tools (the tracing profile must be up and
// scripts/85_export_traces.py must have sent the run to both). Usage: node tools/trace_screenshots.mjs <trace id>
import { chromium } from "@playwright/test";

const id = process.argv[2];
if (!id) throw new Error("trace id");
const out = new URL("../../results/plots/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1500, height: 900 } });

await page.goto(`http://127.0.0.1:5000/#/experiments/1/traces?selectedEvaluationId=tr-${id}`);
await page.waitForTimeout(6000);
// the trace opens as a panel over the left of the page; the assistant sits to its right
await page.screenshot({ path: out + "trace_mlflow.png", clip: { x: 0, y: 0, width: 1048, height: 900 } });

await page.goto("http://127.0.0.1:3000/auth/sign-in");
await page.getByLabel("Email").fill("analyst@example.com");
await page.getByLabel("Password").fill("analyst-local");
await page.getByRole("button", { name: /sign in/i }).click();
await page.waitForURL((u) => !u.pathname.startsWith("/auth"), { timeout: 30000 });
await page.goto(`http://127.0.0.1:3000/project/analyst/traces/${id}`);
await page.waitForTimeout(6000);
await page.screenshot({ path: out + "trace_langfuse.png" });
await browser.close();
