import { defineConfig } from "@playwright/test";

// The static demo, served from a path by a file server with no API: `npm run build:static` and
// `uv run python scripts/95_static_site.py` first.
export default defineConfig({
  testDir: "e2e-static",
  timeout: 30_000,
  use: { baseURL: "http://127.0.0.1:8766/analyst/", trace: "retain-on-failure" },
  webServer: {
    command: "node e2e-static/serve.mjs",
    url: "http://127.0.0.1:8766/analyst/",
    reuseExistingServer: false,
    timeout: 30_000,
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
