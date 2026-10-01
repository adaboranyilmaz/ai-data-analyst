import { defineConfig } from "@playwright/test";

// The end-to-end tests run against the API in replay mode, serving the built UI, at a high replay
// speed. `npm run build` first; the web server below starts the API.
export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  use: { baseURL: "http://127.0.0.1:8765", trace: "retain-on-failure" },
  webServer: {
    command: "uv run python -m src.serving --port 8765",
    cwd: "..",
    url: "http://127.0.0.1:8765/health",
    reuseExistingServer: false,
    timeout: 60_000,
    env: { ANALYST_SERVING_MODE: "replay", ANALYST_REPLAY_SPEED: "200" },
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});