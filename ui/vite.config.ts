import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The API runs on 8000; in development the UI proxies to it.
const api = "http://127.0.0.1:8000";

// `vite build --mode static` builds the demo that needs no server: relative asset paths, so it
// works under any path (GitHub Pages serves it under /<repository>/), and it reads data/ files.
export default defineConfig(({ mode }) => ({
  base: mode === "static" ? "./" : "/",
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/ask": api,
      "/runs": api,
      "/api": api,
      "/connections": api,
      "/health": api,
    },
  },
  build: { outDir: "dist", sourcemap: false },
  test: { environment: "jsdom", include: ["src/**/*.test.{ts,tsx}"] },
}));
