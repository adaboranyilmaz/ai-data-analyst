// Serves dist-static under /analyst/ the way GitHub Pages serves a project site: from a path,
// by a plain file server with no API behind it. Anything outside the path or the files is a 404.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const root = new URL("../dist-static/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".woff2": "font/woff2", ".woff": "font/woff", ".svg": "image/svg+xml" };

createServer(async (req, res) => {
  const path = new URL(req.url, "http://x").pathname;
  if (!path.startsWith("/analyst/")) return res.writeHead(404).end("not found");
  const rel = normalize(path.slice("/analyst/".length) || "index.html").replace(/^(\.\.[\/])+/, "");
  try {
    const body = await readFile(join(root, rel.endsWith("/") ? rel + "index.html" : rel));
    res.writeHead(200, { "content-type": types[extname(rel)] ?? "application/octet-stream" }).end(body);
  } catch {
    res.writeHead(404).end("not found");
  }
}).listen(8766, "127.0.0.1");
