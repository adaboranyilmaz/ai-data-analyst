"""Load the replay service and record how it holds up.

Against a running service in replay mode (`docker compose up -d --wait`, with
ANALYST_REPLAY_SPEED set so a replayed run takes milliseconds, not its recorded seconds). For
each of three routes and each concurrency level, a fixed number of requests spread over the
curated runs: the latency percentiles, the requests per second and the errors. A streamed answer
counts as served only if its stream ends with a `done` event.

The numbers are one machine's: the service in a container on a laptop, the client on the same
laptop. They show the shape (where latency starts to climb and whether anything fails), not a
capacity for another machine. Writes results/metrics/load_test.json.

Usage:
    ANALYST_REPLAY_SPEED=100 docker compose up -d --wait
    uv run python scripts/96_load_test.py --replay-speed 100
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.bird import write_json  # noqa: E402

OUT = ROOT / "results/metrics/load_test.json"
LEVELS = (1, 8, 32)
REQUESTS = 200  # per route and level


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def one(client: httpx.Client, route: str, run_id: str) -> tuple[float, bool]:
    start = time.perf_counter()
    try:
        if route == "GET /api/meta":
            ok = client.get("/api/meta").status_code == 200
        elif route == "GET /runs/{id}":
            ok = client.get(f"/runs/{run_id}").status_code == 200
        else:  # POST /ask
            with client.stream("POST", "/ask", json={"run_id": run_id}) as r:
                body = "".join(r.iter_text())
                ok = r.status_code == 200 and "event: done" in body
    except httpx.HTTPError:
        ok = False
    return (time.perf_counter() - start) * 1000, ok


def measure(base: str, route: str, ids: list[str], workers: int) -> dict:
    jobs = [ids[i % len(ids)] for i in range(REQUESTS)]
    with httpx.Client(base_url=base, timeout=120) as client:
        started = time.perf_counter()
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(lambda rid: one(client, route, rid), jobs))
        elapsed = time.perf_counter() - started
    ms = [t for t, _ in results]
    return {
        "route": route,
        "concurrency": workers,
        "requests": len(results),
        "errors": sum(not ok for _, ok in results),
        "p50_ms": round(statistics.median(ms), 1),
        "p95_ms": round(percentile(ms, 0.95), 1),
        "p99_ms": round(percentile(ms, 0.99), 1),
        "requests_per_second": round(len(results) / elapsed, 1),
    }


def container_memory() -> str | None:
    try:
        out = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.MemUsage}}",
                "ai-data-analyst-api-1",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--replay-speed", type=float, required=True, help="the service's, recorded")
    args = ap.parse_args()
    meta = httpx.get(f"{args.base}/api/meta", timeout=30).json()
    if meta["mode"] != "replay":
        sys.exit("load test the replay service only: a live run costs money")
    ids = [e["id"] for e in meta["suggested"]]
    rows = [
        measure(args.base, route, ids, workers)
        for route in ("GET /api/meta", "GET /runs/{id}", "POST /ask")
        for workers in LEVELS
    ]
    doc = {
        "note": "one machine: the service in a container on a laptop, the client on the same "
        "laptop; the shape, not a capacity for another machine. A replayed run is paced to its "
        "recorded timing divided by the replay speed.",
        "replay_speed": args.replay_speed,
        "requests_per_row": REQUESTS,
        "runs": len(ids),
        "container_memory_after": container_memory(),
        "results": rows,
        "errors_total": sum(r["errors"] for r in rows),
    }
    write_json(OUT, doc)
    for r in rows:
        print(
            f"{r['route']:<16} x{r['concurrency']:<3} p50 {r['p50_ms']:>8} ms  p95 {r['p95_ms']:>8}"
            f"  p99 {r['p99_ms']:>8}  {r['requests_per_second']:>7} req/s  errors {r['errors']}"
        )
    print(f"memory {doc['container_memory_after']}; wrote {OUT.relative_to(ROOT)}")
    sys.exit(0 if doc["errors_total"] == 0 else 1)


if __name__ == "__main__":
    main()
