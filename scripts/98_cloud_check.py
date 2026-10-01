"""Check a deployed replay service and record what it did (the cloud deployment's results).

Given the deployment's address, it checks health and readiness, that the page and its metadata are
served, that a recorded run streams to its `done` event with the same answer the committed record
holds, that a free-text question is refused as the replay service refuses it, and the latency of
repeated asks. The first request is timed on its own: a container app scaled to zero starts on
demand, so that is the cold start. Uses only the standard library. Writes
results/metrics/cloud_deploy.json (no address: the deployment is destroyed afterwards).

Usage:
    uv run python scripts/98_cloud_check.py --url https://<the app's address> --region westeurope \
        [--cost-usd 0.12]   # the cost the portal reported, if the author has it
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.bird import write_json  # noqa: E402

OUT = ROOT / "results/metrics/cloud_deploy.json"
RUN = "bench-782"
ASKS = 20


def call(url: str, body: dict | None = None, timeout: int = 120) -> tuple[int, str, float]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(), time.perf_counter() - start
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), time.perf_counter() - start


def final_answer(stream: str) -> dict | None:
    for block in stream.strip().split("\n\n"):
        if block.startswith("event: answer"):
            return json.loads(block.split("data: ", 1)[1])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--cost-usd", type=float, default=None)
    args = ap.parse_args()
    base = args.url.rstrip("/")

    # the first request to an app scaled to zero starts it: that is the cold start
    status, body, cold = call(f"{base}/health")
    cold_start = {"status": status, "seconds": round(cold, 1)}
    checks = {"health": status == 200 and json.loads(body)["status"] == "ok"}

    status, body, _ = call(f"{base}/ready")
    checks["ready"] = status == 200 and json.loads(body)["ready"] is True

    status, body, _ = call(f"{base}/")
    checks["page_served"] = status == 200 and "<div id=" in body

    status, body, _ = call(f"{base}/api/meta")
    meta = json.loads(body) if status == 200 else {}
    checks["replay_mode"] = meta.get("mode") == "replay" and not meta.get("local_mode", True)
    runs = len(meta.get("suggested", []))

    committed = json.loads((ROOT / "results/demo/runs" / f"{RUN}.json").read_text(encoding="utf-8"))
    times = []
    same = True
    for _ in range(ASKS):
        status, stream, t = call(f"{base}/ask", {"run_id": RUN})
        times.append(t * 1000)
        got = final_answer(stream)
        same &= (
            status == 200
            and "event: done" in stream
            and got is not None
            and got["sql"] == committed["answer"]["sql"]
        )
    checks["recorded_run_matches_committed_answer"] = same

    status, _, _ = call(f"{base}/ask", {"question": "A question nobody recorded?"})
    checks["unrecorded_question_refused"] = status == 404
    status, _, _ = call(f"{base}/connections", {"host": "x"})
    checks["connections_not_available"] = status in (404, 405, 403, 409, 422)

    ordered = sorted(times)
    doc = {
        "note": "a deployed replay service, checked from outside; one run from one client; the "
        "deployment was destroyed afterwards",
        "region": args.region,
        "recorded_runs_served": runs,
        "cold_start": cold_start,
        "asks": {
            "requests": ASKS,
            "p50_ms": round(statistics.median(times), 1),
            "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1),
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "cost_usd_reported_by_the_portal": args.cost_usd,
    }
    write_json(OUT, doc)
    print(json.dumps(doc, indent=1))
    sys.exit(0 if doc["all_checks_passed"] else 1)


if __name__ == "__main__":
    main()
