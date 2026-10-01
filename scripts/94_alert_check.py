"""Does the drift alert fire, through a real Prometheus, on shifted traffic, and clear afterwards?

Against the running stack (`docker compose --profile monitoring up -d --wait`, replay mode). The
service is restarted (an empty drift window). Traffic is then replayed through POST /ask:

1. a few answers, fewer than the window needs: the service says there is not enough to judge, and
   no drift series or alert exists;
2. the recorded run with the highest calibrated confidence asked over and over: a distribution
   piled into the top bin, far from the held-out one. The drift index passes the alert threshold
   and Prometheus moves the ConfidenceDrift alert from pending to firing;
3. the service is restarted: the window is empty, the series is gone, the alert resolves.

Replayed recorded runs stand in for live traffic here: this checks the pipeline from the
service's metric through Prometheus's rule to a firing alert, not how often real traffic would
shift. Writes results/metrics/alert_check.json (booleans and values; no timings).

Usage:
    ANALYST_REPLAY_SPEED=100 docker compose --profile monitoring up -d --wait
    uv run python scripts/94_alert_check.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.bird import write_json  # noqa: E402

API = "http://127.0.0.1:8000"
PROM = "http://127.0.0.1:9090"
OUT = ROOT / "results/metrics/alert_check.json"
ALERT = "ConfidenceDrift"
FEW, MANY = 20, 120


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def ask(run_id: str) -> None:
    req = urllib.request.Request(
        f"{API}/ask",
        data=json.dumps({"run_id": run_id}).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()


def ask_many(run_id: str, n: int) -> None:
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: ask(run_id), range(n)))


def restart() -> None:
    subprocess.run(["docker", "compose", "restart", "api"], cwd=ROOT, check=True)
    for _ in range(60):
        try:
            if get(f"{API}/health")["status"] == "ok":
                return
        except OSError:
            pass
        time.sleep(1)
    sys.exit("the service did not come back")


def alert_state() -> str | None:
    alerts = get(f"{PROM}/api/v1/alerts")["data"]["alerts"]
    return next((a["state"] for a in alerts if a["labels"]["alertname"] == ALERT), None)


def wait_for(check, seconds: int):
    end = time.time() + seconds
    while time.time() < end:
        got = check()
        if got:
            return got
        time.sleep(3)
    return None


def main() -> None:
    restart()
    suggested = get(f"{API}/api/meta")["suggested"]
    best, top = None, -1.0
    for e in suggested:
        c = get(f"{API}/runs/{e['id']}")["confidence"].get("calibrated")
        if c is not None and c > top:
            best, top = e["id"], c
    out: dict = {
        "note": "replayed recorded runs stand in for traffic; this checks the pipeline from the "
        "service's metric through Prometheus's rule to a firing alert",
        "alert": ALERT,
        "shifted_traffic": {"run_id": best, "calibrated_confidence": top},
    }

    ask_many(best, FEW)
    early = get(f"{API}/api/drift")
    out["before_the_window_is_full"] = {
        "answers": early["window"],
        "ready": early["ready"],
        "psi": early["psi"],
        "alert_state": alert_state(),
    }

    ask_many(best, MANY)
    shifted = get(f"{API}/api/drift")
    out["after_shifted_traffic"] = {
        "answers": shifted["window"],
        "ready": shifted["ready"],
        "psi": round(shifted["psi"], 4),
        "reading": shifted["psi_reading"],
    }
    fired = wait_for(lambda: alert_state() == "firing", 240)
    out["prometheus_alert_fired"] = bool(fired)

    restart()
    cleared = wait_for(lambda: alert_state() in (None, "inactive"), 120)
    out["alert_cleared_after_restart"] = bool(cleared)

    write_json(OUT, out)
    print(json.dumps(out, indent=1))
    ok = (
        out["before_the_window_is_full"]["alert_state"] in (None, "inactive")
        and out["before_the_window_is_full"]["psi"] is None
        and shifted["psi"] > shifted["alert"]
        and fired
        and cleared
    )
    print("alert check " + ("passed" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
