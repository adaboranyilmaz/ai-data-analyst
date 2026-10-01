"""Time a fresh checkout from `docker compose up` to a first recorded answer.

Run it in a fresh clone (the clone itself is timed by the caller). It starts the service, waits
until it is ready, asks for a recorded run and waits for the stream's `done` event, timing each
step, then stops the service and removes its volumes. Uses only the standard library, so a
clone needs nothing but Docker and Python. Writes results/metrics/first_answer.json in the clone.

Usage (PowerShell, in a fresh clone):
    $clone = (Measure-Command { git clone <url> analyst }).TotalSeconds
    cd analyst
    python scripts/97_first_answer.py --clone-seconds $clone
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "http://127.0.0.1:8000"
RUN = "bench-782"
OUT = ROOT / "results/metrics/first_answer.json"


def compose(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone-seconds", type=float, default=None)
    args = ap.parse_args()
    t0 = time.perf_counter()
    try:
        compose("up", "-d", "--build", "--wait", "api")
        up = time.perf_counter()
        with urllib.request.urlopen(f"{API}/ready", timeout=30) as r:
            ready = json.load(r)["ready"]
        req = urllib.request.Request(
            f"{API}/ask",
            data=json.dumps({"run_id": RUN}).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read().decode()
        done = time.perf_counter()
    finally:
        compose("down", "-v")
    answered = "event: done" in body and ready
    doc = {
        "note": "a fresh checkout, `docker compose up --build` on the service, then one recorded "
        "run asked for; the machine's own docker cache decides how much of the build is rebuilt",
        "run": RUN,
        "answered": answered,
        "clone_seconds": args.clone_seconds,
        "compose_up_seconds": round(up - t0, 1),
        "first_answer_seconds": round(done - up, 1),
        "total_seconds_after_clone": round(done - t0, 1),
        "machine": {"os": platform.platform(), "python": platform.python_version()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(doc, indent=1))
    sys.exit(0 if answered else 1)


if __name__ == "__main__":
    main()
