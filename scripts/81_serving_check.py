"""Serving check: every curated run, as the service serves it, equals what was evaluated.

For each run in results/demo: the evidence record against the evaluated record it came from, the
replayed stream against the evidence record, and the API's stream against it too. Writes
results/metrics/serving_check.json. Needs no database and makes no model call.

Usage:
    uv run python scripts/81_serving_check.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["ANALYST_REPLAY_SPEED"] = "1000000"

from fastapi.testclient import TestClient  # noqa: E402

from src.serving import check, replay  # noqa: E402
from src.serving.app import Settings, create_app  # noqa: E402
from src.serving.meter import Meter, config  # noqa: E402
from src.serving.store import RunStore  # noqa: E402

OUT = ROOT / "results/metrics/serving_check.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def parse_sse(text: str) -> list[dict]:
    return [
        json.loads(line.split("data: ", 1)[1])
        for block in text.strip().split("\n\n")
        for line in block.split("\n")
        if line.startswith("data: ")
    ]


def main() -> None:
    cfg = config()
    c = cfg["curated"]
    meter = Meter.load(cfg)
    store = RunStore(ROOT / c["out_dir"])
    main_rec = {
        str(r["question_id"]): r
        for r in read_jsonl(
            ROOT / f"results/runs/{c['held_out']['stage']}/{c['held_out']['run']}.jsonl"
        )
    }
    bank_rec = {
        str(r["question_id"]): r
        for r in read_jsonl(
            ROOT / f"results/runs/{c['banking']['stage']}/{c['banking']['run']}.jsonl"
        )
    }
    guard_rec = {r["id"].split(":")[-1]: r for r in read_jsonl(ROOT / c["guardrail"]["records"])}
    client = TestClient(create_app(Settings(mode="replay", request_log=None)))

    runs = []
    for entry in store.index():
        ev = store.get(entry["id"])
        key = entry["id"].split("-", 1)[1]
        record = {"benchmark": main_rec, "banking": bank_rec, "guardrail": guard_rec}[
            ev["kind"]
        ].get(key)
        stream = [e for e, _ in replay.events(ev, cfg["replay"])]
        api = parse_sse(client.post("/ask", json={"run_id": ev["id"]}).text)
        diffs = {
            "evidence_vs_evaluated": check.against_record(ev, record, meter),
            "stream_vs_evidence": check.stream_against_record(stream, ev),
            "api_vs_evidence": check.stream_against_record(api, ev),
        }
        runs.append(
            {
                "id": ev["id"],
                "kind": ev["kind"],
                "identical": not any(diffs.values()),
                "differences": {
                    k: {f: list(v) for f, v in d.items()} for k, d in diffs.items() if d
                },
            }
        )

    result = {
        "note": "every curated run, as served, compared field by field with the evaluated record "
        "it came from and with its own replayed and API streams; no model call, no database",
        "runs": len(runs),
        "identical": sum(r["identical"] for r in runs),
        "all_identical": all(r["identical"] for r in runs),
        "per_run": runs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{result['identical']} of {result['runs']} runs identical -> {OUT.relative_to(ROOT)}")
    if not result["all_identical"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
