"""Run a stage's runs as configured (configs/agent.yaml `stages`), score them, write the records.

An arm whose design is `winner` runs the design chosen on the ablation set
(results/metrics/ablation.json, written by scripts/43_ablation_report.py).

Batched arms run together (one set of batches per round, src/agent/evaluate.py `execute_many`);
direct arms run one after another. Every model call goes through the stage's response cache and,
for the Anthropic API, the spend ledger and its caps; with ANALYST_REPLAY_ONLY=1 the stage is
rebuilt from the cache at no cost, and a missing response is an error.

Usage:
    uv run python scripts/42_run_stage.py --stage ablation [--models M ...] [--designs D ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.agent.evaluate import execute, execute_many  # noqa: E402
from src.agent.run import config  # noqa: E402
from src.agent.stages import make_spec, resolve_arms  # noqa: E402
from src.eval.records import answered_correct  # noqa: E402


def summary(records: list[dict]) -> str:
    scored = [r for r in records if r["correct"] is not None]
    ex = sum(answered_correct(r) for r in scored) / len(scored) if scored else float("nan")
    cost = sum(r["cost_usd"] for r in records)
    return f"EX {ex:.3f} on {len(scored)} (one run), cost ${cost:.4f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stage", required=True)
    p.add_argument("--models", nargs="*", help="only these models' arms")
    p.add_argument("--designs", nargs="*", help="only these designs' arms")
    a = p.parse_args()

    load_dotenv(ROOT / ".env")  # the API key, for live calls
    stages = config()["stages"]
    resolved, notes = resolve_arms(stages[a.stage], stages["ablation"])
    for note in notes:
        print(note, flush=True)
    arms = [
        arm
        for arm in resolved
        if (not a.models or arm["model"] in a.models)
        and (not a.designs or arm["design"] in a.designs)
    ]
    specs = [
        make_spec(
            a.stage,
            arm["set"],
            arm["design"],
            arm["model"],
            arm["evidence"],
            arm["mode"],
            arm.get("limit"),
        )
        for arm in arms
    ]
    for s in specs:
        if s.spans.exists():
            s.spans.unlink()  # spans describe this execution only
    batched = [s for s in specs if s.mode == "batch"]
    direct = [s for s in specs if s.mode != "batch"]
    print(f"stage {a.stage}: {len(batched)} batched and {len(direct)} direct runs", flush=True)
    if batched:
        for s, records in zip(batched, execute_many(batched), strict=True):
            print(f"{s.name}: {summary(records)}", flush=True)
    for s in direct:
        print(f"{s.name}: {summary(execute(s))}", flush=True)


if __name__ == "__main__":
    main()
