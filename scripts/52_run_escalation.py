"""Run the winning design on the escalation model (configs/confidence.yaml `escalation`).

Each configured run answers one question set with the winning design on Claude Opus 5.5, at the
effort level the config states: the pilot's first ten (to measure the output size, thinking
included, before the rest is sent) and the ablation set, paired with Claude Sonnet 5's run there.
Every call goes through the escalation stage's response cache and the spend ledger's phase cap;
with ANALYST_REPLAY_ONLY=1 the stage is rebuilt from the cache at no cost.

Writes results/runs/escalation/<run>.jsonl, and traces and spans under data/.

Usage:
    uv run python scripts/52_run_escalation.py [--sets pilot ablation]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.agent.confidence import confidence_config  # noqa: E402
from src.agent.escalation import agent_config_with, execute  # noqa: E402
from src.agent.stages import WINNER, make_spec, winner  # noqa: E402
from src.eval.records import answered_correct  # noqa: E402

STAGE = "escalation"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sets", nargs="*", help="only the runs on these question sets")
    a = p.parse_args()

    load_dotenv(ROOT / ".env")  # the API key, for live calls
    conf = confidence_config()
    esc = conf["escalation"]
    cfg = agent_config_with(esc["model"], esc["settings"])
    design = winner() if esc["design"] == WINNER else esc["design"]
    specs = [
        make_spec(
            STAGE,
            r["set"],
            design,
            esc["model"],
            esc["evidence"],
            esc["mode"],
            r.get("limit"),
            phase=conf["phase"],
        )
        for r in esc["runs"]
        if not a.sets or r["set"] in a.sets
    ]
    for s in specs:
        if s.spans.exists():
            s.spans.unlink()  # spans describe this execution only
    print(f"escalation: {len(specs)} runs of {design} on {esc['model']}", flush=True)
    for s, records in zip(specs, execute(specs, cfg), strict=True):
        ex = sum(answered_correct(r) for r in records) / len(records)
        out = sum(r["tokens"]["output"] for r in records) / len(records)
        print(
            f"{s.name}: EX {ex:.3f} on {len(records)} (one run), "
            f"mean output {out:.0f} tokens, cost ${sum(r['cost_usd'] for r in records):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
