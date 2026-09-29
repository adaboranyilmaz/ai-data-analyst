"""Run one design and model over a question set, score it, and write its records.

For single runs outside a stage's configured list (smoke checks, a rerun of one arm); a stage's
runs are run together by scripts/42_run_stage.py. Paths and names are the stage's
(src/agent/stages.py). Every model call goes through the stage's response cache and, for the
Anthropic API, the spend ledger with its phase cap (configs/budget.yaml); with
ANALYST_REPLAY_ONLY=1 a missing response is an error instead of a call.

Usage:
    uv run python scripts/40_run_agent.py --stage pilot --set pilot --design d3 \
        --model claude-sonnet-5 [--no-evidence] [--mode batch] [--limit N]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.agent.evaluate import execute  # noqa: E402
from src.agent.run import config  # noqa: E402
from src.agent.stages import make_spec  # noqa: E402
from src.eval.records import answered_correct  # noqa: E402

SETS = ("pilot", "ablation", "held_out", "all", "own")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stage", required=True, help="pilot, ablation, main, own")
    p.add_argument("--set", required=True, choices=SETS)
    p.add_argument("--design", required=True, choices=sorted(config()["designs"]))
    p.add_argument("--model", required=True, choices=sorted(config()["models"]))
    p.add_argument("--no-evidence", action="store_true", help="leave out the benchmark's hint")
    p.add_argument("--mode", choices=("direct", "batch"), default="direct")
    p.add_argument("--limit", type=int, help="only the first N questions (smoke runs)")
    p.add_argument("--phase", default="phase4", help="the ledger phase whose cap applies")
    a = p.parse_args()

    load_dotenv(ROOT / ".env")  # the API key, for live calls
    spec = make_spec(a.stage, a.set, a.design, a.model, not a.no_evidence, a.mode, a.limit, a.phase)
    if spec.spans.exists():
        spec.spans.unlink()  # spans describe this execution only
    print(f"{spec.name}: {len(spec.items)} questions, mode {a.mode}")
    records = execute(spec)

    scored = [r for r in records if r["correct"] is not None]
    ex = sum(answered_correct(r) for r in scored) / len(scored) if scored else float("nan")
    cost = sum(r["cost_usd"] for r in records)
    conf = statistics.mean(r["confidence"] for r in records)
    errors = sum(bool(r["errors"]) for r in records)
    print(
        f"EX {ex:.3f} on {len(scored)} scored (one run); mean confidence {conf:.3f}; "
        f"declined {sum(r['declined'] for r in records)}; questions with errors {errors}; "
        f"cost ${cost:.4f} (as priced from token usage; cached calls included)"
    )
    print(f"wrote {spec.records.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
