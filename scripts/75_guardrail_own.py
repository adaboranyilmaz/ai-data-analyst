"""The guarded answers on the banking set: each planned question's rows pulled through the
guards, analyzed in the sandbox, and answered from the result, with and without the statistics.

Main run: the banking set (its descriptive questions are recorded as such); `--probe`: the
probe questions, to check the answer prompt's format before it is frozen. Needs the sandbox image
(scripts/70_build_sandbox.py) and the benchmark loaded.

Usage:
    uv run python scripts/75_guardrail_own.py [--probe]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.stats import guardrail as gr  # noqa: E402
from src.stats.calls import config, require_protocol  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--probe", action="store_true", help="the format check (not evaluated)")
    a = p.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = config()
    require_protocol(cfg)
    probes = ROOT / cfg["stages"]["probes"]["dir"]
    if a.probe:
        plans = gr.read_jsonl(probes / "plans.jsonl")
        classified = [{**r, "statistical": True} for r in plans]
        out = probes / "answers.jsonl"
    else:
        plans = [
            r
            for r in gr.read_jsonl(ROOT / cfg["stages"]["plans"]["records"])
            if r["source"] == "own"
        ]
        classified = [
            r
            for r in gr.read_jsonl(ROOT / cfg["stages"]["classify"]["records"])
            if r["source"] == "own"
        ]
        out = ROOT / cfg["stages"]["own"]["records"]
    records = gr.run_own(plans, classified, cfg, probe=a.probe)
    gr.write_jsonl(out, records)
    answered = [r for r in records if "guarded" in r]
    print(
        f"{len(records)} questions, {len(answered)} answered on the statistical path; "
        f"${sum(r.get('cost_usd', 0) for r in records):.4f}"
    )


if __name__ == "__main__":
    main()
