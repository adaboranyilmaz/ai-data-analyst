"""Plan the analysis of every statistical question: the query and the analysis, written by the
analyst from the schema as design 1 reads it.

Main run: the banking set's questions the classifier flagged, and every planted question (they
are statistical by construction; their classification is measured on its own); `--probe`: the
probe questions, to check the prompt's format before it is frozen.

Usage:
    uv run python scripts/74_plans.py [--probe]
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
    if a.probe:
        questions = gr.probe_questions(cfg)
        out = ROOT / f"{cfg['stages']['probes']['dir']}/plans.jsonl"
    else:
        classified = gr.read_jsonl(ROOT / cfg["stages"]["classify"]["records"])
        questions = [
            {k: r[k] for k in ("id", "source", "db_id", "category", "question")}
            for r in classified
            if (r["source"] == "own" and r["statistical"]) or r["source"] == "planted"
        ]
        out = ROOT / cfg["stages"]["plans"]["records"]
    records = gr.run_plans(questions, cfg, probe=a.probe)
    gr.write_jsonl(out, records)
    ok = sum(r["plan"] is not None for r in records)
    print(f"{len(records)} planned, {ok} usable; ${sum(r['cost_usd'] for r in records):.4f}")
    for r in records:
        if r["plan"] is None:
            print(f"  {r['id']}: {r['plan_error']}")


if __name__ == "__main__":
    main()
