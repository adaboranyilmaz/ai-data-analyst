"""Classify questions as statistical or descriptive: keyword rules and a model, combined by OR.

Main run: the banking set, all 500 benchmark questions and the planted questions; `--probe`: the
first pilot questions, to check the prompt's format before it is frozen (src/stats/classify.py,
configs/guardrail.yaml `classify`, `calls.classify`).

Usage:
    uv run python scripts/73_classify.py [--probe]
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
    questions = gr.classify_questions(cfg, probe=a.probe)
    records = gr.run_classify(questions, cfg, probe=a.probe)
    out = ROOT / (
        f"{cfg['stages']['probes']['dir']}/classify.jsonl"
        if a.probe
        else cfg["stages"]["classify"]["records"]
    )
    gr.write_jsonl(out, records)
    flagged = sum(r["statistical"] for r in records)
    cost = sum(r["cost_usd"] for r in records)
    print(f"{len(records)} classified, {flagged} statistical; ${cost:.4f}")


if __name__ == "__main__":
    main()
