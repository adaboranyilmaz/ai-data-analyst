"""The statistical guardrail's results, from the recorded runs: which questions are statistical,
the banking set's comparative and causal questions before and after, and the planted effects at
both levels, each prediction checked against what was found.

Reads the classifier, plan and banking-set records (scripts/73-75), the planted records
(scripts/76), the winning design's banking-set run (its answers are the "before"), and the close
reading in results/reviews/guardrail_review.yaml. Writes results/metrics/guardrail.json and
results/metrics/planted_effects.json. No model is called.

Usage:
    uv run python scripts/77_guardrail_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.bird import write_json  # noqa: E402
from src.stats import guardrail as gr  # noqa: E402
from src.stats import report  # noqa: E402
from src.stats.calls import config  # noqa: E402

REVIEW = ROOT / "results/reviews/guardrail_review.yaml"
LEDGER = ROOT / "results/metrics/api_spend.json"
GUARDRAIL = ROOT / "results/metrics/guardrail.json"
PLANTED = ROOT / "results/metrics/planted_effects.json"
BOOTSTRAP_SEED = 20260927


def main() -> None:
    cfg = config()
    st = cfg["stages"]
    review = yaml.safe_load(REVIEW.read_text(encoding="utf-8"))
    classified = gr.read_jsonl(ROOT / st["classify"]["records"])
    plans = gr.read_jsonl(ROOT / st["plans"]["records"])
    own = gr.read_jsonl(ROOT / st["own"]["records"])
    level1 = gr.read_jsonl(ROOT / st["planted"]["level1"])
    level2 = gr.read_jsonl(ROOT / st["planted"]["level2"])

    cls = report.classification(classified, review)
    bank = report.banking(own, classified, review)
    cells = report.level1_cells(level1, cfg["planted"]["power"])
    checks = report.power_checks(cells)
    l2 = report.level2_summary(
        level2, BOOTSTRAP_SEED, reading=review["level2_causal_wording"]["causal_claims"]
    )
    preds = report.predictions(cls, bank, cells, checks, l2, review)

    budget = yaml.safe_load((ROOT / "configs/budget.yaml").read_text(encoding="utf-8"))
    spend = json.loads(LEDGER.read_text(encoding="utf-8"))["by_phase"].get(cfg["phase"], {})
    reading = {"status": review["status"], "reviewed_on": review["reviewed_on"]}
    planned = {
        r["id"]: {
            "plan": r["plan"],
            "error": r["plan_error"],
            "assumptions": (r["submitted"] or {}).get("assumptions"),
        }
        for r in plans
    }
    write_json(
        GUARDRAIL,
        {
            "close_reading": reading,
            "classification": cls,
            "banking_f": bank,
            "plans": planned,
            "predictions": preds,
            "spend": {
                "usd": round(spend.get("usd", 0.0), 4),
                "calls": spend.get("n_calls", 0),
                "cap_usd": budget["phase_caps_usd"][cfg["phase"]],
            },
        },
    )
    write_json(
        PLANTED,
        {
            "close_reading": {
                **reading,
                "level2_causal_wording": review["level2_causal_wording"]["status"],
            },
            "copies": cfg["planted"]["copies"],
            "power": cfg["planted"]["power"],
            "level1": {
                "cells": cells,
                "pooled": report.pooled(level1),
                "power_checks": checks,
                "headline": report.headline(level1),
                "confounder_strata": review["confounder_strata"],
            },
            "level2": {**l2, "bootstrap": {"resamples": 10_000, "seed": BOOTSTRAP_SEED}},
            "predictions": [p for p in preds if p["id"] in {f"G{i}" for i in range(6, 14)}],
        },
    )
    held = sum(p["held"] for p in preds)
    print(f"predictions held: {held} of {len(preds)} (close reading: {review['status']})")
    for p in preds:
        print(
            f"  {p['id']}: {'held' if p['held'] else 'missed'}  observed {p['observed']}"
            f"  range {p['range']}"
        )
    t = bank["totals"]
    print(
        f"banking f: before {t['before']['success']}/9, after {t['after']['success']}/9, "
        f"reviewed {t['after']['reviewed_success']}/9"
    )
    pr = l2["primary"]
    d = pr["difference"]
    print(
        f"level 2 primary: guarded {pr['guarded']:.3f}, numbers only {pr['numbers_only']:.3f}, "
        f"difference {d['estimate']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}] "
        f"on {pr['copies']} copies"
    )


if __name__ == "__main__":
    main()
