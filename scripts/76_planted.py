"""The planted-effect checks: does the guardrail find effects that are there, and report none
where there is none?

Level 1: every planted question on `planted.copies.level1` copies per condition, with the
reference plan(s) and the analyst's plan (scripts/74_plans.py); the sandbox's verdict per copy
($0: no model call). Level 2: on the first `planted.copies.level2` copies, the answer written from
the analyst plan's result, with the statistics and on the numbers alone (paid): what
the analyst says, not only what the test says. Writes results/runs/planted/level1.jsonl and
level2.jsonl (src/stats/planted.py explains the copies and the effects).

Usage:
    uv run python scripts/76_planted.py                 # both levels
    uv run python scripts/76_planted.py --level1-only   # no model call
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.stats import guardrail as gr  # noqa: E402
from src.stats import planted as pl  # noqa: E402
from src.stats.answer import answer_text, checks  # noqa: E402
from src.stats.calls import Item, config, require_protocol, run_calls  # noqa: E402
from src.stats.plans import Plan  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--level1-only", action="store_true", help="no model call")
    a = p.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = config()
    require_protocol(cfg)
    pcfg = cfg["planted"]
    st = cfg["stages"]["planted"]
    info = pl.setup_database(pcfg)
    print(f"copy database: {info['rows']['loan']} loans, copied now: {info['copied_now']}")
    units = pl.Units.load()

    analyst: dict[str, Plan] = {}
    plans_path = ROOT / cfg["stages"]["plans"]["records"]
    for r in gr.read_jsonl(plans_path) if plans_path.exists() else []:
        if r["source"] == "planted" and r["plan"] is not None:
            analyst[r["id"].split(":", 1)[1]] = Plan.from_dict(r["plan"])
    print(f"analyst plans: {sorted(analyst)}")

    keep = 0 if a.level1_only else pcfg["copies"]["level2"]
    records, sizes, kept = pl.run_level1(
        pcfg, units, analyst, pcfg["copies"]["level1"], keep=keep, clock_dir=ROOT / st["cache"]
    )
    for r in records:
        if r.get("template") in sizes and r["condition"] in sizes[r["template"]]:
            r["planted_effect"] = sizes[r["template"]][r["condition"]]["effect"]
    gr.write_jsonl(ROOT / st["level1"], records)
    print(f"level 1: {len(records)} records")
    if a.level1_only:
        return

    exposed = {
        (r["template"], r["condition"], r["copy"]): r.get("exposed")
        for r in records
        if r["plan"] == "analyst" and r["ok"]
    }
    questions = {t["id"]: t for t in pcfg["templates"]}
    items, meta = [], []
    for (tid, cond, copy), (plan, pulled, result) in sorted(kept.items()):
        for arm in gr.ARMS:
            res = result if arm == "guarded" else None
            item_id = f"{tid}/{cond}/{copy}#{arm}"
            text = answer_text(questions[tid]["question"], plan, pulled, res)
            items.append(Item(item_id, f"Database: {gr.DB}", text))
            meta.append((tid, cond, copy, arm, result))
    results = run_calls(
        items,
        "answer",
        cfg,
        cfg["calls"]["answer"]["planted_mode"],
        ROOT / st["cache"],
        cfg["phase"],
    )
    level2 = []
    for (tid, cond, copy, arm, result), r in zip(meta, results, strict=True):
        finding = r["submitted"] if isinstance(r["submitted"], dict) else {}
        level2.append(
            {
                "template": tid,
                "family": questions[tid]["family"],
                "condition": cond,
                "copy": copy,
                "arm": arm,
                "planted_sign": pl.planted_sign(cond),
                "analysis_detected": result["detected"],
                "finding": finding or None,
                "claims_effect": finding.get("claims_effect"),
                "claimed_direction": pl.claimed_direction(
                    finding, result, exposed.get((tid, cond, copy))
                ),
                "checks": checks(finding, result),
                "tokens": r["tokens"],
                "cost_usd": r["cost_usd"],
                "errors": r["errors"],
                "cache_keys": r["cache_keys"],
            }
        )
    gr.write_jsonl(ROOT / st["level2"], level2)
    print(f"level 2: {len(level2)} answers; ${sum(x['cost_usd'] for x in level2):.4f}")


if __name__ == "__main__":
    main()
