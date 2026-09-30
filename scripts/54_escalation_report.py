"""The escalation decision: the winning design on Claude Opus 5.5 against Claude Sonnet 5.

Refuses to run unless the pre-registration is frozen and unchanged. From the escalation stage's
run on the ablation set (results/runs/escalation/) and Claude Sonnet 5's run of the same design
there (results/runs/ablation/), paired on the same questions:

- both runs summarized (execution accuracy, calibration, cost), and Opus 5.5 minus Sonnet 5 with
  paired intervals;
- the pre-registered rule (configs/confidence.yaml `escalation.adopt_min_gain`): Opus 5.5 is
  adopted only if its execution accuracy is higher by at least the minimum gain, with a paired
  95% interval excluding zero; the cost per correct answer of both beside it;
- how Opus 5.5 answered without a forced tool choice: replies needing the reminder, questions
  left without an answer, refusals, and its output tokens (thinking included) at the stated
  effort;
- the probe on the pilot questions, which is in no comparison;
- the prediction P14. A router is built and evaluated on the held-out set only if Opus 5.5 is
  adopted.

Every result is from one run. Writes results/metrics/escalation.json.

Usage:
    uv run python scripts/54_escalation_report.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.agent.confidence import WINNER, confidence_config, run_name, winning_design  # noqa: E402
from src.data.bird import write_json  # noqa: E402
from src.eval import preregistration  # noqa: E402
from src.eval.records import read_records  # noqa: E402
from src.eval.reports import NOT_MEASURED, cost_row, with_predictions  # noqa: E402
from src.eval.summary import compare, summarise  # noqa: E402

OUT = ROOT / "results/metrics/escalation.json"
SONNET = "claude-sonnet-5"


def behaviour(records: list[dict]) -> dict:
    """How the model answered: model calls, and the ways a question ended without an answer."""
    kinds = Counter(e["kind"] for r in records for e in r["errors"])
    output = np.array([r["tokens"]["output"] for r in records], dtype=float)
    return {
        "questions": len(records),
        "needed_the_reminder": sum(r["steps"] > 1 for r in records),
        "no_answer": kinds.get("no_answer", 0),
        "refusals": kinds.get("refusal", 0),
        "cut_off_at_max_tokens": kinds.get("max_tokens", 0),
        "model_errors": kinds.get("model_error", 0),
        "output_tokens_per_question": {
            "mean": float(output.mean()),
            "p95": float(np.percentile(output, 95)),
            "max": float(output.max()),
        },
    }


def report(
    opus: list[dict],
    sonnet: list[dict],
    probe: list[dict] | None,
    esc: dict,
    design: str,
    predictions: dict,
    cfg: dict | None = None,
) -> dict:
    def summary(records: list[dict]) -> dict:
        s = summarise(records, cfg)
        s["latency_s"] = dict(NOT_MEASURED)  # both runs were batched
        return s

    diff = compare(opus, sonnet, cfg)
    gain = diff["execution_accuracy"]
    adopted = bool(gain["estimate"] >= esc["adopt_min_gain"] and gain["low"] > 0)
    s_opus, s_sonnet = summary(opus), summary(sonnet)
    out = {
        "note": "one run per model; paired on the ablation set; intervals are 95% bootstrap "
        "over questions",
        "design": design,
        "model": esc["model"],
        "settings": esc["settings"],
        "tool_choice": "auto: the model refuses a forced tool choice; one reminder when a reply "
        "has no submit_answer call",
        "opus": s_opus,
        "sonnet": s_sonnet,
        "opus_minus_sonnet": diff,
        "rule": {
            "adopt_min_gain": esc["adopt_min_gain"],
            "requires": "execution accuracy gain at least adopt_min_gain and a paired 95% "
            "interval excluding zero",
            "gain": gain,
            "adopted": adopted,
        },
        "opus_behaviour": behaviour(opus),
        "cost_per_correct": [
            cost_row(s_opus, model=esc["model"], design=design, set="ablation", evidence=True),
            cost_row(s_sonnet, model=SONNET, design=design, set="ablation", evidence=True),
        ],
        "router": {"built": False, "reason": "escalation not adopted"}
        if not adopted
        else {"built": False, "reason": "adopted: the router stage is next"},
    }
    if probe is not None:
        out["probe_pilot"] = {
            "note": "the first pilot questions, run to measure the output size before the "
            "ablation set; in no comparison",
            **behaviour(probe),
            "cost_usd": sum(r["cost_usd"] for r in probe),
        }
    out["predictions"] = with_predictions({"P14": {"adopted": adopted, "gain": gain}}, predictions)
    return out


def main() -> None:
    preregistration.require()
    esc = confidence_config()["escalation"]
    design = winning_design() if esc["design"] == WINNER else esc["design"]
    runs = ROOT / "results/runs"
    opus = read_records(
        runs / "escalation" / f"{run_name('ablation', design, esc['model'], True, None)}.jsonl"
    )
    sonnet = read_records(
        runs / "ablation" / f"{run_name('ablation', design, SONNET, True, None)}.jsonl"
    )
    probe = None
    for r in esc["runs"]:
        if r["set"] == "pilot":
            rid = run_name("pilot", design, esc["model"], esc["evidence"], r.get("limit"))
            probe = read_records(runs / "escalation" / f"{rid}.jsonl")
    predictions = {
        p["id"]: p
        for p in preregistration.parse(
            (ROOT / "results/metrics/preregistration.md").read_text(encoding="utf-8")
        )["predictions"]
    }
    out = report(opus, sonnet, probe, esc, design, predictions)
    write_json(OUT, out)
    g = out["rule"]["gain"]
    print(
        f"Opus 5.5 - Sonnet 5 EX: {g['estimate']:+.3f} [{g['low']:+.3f}, {g['high']:+.3f}]; "
        f"adopted: {out['rule']['adopted']}"
    )
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
