"""The winning design on the benchmark: all 500 questions, the held-out set, the effect of the
evidence hint, and Claude Haiku 4.5 on the same design.

Refuses to run unless the pre-registration is frozen and unchanged. From the records of the
`main` stage (results/runs/main/; Claude Haiku 4.5's run is the ablation stage's when it ran the
winning design there):

- the winner with evidence, summarized on all 500 questions, on the held-out set (the headline:
  nothing was chosen or fitted on it) and on the ablation set; the pilot's 30 questions, on
  which the prompts were tuned, are in the 500 but in no comparison;
- the evidence hint's effect: the winner with evidence minus without, paired on the ablation
  set;
- Claude Haiku 4.5 minus Claude Sonnet 5 on the winning design, paired on the ablation set;
- the cost-per-correct-answer table of these runs;
- the pre-registered predictions this stage can check (P06: the raw confidence's AUROC on the
  held-out set; P11: the evidence effect).

Every result is from one run. Writes results/metrics/benchmark_main.json.

Usage:
    uv run python scripts/45_main_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.run import config as agent_config  # noqa: E402
from src.agent.stages import WINNER, run_id, winner  # noqa: E402
from src.data.bird import write_json  # noqa: E402
from src.eval import preregistration  # noqa: E402
from src.eval.records import read_records  # noqa: E402
from src.eval.reports import (  # noqa: E402
    NOT_MEASURED,
    cost_row,
    split_ids,
    subset,
    with_predictions,
)
from src.eval.summary import compare, summarise  # noqa: E402

RUNS = ROOT / "results/runs"
OUT = ROOT / "results/metrics/benchmark_main.json"
MAIN = "claude-sonnet-5"
HAIKU = "claude-haiku-4-5"


def arm_records(arm: dict, design: str) -> tuple[list[dict], dict]:
    """A main-stage arm's records (the main stage's file, else the ablation stage's, where
    that arm was not repeated) and the arm with its design filled in."""
    arm = {**arm, "design": design}
    rid = run_id(arm["set"], design, arm["model"], arm["evidence"], None)
    for stage in ("main", "ablation"):
        path = RUNS / stage / f"{rid}.jsonl"
        if path.exists():
            return read_records(path), arm
    sys.exit(f"missing run: {rid}")


def main() -> None:
    preregistration.require()
    stage = agent_config()["stages"]["main"]
    design = winner()
    runs = {}
    for arm in stage["arms"]:
        arm = {"set": stage["set"], "evidence": stage.get("evidence", True), **arm}
        if arm["design"] != WINNER:
            sys.exit(f"the main stage runs the winner only, not {arm['design']}")
        records, arm = arm_records(arm, design)
        runs[(arm["model"], arm["set"], arm["evidence"])] = (records, arm)
    predictions = {
        p["id"]: p
        for p in preregistration.parse(
            (ROOT / "results/metrics/preregistration.md").read_text(encoding="utf-8")
        )["predictions"]
    }
    ids = {name: split_ids(name) for name in ("pilot", "ablation", "held_out")}
    out = report(design, runs, ids, predictions)
    write_json(OUT, out)
    ex = out["winner"]["held_out"]["execution_accuracy"]
    print(f"winner {design}: held-out EX {ex['estimate']:.3f} [{ex['low']:.3f}, {ex['high']:.3f}]")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for row in out["cost_per_correct"]:
        print(row)


def report(
    design: str,
    runs: dict[tuple[str, str, bool], tuple[list[dict], dict]],
    ids: dict[str, set],
    predictions: dict,
) -> dict:
    """The report for the winning design's runs, keyed by (model, set, evidence), each with its
    arm; `ids`: the question ids of the pilot, ablation and held-out splits."""
    full, full_arm = runs[(MAIN, "all", True)]
    if {r["question_id"] for r in full} != set().union(*ids.values()):
        raise ValueError("the winner's run with evidence must cover all 500 questions")
    no_ev, no_ev_arm = runs[(MAIN, "ablation", False)]
    haiku, haiku_arm = runs[(HAIKU, "ablation", True)]
    on_ablation = subset(full, ids["ablation"])

    def summary(records: list[dict], arm: dict) -> dict:
        s = summarise(records)
        if arm["mode"] == "batch":
            s["latency_s"] = dict(NOT_MEASURED)
        return s

    out: dict = {
        "note": "one run per model and setting; intervals are 95% bootstrap over questions",
        "winner": {
            "design": design,
            "all": summary(full, full_arm),
            "held_out": summary(subset(full, ids["held_out"]), full_arm),
            "ablation": summary(on_ablation, full_arm),
            "pilot_questions": len(subset(full, ids["pilot"])),
            "pilot_note": "the prompts were tuned on the pilot questions: they count in 'all' "
            "and in no comparison",
        },
        "without_evidence": {"ablation": summary(no_ev, no_ev_arm)},
        "haiku": {"ablation": summary(haiku, haiku_arm)},
        "evidence_minus_no_evidence": compare(on_ablation, no_ev),
        "haiku_minus_sonnet": compare(haiku, on_ablation),
    }
    out["cost_per_correct"] = [
        cost_row(out["winner"]["all"], model=MAIN, design=design, set="all", evidence=True),
        cost_row(
            out["winner"]["held_out"], model=MAIN, design=design, set="held_out", evidence=True
        ),
        cost_row(
            out["without_evidence"]["ablation"],
            model=MAIN,
            design=design,
            set="ablation",
            evidence=False,
        ),
        cost_row(
            out["haiku"]["ablation"], model=HAIKU, design=design, set="ablation", evidence=True
        ),
    ]
    auroc = out["winner"]["held_out"].get("calibration", {}).get("auroc")
    observed = {
        "P06": {"raw_confidence_auroc_held_out": auroc},
        "P11": {
            "evidence_minus_no_evidence": out["evidence_minus_no_evidence"]["execution_accuracy"]
        },
    }
    out["predictions"] = with_predictions(observed, predictions)
    return out


if __name__ == "__main__":
    main()
