"""The design comparison on the ablation set: every run summarised, the designs compared, and
the winner chosen by the pre-registered rule.

Refuses to run unless the pre-registration is frozen and unchanged (src/eval/preregistration.py).
From the records of the `ablation` stage (results/runs/ablation/):

- each run: execution accuracy with its interval, by database and difficulty, selective
  prediction and calibration of the confidence it gives (stated, or agreement for designs 4
  and 5), cost, latency, steps, errors (src/eval/summary.py `summarise`);
- every pair of Claude Sonnet 5 designs, paired over the same questions (`compare`);
- the other models against Claude Sonnet 5 on the same design;
- the winner by the selection rule (`select_design`) over the Claude Sonnet 5 designs;
- the cost-per-correct-answer table;
- the pre-registered predictions this stage can check, beside what was observed.

Every result is from one run. Writes results/metrics/ablation.json.

Usage:
    uv run python scripts/43_ablation_report.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.run import config as agent_config  # noqa: E402
from src.data.bird import write_json  # noqa: E402
from src.eval import preregistration  # noqa: E402
from src.eval.records import read_records  # noqa: E402
from src.eval.reports import NOT_MEASURED, cost_row, with_predictions  # noqa: E402
from src.eval.summary import compare, select_design, summarise  # noqa: E402

RUNS = ROOT / "results/runs/ablation"
OUT = ROOT / "results/metrics/ablation.json"
MAIN = "claude-sonnet-5"


def load_runs() -> dict[tuple[str, str], list[dict]]:
    runs = {}
    for f in sorted(RUNS.glob("*.jsonl")):
        records = read_records(f)
        (design,) = {r["design"] for r in records}
        (model,) = {r["model"] for r in records}
        runs[(model, design)] = records
    return runs


def main() -> None:
    preregistration.require()
    arms = agent_config()["stages"]["ablation"]["arms"]
    expected = {(arm["model"], arm["design"]) for arm in arms}
    runs = load_runs()
    missing = sorted(expected - set(runs))
    if missing:
        sys.exit(f"missing ablation runs: {missing}")
    if len({len(r) for r in runs.values()}) != 1:
        sys.exit("the runs cover different numbers of questions")
    predictions = {
        p["id"]: p
        for p in preregistration.parse(
            (ROOT / "results/metrics/preregistration.md").read_text(encoding="utf-8")
        )["predictions"]
    }
    out = report(runs, arms, predictions)
    write_json(OUT, out)
    selection = out["selection"]
    print(f"winner: {selection['winner']} (lowest AURC: {selection['best_aurc']})")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for row in out["cost_per_correct"]:
        print(row)


def report(runs: dict[tuple[str, str], list[dict]], arms: list[dict], predictions: dict) -> dict:
    """The report for runs keyed by (model, design), the stage's arms and the pre-registered
    predictions (by id)."""
    out: dict = {
        "note": "one run per design and model; intervals are 95% bootstrap over questions",
        "questions": len(next(iter(runs.values()))),
        "evidence": True,
        "runs": {f"{m}/{d}": summarise(r) for (m, d), r in sorted(runs.items())},
    }
    for arm in arms:  # a batched run's latency is not measured (src/eval/reports.py)
        if arm["mode"] == "batch" and (arm["model"], arm["design"]) in runs:
            out["runs"][f"{arm['model']}/{arm['design']}"]["latency_s"] = dict(NOT_MEASURED)
    sonnet = {d: r for (m, d), r in runs.items() if m == MAIN}
    out["pairs"] = {
        f"{a} - {b}": compare(sonnet[a], sonnet[b]) for a, b in combinations(sorted(sonnet), 2)
    }
    out["models_vs_sonnet"] = {
        f"{m}/{d} - {MAIN}/{d}": compare(r, sonnet[d])
        for (m, d), r in sorted(runs.items())
        if m != MAIN and d in sonnet
    }
    selection = select_design(sonnet)
    out["selection"] = selection
    out["cost_per_correct"] = [
        cost_row(out["runs"][f"{m}/{d}"], model=m, design=d)
        for m, d in sorted(runs, key=lambda k: (k[1], k[0]))  # by design, then model
    ]

    winner = selection["winner"]
    ex = {k: v["execution_accuracy"] for k, v in out["runs"].items()}
    best = selection["best_aurc"]
    observed = {
        "P01": {"execution_accuracy": ex[f"{MAIN}/d1"]},
        "P02": {
            "best_ex_design": max(sonnet, key=lambda d: ex[f"{MAIN}/{d}"]["estimate"]),
            "best_ex": max((ex[f"{MAIN}/{d}"] for d in sonnet), key=lambda e: e["estimate"]),
        },
        "P03": {"winner": winner, "lowest_aurc": best},
        "P04": {"d4_minus_d3": compare(sonnet["d4"], sonnet["d3"])["execution_accuracy"]},
        "P05": {
            "winner_minus_d1_aurc": None
            if winner == "d1"
            else compare(sonnet[winner], sonnet["d1"]).get("aurc")
        },
    }
    if "claude-haiku-4-5/d3" in out["runs"]:
        observed["P09"] = {
            "haiku_minus_sonnet_d3": out["models_vs_sonnet"][f"claude-haiku-4-5/d3 - {MAIN}/d3"][
                "execution_accuracy"
            ]
        }
    if "qwen2.5:3b-instruct/d1" in out["runs"]:
        observed["P10"] = {"execution_accuracy": ex["qwen2.5:3b-instruct/d1"]}
    out["predictions"] = with_predictions(observed, predictions)
    return out


if __name__ == "__main__":
    main()
