"""What errors cost: the expected cost per held-out question of the systems measured in Phase 5,
over the cost of a wrong answer and the cost of declining (src/eval/decisions.py).

The systems, on the 320 held-out questions:
- Claude Sonnet 5 answering every question (the winning design's run);
- Sonnet with the decline threshold chosen on the calibration split (calibration.json);
- the router: Sonnet where its calibrated confidence reaches that threshold, Claude Opus 5.5 where
  it does not (Sonnet's cost is spent on every question);
- Opus answering every question;
- exploratory, not pre-registered: Opus with its own decline threshold, chosen by the same rule
  (90% accuracy, at least 15 answered) on its run on the calibration split, with its stated
  confidence calibrated there (Platt) as well.
Every answer is priced from its recorded tokens at the batch price (part of the Opus run was
direct calls at twice the price), with the prices the spend ledger charges (configs/budget.yaml);
no model is called. A declined question's API cost is spent all the same.

Reports each system's API cost, share wrong and share declined with bootstrap intervals over the
questions; the cost of a wrong answer above which Opus is cheaper than Sonnet; the ratio of the
two costs below which declining pays; the cheapest system over a grid of both costs; and the
cost-optimal decline rule (decline when the calibrated probability of being right is below
1 - C_d / C_w) against the fixed thresholds, in units of a wrong answer. One run per model.

Writes results/metrics/decision_analysis.json.

Usage:
    uv run python scripts/65_decision_analysis.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.confidence import answers_path, confidence_config, routed_ids  # noqa: E402
from src.data.bird import write_json  # noqa: E402
from src.eval import decisions as dec  # noqa: E402
from src.eval.bootstrap import bootstrap  # noqa: E402
from src.eval.calibrate import Platt, choose_threshold  # noqa: E402
from src.eval.config import config as eval_config  # noqa: E402
from src.eval.records import read_records  # noqa: E402
from src.eval.reports import split_ids, subset  # noqa: E402
from src.llm.ledger import load_ledger  # noqa: E402
from src.llm.types import TokenUsage  # noqa: E402

OUT = ROOT / "results/metrics/decision_analysis.json"
CALIBRATION = ROOT / "results/metrics/calibration.json"
OPUS_CALIBRATION_RUN = ROOT / "results/runs/escalation/ablation-d1-claude-opus-5-5-evidence.jsonl"
OPUS_HELD_OUT_RUN = ROOT / "results/runs/router/held_out-d1-claude-opus-5-5-evidence.jsonl"
PRICE_BOOK = "phase5"  # only the ledger's price table is used; nothing is charged
WRONG_GRID = np.logspace(-2, 4, 61)  # USD per wrong answer, $0.01 to $10,000
DECLINE_GRID = np.logspace(-2, 3, 51)  # USD per declined question, $0.01 to $1,000
RATIO_GRID = np.logspace(-3, 0, 31)  # C_d / C_w
RATIO_TABLE = (0.01, 0.05, 0.1, 0.2, 0.5)


def priced(records: list[dict], cost) -> np.ndarray:
    return np.array([cost(r["model"], TokenUsage(**r["tokens"]), True) for r in records])


def right(records: list[dict]) -> np.ndarray:
    return np.array([r["correct"] == 1 and not r["declined"] for r in records], dtype=bool)


def stated(records: list[dict]) -> np.ndarray:
    return np.array([0.0 if r["declined"] else r["confidence"] for r in records])


def main() -> None:
    conf = confidence_config()
    boot_cfg = eval_config()["bootstrap"]
    boot = {
        "count": boot_cfg["resamples"],
        "confidence": boot_cfg["confidence"],
        "seed": boot_cfg["seed"],
    }
    cost = load_ledger(PRICE_BOOK).cost
    cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))

    path, sonnet_run = answers_path(conf)
    sonnet = sorted(
        subset(read_records(path), split_ids("held_out")), key=lambda r: r["question_id"]
    )
    by_id = {r["question_id"]: r for r in read_records(OPUS_HELD_OUT_RUN)}
    if set(by_id) != {r["question_id"] for r in sonnet}:
        raise SystemExit("the Opus run does not cover the same held-out questions")
    opus = [by_id[r["question_id"]] for r in sonnet]
    n = len(sonnet)

    s_api, o_api = priced(sonnet, cost), priced(opus, cost)
    s_right, o_right = right(sonnet), right(opus)

    # Sonnet: the Platt calibration and the threshold chosen on the calibration split
    fit = cal["calibration_split"]["calibrators"]["platt"]
    s_platt = Platt(fit["intercept"], fit["slope"])
    s_p = np.where([r["declined"] for r in sonnet], 0.0, s_platt(stated(sonnet)))
    s_threshold = cal["decline"]["chosen_on_calibration_split"]["threshold"]
    s_decline = s_p < s_threshold

    # Opus (exploratory): calibrated and thresholded on its calibration-split run, the same way
    o_cal = read_records(OPUS_CALIBRATION_RUN)
    o_platt = Platt.fit(stated(o_cal), [int(r["correct"] == 1) for r in o_cal])
    o_cal_p = np.where([r["declined"] for r in o_cal], 0.0, o_platt(stated(o_cal)))
    o_choice = choose_threshold(
        o_cal_p,
        [int(r["correct"] == 1) for r in o_cal],
        [r["declined"] for r in o_cal],
        conf["decline"]["target_accuracy"],
        conf["decline"]["min_answered"],
    )
    o_p = np.where([r["declined"] for r in opus], 0.0, o_platt(stated(opus)))
    o_decline = o_p < o_choice["threshold"]

    # the router's own rule (Phase 5): exactly the questions Sonnet with the threshold declines
    ids = routed_ids(sonnet, cal)
    routed = np.array([r["question_id"] in ids for r in sonnet])
    if not np.array_equal(routed, s_decline):
        raise SystemExit("the router's questions differ from the ones Sonnet declines")
    systems = {
        "sonnet": dec.outcomes("sonnet", s_api, s_right),
        "sonnet_declining": dec.outcomes("sonnet_declining", s_api, s_right, s_decline),
        "router": dec.outcomes(
            "router", s_api + np.where(routed, o_api, 0.0), np.where(routed, o_right, s_right)
        ),
        "opus": dec.outcomes("opus", o_api, o_right),
        "opus_declining": dec.outcomes("opus_declining", o_api, o_right, o_decline),
    }

    def iv(stat) -> dict:
        return bootstrap(stat, n, **boot).to_dict()

    table = {}
    for key, s in systems.items():
        table[key] = {
            "api_usd_per_question": iv(lambda i, s=s: s.shares(i)[0]),
            "share_wrong": iv(lambda i, s=s: s.shares(i)[1]),
            "share_declined": iv(lambda i, s=s: s.shares(i)[2]),
            "answered_accuracy": iv(
                lambda i, s=s: (
                    float(np.mean(~s.wrong[i][~s.declined[i]]))
                    if (~s.declined[i]).any()
                    else float("nan")
                )
            ),
        }
    table["opus_declining"]["exploratory"] = True

    def ratio_or_nan(x: float) -> float:
        return x if math.isfinite(x) else float("nan")

    break_even = {
        "opus_over_sonnet_usd": iv(
            lambda i: ratio_or_nan(dec.break_even_wrong_cost(systems["sonnet"], systems["opus"], i))
        ),
        "router_minus_opus": {
            "api_usd_per_question": iv(
                lambda i: systems["router"].shares(i)[0] - systems["opus"].shares(i)[0]
            ),
            "share_wrong": iv(
                lambda i: systems["router"].shares(i)[1] - systems["opus"].shares(i)[1]
            ),
        },
        "declining_pays_below_ratio": {
            "sonnet": iv(
                lambda i: dec.decline_pays_below(systems["sonnet"], systems["sonnet_declining"], i)
            ),
            "opus": iv(
                lambda i: dec.decline_pays_below(systems["opus"], systems["opus_declining"], i)
            ),
        },
    }

    fixed = ("sonnet", "sonnet_declining", "router", "opus", "opus_declining")
    grid = dec.cheapest([systems[k] for k in fixed], WRONG_GRID, DECLINE_GRID)

    curves = {"ratio": RATIO_GRID.tolist()}
    for key in ("sonnet", "opus", "sonnet_declining", "opus_declining"):
        curves[key] = [dec.cost_in_wrong_answers(systems[key], r) for r in RATIO_GRID]
    for key, p, api, ok in (
        ("sonnet_optimal", s_p, s_api, s_right),
        ("opus_optimal", o_p, o_api, o_right),
    ):
        curves[key] = [
            dec.cost_in_wrong_answers(dec.optimal_decline(key, api, ok, p, r), r)
            for r in RATIO_GRID
        ]

    optimal = []
    for r in RATIO_TABLE:
        row = {"ratio": r, "decline_below": 1 - r}
        for key, p, api, ok in (("sonnet", s_p, s_api, s_right), ("opus", o_p, o_api, o_right)):
            o = dec.optimal_decline(key, api, ok, p, r)
            _, wrong, declined = o.shares()
            row[key] = {
                "share_declined": declined,
                "share_wrong": wrong,
                "cost_in_wrong_answers": iv(lambda i, o=o, r=r: dec.cost_in_wrong_answers(o, r, i)),
                "answering_all_in_wrong_answers": dec.cost_in_wrong_answers(systems[key], r),
            }
        optimal.append(row)

    out = {
        "note": "one run per model; the held-out questions; costs from recorded tokens at batch "
        "prices; intervals are 95% bootstrap over questions; the Opus decline threshold and "
        "calibrator are exploratory, fitted on its calibration-split run",
        "assumptions": [
            "a declined question is answered by a person, rightly, at the cost C_d",
            "every wrong answer costs the same C_w; a failing or refused query counts as wrong",
            "a declined question's API cost is spent all the same",
            "no price is assumed: results are given over grids of C_w and C_d",
        ],
        "questions": n,
        "answers": {"sonnet": sonnet_run, "opus": OPUS_HELD_OUT_RUN.stem},
        "prices": "configs/budget.yaml (the spend ledger's price table), batch discount applied",
        "thresholds": {
            "sonnet": {"calibrated": s_threshold, "source": "calibration.json"},
            "opus_exploratory": {
                "calibrated": o_choice["threshold"],
                "reached_target": o_choice["reached_target"],
                "on_calibration_split": o_choice["chosen"],
                "platt": o_platt.to_dict(),
            },
        },
        "systems": table,
        "break_even": break_even,
        "cheapest": {
            "cost_wrong_usd": WRONG_GRID.tolist(),
            "cost_decline_usd": DECLINE_GRID.tolist(),
            "rows_by_cost_decline": grid,
            "share_of_grid": {
                k: sum(row.count(k) for row in grid) / (len(grid) * len(grid[0])) for k in fixed
            },
        },
        "curves_in_wrong_answers": curves,
        "optimal_rule": optimal,
    }
    write_json(OUT, out)
    be = break_even["opus_over_sonnet_usd"]
    print(f"Opus cheaper than Sonnet above ${be['estimate']:.4f} a wrong answer")
    print(f"cheapest over the grid: {out['cheapest']['share_of_grid']}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
