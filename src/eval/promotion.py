"""Scoring two agent configurations on the held-out questions, and the promotion rule.

A configuration's *reported confidence* is calibrated on the calibration split only (a Platt fit,
fixed in the configuration), and both configurations are scored on the same held-out questions
with that confidence: execution accuracy, the area under the risk-coverage curve (AURC), the
calibration error and the cost per question. A challenger replaces the champion only if it passes
the rule in configs/promotion.yaml: execution accuracy not lower, AURC not worse, and a worst-case
question that fits the service's per-question reserve, each within the paired 95% interval.

Pure functions over records: the run files, the registry and the script that writes the results
(scripts/87_promotion.py) are around it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from src.eval.summary import compare, summarise
from src.llm.types import TokenUsage

CostFn = Callable[[str, TokenUsage, bool], float]
SUMMARY_KEYS = ("questions", "declined", "execution_accuracy")


def platt(calibrator: dict[str, Any], stated: float) -> float:
    return 1 / (1 + math.exp(-(calibrator["slope"] * stated + calibrator["intercept"])))


def calibrated(records: Sequence[dict], calibrator: dict[str, Any]) -> list[dict]:
    """The records with the calibrated confidence in place of the stated one; a declined answer
    returns no result, so its confidence is 0."""
    return [
        {**r, "confidence": 0.0 if r["declined"] else platt(calibrator, r["confidence"])}
        for r in records
    ]


def direct_price(records: Sequence[dict], cost: CostFn) -> list[dict]:
    """The records priced as direct calls from their recorded tokens (the batch price is half)."""
    return [{**r, "cost_usd": cost(r["model"], TokenUsage(**r["tokens"]), False)} for r in records]


def routed_system(
    primary: Sequence[dict], larger: Sequence[dict], threshold: float
) -> tuple[list[dict], int]:
    """The router's answers: the primary model's where its calibrated confidence reaches the
    threshold, otherwise the larger model's (both given with calibrated confidence). The routed
    question costs both calls. Returns the records and how many were routed."""
    by_id = {r["question_id"]: r for r in larger}
    if set(by_id) != {r["question_id"] for r in primary}:
        raise ValueError("both models must have answered the same questions")
    out, routed = [], 0
    for p in primary:
        if p["confidence"] < threshold:  # a declined answer has confidence 0: routed too
            o = by_id[p["question_id"]]
            out.append({**o, "cost_usd": p["cost_usd"] + o["cost_usd"]})
            routed += 1
        else:
            out.append(p)
    return out, routed


def system_records(
    resolved: dict[str, Any], root: Path, cost: CostFn | None = None, split: str = "held_out"
) -> tuple[list[dict], int]:
    """The records of a resolved configuration's system on a split, each with its calibrated
    confidence and (given `cost`) its direct-price cost, and how many questions were routed to a
    larger model.

    The primary model's answers are the winning design's run on all questions
    (results/runs/main). A larger model's are its run on the same split: the escalation arm's on
    the calibration split (`ablation`), the router's on the held-out one (results/runs/). All of
    them are scored already; nothing is run again."""
    from src.agent import confidence as conf
    from src.eval.records import read_records
    from src.eval.reports import split_ids, subset

    c = conf.confidence_config()
    esc = c["escalation"]
    ids = split_ids(split)
    path, _ = conf.answers_path(c)
    if resolved["model"] != c["answers"]["model"] or resolved["design"] != conf.winning_design():
        raise ValueError("only the evaluated winning design on its model has recorded answers")

    def priced(records: list[dict]) -> list[dict]:
        return direct_price(records, cost) if cost else records

    primary = priced(
        calibrated(subset(read_records(path), ids), resolved["calibration"]["calibrator"])
    )
    router = resolved["router"]
    if router is None:
        return primary, 0
    stage = "router" if split == "held_out" else "escalation"
    rid = conf.run_name(split, resolved["design"], router["model"], esc["evidence"])
    larger = subset(read_records(root / "results/runs" / stage / f"{rid}.jsonl"), ids)
    larger = priced(calibrated(larger, router["calibrator"]))
    return routed_system(primary, larger, resolved["calibration"]["decline_threshold"])


def decline_threshold(resolved: dict[str, Any], root: Path, rule: dict[str, Any]) -> dict[str, Any]:
    """The system's decline threshold, chosen on the calibration split as for the primary model
    (the highest coverage whose answered questions reach the target accuracy); `rule`: the
    target accuracy and the fewest answered questions (configs/confidence.yaml `decline`)."""
    from src.eval.calibrate import choose_threshold
    from src.eval.records import answered_correct

    records, _ = system_records(resolved, root, split="ablation")
    return choose_threshold(
        [r["confidence"] for r in records],
        [answered_correct(r) for r in records],
        [r["declined"] for r in records],
        rule["target_accuracy"],
        rule["min_answered"],
    )


def held_out_confidence(
    records: Sequence[dict], threshold: float, cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """What a confidence means for a system's held-out answers: coverage and the answered
    questions' accuracy at the threshold, and the reliability table by bin of calibrated
    confidence."""
    import numpy as np

    from src.eval import calibration
    from src.eval.calibrate import threshold_outcome
    from src.eval.config import config
    from src.eval.records import answered_correct

    cfg = cfg or config()
    p = np.array([r["confidence"] for r in records], dtype=float)
    y = np.array([answered_correct(r) for r in records], dtype=float)
    d = np.array([r["declined"] for r in records], dtype=bool)
    out = threshold_outcome(p, y, d, threshold, _partial_boot(cfg["bootstrap"]))
    out["reliability"] = calibration.reliability(p, y, cfg["calibration"]["ece_bins"])
    return out


def _partial_boot(b: dict[str, Any]):
    from functools import partial

    from src.eval.bootstrap import bootstrap

    return partial(bootstrap, count=b["resamples"], confidence=b["confidence"], seed=b["seed"])


def summary(records: Sequence[dict], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    s = summarise(records, cfg)
    sel = s.get("selective", {})
    cal = s.get("calibration", {})
    return {
        **{k: s[k] for k in SUMMARY_KEYS},
        "aurc": sel.get("aurc"),
        "accuracy_at_80": sel.get("accuracy_at_80"),
        "accuracy_at_90": sel.get("accuracy_at_90"),
        "ece": cal.get("ece"),
        "brier": cal.get("brier"),
        "auroc": cal.get("auroc"),
        "cost_per_question_usd": s["cost"]["per_question_usd"],
        "cost_per_correct_answer_usd": s["cost"]["per_correct_answer_usd"],
        "max_question_cost_usd": max(r["cost_usd"] for r in records),
    }


def compare_systems(
    challenger: Sequence[dict], champion: Sequence[dict], cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Challenger minus champion, paired over the same questions."""
    return compare(challenger, champion, cfg)


def decide(
    rule: dict[str, Any], comparison: dict[str, Any], challenger: dict[str, Any]
) -> dict[str, Any]:
    """Each condition of the rule against the challenger's numbers; promote only if all hold."""
    cond = rule["conditions"]
    ex, aurc = comparison["execution_accuracy"], comparison["aurc"]
    checks = {
        "execution_accuracy_not_lower": {
            "difference": ex["estimate"],
            "lower_bound": ex["low"],
            "required_at_least": cond["ex_difference_lower_bound_at_least"],
            "passed": ex["low"] >= cond["ex_difference_lower_bound_at_least"],
        },
        "aurc_not_worse": {
            "difference": aurc["estimate"],
            "upper_bound": aurc["high"],
            "required_at_most": cond["aurc_difference_upper_bound_at_most"],
            "passed": aurc["high"] <= cond["aurc_difference_upper_bound_at_most"],
        },
        "worst_case_question_cost": {
            "max_question_cost_usd": challenger["max_question_cost_usd"],
            "required_at_most": cond["max_question_cost_usd_at_most"],
            "passed": challenger["max_question_cost_usd"] <= cond["max_question_cost_usd_at_most"],
        },
    }
    return {"checks": checks, "promote": all(c["passed"] for c in checks.values())}
