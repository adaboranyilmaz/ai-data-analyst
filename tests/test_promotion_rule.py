"""Scoring two configurations and the promotion rule, on synthetic records with known answers."""

from __future__ import annotations

import pytest
import yaml

from src.eval import promotion
from src.tracking import registry

PLATT = {"slope": 4.0, "intercept": -2.0}  # 0.5 stated -> 0.5 calibrated


def rec(qid, confidence, correct, declined=False, cost=0.01, model="m"):
    return {
        "question_id": qid,
        "confidence": confidence,
        "correct": correct,
        "declined": declined,
        "soft_f1": float(correct),
        "cost_usd": cost,
        "model": model,
        "difficulty": "simple",
        "db_id": "d",
        "steps": 1,
        "tool_calls": 0,
        "errors": [],
        "score_outcome": "ok",
        "latency_s": 0.0,
    }


def test_platt_is_a_sigmoid_of_the_stated_confidence():
    assert promotion.platt(PLATT, 0.5) == pytest.approx(0.5)
    assert promotion.platt(PLATT, 1.0) > promotion.platt(PLATT, 0.5) > promotion.platt(PLATT, 0.0)


def test_a_declined_answer_has_calibrated_confidence_zero():
    got = promotion.calibrated([rec(1, 0.9, 0, declined=True), rec(2, 0.5, 1)], PLATT)
    assert got[0]["confidence"] == 0.0 and got[1]["confidence"] == pytest.approx(0.5)


def test_the_router_sends_low_confidence_and_declined_questions_to_the_larger_model():
    primary = [rec(1, 0.9, 1, cost=1.0), rec(2, 0.2, 0, cost=1.0), rec(3, 0.0, 0, True, cost=1.0)]
    larger = [rec(q, 0.8, 1, cost=5.0, model="big") for q in (1, 2, 3)]
    out, routed = promotion.routed_system(primary, larger, threshold=0.5)
    assert routed == 2
    assert [r["model"] for r in out] == ["m", "big", "big"]
    assert [r["cost_usd"] for r in out] == [1.0, 6.0, 6.0]  # a routed question pays both calls
    with pytest.raises(ValueError):
        promotion.routed_system(primary, larger[:2], 0.5)


def comparison(ex_low, aurc_high):
    return {
        "execution_accuracy": {"estimate": ex_low + 0.05, "low": ex_low, "high": ex_low + 0.1},
        "aurc": {"estimate": aurc_high - 0.05, "low": aurc_high - 0.1, "high": aurc_high},
    }


RULE = yaml.safe_load((registry.ROOT / "configs/promotion.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "ex_low, aurc_high, cost, promote, failing",
    [
        (0.10, -0.05, 0.13, True, None),
        (0.0, 0.0, 0.20, True, None),  # the rule's bounds are inclusive
        (-0.01, -0.05, 0.13, False, "execution_accuracy_not_lower"),
        (0.10, 0.01, 0.13, False, "aurc_not_worse"),
        (0.10, -0.05, 0.21, False, "worst_case_question_cost"),
    ],
)
def test_the_rule_needs_every_condition(ex_low, aurc_high, cost, promote, failing):
    out = promotion.decide(RULE, comparison(ex_low, aurc_high), {"max_question_cost_usd": cost})
    assert out["promote"] is promote
    failed = [k for k, c in out["checks"].items() if not c["passed"]]
    assert failed == ([failing] if failing else [])


def test_the_committed_rule_is_the_one_in_the_text_of_this_phase():
    c = RULE["conditions"]
    assert c["ex_difference_lower_bound_at_least"] == 0.0
    assert c["aurc_difference_upper_bound_at_most"] == 0.0
    assert c["max_question_cost_usd_at_most"] == 0.20


def test_summary_reports_what_the_rule_and_the_page_read():
    records = promotion.calibrated(
        [rec(i, 0.9 if i % 2 else 0.3, i % 2) for i in range(1, 41)], PLATT
    )
    s = promotion.summary(records)
    for key in ("execution_accuracy", "aurc", "ece", "brier", "auroc", "accuracy_at_80"):
        assert s[key]["estimate"] is not None
    assert s["max_question_cost_usd"] == 0.01 and s["questions"] == 40
    outcome = promotion.held_out_confidence(records, threshold=0.5)
    assert outcome["answered"] == 20 and outcome["accuracy"]["estimate"] == 1.0
    assert sum(b["n"] for b in outcome["reliability"]) == 40


def test_the_baseline_evaluation_reproduces_the_phase_5_threshold_and_held_out_result():
    import json

    state = registry.read_state()
    cal = json.loads((registry.ROOT / "results/metrics/calibration.json").read_text("utf-8"))
    ev = json.loads(
        (registry.ROOT / state["versions"]["d1-sonnet-5"]["evaluation"]).read_text("utf-8")
    )["confidence"]
    chosen = cal["decline"]["chosen_on_calibration_split"]
    assert ev["decline_threshold"] == chosen["threshold"]
    assert ev["held_out"]["answered"] == cal["decline"]["held_out"]["answered"]
    assert ev["held_out"]["accuracy"] == cal["decline"]["held_out"]["accuracy"]
    assert ev["held_out"]["reliability"] == cal["held_out"]["stated_platt"]["reliability"]
