"""Run summaries, paired comparisons, the own set's behavior scores and the selection rule,
on synthetic records with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.config import config
from src.eval.summary import compare, own_set_behaviour, select_design, summarise

CFG = config()
CFG["bootstrap"] = {**CFG["bootstrap"], "resamples": 500}  # fast; the rules are the same


def rec(qid, correct, confidence, *, db="financial", difficulty="simple", declined=False,
        cost=0.01, **extra) -> dict:  # fmt: skip
    r = {
        "run": "test", "question_id": qid, "source": "bird", "db_id": db,
        "difficulty": difficulty, "category": None, "design": "d", "model": "m",
        "evidence": True, "final_sql": "SELECT 1", "answer": None, "confidence": confidence,
        "declined": declined, "decline_reason": "unsure" if declined else None,
        "clarifying_question": None, "assumptions": [], "premise_correction": None,
        "correct": correct, "soft_f1": float(correct) if correct is not None else None,
        "score_outcome": "ok",
        "tokens": {"input": 0, "output": 0, "cache_write_5m": 0, "cache_write_1h": 0,
                   "cache_read": 0},
        "cost_usd": cost, "latency_s": 1.0, "steps": 2, "tool_calls": 1, "errors": [],
        "trace": None, "evaluated_at": "2026-09-28T10:00:00+00:00",
    }  # fmt: skip
    r.update(extra)
    return r


def test_summary_of_a_small_run():
    records = [
        rec(1, 1, 0.9),
        rec(2, 1, 0.8, db="formula_1", difficulty="moderate"),
        rec(3, 0, 0.3, db="formula_1", difficulty="moderate"),
        rec(4, 1, 0.7, declined=True),  # right SQL, but declined: not a correct answer
    ]
    s = summarise(records, CFG)
    assert s["questions"] == 4 and s["declined"] == 1
    assert s["execution_accuracy"]["estimate"] == pytest.approx(0.5)
    assert s["by_db_id"]["formula_1"]["execution_accuracy"]["estimate"] == pytest.approx(0.5)
    assert s["by_difficulty"]["simple"]["questions"] == 2
    # ranked: 0.9 right, 0.8 right, 0.3 wrong, then the declined one (an error)
    assert s["selective"]["aurc"]["estimate"] == pytest.approx((0 + 0 + 1 / 3 + 2 / 4) / 4)
    assert s["cost"]["per_correct_answer_usd"]["estimate"] == pytest.approx(0.04 / 2)
    assert s["calibration"]["auroc"]["estimate"] == pytest.approx(1.0)
    assert len(s["selective"]["curve"]["risk"]) == 4


def test_a_declined_answer_enters_calibration_at_confidence_zero():
    records = [
        rec(1, 1, 0.9),
        rec(2, 1, 0.8),
        rec(3, 0, 0.3),
        rec(4, 0, 1.0, declined=True),  # "certain" it cannot answer: no result, so 0
    ]
    s = summarise(records, CFG)
    cal = s["calibration"]
    assert cal["declines_at_zero"] == 1
    # at its stated 1.0 it would outrank both right answers (AUROC 0.5); at 0 it ranks last
    assert cal["auroc"]["estimate"] == pytest.approx(1.0)
    assert cal["brier"]["estimate"] == pytest.approx((0.1**2 + 0.2**2 + 0.3**2 + 0) / 4)
    assert records[3]["confidence"] == 1.0  # the record keeps what the model stated


def test_questions_without_a_gold_result_are_left_out_of_accuracy():
    records = [rec(1, 1, 0.9), rec(2, None, 0.5)]
    s = summarise(records, CFG)
    assert (s["questions"], s["scored_questions"]) == (2, 1)
    assert s["execution_accuracy"]["estimate"] == 1.0


def test_no_confidence_no_selective_measures():
    s = summarise([rec(1, 1, None), rec(2, 0, None)], CFG)
    assert "selective" not in s and "calibration" not in s


def test_compare_is_paired_by_question():
    a = [rec(i, 1, 0.9) for i in range(20)]
    b = [rec(i, int(i % 2 == 0), 0.9) for i in reversed(range(20))]
    c = compare(a, b, CFG)
    assert c["execution_accuracy"]["estimate"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        compare(a, b[:-1], CFG)


def run(correct: np.ndarray, confidence: np.ndarray, cost: float) -> list[dict]:
    return [rec(i, int(y), float(c), cost=cost)
            for i, (y, c) in enumerate(zip(correct, confidence, strict=True))]  # fmt: skip


def test_the_selection_rule():
    rng = np.random.default_rng(0)
    n = 150
    y = (rng.random(n) < 0.7).astype(int)
    informative = np.clip(0.5 + 0.4 * (y - 0.5) + rng.normal(0, 0.15, n), 0, 1)
    best = run(y, informative, cost=0.05)
    y_close = y.copy()
    y_close[np.flatnonzero(y)[:2]] = 0  # two more errors: not shown to be worse
    close_and_cheap = run(y_close, informative, cost=0.02)
    noise = rng.random(n)
    y_bad = (rng.random(n) < 0.4).astype(int)
    bad_and_cheapest = run(y_bad, noise, cost=0.01)
    pick = select_design(
        {"best": best, "close_and_cheap": close_and_cheap, "bad": bad_and_cheapest}, CFG
    )
    assert pick["best_aurc"] == "best"
    assert pick["cheaper_not_worse"] == ["close_and_cheap"]
    assert pick["winner"] == "close_and_cheap"
    # without a cheaper design within its interval, the best stays
    assert select_design({"best": best, "bad": bad_and_cheapest}, CFG)["winner"] == "best"


def own(qid, category, **fields) -> dict:
    base = {"source": "own", "db_id": "financial", "difficulty": None, "category": category,
            "correct": fields.pop("correct", None)}  # fmt: skip
    return rec(qid, base.pop("correct"), 0.5, **base, **fields)


def test_own_set_behaviour_rules():
    records = [
        own("own-a01", "a", correct=1),
        own("own-a02", "a", correct=1, declined=True),  # a false decline
        own("own-c01", "c", clarifying_question="Best by what: balance or activity?"),
        own("own-c02", "c", assumptions=["best = highest balance"], correct=1),
        own("own-c03", "c", assumptions=["best = highest balance"], correct=0),
        own("own-d01", "d", declined=True),
        own("own-d02", "d"),
        own("own-e01", "e", premise_correction="The bank issued no mortgages."),
        own("own-e02", "e"),
        own("own-f01", "f", correct=1),
    ]
    b = own_set_behaviour(records, CFG)
    assert b["a"]["success"]["estimate"] == 0.5
    assert b["c"]["success"]["estimate"] == pytest.approx(2 / 3)
    assert b["d"]["success"]["estimate"] == 0.5
    assert b["e"]["success"]["estimate"] == 0.5
    assert "success" not in b["f"] and b["f"]["questions"] == 1
    assert b["false_decline_rate_ab"]["estimate"] == 0.5
    assert b["clarification_rate_ab"]["estimate"] == 0.0
    assert b["premise_correction_rate_ab"]["estimate"] == 0.0


def test_asking_or_correcting_everywhere_shows_on_clear_questions():
    ask = "Which year do you mean?"
    fix = "That did not happen."
    records = [
        own("own-a01", "a", correct=1, clarifying_question=ask, premise_correction=fix),
        own("own-b01", "b", correct=1, clarifying_question=ask),
        own("own-c01", "c", clarifying_question=ask),
        own("own-e01", "e", premise_correction=fix),
    ]
    b = own_set_behaviour(records, CFG)
    assert b["c"]["success"]["estimate"] == b["e"]["success"]["estimate"] == 1.0
    assert b["clarification_rate_ab"]["estimate"] == 1.0
    assert b["premise_correction_rate_ab"]["estimate"] == 0.5
