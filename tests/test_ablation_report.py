"""The design comparison report on synthetic records with known answers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("ablation", ROOT / "scripts/43_ablation_report.py")
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)

N = 40
SONNET = "claude-sonnet-5"
PREDICTIONS = {
    f"P{i:02d}": {"id": f"P{i:02d}", "claim": "c", "prediction": "p"} for i in range(1, 16)
}


def records(design, model, n_correct, cost, confidence=None):
    """n_correct of N questions right; the right ones carry the higher confidence."""
    out = []
    for i in range(N):
        right = i < n_correct
        conf = confidence if confidence is not None else (0.9 if right else 0.3)
        out.append(
            {
                "question_id": i,
                "source": "bird",
                "db_id": "financial",
                "difficulty": "simple",
                "category": None,
                "design": design,
                "model": model,
                "correct": int(right),
                "declined": False,
                "confidence": conf,
                "soft_f1": float(right),
                "cost_usd": cost,
                "latency_s": 1.0,
                "steps": 1,
                "tool_calls": 0,
                "errors": [],
                "score_outcome": "ok",
            }
        )
    return out


@pytest.fixture(scope="module")
def out():
    """The report, built once with few bootstrap resamples (the statistics are tested
    elsewhere; here only what the report does with them)."""
    import src.eval.summary as summary
    from src.eval.config import config

    cfg = config()
    cfg["bootstrap"]["resamples"] = 200
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(summary, "config", lambda: cfg)
        yield ablation.report(runs(), ARMS, PREDICTIONS)


def runs():
    r = {
        (SONNET, "d1"): records("d1", SONNET, 20, 0.01, confidence=0.5),  # uninformative
        (SONNET, "d2"): records("d2", SONNET, 22, 0.02),
        (SONNET, "d3"): records("d3", SONNET, 24, 0.03),
        (SONNET, "d4"): records("d4", SONNET, 30, 0.09),
        (SONNET, "d5"): records("d5", SONNET, 30, 0.10),
        ("claude-haiku-4-5", "d3"): records("d3", "claude-haiku-4-5", 16, 0.02),
    }
    return r


ARMS = [{"model": SONNET, "design": d, "mode": "batch"} for d in ("d1", "d2", "d3", "d4", "d5")] + [
    {"model": "claude-haiku-4-5", "design": "d3", "mode": "direct"}
]


def test_signs_and_labels(out):
    # design 4 is 6 questions of 40 above design 3: positive, not negative
    assert out["predictions"]["P04"]["d4_minus_d3"]["estimate"] == pytest.approx(6 / N)
    assert out["pairs"]["d3 - d4"]["execution_accuracy"]["estimate"] == pytest.approx(-6 / N)
    p09 = out["predictions"]["P09"]["haiku_minus_sonnet_d3"]["estimate"]
    assert p09 == pytest.approx((16 - 24) / N)


def test_batched_latency_is_not_reported(out):
    assert out["runs"][f"{SONNET}/d3"]["latency_s"]["p50"] is None
    assert out["runs"]["claude-haiku-4-5/d3"]["latency_s"]["p50"] == 1.0  # direct: measured


def test_cost_table(out):
    rows = out["cost_per_correct"]
    d3 = next(r for r in rows if r["model"] == SONNET and r["design"] == "d3")
    assert d3["cost_per_correct_answer_usd"] == pytest.approx(0.03 * N / 24)
    assert [r["design"] for r in rows] == sorted(r["design"] for r in rows)


def test_selection_is_reported(out):
    assert out["selection"]["winner"] in {"d1", "d2", "d3", "d4", "d5"}
    assert out["predictions"]["P03"]["winner"] == out["selection"]["winner"]
