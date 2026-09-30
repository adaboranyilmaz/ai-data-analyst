"""The confidence and escalation reports on synthetic records with known answers: nothing fitted
or chosen on the held-out set, the critic's comparison, and the escalation rule."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.agent.confidence import confidence_config
from src.eval.calibrate import Platt, choose_threshold
from src.eval.config import config

ROOT = Path(__file__).resolve().parent.parent


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


confidence_report = load("confidence_report", "53_confidence_report.py")
escalation_report = load("escalation_report", "54_escalation_report.py")

PREDICTIONS = {
    f"P{i:02d}": {"id": f"P{i:02d}", "claim": "c", "prediction": "p"} for i in range(1, 16)
}
IDS = {"ablation": set(range(0, 80)), "held_out": set(range(100, 200))}


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(config())
    c["bootstrap"]["resamples"] = 200  # the statistics are tested elsewhere
    return c


def record(qid, right, confidence, declined=False, **kw):
    base = {
        "question_id": qid,
        "source": "bird",
        "db_id": "financial",
        "difficulty": "simple",
        "category": None,
        "correct": int(right),
        "declined": declined,
        "decline_reason": "missing" if declined else None,
        "clarifying_question": None,
        "assumptions": [],
        "premise_correction": None,
        "confidence": confidence,
        "soft_f1": float(right),
        "cost_usd": 0.01,
        "tokens": {
            "input": 0,
            "output": 500,
            "cache_write_5m": 0,
            "cache_write_1h": 0,
            "cache_read": 0,
        },
        "latency_s": 0.0,
        "steps": 1,
        "tool_calls": 0,
        "errors": [],
        "score_outcome": "ok",
    }
    return {**base, **kw}


def answers(seed=0, held_out_flip=False):
    """Overconfident stated confidences: right about (confidence - 0.25) of the time."""
    rng = np.random.default_rng(seed)
    out = []
    for q in sorted(IDS["ablation"] | IDS["held_out"]):
        c = float(rng.choice([0.6, 0.7, 0.8, 0.9, 0.95]))
        right = rng.uniform() < c - 0.25
        if held_out_flip and q in IDS["held_out"]:
            right = not right
        out.append(record(q, right, c, declined=(q == 150)))
    return out


def review(a, confidence):
    declined = a["declined"]
    return {
        "question_id": a["question_id"],
        "reviewed": not declined,
        "not_reviewed": "declined" if declined else None,
        "verdict": None if declined else "unsure",
        "confidence": 0.0 if declined else confidence,
        "cost_usd": 0.0 if declined else 0.005,
    }


def reviews(ans, seed=1, missing=()):
    rng = np.random.default_rng(seed)
    out = []
    for a in ans:
        c = float(np.clip((0.75 if a["correct"] else 0.35) + rng.normal(0, 0.2), 0, 1))
        out.append(review(a, None if a["question_id"] in missing else c))
    return out


class TestConfidenceReport:
    @pytest.fixture(scope="class")
    @classmethod
    def made(cls, cfg):
        ans = answers()
        conf = confidence_config()
        cal, risk = confidence_report.report(
            ans, reviews(ans, missing={120}), IDS, conf, PREDICTIONS, "main/x", cfg
        )
        return ans, cal, risk

    def test_the_calibrators_are_fitted_on_the_calibration_split_only(self, made, cfg):
        ans, cal, _ = made
        fit = [a for a in ans if a["question_id"] in IDS["ablation"]]
        expected = Platt.fit([a["confidence"] for a in fit], [a["correct"] for a in fit])
        assert cal["calibration_split"]["calibrators"]["platt"] == expected.to_dict()
        # changing every held-out outcome changes nothing that was fitted or chosen
        flipped = answers(held_out_flip=True)
        cal2, _ = confidence_report.report(
            flipped, reviews(flipped), IDS, confidence_config(), PREDICTIONS, "main/x", cfg
        )
        assert cal2["calibration_split"] == cal["calibration_split"]
        assert (
            cal2["decline"]["chosen_on_calibration_split"]
            == cal["decline"]["chosen_on_calibration_split"]
        )
        assert cal2["held_out"]["stated_platt"] != cal["held_out"]["stated_platt"]

    def test_the_threshold_is_chosen_on_the_calibration_split(self, made):
        ans, cal, _ = made
        fit = [a for a in ans if a["question_id"] in IDS["ablation"]]
        platt = Platt(
            **{
                k: v
                for k, v in cal["calibration_split"]["calibrators"]["platt"].items()
                if k != "method"
            }
        )
        p = platt([a["confidence"] for a in fit])
        rule = confidence_config()["decline"]
        want = choose_threshold(
            p,
            [a["correct"] for a in fit],
            [False] * len(fit),
            rule["target_accuracy"],
            rule["min_answered"],
        )
        d = cal["decline"]
        assert d["chosen_on_calibration_split"]["threshold"] == want["threshold"]
        assert d["held_out"]["threshold"] == want["threshold"]
        assert d["held_out"]["questions"] == len(IDS["held_out"])

    def test_the_declined_answer_counts_at_zero(self, made):
        _, cal, _ = made
        for name in ("stated_raw", "stated_platt", "stated_isotonic"):
            assert cal["held_out"][name]["declines_at_zero"] == 1
        assert cal["held_out"]["declined"] == 1

    def test_platt_keeps_the_ranking(self, made):
        _, _, risk = made
        h = risk["held_out"]
        assert h["stated_platt"]["aurc"]["estimate"] == pytest.approx(
            h["stated_raw"]["aurc"]["estimate"]
        )

    def test_the_critic_comparison_and_counts(self, made):
        _, cal, risk = made
        c = risk["comparisons"]["critic_minus_stated_platt"]
        assert c["pre_registered"] and c["questions"] == len(IDS["held_out"])
        diff = (
            risk["held_out"]["critic_raw"]["aurc"]["estimate"]
            - (risk["held_out"]["stated_platt"]["aurc"]["estimate"])
        )
        assert c["aurc"]["estimate"] == pytest.approx(diff)
        held = cal["critic"]["held_out"]
        assert held["without_verdict"] == 1 and held["not_reviewed"] == {"declined": 1}
        assert risk["comparisons"]["combined_minus_stated_platt"]["pre_registered"] is False
        assert "critic_minus_stated_zeroed_without_result" in risk["comparisons"]

    def test_the_zeroed_baseline_zeroes_only_answers_without_a_result(self, cfg):
        ans = answers()
        revs = reviews(ans)
        # the most confident wrong held-out answer: its query "failed"
        wrong = max(
            (a for a in ans if a["question_id"] in IDS["held_out"] and not a["correct"]),
            key=lambda a: a["confidence"],
        )
        for r in revs:
            if r["question_id"] == wrong["question_id"]:
                r |= {"reviewed": False, "not_reviewed": "final_sql_sql_error", "confidence": 0.0}
        _, risk = confidence_report.report(
            ans, revs, IDS, confidence_config(), PREDICTIONS, "main/x", cfg
        )
        h = risk["held_out"]
        assert (
            h["stated_zeroed_without_result"]["aurc"]["estimate"]
            < h["stated_raw"]["aurc"]["estimate"]
        )
        # without any failing answer, the baseline is the stated confidence itself
        _, plain = confidence_report.report(
            ans, reviews(ans), IDS, confidence_config(), PREDICTIONS, "main/x", cfg
        )
        assert (
            plain["held_out"]["stated_zeroed_without_result"]["aurc"]
            == plain["held_out"]["stated_raw"]["aurc"]
        )

    def test_platt_keeps_the_order_is_recorded(self, made):
        _, cal, _ = made
        assert cal["calibration_split"]["platt_keeps_order"] is True

    def test_predictions(self, made):
        _, cal, _ = made
        assert set(cal["predictions"]) == {"P07", "P08"}
        assert "ece_platt_held_out" in cal["predictions"]["P07"]

    def test_every_answer_needs_a_review(self, cfg):
        ans = answers()
        with pytest.raises(ValueError, match="not reviewed every answer"):
            confidence_report.report(
                ans, reviews(ans)[1:], IDS, confidence_config(), PREDICTIONS, "main/x", cfg
            )


class TestEscalationReport:
    esc = confidence_config()["escalation"]

    def runs(self, sonnet_right, opus_right, n=40):
        sonnet = [record(q, q < sonnet_right, 0.8) for q in range(n)]
        opus = [
            record(q, q < opus_right, 0.8, steps=2 if q == 0 else 1, cost_usd=0.03)
            for q in range(n)
        ]
        return opus, sonnet

    def test_adopted_on_a_large_gain(self, cfg):
        opus, sonnet = self.runs(10, 30)
        out = escalation_report.report(opus, sonnet, None, self.esc, "d1", PREDICTIONS, cfg)
        assert out["rule"]["adopted"] and out["rule"]["gain"]["estimate"] == pytest.approx(0.5)
        assert out["predictions"]["P14"]["adopted"] is True
        assert out["opus_behaviour"]["needed_the_reminder"] == 1

    def test_not_adopted_below_the_minimum_gain(self, cfg):
        opus, sonnet = self.runs(10, 11)
        out = escalation_report.report(opus, sonnet, None, self.esc, "d1", PREDICTIONS, cfg)
        assert out["rule"]["gain"]["estimate"] == pytest.approx(0.025)
        assert not out["rule"]["adopted"] and out["router"]["built"] is False

    def test_cost_per_correct_and_probe(self, cfg):
        opus, sonnet = self.runs(10, 20)
        probe = [
            record(q, True, 0.9, errors=[{"kind": "no_answer", "message": "m"}]) for q in range(3)
        ]
        out = escalation_report.report(opus, sonnet, probe, self.esc, "d1", PREDICTIONS, cfg)
        rows = {r["model"]: r for r in out["cost_per_correct"]}
        assert rows["claude-opus-5-5"]["cost_per_correct_answer_usd"] == pytest.approx(
            0.03 * 40 / 20
        )
        assert out["probe_pilot"]["no_answer"] == 3


def test_the_figures_render_from_the_reports(tmp_path, cfg):
    plots = load("plots", "57_plots.py")
    ans = answers()
    cal, risk = confidence_report.report(
        ans, reviews(ans), IDS, confidence_config(), PREDICTIONS, "main/x", cfg
    )
    plots.reliability(cal, tmp_path / "r.png")
    plots.risk_coverage(risk, tmp_path / "c.png")
    assert (tmp_path / "r.png").stat().st_size > 10_000
    assert (tmp_path / "c.png").stat().st_size > 10_000


framework_report = load("framework_report", "56_framework_report.py")


def test_logical_lines_skip_blanks_comments_and_docstrings(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(
        '"""Module\ndocstring."""\n\n# a comment\nx = 1  # trailing\n\n\ndef f():\n'
        '    """Doc."""\n    return (\n        x\n    )\n',
        encoding="utf-8",
    )
    assert framework_report.logical_lines(f) == 5  # x = 1, def, return, x, )


class TestFrameworkReport:
    def made(self, cfg, graph_right=None, same=True):
        ids = list(range(40))
        answers = [record(q, q < 20, 0.8) for q in ids]
        reviews = [
            {"question_id": q, "confidence": 0.9 if q < 20 else 0.2, "cost_usd": 0.005} for q in ids
        ]
        right = graph_right if graph_right is not None else 20
        graph = [
            {
                "question_id": q,
                "correct": int(q < right),
                "declined": False,
                "stated_confidence": 0.8,
                "critic_confidence": 0.9 if q < 20 else 0.2,
                "cost_usd": 0.015,
                "model_calls": 2,
                "own_loop_request_keys": ["a", "b"],
                "same_requests": same,
                "checkpoints": 6,
                "graph_seconds": 0.2,
                "own_loop_seconds": 0.1,
            }
            for q in ids
        ]
        code = {"graph_arm": {"files": [], "covers": "", "lines": 1}}
        return framework_report.report(graph, answers, reviews, 0.5, PREDICTIONS, code, cfg)

    def test_identical_pipelines_differ_by_nothing(self, cfg):
        out = self.made(cfg)
        d = out["graph_minus_own_loop"]
        assert d["execution_accuracy"]["estimate"] == 0 and d["execution_accuracy"]["low"] == 0
        assert d["aurc_pipeline_confidence"]["estimate"] == pytest.approx(0)
        assert out["requests_identical"] == 40 and not out["rule"]["replaces_own_loop"]
        c = out["cost_per_correct_answer_usd"]
        assert c["graph"] == pytest.approx(c["own_loop"]) == pytest.approx(0.015 * 40 / 20)
        assert out["predictions"]["P15"]["replaces_own_loop"] is False

    def test_the_rule_needs_both_better(self, cfg):
        out = self.made(cfg, graph_right=35)  # more right, but the ranking is not better
        assert out["graph_minus_own_loop"]["execution_accuracy"]["low"] > 0
        assert out["rule"]["replaces_own_loop"] is (
            out["graph_minus_own_loop"]["aurc_pipeline_confidence"]["high"] < 0
        )


router_report = load("router_report", "60_router_report.py")


class TestRouterReport:
    def test_routing_costs_and_differences(self, cfg):
        ids = list(range(20))
        sonnet = [record(q, q < 8, 0.8, cost_usd=0.01) for q in ids]
        opus = [record(q, q < 14, 0.8, cost_usd=0.02) for q in ids]
        routed = set(range(10, 20))  # Sonnet keeps 0-9 (8 right), Opus takes 10-19 (4 right)
        out = router_report.report(sonnet, opus, routed, "claude-opus-5-5", "d1", cfg)
        assert (out["routed"], out["kept"]) == (10, 10)
        ex = out["routed_system"]["execution_accuracy"]["estimate"]
        assert ex == pytest.approx((8 + 4) / 20)
        cost = out["routed_system"]["cost"]["total_usd"]
        assert cost == pytest.approx(20 * 0.01 + 10 * 0.02)  # Sonnet always, Opus where routed
        assert out["routed_minus_sonnet"]["execution_accuracy"]["estimate"] == pytest.approx(
            (12 - 8) / 20
        )
        e = out["exploratory"]
        assert e["opus_minus_sonnet"]["execution_accuracy"]["estimate"] == pytest.approx(6 / 20)
        assert e["on_kept_questions"]["opus"]["execution_accuracy"]["estimate"] == pytest.approx(1)
        assert "selective" not in out["routed_system"]

    def test_direct_calls_are_repriced_at_the_batch_price(self):
        def cost(model, usage, batch):  # $10 per million output tokens, half through batches
            return usage.output * 10 / 1e6 * (0.5 if batch else 1.0)

        batch, direct = 500 * 10 / 1e6 * 0.5, 500 * 10 / 1e6
        records = [record(0, True, 0.8, model="m", cost_usd=batch)]
        records.append(record(1, True, 0.8, model="m", cost_usd=direct))
        out = router_report.at_batch_price(records, cost)
        assert [r["cost_usd"] for r in out["records"]] == pytest.approx([batch, batch])
        assert out["spent_usd"] == pytest.approx(batch + direct)
        assert out["at_batch_price_usd"] == pytest.approx(2 * batch)
        assert out["questions_with_direct_calls"] == 1

    def test_both_models_must_answer_the_same_questions(self, cfg):
        sonnet = [record(q, True, 0.8) for q in range(3)]
        with pytest.raises(ValueError):
            router_report.report(sonnet, sonnet[:2], set(), "m", "d1", cfg)
