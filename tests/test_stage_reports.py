"""The winner's benchmark report and the banking-set report, on synthetic records with known
answers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main_report = load("main_report", "45_main_report.py")
own_report = load("own_report", "46_own_set_report.py")

PREDICTIONS = {
    f"P{i:02d}": {"id": f"P{i:02d}", "claim": "c", "prediction": "p"} for i in range(1, 16)
}
IDS = {"pilot": set(range(4)), "ablation": set(range(4, 24)), "held_out": set(range(24, 40))}
BATCH = {"mode": "batch"}


@pytest.fixture(autouse=True, scope="module")
def few_resamples():
    """Few bootstrap resamples: the statistics are tested elsewhere."""
    import src.eval.summary as summary
    from src.eval.config import config

    cfg = config()
    cfg["bootstrap"]["resamples"] = 200
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(summary, "config", lambda: cfg)
        yield


def record(qid, right, source="bird", category=None, **kw):
    base = {
        "question_id": qid,
        "source": source,
        "db_id": "financial",
        "difficulty": "simple" if source == "bird" else None,
        "category": category,
        "correct": int(right),
        "declined": False,
        "decline_reason": None,
        "clarifying_question": None,
        "assumptions": [],
        "premise_correction": None,
        "confidence": 0.9 if right else 0.3,
        "soft_f1": float(right),
        "cost_usd": 0.02,
        "latency_s": 1.0,
        "steps": 3,
        "tool_calls": 2,
        "errors": [],
        "score_outcome": "ok",
    }
    return {**base, **kw}


def run(ids, n_right):
    """Records for the ids in order: the first n_right correct."""
    return [record(q, i < n_right) for i, q in enumerate(sorted(ids))]


class TestMainReport:
    @pytest.fixture(scope="class")
    @classmethod
    def out(cls):
        full = run(IDS["pilot"], 2) + run(IDS["ablation"], 15) + run(IDS["held_out"], 12)
        runs = {
            ("claude-sonnet-5", "all", True): (full, BATCH),
            ("claude-sonnet-5", "ablation", False): (run(IDS["ablation"], 10), BATCH),
            ("claude-haiku-4-5", "ablation", True): (run(IDS["ablation"], 8), BATCH),
        }
        return main_report.report("d3", runs, IDS, PREDICTIONS)

    def test_the_splits(self, out):
        w = out["winner"]
        assert (w["all"]["questions"], w["held_out"]["questions"]) == (40, 16)
        assert w["ablation"]["questions"] == 20 and w["pilot_questions"] == 4
        assert w["held_out"]["execution_accuracy"]["estimate"] == pytest.approx(12 / 16)

    def test_signs_of_the_paired_differences(self, out):
        ev = out["predictions"]["P11"]["evidence_minus_no_evidence"]["estimate"]
        assert ev == pytest.approx((15 - 10) / 20)
        assert out["haiku_minus_sonnet"]["execution_accuracy"]["estimate"] == pytest.approx(
            (8 - 15) / 20
        )

    def test_batched_latency_and_cost_table(self, out):
        assert out["winner"]["all"]["latency_s"]["p50"] is None
        rows = {(r["model"], r["set"], r["evidence"]): r for r in out["cost_per_correct"]}
        assert len(rows) == 4
        held = rows[("claude-sonnet-5", "held_out", True)]
        assert held["cost_per_correct_answer_usd"] == pytest.approx(0.02 * 16 / 12)

    def test_raw_confidence_auroc_is_reported(self, out):
        # right answers carry the higher confidence: perfectly separated
        auroc = out["predictions"]["P06"]["raw_confidence_auroc_held_out"]
        assert auroc["estimate"] == pytest.approx(1.0)

    def test_the_full_run_must_cover_every_question(self):
        runs = {
            ("claude-sonnet-5", "all", True): (run(IDS["held_out"], 5), BATCH),
            ("claude-sonnet-5", "ablation", False): (run(IDS["ablation"], 1), BATCH),
            ("claude-haiku-4-5", "ablation", True): (run(IDS["ablation"], 1), BATCH),
        }
        with pytest.raises(ValueError, match="all 500"):
            main_report.report("d3", runs, IDS, PREDICTIONS)


class TestOwnSetReport:
    @pytest.fixture(scope="class")
    @classmethod
    def out(cls):
        held_out = {"estimate": 0.6, "low": 0.5, "high": 0.7}
        return own_report.report("d3", cls.records(), held_out, PREDICTIONS)

    @staticmethod
    def records():
        own = {"source": "own"}
        return [
            *(record(f"a{i}", i < 3, category="a", **own) for i in range(4)),
            *(record(f"b{i}", i < 1, category="b", **own) for i in range(2)),
            # ambiguous: one asks; one assumes, but its SQL matches no accepted reading
            record("c0", False, category="c", clarifying_question="Which year?", **own),
            record("c1", False, category="c", assumptions=["calendar year"], **own),
            # unanswerable: declined with a reason, and answered anyway
            record(
                "d0",
                False,
                category="d",
                declined=True,
                decline_reason="not recorded",
                correct=None,
                **own,
            ),
            record("d1", False, category="d", correct=None, **own),
            *(
                record(
                    f"e{i}",
                    False,
                    category="e",
                    premise_correction="none closed",
                    correct=None,
                    **own,
                )
                for i in range(2)
            ),
            record("f0", True, category="f", correct=None, **own),
        ]

    def test_categories(self, out):
        cats = out["categories"]
        assert cats["a"]["success"]["estimate"] == pytest.approx(3 / 4)
        assert cats["c"]["success"]["estimate"] == pytest.approx(1 / 2)
        assert cats["d"]["success"]["estimate"] == pytest.approx(1 / 2)
        assert cats["e"]["success"]["estimate"] == pytest.approx(1.0)
        assert "success" not in cats["f"] and cats["f"]["name"] == "comparative or causal"

    def test_predictions(self, out):
        p12 = out["predictions"]["P12"]
        assert p12["standard_and_multi_step_ex"]["estimate"] == pytest.approx(4 / 6)
        assert p12["held_out_benchmark_ex"]["estimate"] == 0.6
        p13 = out["predictions"]["P13"]
        assert p13["unanswerable"]["estimate"] == pytest.approx(1 / 2)

    def test_behaviours_where_not_called_for(self, out):
        nc = out["not_called_for_ab"]
        assert nc["clarifying_question"]["estimate"] == 0.0
        assert nc["premise_correction"]["estimate"] == 0.0
        assert out["review"]["done"] is False
        assert out["categories"]["c"]["reviewed_success"] is None

    def test_the_review_checks_answers_that_pass_on_form(self):
        from src.eval import own_review

        records = self.records()
        questions = {
            "c0": {"question": "Who are our best customers?",
                   "interpretations": [{"assumption": "by balance"}, {"assumption": "by income"}]},
            "e0": {"question": "Why did X stop?", "premise": "X stopped", "correction": "No"},
            "e1": {"question": "Why did Y stop?", "premise": "Y stopped", "correction": "No"},
        }  # fmt: skip
        t = own_review.template(records, questions, "own/test")
        # the clarifying question and the two corrections; the assumption (c1) is checked by SQL
        assert [(i["id"], i["check"]) for i in t["items"]] == [
            ("c0", "clarifying_question"),
            ("e0", "premise_correction"),
            ("e1", "premise_correction"),
        ]
        assert t["items"][0]["expected"] == ["by balance", "by income"]
        with pytest.raises(ValueError, match="no valid verdict"):
            own_review.reviewed_success(records, t)  # verdicts still empty
        for item, verdict in zip(t["items"], ("correct", "correct", "incorrect"), strict=True):
            item["verdict"] = verdict
        assert own_review.reviewed_success(records, t) == {"c": [1, 0], "e": [1, 0]}
        out = own_report.report("d3", records, {"estimate": 0.6}, PREDICTIONS, t)
        assert out["categories"]["e"]["success"]["estimate"] == 1.0  # the rule, unchanged
        assert out["categories"]["e"]["reviewed_success"]["estimate"] == 0.5
        assert out["review"]["done"] and out["review"]["checked"] == 3

    def test_a_review_that_misses_an_answer_is_refused(self):
        from src.eval import own_review

        review = {"items": [{"id": "c0", "check": "clarifying_question", "verdict": "correct"}]}
        with pytest.raises(ValueError, match=r"missing \['e0', 'e1'\]"):
            own_review.reviewed_success(self.records(), review)

    def test_benchmark_records_are_refused(self):
        with pytest.raises(ValueError, match="banking-set records only"):
            own_report.report("d3", [record(1, True)], {}, PREDICTIONS)
