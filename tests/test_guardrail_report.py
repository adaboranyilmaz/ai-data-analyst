"""The guardrail report's measures, on small records made up for the purpose."""

from __future__ import annotations

import pytest

from src.stats import report


def test_wilson_matches_the_sandbox():
    assert report.wilson(0, 10)[0] == 0.0
    assert report.wilson(10, 10)[1] == 1.0
    lo, hi = report.wilson(81, 263)  # Newcombe (1998), table I
    assert lo == pytest.approx(0.2553, abs=1e-4) and hi == pytest.approx(0.3662, abs=1e-4)
    assert report.wilson(0, 0) is None


def classified(qid, source, category, rules, model):
    return {
        "id": qid,
        "source": source,
        "category": category,
        "rule_flag": rules,
        "model_flag": model,
        "statistical": rules or model,
    }


def test_classification_counts_and_the_reading():
    records = [
        classified("own:f1", "own", "f", True, True),
        classified("own:f2", "own", "f", False, True),
        classified("own:a1", "own", "a", False, True),
        classified("own:a2", "own", "a", False, False),
        classified("planted:A1", "planted", None, True, True),
        classified("bird:1", "bird", None, True, False),
        classified("bird:2", "bird", None, False, True),
        classified("bird:3", "bird", None, False, False),
        classified("bird:4", "bird", None, False, False),
    ]
    review = {
        "benchmark_flagged": [
            {"id": "bird:1", "label": "statistical"},
            {"id": "bird:2", "label": "descriptive"},
        ]
    }
    out = report.classification(records, review)
    assert out["or"]["sensitivity_own_f"]["k"] == 2
    assert out["rules"]["sensitivity_own_f"]["k"] == 1
    assert out["model"]["false_positive_own_other"] == report.rate(1, 2)
    assert out["or"]["false_positive_benchmark_as_flagged"] == report.rate(2, 4)
    # bird:1 is statistical on reading: it is no false positive and leaves the denominator
    assert out["or"]["false_positive_benchmark_after_reading"] == report.rate(1, 3)
    with pytest.raises(ValueError, match="without a reading"):
        report.classification(records, {"benchmark_flagged": []})


def analysis(groups, warnings=()):
    return {"groups": groups, "warnings": list(warnings)}


def test_the_small_sample_rule():
    assert report.small_sample(analysis([{"n": 29, "events": 10}, {"n": 100, "events": 50}]))
    assert report.small_sample(analysis([{"n": 145, "events": 145}, {"n": 537, "events": 461}]))
    assert not report.small_sample(analysis([{"n": 348, "events": 41}, {"n": 334, "events": 35}]))
    assert not report.small_sample(analysis([{"n": 40}, {"n": 50}]))  # a numeric outcome


def test_banking_criteria():
    reading = {"causal_claim_after": False, "reviewed_success": True}
    small = analysis([{"n": 145, "events": 0}, {"n": 537, "events": 76}])
    warned = analysis(small["groups"], [{"kind": "few_events"}])
    ran = {"analysis": warned, "guarded": {"checks": {"causal_sentences": []}}}
    assert report.banking_after(ran, reading)["success"]
    unwarned = {**ran, "analysis": small}
    assert not report.banking_after(unwarned, reading)["uncertainty"]
    causal = report.banking_after(ran, {**reading, "causal_claim_after": True})
    assert not causal["success"] and not causal["reviewed_success"]
    failed = {"analysis": None, "guarded": None, "analysis_error": {"kind": "timeout"}}
    out = report.banking_after(failed, reading)
    assert not out["success"] and not out["reviewed_success"]
    assert out["failed_at"] == {"kind": "timeout"}


def l1(tid, family, cond, plan, detected, direction=None):
    return {
        "template": tid,
        "family": family,
        "condition": cond,
        "plan": plan,
        "ok": True,
        "detected": detected,
        "direction": direction,
    }


def test_level1_cells_and_the_power_checks():
    records = []
    for plan, small_hits in (("reference", 5), ("analyst", 2)):
        for i in range(10):
            records.append(l1("A1", "rates", "small", plan, i < small_hits, "planted"))
            records.append(l1("A1", "rates", "large", plan, i < 9, "planted"))
    cells = report.level1_cells(records, {"small": 0.5, "large": 0.9})
    ref_small = next(c for c in cells if c["plan"] == "reference" and c["condition"] == "small")
    assert ref_small["detected"]["rate"] == 0.5 and ref_small["theoretical_power"] == 0.5
    checks = report.power_checks(cells)
    assert all(x["within"] for x in checks["reference_vs_power"])
    by = {x["condition"]: x for x in checks["analyst_vs_reference"]}
    assert not by["small"]["within"] and by["large"]["within"]


def l2(tid, cond, copy, arm, claims, direction=0, unsupported=False):
    return {
        "template": tid,
        "condition": cond,
        "copy": copy,
        "arm": arm,
        "planted_sign": {"none": 0, "confounded": 0, "reversed": -1}.get(cond, 1),
        "finding": {"claims_effect": claims},
        "claims_effect": claims,
        "claimed_direction": direction,
        "checks": {
            "unsupported_claim": unsupported,
            "missed_effect": False,
            "direction_mismatch": False,
            "causal_sentences": [],
        },
    }


def test_level2_primary_is_paired_over_copies():
    records = []
    for copy in range(10):
        records.append(l2("A1", "none", copy, "numbers_only", "yes" if copy < 4 else "no", 1))
        records.append(l2("A1", "none", copy, "guarded", "yes" if copy < 1 else "no", 1))
        records.append(l2("D1", "reversed", copy, "guarded", "yes", -1 if copy < 6 else 1))
        records.append(l2("D1", "reversed", copy, "numbers_only", "yes", 1))
    out = report.level2_summary(records, seed=1, resamples=500)
    p = out["primary"]
    assert p["copies"] == 10 and p["guarded"] == 0.1 and p["numbers_only"] == 0.4
    assert p["difference"]["estimate"] == pytest.approx(-0.3)
    assert p["difference"]["low"] <= -0.3 <= p["difference"]["high"]
    rev = out["arms"]["guarded"]["planted_direction_by_condition"]["reversed"]
    assert rev["k"] == 6 and rev["n"] == 10
    assert out["arms"]["numbers_only"]["planted_direction_by_condition"]["reversed"]["k"] == 0


def test_level2_leaves_failed_answers_out():
    records = [l2("A1", "large", 0, "guarded", "yes", 1), l2("A1", "large", 1, "guarded", "yes", 1)]
    records[1]["finding"] = None
    records[1]["claims_effect"] = None
    records[1]["errors"] = [{"kind": "max_tokens"}]
    records += [l2("A1", "none", 0, arm, "no") for arm in ("guarded", "numbers_only")]
    out = report.level2_summary(records, seed=1, resamples=50, reading={"guarded": 0})
    g = out["arms"]["guarded"]
    assert g["answers"] == 3 and len(g["failed"]) == 1
    assert g["planted_direction_by_condition"]["large"] == report.rate(1, 1)
    assert g["causal_claims_on_reading"] == 0
