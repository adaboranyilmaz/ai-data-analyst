"""The guardrail's pieces without a model API: the keyword rules and the classification's checks,
the single-call runs (a scripted model, then replay at $0), the answer's inputs and the checks
after it, and the analyst's plans."""

from __future__ import annotations

import json

import pytest
import yaml

from src.llm.ledger import ModelPrice, SpendLedger
from src.llm.types import LLMResponse
from src.stats import answer as ans
from src.stats import calls, classify, guardrail, planted
from src.stats.plans import Plan, Pulled

CFG = calls.config()


# --- the classifier -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,hit",
    [
        ("Do stores with longer opening hours sell more often to new customers?", True),
        ("Does the new menu lead to higher sales?", True),
        ("Has the share of late deliveries changed over time?", True),
        ("Are larger schools associated with lower test scores?", True),
        ("Are students who sleep more better at exams?", True),
        ("How many orders were placed in 2019?", False),
        ("What is the difference between the average price in store A and store B?", False),
        ("Which store had the highest revenue in March?", False),
        ("List the five customers with the most orders.", False),
    ],
)
def test_rules(question, hit):
    assert bool(classify.rule_hits(question, classify.rules(CFG))) is hit


def test_parse_classification():
    good = classify.parse({"statistical": True, "kind": "causal", "reason": "r"})
    assert good["statistical"] is True and not good["format_problems"]
    clash = classify.parse({"statistical": True, "kind": "descriptive", "reason": "r"})
    assert clash["format_problems"]
    bad = classify.parse({"statistical": "yes", "kind": "maybe"})
    assert bad["statistical"] is None and bad["kind"] is None and len(bad["format_problems"]) == 2
    assert classify.parse(None)["statistical"] is None
    assert classify.combine(False, None) is False and classify.combine(True, False) is True


# --- one call per item --------------------------------------------------------------------------


def response(name: str, args: dict) -> LLMResponse:
    return LLMResponse(
        text="",
        content=[{"type": "tool_use", "id": "t1", "name": name, "input": args}],
        model_reported="m",
        stop_reason="tool_use",
        usage={"input_tokens": 1000, "output_tokens": 50},
        latency_ms=5.0,
        created_utc="2026-10-01T00:00:00+00:00",
    )


class Scripted:
    name = "fake"

    def __init__(self):
        self.requests = []

    def check(self, request):
        pass

    def generate(self, request):
        self.requests.append(request)
        tool = request.tools[0]["name"]
        if tool == "classify_question":
            stat = "more" in request.messages[0]["content"][1]["text"]
            return response(
                tool,
                {
                    "statistical": stat,
                    "kind": "comparison" if stat else "descriptive",
                    "reason": "r",
                },
            )
        return response(tool, {"answer": "A.", "claims_effect": "no", "higher": None})


class NoCalls:
    name = "fake"

    def check(self, request):
        pass

    def generate(self, request):
        raise AssertionError("replay must not call the model")


def ledger(tmp_path) -> SpendLedger:
    prices = yaml.safe_load(open("configs/budget.yaml", encoding="utf-8"))["prices_usd_per_mtok"]
    return SpendLedger(
        tmp_path / "spend.json",
        {m: ModelPrice(**p) for m, p in prices.items()},
        80.0,
        "phase7",
        1.0,
    )


def test_prompt_hashes_are_enforced(tmp_path):
    cfg = json.loads(json.dumps(CFG))
    cfg["calls"]["classify"]["prompt_sha256"] = None
    with pytest.raises(RuntimeError, match="not frozen"):
        calls.prompt_for("classify", cfg)
    assert calls.prompt_for("classify", cfg, probe=True)[0]
    cfg["calls"]["classify"]["prompt_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="does not match"):
        calls.prompt_for("classify", cfg, probe=True)


def test_requests_force_the_tool_and_keep_the_settings():
    items = [calls.Item("q1", "Database: financial", "Question: Do A go bad more often?")]
    (req,) = calls.requests_for(items, "classify", CFG, probe=True)
    assert req.model == "claude-haiku-4-5" and req.params["temperature"] == 0
    assert req.params["tool_choice"] == {"type": "tool", "name": "classify_question"}
    assert [t["name"] for t in req.tools] == ["classify_question"]
    (plan,) = calls.requests_for(items, "plan", CFG, probe=True)
    assert plan.params["thinking"] == {"type": "disabled"} and plan.model == "claude-sonnet-5"


def test_run_calls_then_replay(tmp_path, monkeypatch):
    items = [
        calls.Item("a", "Database: financial", "Question: Do A go bad more often than B?"),
        calls.Item("b", "Database: financial", "Question: How many loans are there?"),
    ]
    kw = dict(probe=True, ledger=ledger(tmp_path), log=lambda _: None)
    first = calls.run_calls(
        items, "classify", CFG, "direct", tmp_path / "cache", "phase7", backend=Scripted(), **kw
    )
    assert [r["submitted"]["statistical"] for r in first] == [True, False]
    assert all(r["cost_usd"] > 0 and len(r["cache_keys"]) == 1 for r in first)
    monkeypatch.setenv("ANALYST_REPLAY_ONLY", "1")
    again = calls.run_calls(
        items, "classify", CFG, "direct", tmp_path / "cache", "phase7", backend=NoCalls(), **kw
    )
    assert [r["submitted"] for r in again] == [r["submitted"] for r in first]


# --- the answer's inputs and the checks after it ------------------------------------------------


PLAN = Plan.from_dict(
    {
        "sql": "SELECT 1",
        "analysis": "compare_groups",
        "outcome": "bad",
        "outcome_type": "binary",
        "group": "card",
        "reference_group": "false",
        "unit": "a loan",
    }
)
PULLED = Pulled(
    columns=["card", "bad"],
    rows=[[False, 1]] * 76 + [[False, 0]] * 461 + [[True, 0]] * 145 + [[None, 1]],
)


def result(detected=True, est=-0.142, ci=(-0.174, -0.104)):
    return {
        "analysis": "compare_groups",
        "outcome_type": "binary",
        "n": 682,
        "n_dropped": 0,
        "detected": detected,
        "primary": "comparison",
        "reference": "false",
        "groups": [
            {"label": "false", "n": 537, "events": 76, "estimate": 0.1415, "ci": [0.115, 0.174]},
            {"label": "true", "n": 145, "events": 0, "estimate": 0.0, "ci": [0.0, 0.026]},
        ],
        "comparisons": [
            {
                "group": "true",
                "estimate": est,
                "ci": list(ci),
                "p_value": 1e-9,
                "method": "Newcombe's hybrid score interval; Fisher's exact test",
            }
        ],
        "warnings": [{"kind": "few_events", "groups": [{"label": "true", "events": 0, "n": 145}]}],
    }


def test_numbers_only_input_has_no_statistics():
    text = ans.answer_text("Do card holders go bad less?", PLAN, PULLED, None)
    assert "- card = false: 76 of 537 with the outcome (14.2%)" in text
    assert "- card = true: 0 of 145 with the outcome (0.0%)" in text
    assert "682 units with every value (1 left out)" in text
    for word in ("interval", "Verdict", "observational", "Warning"):
        assert word not in text


def test_guarded_input_adds_the_statistics():
    text = ans.answer_text("Do card holders go bad less?", PLAN, PULLED, result())
    assert "true minus false: -14.2 percentage points, interval -17.4 to -10.4" in text
    assert "Verdict of the test (the difference): the interval excludes zero" in text
    assert ans.OBSERVATIONAL in text and "Warning: Few units" in text
    null = ans.answer_text("q", PLAN, PULLED, result(False, 0.01, (-0.02, 0.04)))
    assert "the interval includes zero, so the data do not show a difference" in null


def test_checks_after_the_answer():
    r = result()
    assert ans.checks({"claims_effect": "yes", "higher": "false", "answer": "x"}, r) == {
        "unsupported_claim": False,
        "missed_effect": False,
        "direction_mismatch": False,
        "causal_sentences": [],
    }
    wrong = ans.checks({"claims_effect": "yes", "higher": "true", "answer": "x"}, r)
    assert wrong["direction_mismatch"]
    assert ans.checks({"claims_effect": "no", "higher": None, "answer": ""}, r)["missed_effect"]
    null = result(False, 0.01, (-0.02, 0.04))
    assert ans.checks({"claims_effect": "yes", "higher": "true", "answer": ""}, null)[
        "unsupported_claim"
    ]


def test_causal_wording_outside_a_negation():
    text = (
        "Cards make loans less likely to go bad. This does not show that cards cause "
        "repayment. Defaults fell because of the card."
    )
    assert ans.causal_sentences(text) == [
        "Cards make loans less likely to go bad.",
        "Defaults fell because of the card.",
    ]


def test_the_delivered_answer_always_carries_the_interval_and_the_caveat():
    out = ans.delivered(
        {"answer": "They differ.", "claims_effect": "yes", "higher": "false"}, result(), PLAN
    )
    assert out.startswith("They differ.")
    assert "interval -17.4 to -10.4" in out and ans.OBSERVATIONAL in out
    corrected = ans.delivered(
        {"answer": "Yes.", "claims_effect": "yes", "higher": "true"},
        result(False, 0.01, (-0.02, 0.04)),
        PLAN,
    )
    assert "do not show an effect" in corrected


def test_leaked_markup_is_cut_from_the_delivered_answer():
    leaked = 'They differ.</answer>\n<parameter name="claims_effect">yes'
    assert ans.answer_body(leaked) == "They differ."
    assert ans.answer_body("They differ.</answer>\n<claims_effect>no</claims_effect>") == (
        "They differ."
    )
    assert ans.answer_body("No markup here.") == "No markup here."
    out = ans.delivered(
        {"answer": leaked, "claims_effect": "yes", "higher": "false"}, result(), PLAN
    )
    assert out.startswith("They differ.\n") and "parameter" not in out
    area = Plan(**{**PLAN.__dict__, "x_describes": "area"})
    assert "individuals in them.\n" in ans.delivered(
        {"answer": "x", "claims_effect": "yes", "higher": "false"}, result(), area
    )


def test_labels_match_the_sandbox_rules():
    assert [ans.label(v) for v in (True, False, 3, 2.0, 2.5, "F")] == [
        "true",
        "false",
        "3",
        "2",
        "2.5",
        "F",
    ]
    assert sorted(["10", "9", "b", "a"], key=ans.order_key) == ["9", "10", "a", "b"]


# --- plans ----------------------------------------------------------------------------------------


def submitted(**over):
    base = {
        "unit": "a loan",
        "sql": "SELECT 1",
        "analysis": "compare_groups",
        "outcome": "bad",
        "outcome_type": "binary",
        "group": "g",
        "reference_group": "0",
        "x": "ignored",
        "x_describes": "unit",
        "strata": [],
        "strata_reason": "",
        "assumptions": [],
        "declined": False,
        "decline_reason": None,
    }
    return {**base, **over}


def test_to_plan():
    plan, err = guardrail.to_plan(submitted())
    assert err == {} and plan.x is None and plan.group == "g"
    trend, _ = guardrail.to_plan(submitted(analysis="trend", x="year"))
    assert trend.group is None and trend.reference_group is None and trend.x == "year"
    assert guardrail.to_plan(submitted(declined=True, decline_reason="no data"))[1]["kind"] == (
        "declined"
    )
    assert guardrail.to_plan(submitted(sql=None))[1]["kind"] == "no_sql"
    assert guardrail.to_plan(submitted(strata=["a", "b", "c"]))[1]["kind"] == "invalid_plan"
    assert guardrail.to_plan(None)[1]["kind"] == "no_plan"


def test_the_plan_tool_names_every_plan_field():
    props = calls.PLAN_TOOL["input_schema"]["properties"]
    assert set(guardrail.PLAN_FIELDS) <= set(props)
    assert set(calls.PLAN_TOOL["input_schema"]["required"]) == set(props)


def test_claimed_direction():
    r = result()
    assert planted.claimed_direction({"claims_effect": "no"}, r, "true") == 0
    assert planted.claimed_direction({"claims_effect": "yes", "higher": "true"}, r, "true") == 1
    assert planted.claimed_direction({"claims_effect": "yes", "higher": "false"}, r, "true") == -1
    assert planted.claimed_direction({"claims_effect": "yes", "higher": "x"}, r, "true") is None
    trend = {"analysis": "trend"}
    assert (
        planted.claimed_direction({"claims_effect": "yes", "higher": "decreasing"}, trend, None)
        == -1
    )


def test_the_question_sets():
    own = guardrail.own_questions()
    assert len(own) == 60 and sum(q["category"] == "f" for q in own) == 9
    assert len(guardrail.planted_questions(CFG)) == 8
    texts = {q["question"] for q in own + guardrail.planted_questions(CFG)}
    assert not texts & set(CFG["probes"]["questions"])  # the probes are other questions
