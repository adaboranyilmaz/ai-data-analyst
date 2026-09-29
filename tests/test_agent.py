"""The agent's loop, answer, checks and vote, with a scripted model and scripted tools (no
database, no API)."""

from __future__ import annotations

import json

import pytest

from src.agent import verify
from src.agent.answer import SUBMIT, SUBMIT_TOOL, parse_answer, parse_selection
from src.agent.conversation import BUDGET_SPENT, REMINDER, Conversation, Settings
from src.agent.tools import ToolOutcome, present
from src.llm.types import LLMResponse


def response(*blocks, stop="tool_use", usage=(100, 20)) -> LLMResponse:
    return LLMResponse(
        text="".join(b.get("text", "") for b in blocks if b["type"] == "text"),
        content=list(blocks),
        model_reported="m",
        stop_reason=stop,
        usage={"input_tokens": usage[0], "output_tokens": usage[1]},
        latency_ms=5.0,
        created_utc="2026-09-28T00:00:00+00:00",
    )


def call(name, args, id_="t1") -> dict:
    return {"type": "tool_use", "id": id_, "name": name, "input": args}


def submit(sql="SELECT 1", confidence=0.8, declined=False, id_="s1", **kw) -> dict:
    args = {
        "sql": sql,
        "answer": "one",
        "confidence": confidence,
        "declined": declined,
        "decline_reason": None,
        "clarifying_question": None,
        "assumptions": [],
        "premise_correction": None,
        "chart_spec": None,
        **kw,
    }
    return call(SUBMIT, args, id_)


class FakeTools:
    """Tools that answer every call with a fixed result; `final` is what run_final returns."""

    def __init__(self, final=None):
        self.calls: list[tuple[str, object]] = []
        self.final = final or {"ok": True, "rows": [[1]], "row_count": 1, "total_rows": 1}
        self.finals: list[str] = []
        self.ats: list = []  # the call each tool call or final check answered

    def call(self, name, args, at=None):
        self.calls.append((name, args))
        self.ats.append(at)
        result = {"ok": True, "rows": [[1]], "row_count": 1, "total_rows": 1, "seconds": 0.123}
        return ToolOutcome(name, args, result, json.dumps(present(result, 20)), False)

    def run_final(self, sql, at=None):
        self.finals.append(sql)
        self.ats.append(at)
        return self.final


def settings(**kw) -> Settings:
    base = dict(
        backend="anthropic",
        model="claude-sonnet-5",
        max_tokens=100,
        params={"thinking": {"type": "disabled"}},
        prompt_cache=True,
        max_tool_calls=3,
        resubmits=2,
        reminders=1,
        max_model_calls=10,
    )
    return Settings(**{**base, **kw})


def conversation(tools=None, **kw) -> Conversation:
    return Conversation(
        settings(**kw), "system", "Database: x", "Question: q", [SUBMIT_TOOL], tools or FakeTools()
    )


# ------------------------------------------------------------------------------ the loop


class TestConversation:
    def test_submit_first_turn(self):
        c = conversation()
        c.feed("k1", response(submit()))
        assert c.done and c.answer.sql == "SELECT 1" and c.answer.confidence == 0.8
        assert c.errors == [] and len(c.turns) == 1

    def test_tool_then_submit(self):
        tools = FakeTools()
        c = conversation(tools)
        c.feed("k1", response(call("describe_table", {"table": "a"})))
        assert not c.done
        results = c.messages[-1]["content"]
        assert results[0]["type"] == "tool_result" and results[0]["tool_use_id"] == "t1"
        assert "seconds" not in results[0]["content"]  # a timing never reaches the model
        c.feed("k2", response(submit()))
        assert c.done and tools.calls == [("describe_table", {"table": "a"})]
        assert c.tool_calls == 1 and len(c.step_lines) == 2

    def test_parallel_calls_answered_in_one_message(self):
        c = conversation()
        c.feed(
            "k", response(call("list_tables", {}, "a"), call("sample_rows", {"table": "x"}, "b"))
        )
        ids = [b["tool_use_id"] for b in c.messages[-1]["content"]]
        assert ids == ["a", "b"] and c.messages[-1]["role"] == "user"

    def test_tool_budget(self):
        tools = FakeTools()
        c = conversation(tools, max_tool_calls=1)
        c.feed("k1", response(call("list_tables", {}, "a"), call("list_tables", {}, "b")))
        second = c.messages[-1]["content"][1]
        assert second["is_error"] and second["content"] == BUDGET_SPENT
        assert len(tools.calls) == 1 and c.errors[0]["kind"] == "tool_budget"

    def test_reminder_then_give_up(self):
        c = conversation(reminders=1)
        c.feed("k1", response({"type": "text", "text": "I think it is 3."}, stop="end_turn"))
        assert not c.done and c.messages[-1] == {"role": "user", "content": REMINDER}
        c.feed("k2", response({"type": "text", "text": "3"}, stop="end_turn"))
        assert c.done and c.answer.sql is None and c.errors[-1]["kind"] == "no_answer"

    @pytest.mark.parametrize(
        "stop,kind",
        [
            ("refusal", "refusal"),
            ("max_tokens", "max_tokens"),
            ("context_overflow", "context_overflow"),
        ],
    )
    def test_abnormal_stops_end_without_answer(self, stop, kind):
        c = conversation()
        c.feed("k1", response(submit(), stop=stop))
        assert c.done and c.answer.sql is None and c.errors[-1]["kind"] == kind

    def test_resubmission_on_empty_result_bounded(self):
        tools = FakeTools(final={"ok": True, "rows": [], "row_count": 0, "total_rows": 0})
        c = conversation(tools, resubmits=2)
        for i in range(2):
            c.feed(f"k{i}", response(submit(sql=f"SELECT {i}")))
            assert not c.done
            last = c.messages[-1]["content"][-1]
            assert last["is_error"] and "returned no rows" in last["content"]
        c.feed("k3", response(submit(sql="SELECT 9")))
        assert c.done and c.answer.sql == "SELECT 9" and c.resubmits_used == 2
        assert tools.finals == ["SELECT 0", "SELECT 1"]  # the third is accepted unchecked

    def test_failed_query_goes_back(self):
        tools = FakeTools(final={"ok": False, "error": {"kind": "sql_error", "message": "boom"}})
        c = conversation(tools, resubmits=1)
        c.feed("k1", response(submit()))
        assert "failed: boom" in c.messages[-1]["content"][-1]["content"]

    def test_declined_is_accepted_without_check(self):
        tools = FakeTools(final={"ok": True, "rows": [], "row_count": 0, "total_rows": 0})
        c = conversation(tools)
        c.feed("k1", response(submit(sql=None, declined=True, decline_reason="not recorded")))
        assert c.done and c.answer.declined and tools.finals == []

    def test_no_resubmission_without_budget(self):
        tools = FakeTools(final={"ok": True, "rows": [], "row_count": 0, "total_rows": 0})
        c = conversation(tools, resubmits=0)
        c.feed("k1", response(submit()))
        assert c.done and tools.finals == []

    def test_model_call_limit(self):
        c = conversation(max_model_calls=2, max_tool_calls=99)
        c.feed("k1", response(call("list_tables", {})))
        c.feed("k2", response(call("list_tables", {})))
        assert c.done and c.errors[-1]["kind"] == "model_call_limit"

    def test_feed_after_done_raises(self):
        c = conversation()
        c.feed("k1", response(submit()))
        with pytest.raises(RuntimeError):
            c.feed("k2", response(submit()))

    def test_assistant_turn_sent_back_as_input(self):
        c = conversation()
        extra = {"type": "tool_use", "id": "t1", "name": "list_tables", "input": {}, "caller": {}}
        c.feed("k1", response({"type": "text", "text": ""}, extra))
        sent = c.messages[1]["content"]
        assert sent == [{"type": "tool_use", "id": "t1", "name": "list_tables", "input": {}}]


class TestRequests:
    def test_prompt_cache_markers(self):
        r = conversation().request()
        assert r.system[0]["cache_control"] == {"type": "ephemeral"}
        assert r.messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert r.params["cache_control"] == {"type": "ephemeral"}  # multi-turn: the tail too
        assert "tool_choice" not in r.params

    def test_single_shot_forces_the_tool_and_skips_tail_cache(self):
        r = conversation(force_tool=SUBMIT).request()
        assert r.params["tool_choice"] == {"type": "tool", "name": SUBMIT}
        assert "cache_control" not in r.params

    def test_local_model_gets_no_anthropic_params(self):
        r = conversation(
            backend="ollama",
            params={"options": {"num_ctx": 1}},
            prompt_cache=False,
            force_tool=SUBMIT,
        ).request()
        assert r.params == {"options": {"num_ctx": 1}} and r.system == "system"

    def test_sample_zero_is_the_plain_request(self):
        assert conversation(sample=0).request() == conversation().request()
        r1 = conversation(sample=1).request()
        assert r1.params["_sample"] == 1 and r1.cache_key != conversation().request().cache_key

    def test_requests_are_rebuilt_identically(self):
        a, b = conversation(), conversation()
        for c in (a, b):
            c.feed("k1", response(call("describe_table", {"table": "a"})))
        assert a.request().cache_key == b.request().cache_key


# ------------------------------------------------------------------------------ the answer


class TestAnswer:
    def test_valid(self):
        a = parse_answer(submit(assumptions=[" x ", ""])["input"])
        assert (a.sql, a.confidence, a.declined, a.assumptions, a.problems) == (
            "SELECT 1",
            0.8,
            False,
            ["x"],
            [],
        )

    @pytest.mark.parametrize("value,expected", [(1.7, 1.0), (-0.2, 0.0)])
    def test_confidence_clipped_and_recorded(self, value, expected):
        a = parse_answer(submit(confidence=value)["input"])
        assert a.confidence == expected and "clipped" in a.problems[0]

    def test_missing_confidence(self):
        args = submit()["input"]
        del args["confidence"]
        a = parse_answer(args)
        assert a.confidence == 0.0 and any("missing" in p for p in a.problems)

    def test_chart_json(self):
        good = parse_answer(submit(chart_spec='{"mark": "bar"}')["input"])
        bad = parse_answer(submit(chart_spec="{not json")["input"])
        assert good.chart_spec == {"mark": "bar"} and bad.chart_spec is None and bad.problems

    def test_not_an_object(self):
        assert parse_answer("SELECT 1").sql is None

    def test_submit_tool_is_strict_and_complete(self):
        schema = SUBMIT_TOOL["input_schema"]
        assert SUBMIT_TOOL["strict"] and schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


class TestSelection:
    KNOWN = {"account": ["account_id", "district_id"], "loan": ["loan_id", "account_id", "amount"]}

    def test_order_and_unknowns(self):
        chosen, problems = parse_selection(
            {
                "tables": [
                    {"table": "LOAN", "columns": ["amount", "loan_id", "nope"]},
                    {"table": "ghost", "columns": []},
                    {"table": "account", "columns": []},
                ]
            },
            self.KNOWN,
        )
        assert chosen == {"account": ["account_id", "district_id"], "loan": ["loan_id", "amount"]}
        assert problems == ["unknown column loan.nope", "unknown table 'ghost'"]

    def test_malformed(self):
        assert parse_selection(None, self.KNOWN) == (
            {},
            ["select_schema input has no `tables` list"],
        )


# ------------------------------------------------------------------------------ checks and vote


def result(rows, columns=("x",), ok=True, truncated=False):
    return verify.FinalResult(ok, list(columns), [tuple(r) for r in rows], truncated)


class TestChecks:
    def test_clean(self):
        c = verify.checks("SELECT a FROM t", result([[1], [2]]), 1000)
        assert c["applicable"] and not c["failed"]

    def test_empty_and_repeats_and_size(self):
        assert verify.checks("SELECT a FROM t", result([]), 1000)["empty"]
        assert verify.checks("SELECT a FROM t", result([[1], [1]]), 1000)["repeated_rows"]
        assert verify.checks("SELECT a FROM t", result([[i] for i in range(5)]), 3)["too_many_rows"]

    def test_negative_count_and_amount(self):
        c = verify.checks("SELECT COUNT(*) FROM t", result([[-1]], ["count"]), 1000)
        assert c["out_of_range"][0]["kind"] == "count"
        c = verify.checks("SELECT SUM(amount) AS s FROM t", result([[-5]], ["s"]), 1000)
        assert c["out_of_range"][0]["kind"] == "amount"

    def test_a_difference_may_be_negative(self):
        c = verify.checks("SELECT SUM(amount) - 10 AS d FROM t", result([[-5]], ["d"]), 1000)
        assert not c["failed"]

    def test_share_above_100(self):
        sql = "SELECT CAST(SUM(x) AS REAL) * 100 / COUNT(*) FROM t"
        assert verify.checks(sql, result([[120.0]], ["?column?"]), 1000)["out_of_range"]
        assert not verify.checks(sql, result([[42.0]], ["?column?"]), 1000)["failed"]
        named = verify.checks("SELECT r AS pct FROM t", result([[101]], ["pct"]), 1000)
        assert named["out_of_range"]

    def test_not_applicable_on_failure(self):
        c = verify.checks("SELECT 1", verify.FinalResult(False, error={"kind": "x"}), 10)
        assert c == {"applicable": False, "failed": False}


class TestVote:
    def test_majority_and_confidence(self):
        v = verify.vote(
            [(False, result([[1]])), (False, result([[2]])), (False, result([[1.0]]))],
            [False, False, False],
        )
        assert v.chosen == 0 and v.group == [0, 2] and v.confidence == pytest.approx(2 / 3)

    def test_tie_goes_to_earliest(self):
        v = verify.vote(
            [(False, result([[3]])), (False, result([[1]])), (False, result([[2]]))], [False] * 3
        )
        assert v.chosen == 0 and v.confidence == pytest.approx(1 / 3)

    def test_failures_never_agree_and_declines_do(self):
        fail = verify.FinalResult(False, error={"kind": "sql_error"})
        v = verify.vote([(False, fail), (True, fail), (True, fail)], [False] * 3)
        assert v.chosen == 1 and v.group == [1, 2]
        v = verify.vote([(False, fail), (False, fail), (False, result([[1]]))], [False] * 3)
        assert all(len(g) == 1 for g in v.groups)

    def test_order_and_repeats_do_not_matter(self):
        v = verify.vote(
            [(False, result([[1], [2]])), (False, result([[2], [1], [1]]))], [False, False]
        )
        assert v.group == [0, 1] and v.confidence == 1.0

    def test_failed_check_halves(self):
        v = verify.vote([(False, result([[1]]))] * 3, [True, False, False])
        assert v.confidence == 0.5


def test_present_drops_timings_and_cuts_rows():
    shown = present({"ok": True, "rows": [[i] for i in range(30)], "seconds": 1.2}, 20)
    assert "seconds" not in shown and len(shown["rows"]) == 20 and shown["rows_shown"] == 20


def test_undocumented_column_reads_as_undocumented():
    from src.agent.tools import NO_DESCRIPTION, readable_description

    out = readable_description(
        {"ok": True, "columns": [{"name": "country", "status": "missing"}, {"name": "id"}]}
    )
    assert out["columns"] == [{"name": "country", "description": NO_DESCRIPTION}, {"name": "id"}]


def test_pool_keeps_one_connection_per_thread():
    from src.agent.driver import ToolboxPool

    class Box:
        def __init__(self, db):
            self.db, self.closed = db, 0

        def close(self):
            self.closed += 1

    pool = ToolboxPool(Box)
    a = pool.get("a")
    b = pool.get("b")  # the same thread moves on: its box for "a" is closed
    assert a.closed == 1 and b.closed == 0 and pool.get("b") is b
    pool.close()
    assert b.closed == 1


class TestStages:
    ABLATION = {
        "set": "ablation",
        "evidence": True,
        "arms": [{"model": "sonnet", "design": d, "mode": "batch"} for d in ("d1", "d2", "d3")]
        + [{"model": "haiku", "design": d, "mode": "batch"} for d in ("d1", "d3")],
    }
    MAIN = {
        "set": "all",
        "evidence": True,
        "arms": [
            {"model": "sonnet", "design": "winner", "mode": "batch"},
            {
                "model": "sonnet",
                "design": "winner",
                "mode": "batch",
                "set": "ablation",
                "evidence": False,
            },
            {"model": "haiku", "design": "winner", "mode": "batch", "set": "ablation"},
        ],
    }

    @staticmethod
    def report(tmp_path, winner):
        path = tmp_path / "ablation.json"
        path.write_text(json.dumps({"selection": {"winner": winner}}), encoding="utf-8")
        return path

    def test_winner_is_filled_in_with_each_arms_set(self, tmp_path):
        from src.agent.stages import resolve_arms

        arms, notes = resolve_arms(self.MAIN, self.ABLATION, self.report(tmp_path, "d2"))
        assert [(a["model"], a["design"], a["set"], a["evidence"]) for a in arms] == [
            ("sonnet", "d2", "all", True),
            ("sonnet", "d2", "ablation", False),
            ("haiku", "d2", "ablation", True),
        ]
        assert notes == []

    def test_an_arm_the_ablation_ran_is_not_repeated(self, tmp_path):
        from src.agent.stages import resolve_arms

        arms, notes = resolve_arms(self.MAIN, self.ABLATION, self.report(tmp_path, "d3"))
        assert ("haiku", "d3") not in {(a["model"], a["design"]) for a in arms}
        assert len(arms) == 2 and len(notes) == 1 and "haiku" in notes[0]

    def test_winner_needs_the_ablation_report(self, tmp_path):
        from src.agent.stages import resolve_arms

        with pytest.raises(FileNotFoundError, match="43_ablation_report"):
            resolve_arms(self.MAIN, self.ABLATION, tmp_path / "missing.json")
        # a stage that names no winner does not need it
        arms, _ = resolve_arms(self.ABLATION, self.ABLATION, tmp_path / "missing.json")
        assert len(arms) == 5

    def test_configured_stages_resolve(self, tmp_path):
        from src.agent.run import config
        from src.agent.stages import resolve_arms

        stages = config()["stages"]
        for name in ("main", "own"):
            arms, _ = resolve_arms(stages[name], stages["ablation"], self.report(tmp_path, "d4"))
            assert arms and all(a["design"] == "d4" for a in arms)
        (own,) = resolve_arms(stages["own"], stages["ablation"], self.report(tmp_path, "d4"))[0]
        assert own["set"] == "own"
