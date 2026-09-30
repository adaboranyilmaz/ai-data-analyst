"""The escalation arm: design 1 on Claude Opus 5.5, which refuses a forced tool choice. With a
scripted model in place of an API: its request, the reminder, its records and replay at $0; and
that for a model that accepts a forced choice nothing changes."""

from __future__ import annotations

import pytest

from src.agent.answer import SUBMIT
from src.agent.confidence import confidence_config
from src.agent.conversation import REMINDER
from src.agent.escalation import AutoToolRun, agent_config_with, execute
from src.agent.evaluate import RunSpec, benchmark_set
from src.agent.run import QuestionRun, config
from src.llm.backends import AnthropicBackend
from src.llm.types import LLMResponse
from src.tools.toolbox import Toolbox

OPUS = "claude-opus-5-5"
SONNET = "claude-sonnet-5"
GOOD = "SELECT COUNT(*) FROM loan"


def escalation_cfg():
    esc = confidence_config()["escalation"]
    return agent_config_with(esc["model"], esc["settings"])


def test_the_escalation_config():
    esc = confidence_config()["escalation"]
    assert esc["model"] == OPUS and esc["design"] == "winner"
    assert esc["settings"]["params"] == {"output_config": {"effort": "low"}}
    assert esc["settings"]["max_tokens"] == 8192
    assert esc["adopt_min_gain"] == 0.05


def test_adding_a_model_leaves_the_agent_config_unchanged():
    before = config()
    cfg = escalation_cfg()
    assert OPUS in cfg["models"] and OPUS not in config()["models"]
    assert config() == before


def text_response(text="Let me think."):
    return LLMResponse(
        text=text,
        content=[
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": text},
        ],
        model_reported=OPUS,
        stop_reason="end_turn",
        usage={"input_tokens": 100, "output_tokens": 20},
        latency_ms=5.0,
        created_utc="2026-09-29T00:00:00+00:00",
    )


def submit_response(sql=GOOD):
    args = {
        "sql": sql,
        "answer": "The number of loans.",
        "confidence": 0.8,
        "declined": False,
        "decline_reason": None,
        "clarifying_question": None,
        "assumptions": [],
        "premise_correction": None,
        "chart_spec": None,
    }
    return LLMResponse(
        text="",
        content=[
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "tool_use", "id": "s1", "name": SUBMIT, "input": args},
        ],
        model_reported=OPUS,
        stop_reason="tool_use",
        usage={"input_tokens": 100, "output_tokens": 300, "cache_creation_input_tokens": 8000},
        latency_ms=5.0,
        created_utc="2026-09-29T00:00:00+00:00",
    )


@pytest.fixture
def financial(bird_ready):
    q = next(q for q, _ in benchmark_set("pilot") if q.db_id == "financial")
    box = Toolbox("financial")
    yield q, box
    box.close()


@pytest.mark.bird
class TestAutoToolRun:
    def test_opus_asks_for_the_call_without_forcing_it(self, financial):
        q, box = financial
        cfg = escalation_cfg()
        ((_, req),) = AutoToolRun(q, "d1", OPUS, True, box, lambda *a: 0.0, None, cfg).pending()
        assert "tool_choice" not in req.params and "thinking" not in req.params
        assert req.params["output_config"] == {"effort": "low"}
        assert [t["name"] for t in req.tools] == [SUBMIT]
        AnthropicBackend.check(req)  # the parameters Opus 5.5 accepts
        # the same system prompt, schema and question as Claude Sonnet 5's design 1
        ((_, sonnet),) = QuestionRun(q, "d1", SONNET, True, box, lambda *a: 0.0).pending()
        assert req.system == sonnet.system and req.messages == sonnet.messages

    def test_a_model_that_accepts_a_forced_choice_sends_design_1_unchanged(self, financial):
        q, box = financial
        ((_, auto),) = AutoToolRun(q, "d1", SONNET, True, box, lambda *a: 0.0).pending()
        ((_, plain),) = QuestionRun(q, "d1", SONNET, True, box, lambda *a: 0.0).pending()
        assert auto.cache_key == plain.cache_key

    def test_a_reply_without_the_call_gets_one_reminder(self, financial):
        q, box = financial
        run = AutoToolRun(q, "d1", OPUS, True, box, lambda *a: 0.0, None, escalation_cfg())
        ((conv, req),) = run.pending()
        run.feed(conv, req.cache_key, text_response())
        ((conv, second),) = run.pending()
        assert second.messages[-1] == {"role": "user", "content": REMINDER}
        assert second.messages[-2]["content"][0]["type"] == "thinking"  # passed back unchanged
        run.feed(conv, second.cache_key, submit_response())
        fin = run.finish()
        assert fin.final_sql == GOOD and fin.steps == 2 and fin.confidence == 0.8

    def test_no_call_after_the_reminder_is_no_answer(self, financial):
        q, box = financial
        run = AutoToolRun(q, "d1", OPUS, True, box, lambda *a: 0.0, None, escalation_cfg())
        for _ in range(2):
            ((conv, req),) = run.pending()
            run.feed(conv, req.cache_key, text_response())
        fin = run.finish()
        assert fin.final_sql is None and fin.errors[-1]["kind"] == "no_answer"


class Opus:
    name = "fake"

    def __init__(self):
        self.calls = []

    def check(self, request):
        AnthropicBackend.check(request)

    def generate(self, request):
        self.calls.append(request)
        return submit_response()


class NoCalls:
    name = "fake"

    def check(self, request):
        pass

    def generate(self, request):
        raise AssertionError("replay must not call the model")


@pytest.mark.bird
def test_records_and_replay_at_zero_cost(tmp_path, bird_ready, monkeypatch):
    items = [x for x in benchmark_set("ablation") if x[0].db_id == "financial"][:2]
    spec = RunSpec(
        name="escalation/test",
        items=items,
        design="d1",
        model=OPUS,
        evidence=True,
        phase="phase5",
        cache_dir=tmp_path / "cache",
        records=tmp_path / "records.jsonl",
        traces_dir=tmp_path / "traces",
        spans=tmp_path / "spans.jsonl",
        ledger_path=tmp_path / "spend.json",
        batch_dir=tmp_path / "batches",
    )
    cfg = escalation_cfg()
    (first,) = execute([spec], cfg, lambda _: None, Opus())
    assert [r["model"] for r in first] == [OPUS, OPUS]
    assert all(r["final_sql"] == GOOD and r["cost_usd"] > 0 for r in first)
    monkeypatch.setenv("ANALYST_REPLAY_ONLY", "1")
    (again,) = execute([spec], cfg, lambda _: None, NoCalls())

    def stable(rs):
        return [{k: v for k, v in r.items() if k not in ("latency_s", "evaluated_at")} for r in rs]

    assert stable(again) == stable(first)


def test_the_light_modules_run_names_are_the_stages():
    from src.agent.confidence import answers_path, run_name, winning_design
    from src.agent.stages import run_id, winner

    assert winning_design() == winner()
    for args in [
        ("all", "d1", "claude-sonnet-5", True, None),
        ("pilot", "d1", OPUS, True, 10),
        ("ablation", "d3", "qwen2.5:3b-instruct", False, None),
    ]:
        assert run_name(*args) == run_id(*args)
    path, name = answers_path(confidence_config())
    assert path.exists() and name == f"main/{run_id('all', winner(), SONNET, True, None)}"
