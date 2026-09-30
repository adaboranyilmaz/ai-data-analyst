"""The LangGraph arm: its chat-model adapter, and (on the loaded benchmark, with a scripted model
in place of an API) that its requests are exactly the own loop's, its route, its checkpoints and
its spans. Also that the main system never imports the framework."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

from src.agent.answer import SUBMIT, SUBMIT_TOOL  # noqa: E402
from src.agent.confidence import confidence_config  # noqa: E402
from src.agent.critic import VERDICT, CriticRun, load_prompt  # noqa: E402
from src.agent.driver import Caller, LayeredCache  # noqa: E402
from src.agent.evaluate import benchmark_set  # noqa: E402
from src.agent.run import QuestionRun, config  # noqa: E402
from src.agent_langgraph.chat_model import CachedChatModel  # noqa: E402
from src.agent_langgraph.graph import GraphContext, bound_models, build, run_question  # noqa: E402
from src.llm.cache import ResponseCache  # noqa: E402
from src.llm.ledger import load_ledger  # noqa: E402
from src.llm.types import LLMResponse  # noqa: E402
from src.tools.toolbox import Toolbox  # noqa: E402
from src.tracking.otel import MemoryExporter, make_tracer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SONNET = "claude-sonnet-5"


def test_the_main_system_never_imports_the_framework():
    pattern = re.compile(r"^\s*(from|import)\s+(langgraph|langchain)", re.MULTILINE)
    for sub in ("agent", "llm", "tools", "eval", "db", "tracking"):
        for path in (ROOT / "src" / sub).rglob("*.py"):
            assert not pattern.search(path.read_text(encoding="utf-8")), path


class TestAdapter:
    def model(self):
        return CachedChatModel(
            model=SONNET, max_tokens=100, params={"thinking": {"type": "disabled"}}
        )

    def test_messages_become_the_projects_request_unchanged(self):
        m = self.model().bind_tools([SUBMIT_TOOL], tool_choice={"type": "tool", "name": SUBMIT})
        system = [{"type": "text", "text": "S", "cache_control": {"type": "ephemeral"}}]
        user = [
            {"type": "text", "text": "ctx", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Q"},
        ]
        req = m.request([SystemMessage(content=system), HumanMessage(content=user)])
        assert req.system == system and req.messages == [{"role": "user", "content": user}]
        assert req.tools == [SUBMIT_TOOL] and req.max_tokens == 100
        assert req.params == {
            "thinking": {"type": "disabled"},
            "tool_choice": {"type": "tool", "name": SUBMIT},
        }

    def test_binding_tools_leaves_the_unbound_model_unchanged_and_shares_the_call_log(self):
        base = self.model()
        bound = base.bind_tools([SUBMIT_TOOL])
        assert base.tools == [] and bound.tools == [SUBMIT_TOOL]
        assert bound.calls is base.calls

    def test_stop_sequences_and_unknown_messages_are_refused(self):
        from langchain_core.messages import ToolMessage

        with pytest.raises(TypeError):
            self.model().request([ToolMessage(content="x", tool_call_id="1")])
        with pytest.raises(ValueError):
            self.model()._generate([HumanMessage(content="x")], stop=["END"])

    def test_assistant_messages_pass_through(self):
        req = self.model().request(
            [HumanMessage(content="a"), AIMessage(content=[{"type": "text", "text": "b"}])]
        )
        assert req.messages[1] == {"role": "assistant", "content": [{"type": "text", "text": "b"}]}


# ------------------------------------------------------------------------------ on the benchmark

GOOD = "SELECT COUNT(*) FROM loan"


def reply(name, args):
    return LLMResponse(
        text="",
        content=[{"type": "tool_use", "id": "t1", "name": name, "input": args}],
        model_reported=SONNET,
        stop_reason="tool_use",
        usage={"input_tokens": 100, "output_tokens": 50},
        latency_ms=5.0,
        created_utc="2026-09-29T00:00:00+00:00",
    )


class Scripted:
    name = "fake"

    def __init__(self, declined=False):
        self.calls, self.declined = [], declined

    def check(self, request):
        pass

    def generate(self, request):
        self.calls.append(request)
        if request.tools[0]["name"] == VERDICT:
            return reply(VERDICT, {"verdict": "correct", "confidence": 0.7, "problems": []})
        return reply(
            SUBMIT,
            {
                "sql": None if self.declined else GOOD,
                "answer": "The number of loans.",
                "confidence": 0.8,
                "declined": self.declined,
                "decline_reason": "not recorded" if self.declined else None,
                "clarifying_question": None,
                "assumptions": ["loans are rows of loan"],
                "premise_correction": None,
                "chart_spec": None,
            },
        )


@pytest.fixture
def setup(tmp_path, bird_ready):
    q = next(q for q, _ in benchmark_set("pilot") if q.db_id == "financial")
    box = Toolbox("financial")
    agent_cfg = config()
    crit = confidence_config()["critic"]
    ledger = load_ledger("phase5", ledger_path=tmp_path / "spend.json")

    def make(backend, tracer=None):
        caller = Caller(
            backend, LayeredCache(ResponseCache(tmp_path / "c")), ledger, log=lambda _: None
        )
        sql_model, critic_model = bound_models(
            caller,
            {"model": SONNET, **agent_cfg["models"][SONNET]},
            {"model": crit["model"], **crit["settings"]},
        )
        ctx = GraphContext(
            toolbox=lambda db: box,
            sql_model=sql_model,
            critic_model=critic_model,
            sql_system=(ROOT / agent_cfg["prompts"]["single_shot"])
            .read_text(encoding="utf-8")
            .replace("\r\n", "\n"),
            critic_system=load_prompt(ROOT / crit["prompt"])[0],
            agent_cfg=agent_cfg,
            crit=crit,
            evidence=True,
            answer_run="main/x",
            cost=ledger.cost,
            tracer=tracer,
        )
        return build(ctx), ctx

    yield q, box, agent_cfg, crit, make
    box.close()


@pytest.mark.bird
def test_the_graphs_requests_are_the_own_loops(setup):
    q, box, agent_cfg, crit, make = setup
    backend = Scripted()
    app, ctx = make(backend)
    out = run_question(app, ctx, q)
    sql_req, critic_req = backend.calls
    ((_, own_sql),) = QuestionRun(
        q, "d1", SONNET, True, box, lambda *a: 0.0, None, agent_cfg
    ).pending()
    assert sql_req.cache_key == own_sql.cache_key
    a = out["state"]["answer"]
    answer = {k: a[k] for k in ("final_sql", "answer", "assumptions", "declined")}
    system, _ = load_prompt(ROOT / crit["prompt"])
    ((_, own_critic),) = CriticRun(
        q, answer, "main/x", box, crit, system, agent_cfg, lambda *a: 0.0
    ).pending()
    assert critic_req.cache_key == own_critic.cache_key
    assert [r.cache_key for r, _ in out["calls"]] == [sql_req.cache_key, critic_req.cache_key]
    assert out["state"]["verdict"]["confidence"] == 0.7 and a["confidence"] == 0.8


@pytest.mark.bird
def test_route_and_checkpoints(setup):
    q, _, _, _, make = setup
    app, ctx = make(Scripted())
    out = run_question(app, ctx, q)
    assert out["state"]["route"] == [
        "orchestrator",
        "sql_agent",
        "orchestrator",
        "verifier",
        "orchestrator",
    ]
    assert out["checkpoints"] >= 5  # one after every step, and the input


@pytest.mark.bird
def test_a_declined_answer_is_not_reviewed(setup):
    q, _, _, _, make = setup
    backend = Scripted(declined=True)
    app, ctx = make(backend)
    out = run_question(app, ctx, q)
    assert len(backend.calls) == 1
    v = out["state"]["verdict"]
    assert (v["reviewed"], v["not_reviewed"], v["confidence"]) == (False, "declined", 0.0)


@pytest.mark.bird
def test_span_tree(setup):
    q, _, _, _, make = setup
    exporter = MemoryExporter()
    tracer, provider = make_tracer(exporter)
    app, ctx = make(Scripted(), tracer)
    run_question(app, ctx, q)
    provider.shutdown()
    spans = {s.name: [] for s in exporter.spans}
    for s in exporter.spans:
        spans[s.name].append(s)
    (root,) = spans["graph.run"]
    assert all(
        s.parent.span_id == root.context.span_id for s in spans["graph.node"] + spans["llm.call"]
    )
    assert [s.attributes["node"] for s in spans["graph.node"]].count("orchestrator") == 3
    assert len(spans["llm.call"]) == 2


@pytest.mark.bird
def test_the_runner_compares_both_arms_question_by_question(tmp_path, bird_ready, monkeypatch):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("run_graph", ROOT / "scripts/55_run_graph.py")
    runner = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "run_graph", runner)  # its dataclass looks itself up there
    spec.loader.exec_module(runner)
    from src.agent.clock import ClockStore
    from src.agent.driver import ToolboxPool

    agent_cfg = config()
    crit = confidence_config()["critic"]
    ledger = load_ledger("phase5", ledger_path=tmp_path / "spend.json")
    caller = Caller(
        Scripted(), LayeredCache(ResponseCache(tmp_path / "c")), ledger, log=lambda _: None
    )
    critic_system = load_prompt(ROOT / crit["prompt"])[0]
    sql_model, critic_model = bound_models(
        caller,
        {"model": SONNET, **agent_cfg["models"][SONNET]},
        {"model": crit["model"], **crit["settings"]},
    )
    pool = ToolboxPool()
    arms = runner.Arms(
        caller,
        pool,
        ledger,
        agent_cfg,
        crit,
        critic_system,
        ClockStore(tmp_path / "c"),
        "main/x",
        "d1",
        True,
    )
    ctx = GraphContext(
        toolbox=pool.get,
        sql_model=sql_model,
        critic_model=critic_model,
        sql_system=(ROOT / agent_cfg["prompts"]["single_shot"])
        .read_text(encoding="utf-8")
        .replace("\r\n", "\n"),
        critic_system=critic_system,
        agent_cfg=agent_cfg,
        crit=crit,
        evidence=True,
        answer_run="main/x",
        cost=ledger.cost,
    )
    pairs = [x for x in benchmark_set("ablation") if x[0].db_id == "financial"][:2]
    try:
        records = runner.compare_arms(
            arms, build(ctx), ctx, [q for q, _ in pairs], {q.question_id: g for q, g in pairs}
        )
    finally:
        pool.close()
    assert len(records) == 2
    for r in records:
        assert r["same_requests"] and r["model_calls"] == 2 and len(r["own_loop_request_keys"]) == 2
        assert r["critic_confidence"] == r["own_loop_critic_confidence"] == 0.7
        assert r["stated_confidence"] == r["own_loop_stated_confidence"] == 0.8
        assert r["correct"] in (0, 1) and r["checkpoints"] >= 5
