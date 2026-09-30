"""The winning design and the critic as a LangGraph graph: an orchestrator routing between a SQL
sub-agent and a verifier sub-agent.

The algorithm is the own loop's winner-plus-critic, unchanged; only the orchestration moves into
LangGraph, so that a comparison of the two measures the framework, not the prompts:

- **orchestrator**: reads the state and routes, without a model call: to the SQL sub-agent while
  there is no answer, then to the verifier while there is no verdict, then to the end;
- **SQL sub-agent**: design 1: one call with the full schema, the same prompt and the same
  `submit_answer` tool, forced; the submitted query is then run through the agent's guard and the
  evaluation limits, and checked (src/agent/verify.py), as the own loop's run does;
- **verifier sub-agent**: the critic (src/agent/critic.py): the same prompt, evidence and
  `submit_verdict` tool; an answer without a result is not reviewed (confidence 0).

The graph's state (the question, the answer, its result, the verdict and the route taken) is
checkpointed after every step (LangGraph's in-memory checkpointer, one thread per question).
Model calls go through `CachedChatModel`, which builds the own loop's requests exactly; spans use
the own loop's names and attributes (`llm.call`), under a `graph.run` span per question with a
`graph.node` span per step.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from opentelemetry import trace as ot
from opentelemetry.trace import Span, Tracer

from src.agent import verify
from src.agent.answer import SUBMIT, SUBMIT_TOOL, Answer, parse_answer
from src.agent.clock import ClockStore
from src.agent.critic import (
    STAGE as CRITIC_STAGE,
)
from src.agent.critic import (
    VERDICT,
    VERDICT_TOOL,
    fetch_evidence,
    parse_verdict,
    review_text,
    schema_context,
)
from src.agent.run import Question, question_text
from src.agent.tools import AgentTools
from src.agent_langgraph.chat_model import CachedChatModel
from src.db.execute import Limits
from src.llm.types import LLMResponse, TokenUsage
from src.tools.toolbox import Toolbox

NO_ANSWER_STOPS = ("model_error", "refusal", "max_tokens", "context_overflow")


class GraphState(TypedDict, total=False):
    question: dict  # the Question's fields
    answer: dict  # the submitted answer, with `final_sql`
    result: dict  # the final query's outcome and checks
    verdict: dict  # the verifier's review
    route: list[str]  # the nodes visited, in order


@dataclass
class GraphContext:
    """What the nodes use: models, prompts, tools and where the spans go."""

    toolbox: Callable[[str], Toolbox]  # a toolbox for a database
    sql_model: CachedChatModel  # bound to submit_answer, forced
    critic_model: CachedChatModel  # bound to submit_verdict, forced
    sql_system: str
    critic_system: str
    agent_cfg: dict[str, Any]
    crit: dict[str, Any]
    evidence: bool
    answer_run: str  # the run name the critic's clock keys use
    cost: Callable[[str, TokenUsage, bool], float]
    clock: ClockStore | None = None
    tracer: Tracer | None = None
    span: Span | None = field(default=None, repr=False)  # the current question's graph.run


def _cached(text: str) -> list[dict]:
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _context_block(text: str) -> dict:
    """What the own loop sends first (src/agent/conversation.py): the text, cached."""
    return {"type": "text", "text": text.rstrip() + "\n\n", "cache_control": {"type": "ephemeral"}}


def _limits(ctx: GraphContext, box: Toolbox) -> Limits:
    return Limits(
        max_rows=box.cfg["evaluation"]["max_rows"],
        timeout_s=float(ctx.agent_cfg["checks"].get("timeout_s", 30)),
        count_total=False,
    )


def _submitted(ai: Any, tool: str) -> dict | None:
    """The input of the reply's call of `tool`, or None if the reply has no answer."""
    if ai.response_metadata.get("stop_reason") in NO_ANSWER_STOPS:
        return None
    calls = [c for c in ai.tool_calls if c["name"] == tool]
    return calls[0]["args"] if calls else None


def _llm_span(ctx: GraphContext, model: str, response: LLMResponse) -> None:
    if ctx.tracer is None or ctx.span is None:
        return
    now = time.time_ns()
    t = response.tokens
    batch = response.extra.get("service") == "batch"
    span = ctx.tracer.start_span(
        "llm.call",
        context=ot.set_span_in_context(ctx.span),
        start_time=now - int(response.latency_ms * 1e6),
    )
    span.set_attributes(
        {
            "model": model,
            "stop_reason": response.stop_reason or "",
            "input_tokens": t.input,
            "output_tokens": t.output,
            "cache_read_tokens": t.cache_read,
            "cache_write_tokens": t.cache_write_5m + t.cache_write_1h,
            "latency_ms": response.latency_ms,
            "batch": batch,
            "cost_usd": ctx.cost(model, t, batch),
        }
    )
    span.end(end_time=now)


def _invoke(ctx: GraphContext, model: CachedChatModel, messages: list) -> Any:
    before = len(model.calls)
    ai = model.invoke(messages)
    for _, response in model.calls[before:]:
        _llm_span(ctx, model.model, response)
    return ai


def _node_span(ctx: GraphContext, name: str) -> Span | None:
    if ctx.tracer is None or ctx.span is None:
        return None
    span = ctx.tracer.start_span("graph.node", context=ot.set_span_in_context(ctx.span))
    span.set_attribute("node", name)
    return span


def build(ctx: GraphContext):
    """The compiled graph, with an in-memory checkpointer."""

    def orchestrator(state: GraphState) -> GraphState:
        span = _node_span(ctx, "orchestrator")
        if span is not None:
            span.end()
        return {"route": [*state.get("route", []), "orchestrator"]}

    def next_step(state: GraphState) -> str:
        if "answer" not in state:
            return "sql_agent"
        if "verdict" not in state:
            return "verifier"
        return END

    def sql_agent(state: GraphState) -> GraphState:
        span = _node_span(ctx, "sql_agent")
        q = Question(**state["question"])
        box = ctx.toolbox(q.db_id)
        tools = AgentTools(box, [], ctx.agent_cfg["presentation"]["rows_shown"])
        context = "Database: " + q.db_id + "\n\nSchema:\n" + _json(tools.full_schema())
        messages = [
            SystemMessage(content=_cached(ctx.sql_system)),
            HumanMessage(
                content=[
                    _context_block(context),
                    {"type": "text", "text": question_text(q, ctx.evidence)},
                ]
            ),
        ]
        args = _submitted(_invoke(ctx, ctx.sql_model, messages), SUBMIT)
        answer = parse_answer(args) if args is not None else Answer.none()
        sql = None if answer.declined else answer.sql
        result = verify.fetch(box.sql.guard, box.executor, sql, _limits(ctx, box))
        checks = verify.checks(sql, result, ctx.agent_cfg["checks"]["max_rows"])
        if span is not None:
            span.end()
        return {
            "answer": {**answer.to_dict(), "final_sql": sql, "answered": args is not None},
            "result": {
                "ok": result.ok,
                "error": result.error,
                "rows": len(result.rows),
                "checks": checks,
            },
            "route": [*state["route"], "sql_agent"],
        }

    def verifier(state: GraphState) -> GraphState:
        span = _node_span(ctx, "verifier")
        q = Question(**state["question"])
        a = state["answer"]
        box = ctx.toolbox(q.db_id)
        not_reviewed, evidence = None, None
        if a["declined"]:
            not_reviewed = "declined"
        elif not a["final_sql"]:
            not_reviewed = "no_sql"
        else:
            evidence = fetch_evidence(
                box,
                a["final_sql"],
                _limits(ctx, box),
                ctx.agent_cfg["checks"]["max_rows"],
                ctx.crit["rows_shown"],
                ctx.clock,
                (CRITIC_STAGE, f"{ctx.answer_run}#{q.question_id}"),
            )
            if not evidence["ok"]:
                not_reviewed = f"final_sql_{(evidence['error'] or {}).get('kind')}"
        verdict = {"reviewed": not_reviewed is None, "not_reviewed": not_reviewed}
        if not_reviewed is None:
            tools = AgentTools(box, [], ctx.crit["rows_shown"])
            messages = [
                SystemMessage(content=_cached(ctx.critic_system)),
                HumanMessage(
                    content=[
                        _context_block(schema_context(tools, q.db_id)),
                        {
                            "type": "text",
                            "text": review_text(
                                question_text(q, ctx.evidence),
                                {k: a[k] for k in ("final_sql", "answer", "assumptions")},
                                evidence,
                                ctx.agent_cfg["checks"]["max_rows"],
                            ),
                        },
                    ]
                ),
            ]
            args = _submitted(_invoke(ctx, ctx.critic_model, messages), VERDICT)
            v = parse_verdict(args) if args is not None else None
            verdict |= {
                "verdict": v.verdict if v else None,
                "confidence": v.confidence if v else None,
                "problems": list(v.problems) if v else [],
            }
        else:
            verdict |= {"verdict": None, "confidence": 0.0, "problems": []}
        if span is not None:
            span.end()
        return {"verdict": verdict, "route": [*state["route"], "verifier"]}

    g = StateGraph(GraphState)
    g.add_node("orchestrator", orchestrator)
    g.add_node("sql_agent", sql_agent)
    g.add_node("verifier", verifier)
    g.add_edge(START, "orchestrator")
    g.add_conditional_edges(
        "orchestrator", next_step, {"sql_agent": "sql_agent", "verifier": "verifier", END: END}
    )
    g.add_edge("sql_agent", "orchestrator")
    g.add_edge("verifier", "orchestrator")
    return g.compile(checkpointer=InMemorySaver())


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def bound_models(caller: Any, sql: dict[str, Any], critic: dict[str, Any]) -> tuple:
    """The SQL sub-agent's and the verifier's chat models, from their model settings (the
    agent's for the winning design, the critic's), sharing one call log."""
    base = CachedChatModel(
        model=sql["model"],
        backend=sql["backend"],
        max_tokens=sql["max_tokens"],
        params=sql["params"],
        caller=caller,
    )
    critic_base = base.model_copy(
        update={
            "model": critic["model"],
            "backend": critic["backend"],
            "max_tokens": critic["max_tokens"],
            "params": critic["params"],
        }
    )
    return (
        base.bind_tools([SUBMIT_TOOL], tool_choice={"type": "tool", "name": SUBMIT}),
        critic_base.bind_tools([VERDICT_TOOL], tool_choice={"type": "tool", "name": VERDICT}),
    )


def run_question(app: Any, ctx: GraphContext, q: Question) -> dict[str, Any]:
    """Run the graph on one question: its final state, the checkpoints saved and the calls."""
    calls = ctx.sql_model.calls  # shared by both models
    before = len(calls)
    if ctx.tracer is not None:
        ctx.span = ctx.tracer.start_span("graph.run")
        ctx.span.set_attributes({"question_id": str(q.question_id), "db_id": q.db_id})
    config = {"configurable": {"thread_id": str(q.question_id)}}
    t0 = time.perf_counter()
    state = app.invoke({"question": asdict(q), "route": []}, config)
    seconds = time.perf_counter() - t0
    checkpoints = len(list(app.get_state_history(config)))
    if ctx.span is not None:
        ctx.span.set_attribute("route", ",".join(state["route"]))
        ctx.span.end()
        ctx.span = None
    return {
        "state": state,
        "seconds": seconds,
        "checkpoints": checkpoints,
        "calls": calls[before:],
    }
