"""Run the LangGraph arm (the winning design and the critic as a graph) on the ablation set, beside
the own loop's same pipeline, from stored responses only.

For every question of the ablation set, in turn: the own loop's winning design and then its
critic (src/agent/run.py, src/agent/critic.py), and the graph (src/agent_langgraph/graph.py), in
alternating order so that neither always runs first; before a database's first question, one
query reconnects to it, so neither arm pays for the connection. Both arms are traced the same
way (spans to JSON lines), since writing spans is part of the time measured. Both run with
ANALYST_REPLAY_ONLY=1 and a backend that refuses every call: each model response must come from
the stored responses of the design comparison, the winner's runs and the critic, so this costs
nothing and a request that differs by a byte fails instead of calling a model. Per question it
records the graph's answer, scored as the own loop's answers are, its critic confidence, both
arms' request keys (the requests must be identical), model calls and time (orchestration, tools,
the database and the spans; model calls come from the cache, so the time is the orchestration
overhead), and the checkpoints LangGraph saved.

Writes results/runs/framework/ablation-graph-claude-sonnet-5.jsonl and both arms' spans under
data/spans/framework/.

Usage:
    uv run --group langgraph python scripts/55_run_graph.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import critic  # noqa: E402
from src.agent.clock import ClockStore  # noqa: E402
from src.agent.confidence import answers_path, confidence_config, winning_design  # noqa: E402
from src.agent.driver import Caller, LayeredCache, ToolboxPool  # noqa: E402
from src.agent.evaluate import REFUSED, benchmark_set, score  # noqa: E402
from src.agent.run import Question, QuestionRun, config  # noqa: E402
from src.agent_langgraph.graph import GraphContext, bound_models, build, run_question  # noqa: E402
from src.eval.execution import Scorer  # noqa: E402
from src.llm.cache import ResponseCache  # noqa: E402
from src.llm.ledger import load_ledger  # noqa: E402
from src.tools.toolbox import benchmark_target  # noqa: E402
from src.tracking.otel import jsonl_tracer  # noqa: E402

OUT = ROOT / "results/runs/framework/ablation-graph-claude-sonnet-5.jsonl"
SPANS = ROOT / "data/spans/framework"
CACHE = ROOT / "data/cache/llm_framework"
READS = ("ablation", "main", "critic")
MODEL = "claude-sonnet-5"


class NoModel:
    """A backend that is never reached: in replay-only mode a missing response is an error."""

    name = "anthropic"

    def check(self, request):
        raise RuntimeError("the graph arm makes no model call")

    def generate(self, request):
        raise RuntimeError("the graph arm makes no model call")


@dataclass
class Arms:
    """What both arms share: the model calls, the tools, the settings."""

    caller: Caller
    pool: ToolboxPool
    ledger: Any
    agent_cfg: dict
    crit: dict
    critic_system: str
    clock: ClockStore
    answer_run: str
    design: str
    evidence: bool
    own_tracer: Any = None


def own_loop(arms: Arms, q: Question) -> tuple[float, dict, list[str], float]:
    """The own loop's winner-plus-critic on one question: (stated confidence, critic record,
    request keys, seconds)."""
    box = arms.pool.get(q.db_id)
    keys = []
    t0 = time.perf_counter()
    cost, tracer = arms.ledger.cost, arms.own_tracer
    run = QuestionRun(
        q, arms.design, MODEL, arms.evidence, box, cost, tracer, arms.agent_cfg, arms.clock
    )
    while not run.done:
        for conv, req in run.pending():
            keys.append(req.cache_key)
            run.feed(conv, req.cache_key, arms.caller.one(req))
    fin = run.finish()
    answer = {
        "final_sql": fin.final_sql,
        "answer": fin.answer.answer,
        "assumptions": list(fin.answer.assumptions),
        "declined": fin.answer.declined,
    }
    review = critic.CriticRun(
        q,
        answer,
        arms.answer_run,
        box,
        arms.crit,
        arms.critic_system,
        arms.agent_cfg,
        cost,
        tracer,
        arms.clock,
        arms.evidence,
    )
    while not review.done:
        for conv, req in review.pending():
            keys.append(req.cache_key)
            review.feed(conv, req.cache_key, arms.caller.one(req))
    record, _ = review.finish()
    return fin.confidence, record, keys, time.perf_counter() - t0


def warm(arms: Arms, db: str) -> None:
    """Reconnect to a database (a toolbox closes its connection when another database is used),
    outside the time measured."""
    box = arms.pool.get(db)
    box.sql.run_sql(f'SELECT 1 FROM "{box.schema.tables[0]}" LIMIT 1')


def compare_arms(arms: Arms, app: Any, ctx: GraphContext, items: list, golds: dict) -> list[dict]:
    """Both arms on every question (alternating which goes first), then the graph's answers
    scored; one record per question."""
    runs = []
    last_db = None
    for i, q in enumerate(sorted(items, key=lambda q: (q.db_id, q.question_id))):
        if q.db_id != last_db:
            warm(arms, q.db_id)
            last_db = q.db_id
        if i % 2:
            g = run_question(app, ctx, q)
            o = own_loop(arms, q)
        else:
            o = own_loop(arms, q)
            g = run_question(app, ctx, q)
        runs.append((q, o, g))

    records = []
    when = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    for db in sorted({q.db_id for q, _, _ in runs}):
        with Scorer({db: benchmark_target(db)}) as scorer:
            for q, o, g in (r for r in runs if r[0].db_id == db):
                st = g["state"]
                a, v, result = st["answer"], st["verdict"], st["result"]
                sql = a["final_sql"]
                refused = (
                    sql is not None
                    and not result["ok"]
                    and (result["error"] or {}).get("kind") == "refused"
                )
                correct, soft_f1, outcome = (
                    (0, 0.0, REFUSED) if refused else score(scorer, q, golds[q.question_id], sql)
                )
                keys = [req.cache_key for req, _ in g["calls"]]
                cost = sum(
                    arms.ledger.cost(req.model, resp.tokens, resp.extra.get("service") == "batch")
                    for req, resp in g["calls"]
                )
                stated, own_record, own_keys, own_seconds = o
                records.append(
                    {
                        "question_id": q.question_id,
                        "db_id": q.db_id,
                        "difficulty": q.difficulty,
                        "final_sql": sql,
                        "declined": a["declined"],
                        "stated_confidence": a["confidence"] or 0.0,
                        "critic_reviewed": v["reviewed"],
                        "critic_not_reviewed": v["not_reviewed"],
                        "critic_verdict": v.get("verdict"),
                        "critic_confidence": v.get("confidence"),
                        "correct": correct,
                        "soft_f1": soft_f1,
                        "score_outcome": outcome,
                        "route": st["route"],
                        "checkpoints": g["checkpoints"],
                        "model_calls": len(keys),
                        "request_keys": keys,
                        "own_loop_request_keys": own_keys,
                        "same_requests": keys == own_keys,
                        "own_loop_critic_confidence": own_record["confidence"],
                        "own_loop_stated_confidence": stated,
                        "cost_usd": round(cost, 8),
                        "graph_seconds": round(g["seconds"], 6),
                        "own_loop_seconds": round(own_seconds, 6),
                        "evaluated_at": when,
                    }
                )
    return sorted(records, key=lambda r: r["question_id"])


def main() -> None:
    os.environ["ANALYST_REPLAY_ONLY"] = "1"  # every response from the cache; a miss is an error
    os.environ["LANGSMITH_TRACING"] = "false"  # nothing leaves the machine
    conf = confidence_config()
    crit = conf["critic"]
    agent_cfg = config()
    _, answer_run = answers_path(conf)
    ledger = load_ledger(conf["phase"])
    cache = LayeredCache(
        ResponseCache(CACHE), [ResponseCache(ROOT / f"data/cache/llm_{s}") for s in READS]
    )
    caller = Caller(NoModel(), cache, ledger, log=lambda _: None)
    critic_system, _ = critic.load_prompt(ROOT / crit["prompt"])
    sql_system = (ROOT / agent_cfg["prompts"]["single_shot"]).read_text(encoding="utf-8")
    sql_settings = {"model": MODEL, **agent_cfg["models"][MODEL]}
    critic_settings = {"model": crit["model"], **crit["settings"]}
    sql_model, critic_model = bound_models(caller, sql_settings, critic_settings)
    for f in SPANS.glob("ablation-*.jsonl"):
        f.unlink()  # spans describe this execution only
    graph_tracer, graph_provider = jsonl_tracer(SPANS / "ablation-graph-claude-sonnet-5.jsonl")
    own_tracer, own_provider = jsonl_tracer(SPANS / "ablation-own-loop-claude-sonnet-5.jsonl")
    pool = ToolboxPool()
    arms = Arms(
        caller=caller,
        pool=pool,
        ledger=ledger,
        agent_cfg=agent_cfg,
        crit=crit,
        critic_system=critic_system,
        clock=ClockStore(CACHE, [ROOT / "data/cache/llm_critic"]),
        answer_run=answer_run,
        design=winning_design(),
        evidence=conf["answers"]["evidence"],
        own_tracer=own_tracer,
    )
    ctx = GraphContext(
        toolbox=pool.get,
        sql_model=sql_model,
        critic_model=critic_model,
        sql_system=sql_system.replace("\r\n", "\n"),
        critic_system=critic_system,
        agent_cfg=agent_cfg,
        crit=crit,
        evidence=arms.evidence,
        answer_run=answer_run,
        cost=ledger.cost,
        clock=arms.clock,
        tracer=graph_tracer,
    )
    pairs = benchmark_set("ablation")
    try:
        records = compare_arms(
            arms, build(ctx), ctx, [q for q, _ in pairs], {q.question_id: g for q, g in pairs}
        )
    finally:
        pool.close()
        graph_provider.shutdown()
        own_provider.shutdown()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    same = sum(r["same_requests"] for r in records)
    print(f"graph on {len(records)} questions: requests identical to the own loop's on {same}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
