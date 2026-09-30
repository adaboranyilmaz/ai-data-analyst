"""The framework comparison: the LangGraph arm against the own loop, on the ablation set.

Refuses to run unless the pre-registration is frozen and unchanged. From the graph's run
(results/runs/framework/) and the own loop's runs of the same pipeline (the winning design's
answers, results/runs/ablation/, and the critic's reviews, results/runs/critic/), paired on the
same questions:

- execution accuracy and AURC of the pipeline's confidence (the critic's), graph minus own loop,
  with paired intervals; AURC of the stated confidence too;
- the pre-registered rule: the own loop stays the main system unless the graph is better on both,
  with paired 95% intervals excluding zero (the prediction P15);
- whether every request the graph sent was byte-identical to the own loop's (same cache key);
- cost per correct answer (from the stored token counts), model calls per question, LangGraph's
  checkpoints per question, and the time per question with every model response served from the
  cache: the orchestration, tools and database, so the overhead each orchestration adds (the
  model's own latency is the same by construction: the same requests);
- code size: logical lines (not blank, not comments, not docstrings) of the graph arm's modules and
  of the own loop's orchestration modules, with what each covers.

Every result is from one run. Writes results/metrics/framework_comparison.json.

Usage:
    uv run python scripts/56_framework_report.py
"""

from __future__ import annotations

import ast
import io
import json
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.agent.confidence import (  # noqa: E402
    confidence_config,
    run_name,
    winning_design,
)
from src.agent.confidence import read_records as read_reviews  # noqa: E402
from src.data.bird import write_json  # noqa: E402
from src.eval import preregistration  # noqa: E402
from src.eval.records import read_records  # noqa: E402
from src.eval.reports import with_predictions  # noqa: E402
from src.eval.summary import compare, mean_interval  # noqa: E402

GRAPH = ROOT / "results/runs/framework/ablation-graph-claude-sonnet-5.jsonl"
OUT = ROOT / "results/metrics/framework_comparison.json"
MODEL = "claude-sonnet-5"
CODE = {
    "graph_arm": {
        "files": ["src/agent_langgraph/graph.py", "src/agent_langgraph/chat_model.py"],
        "covers": "the winning design and the critic as a graph (routing, the two sub-agents, "
        "checkpointing, spans) and the chat-model adapter over the cache and ledger",
    },
    "own_loop": {
        "files": ["src/agent/conversation.py", "src/agent/run.py", "src/agent/driver.py"],
        "covers": "the own loop for all five designs (tools, budgets, resubmission, reminders, "
        "samples and the vote, narrowing), direct and batched rounds, spans",
    },
    "critic_module": {
        "files": ["src/agent/critic.py"],
        "covers": "the critic's tool, evidence and review text, used by both, and the own "
        "loop's critic runs",
    },
}


def logical_lines(path: Path) -> int:
    """Lines holding code: not blank, not only a comment, not part of a docstring."""
    text = path.read_text(encoding="utf-8")
    docstring_lines: set[int] = set()
    for node in ast.walk(ast.parse(text)):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                docstring_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    code_lines: set[int] = set()
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type not in skip and tok.type != tokenize.ENDMARKER:
            code_lines.update(range(tok.start[0], tok.end[0] + 1))
    return len(code_lines - docstring_lines)


def pipeline_records(answers: list[dict], reviews: list[dict], missing: float) -> list[dict]:
    """The own loop's pipeline per question: the answer, with the critic's confidence and both
    calls' cost."""
    by_id = {r["question_id"]: r for r in reviews}
    out = []
    for a in answers:
        r = by_id[a["question_id"]]
        c = missing if r["confidence"] is None else r["confidence"]
        out.append({**a, "confidence": c, "cost_usd": a["cost_usd"] + r["cost_usd"]})
    return out


def graph_records(graph: list[dict], missing: float, stated: bool = False) -> list[dict]:
    out = []
    for g in graph:
        c = g["stated_confidence"] if stated else g["critic_confidence"]
        out.append(
            {
                "question_id": g["question_id"],
                "correct": g["correct"],
                "declined": g["declined"],
                "confidence": missing if c is None else c,
                "cost_usd": g["cost_usd"],
            }
        )
    return out


def percentiles(xs: list[float]) -> dict:
    return {"p50": float(np.percentile(xs, 50)), "p95": float(np.percentile(xs, 95))}


def report(
    graph: list[dict],
    answers: list[dict],
    reviews: list[dict],
    missing: float,
    predictions: dict,
    code: dict,
    cfg: dict | None = None,
) -> dict:
    own = pipeline_records(answers, reviews, missing)
    ours = graph_records(graph, missing)
    diff = compare(ours, own, cfg)
    stated_diff = compare(graph_records(graph, missing, stated=True), answers, cfg)
    ex, aurc = diff["execution_accuracy"], diff["aurc"]
    replaced = bool(ex["low"] > 0 and aurc["high"] < 0)  # higher accuracy and lower AURC

    def per_correct(records: list[dict]) -> float | None:
        right = sum(r["correct"] == 1 and not r["declined"] for r in records)
        return sum(r["cost_usd"] for r in records) / right if right else None

    own_calls = [len(g["own_loop_request_keys"]) for g in graph]
    out = {
        "note": "one run; the ablation set; both arms from the same stored responses "
        "(replay-only); intervals are 95% bootstrap over questions, paired",
        "questions": len(graph),
        "requests_identical": sum(g["same_requests"] for g in graph),
        "graph_minus_own_loop": {
            "execution_accuracy": ex,
            "aurc_pipeline_confidence": aurc,
            "aurc_stated_confidence": stated_diff["aurc"],
            "accuracy_at_80": diff["accuracy_at_80"],
        },
        "rule": {
            "requires": "the graph better on both execution accuracy and AURC, with paired 95% "
            "intervals excluding zero",
            "replaces_own_loop": replaced,
        },
        "cost_per_correct_answer_usd": {"graph": per_correct(ours), "own_loop": per_correct(own)},
        "model_calls_per_question": {
            "graph": mean_interval([g["model_calls"] for g in graph], cfg),
            "own_loop": mean_interval(own_calls, cfg),
        },
        "checkpoints_per_question": float(np.mean([g["checkpoints"] for g in graph])),
        "seconds_per_question_cached_calls": {
            "note": "every model response from the cache: orchestration, tools and the database "
            "only; the model's latency is the same for both (the same requests)",
            "graph": percentiles([g["graph_seconds"] for g in graph]),
            "own_loop": percentiles([g["own_loop_seconds"] for g in graph]),
        },
        "code_lines": code,
    }
    out["predictions"] = with_predictions(
        {"P15": {"replaces_own_loop": replaced, "execution_accuracy": ex, "aurc": aurc}},
        predictions,
    )
    return out


def main() -> None:
    preregistration.require()
    conf = confidence_config()
    crit = conf["critic"]
    design = winning_design()
    graph = [json.loads(line) for line in GRAPH.read_text(encoding="utf-8").splitlines() if line]
    answers = read_records(
        ROOT / "results/runs/ablation" / f"{run_name('ablation', design, MODEL, True)}.jsonl"
    )
    reviews = read_reviews(ROOT / f"results/runs/critic/ablation-critic-{crit['model']}.jsonl")
    code = {
        k: {**v, "lines": sum(logical_lines(ROOT / f) for f in v["files"])} for k, v in CODE.items()
    }
    predictions = {
        p["id"]: p
        for p in preregistration.parse(
            (ROOT / "results/metrics/preregistration.md").read_text(encoding="utf-8")
        )["predictions"]
    }
    out = report(graph, answers, reviews, crit["missing_confidence"], predictions, code)
    write_json(OUT, out)
    d = out["graph_minus_own_loop"]
    print(
        f"requests identical {out['requests_identical']}/{out['questions']}; "
        f"EX diff {d['execution_accuracy']['estimate']:+.3f}, "
        f"AURC diff {d['aurc_pipeline_confidence']['estimate']:+.4f}; "
        f"replaces own loop: {out['rule']['replaces_own_loop']}"
    )
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
