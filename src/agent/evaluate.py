"""Running a design and model over a question set, scoring the answers, and writing the results.

A question set comes as pairs: the `Question` the agent is given (its text and, for the
benchmark, its hint) and the gold it is scored against, kept apart so that no gold query can
reach a prompt. Scoring (src/eval/execution.py) runs the answer's SQL and the gold query in one
read-only transaction and compares their rows as sets, as the benchmark's official evaluator
does. An answer whose final SQL the query guard refuses is wrong without being run (outcome
`refused`): the system would never have run it, so it has no result, as a refusal is the
answer's error in every design. For the hand-written banking set, the standard, multi-step and
comparative questions are
scored against their gold query, an ambiguous one against each accepted reading (correct if any
matches), and the unanswerable and false-premise ones are not scored by execution (their
behavior is scored from the answer's fields, src/eval/summary.py).

Each run writes one record per question (src/eval/records.py) to a JSON-lines file, one trace
per question, and the run's spans.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from src.agent.clock import ClockStore
from src.agent.driver import Caller, LayeredCache, ToolboxPool, drive_batch, drive_direct
from src.agent.run import Finished, Question, QuestionRun, config
from src.data import bird
from src.eval import own_set
from src.eval.config import config as eval_config
from src.eval.execution import Scorer
from src.eval.records import write_records
from src.llm.backends import Backend, make_backend
from src.llm.batch import BATCH_DIR
from src.llm.cache import ResponseCache
from src.llm.ledger import load_ledger
from src.llm.types import TokenUsage
from src.tools.toolbox import benchmark_target
from src.tracking.otel import jsonl_tracer

ROOT = Path(__file__).resolve().parent.parent.parent
OWN_DB = "financial"
REFUSED = "refused"  # the score outcome of a final SQL the query guard refuses


@dataclass(frozen=True)
class Gold:
    """What an answer is scored against: gold queries (several for an ambiguous question, any
    of which counts), or none."""

    queries: tuple[str, ...]


def benchmark_set(name: str) -> list[tuple[Question, Gold]]:
    """The questions of one split (pilot, ablation, held_out) or `all`, in id order."""
    splits = json.loads((ROOT / eval_config()["splits_file"]).read_text(encoding="utf-8"))
    ids = (
        sorted(i for s in splits["sets"].values() for i in s)
        if name == "all"
        else splits["sets"][name]
    )
    by_id = {q["question_id"]: q for q in bird.questions()}
    out = []
    for i in ids:
        q = by_id[i]
        out.append(
            (
                Question(
                    q["question_id"],
                    "bird",
                    q["db_id"],
                    q["question"],
                    q.get("evidence") or None,
                    difficulty=q["difficulty"],
                ),
                Gold((q["SQL"],)),
            )
        )
    return out


def banking_set() -> list[tuple[Question, Gold]]:
    data = own_set.load(ROOT / eval_config()["own_set"]["path"])
    out = []
    for q in data["questions"]:
        golds = tuple(sql for label, sql in own_set.queries(q) if label != "premise")
        out.append(
            (
                Question(q["id"], "own", OWN_DB, q["question"], None, category=q["category"]),
                Gold(golds),
            )
        )
    return out


def refused(fin: Finished) -> bool:
    """Whether the answer's final SQL is one the query guard refuses."""
    return (
        fin.final_sql is not None
        and not fin.result.ok
        and (fin.result.error or {}).get("kind") == "refused"
    )


def score(scorer: Scorer, q: Question, gold: Gold, sql: str | None) -> tuple[Any, Any, str]:
    """(correct, soft_f1, score_outcome) for an answer's SQL."""
    if not gold.queries:
        return None, None, "not_scored"
    best = None
    for g in gold.queries:
        s = scorer.score(q.db_id, sql, g)
        if best is None or (s.ex, s.soft_f1 or 0.0) > (best.ex, best.soft_f1 or 0.0):
            best = s
    return best.ex, best.soft_f1, best.outcome


def _tokens(u: TokenUsage) -> dict[str, int]:
    return {
        "input": u.input,
        "output": u.output,
        "cache_write_5m": u.cache_write_5m,
        "cache_write_1h": u.cache_write_1h,
        "cache_read": u.cache_read,
    }


def make_record(
    run: str,
    q: Question,
    design: str,
    model: str,
    evidence: bool,
    fin: Finished,
    scored: tuple[Any, Any, str],
    trace: str | None,
    evaluated_at: str,
) -> dict[str, Any]:
    a = fin.answer
    correct, soft_f1, outcome = scored
    return {
        "run": run,
        "question_id": q.question_id,
        "source": q.source,
        "db_id": q.db_id,
        "difficulty": q.difficulty,
        "category": q.category,
        "design": design,
        "model": model,
        "evidence": bool(evidence and q.source == "bird"),
        "final_sql": fin.final_sql,
        "answer": a.answer,
        "confidence": fin.confidence,
        "declined": a.declined,
        "decline_reason": a.decline_reason,
        "clarifying_question": a.clarifying_question,
        "assumptions": list(a.assumptions),
        "premise_correction": a.premise_correction,
        "correct": correct,
        "soft_f1": soft_f1,
        "score_outcome": outcome,
        "tokens": _tokens(fin.usage),
        "cost_usd": round(fin.cost_usd, 8),
        "latency_s": round(fin.latency_s, 4),
        "steps": fin.steps,
        "tool_calls": fin.tool_calls,
        "errors": [{"kind": e["kind"], "message": str(e["message"])} for e in fin.errors],
        "trace": trace,
        "evaluated_at": evaluated_at,
    }


@dataclass
class RunSpec:
    name: str  # e.g. "ablation/d3/claude-sonnet-5/evidence"
    items: list[tuple[Question, Gold]]
    design: str
    model: str
    evidence: bool
    phase: str
    cache_dir: Path  # where this stage's responses are stored
    read_caches: tuple[Path, ...] = ()  # earlier stages' responses, read-only
    mode: str = "direct"  # or "batch"
    records: Path | None = None
    traces_dir: Path | None = None
    spans: Path | None = None
    ledger_path: Path | None = None  # the spend ledger file; None is the project's (tests: a copy)
    batch_dir: Path = BATCH_DIR  # where submitted batches are recorded (tests: their own)
    poll_seconds: float | None = None  # between batch status checks; None: configs/agent.yaml


def execute(
    spec: RunSpec, log: Callable[[str], None] = print, backend: Backend | None = None
) -> list[dict]:
    """Run, score and write one run. `backend` replaces the model's configured one (tests)."""
    return execute_many([spec], log, backend)[0]


def execute_many(
    specs: list[RunSpec], log: Callable[[str], None] = print, backend: Backend | None = None
) -> list[list[dict]]:
    """Run several runs of one stage. Direct runs go one after another. Batched runs go
    together: every round sends the pending requests of all of them in one set of batches, so a
    stage takes as many rounds as its longest run, not the sum; and identical requests (design
    4's first sample is design 3's run) are sent once. Returns each run's records."""
    shared = {
        (s.mode, s.cache_dir, s.read_caches, s.phase, s.ledger_path, s.batch_dir) for s in specs
    }
    if len(shared) != 1:
        raise ValueError("runs executed together must share mode, caches, phase and ledger")
    cfg = config()
    backends = {cfg["models"][s.model]["backend"] for s in specs}
    if specs[0].mode == "batch" and backends != {"anthropic"}:
        raise ValueError("batch mode needs the Anthropic backend")
    if specs[0].mode != "batch" and len(specs) > 1:
        return [execute_many([s], log, backend)[0] for s in specs]

    spec = specs[0]
    kind = backends.pop()
    backend = backend or make_backend(kind)
    ledger = load_ledger(spec.phase, ledger_path=spec.ledger_path) if kind == "anthropic" else None
    cost = ledger.cost if ledger is not None else (lambda model, usage, batch=False: 0.0)
    cache = LayeredCache(
        ResponseCache(spec.cache_dir), [ResponseCache(p) for p in spec.read_caches]
    )
    caller = Caller(
        backend,
        cache,
        ledger,
        cfg["run"]["batch_poll_seconds"] if spec.poll_seconds is None else spec.poll_seconds,
        spec.batch_dir,
        log=log,
    )
    tracers = [jsonl_tracer(s.spans) if s.spans else (None, None) for s in specs]
    pool = ToolboxPool()
    clock = ClockStore(spec.cache_dir, spec.read_caches)  # beside the responses, same layers

    def make(s: RunSpec, tracer, q: Question) -> QuestionRun:
        box = pool.get(q.db_id)
        return QuestionRun(q, s.design, s.model, s.evidence, box, cost, tracer, cfg, clock)

    # questions of one database together, so a cached schema prefix is reused while it is warm
    jobs = [
        (k, i)
        for k, s in enumerate(specs)
        for i in sorted(range(len(s.items)), key=lambda i, s=s: (s.items[i][0].db_id, i))
    ]
    makers = [partial(make, specs[k], tracers[k][0], specs[k].items[i][0]) for k, i in jobs]
    try:
        if spec.mode == "batch":
            done = drive_batch(makers, caller, log)
        else:
            workers = 1 if kind == "ollama" else cfg["run"]["workers"]
            done = drive_direct(makers, caller, workers)
    finally:
        pool.close()
        for _, provider in tracers:
            if provider is not None:
                provider.shutdown()

    finished: list[dict] = [{} for _ in specs]
    for (k, i), (_, fin) in zip(jobs, done, strict=True):
        finished[k][specs[k].items[i][0].question_id] = fin
    return [_score_and_write(s, f) for s, f in zip(specs, finished, strict=True)]


def _score_and_write(spec: RunSpec, finished: dict) -> list[dict]:
    """Score one database's questions at a time, so scoring holds one connection (the agent's
    role has a connection limit, shared with any run going on at the same time). Records come
    out in the order of the run's questions."""
    by_position: dict[int, dict] = {}
    for db in sorted({q.db_id for q, _ in spec.items}):
        with Scorer({db: benchmark_target(db)}) as scorer:
            for pos, (q, gold) in enumerate(spec.items):
                if q.db_id == db:
                    by_position[pos] = _score_one(spec, scorer, q, gold, finished[q.question_id])
    records = [by_position[p] for p in range(len(spec.items))]
    if spec.records is not None:
        write_records(spec.records, records)
    return records


def _score_one(spec: RunSpec, scorer: Scorer, q: Question, gold: Gold, fin: Finished) -> dict:
    """Write the question's trace, score its answer, and return its record."""
    trace_rel = None
    if spec.traces_dir is not None:
        path = spec.traces_dir / f"{q.question_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(fin.trace, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
            newline="\n",
        )
        trace_rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    when = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    if refused(fin) and gold.queries:
        scored = (0, 0.0, REFUSED)  # the system would never have run it: no result, wrong
    else:
        scored = score(scorer, q, gold, fin.final_sql)
    return make_record(
        spec.name, q, spec.design, spec.model, spec.evidence, fin, scored, trace_rel, when
    )
