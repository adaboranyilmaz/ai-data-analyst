"""The critic: a second model call reviews an answer with its evidence and gives its own confidence.

For each answer of the winning design, the critic reads what a reviewer of the answer would: the
database's schema (as design 1 read it), the question and its hint, the answer's SQL, its answer
in words and the assumptions it states, the query's result (its columns, the total number of
rows and the first `rows_shown` rows) and the automatic checks (src/agent/verify.py). It answers
through one call of a strict `submit_verdict` tool: a verdict (`correct`, `incorrect`, `unsure`),
the probability that the result is correct, and the problems it found. It never sees the answer's
stated confidence, and no gold query is ever read here.

An answer with no result is not reviewed: a declined answer, one with no SQL, and one whose query
the guard refuses or that fails. It returns no result, so the probability that it is correct is
0 (as calibration counts a declined answer), and the critic's confidence is 0 without a call.

The result is fetched again here, through the same guard, limits and checks as the answer's own
run (src/agent/run.py `finish`). A query that reads the clock goes through the clock store
(src/agent/clock.py), keyed by the answer it belongs to, so a replay shows the critic the result
of the first run and sends the same request. Every review goes through the response cache and the
spend ledger, in batched rounds or directly, like the agent's runs (src/agent/driver.py).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from opentelemetry import trace as ot
from opentelemetry.trace import Tracer

from src.agent import verify
from src.agent.clock import ClockStore
from src.agent.confidence import (
    VERDICTS,
    write_records,
)
from src.agent.conversation import Conversation, Settings
from src.agent.driver import Caller, LayeredCache, ToolboxPool, drive_batch, drive_direct
from src.agent.evaluate import benchmark_set
from src.agent.run import Question, config, question_text
from src.agent.tools import AgentTools
from src.db.execute import Limits
from src.llm.backends import Backend, make_backend
from src.llm.batch import BATCH_DIR
from src.llm.cache import ResponseCache
from src.llm.ledger import load_ledger
from src.llm.types import LLMRequest, LLMResponse, TokenUsage
from src.tools.sql import display_value
from src.tools.toolbox import Toolbox
from src.tracking.otel import jsonl_tracer

ROOT = Path(__file__).resolve().parent.parent.parent
STAGE = "critic"
VERDICT = "submit_verdict"

VERDICT_TOOL: dict[str, Any] = {
    "name": VERDICT,
    "description": (
        "Submit your review of the analyst's answer: your verdict, the probability that the "
        "query's result is correct, and the problems you found."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": list(VERDICTS)},
            "confidence": {
                "type": "number",
                "description": "Probability from 0 to 1 that the query's result is correct.",
            },
            "problems": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The problems found, most serious first; empty if none.",
            },
        },
        "required": ["verdict", "confidence", "problems"],
        "additionalProperties": False,
    },
}


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def load_prompt(path: Path) -> tuple[str, str]:
    """A prompt's text and the sha256 of its file (LF line endings)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return data.decode("utf-8"), hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------ the verdict


@dataclass(frozen=True)
class Verdict:
    verdict: str | None
    confidence: float | None
    problems: tuple[str, ...]
    format_problems: tuple[str, ...] = ()


def parse_verdict(args: Any) -> Verdict:
    """The critic's submitted verdict, checked; a malformed field is recorded, never repaired
    silently. A confidence outside 0-1 is clipped (strict schemas cannot state a range)."""
    if not isinstance(args, dict):
        return Verdict(None, None, (), ("the verdict is not an object",))
    found = []
    verdict = args.get("verdict")
    if verdict not in VERDICTS:
        found.append(f"verdict {verdict!r} is not one of {VERDICTS}")
        verdict = None
    conf = args.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, int | float):
        found.append(f"confidence {conf!r} is not a number")
        conf = None
    elif not 0 <= conf <= 1:
        found.append(f"confidence {conf} clipped to 0-1")
        conf = min(1.0, max(0.0, float(conf)))
    problems = args.get("problems")
    if not isinstance(problems, list) or not all(isinstance(p, str) for p in problems):
        found.append("problems is not a list of text")
        problems = []
    return Verdict(verdict, None if conf is None else float(conf), tuple(problems), tuple(found))


# ------------------------------------------------------------------------------ the evidence


def fetch_evidence(
    toolbox: Toolbox,
    sql: str,
    limits: Limits,
    max_rows: int,
    rows_shown: int,
    clock: ClockStore | None,
    at: tuple[str, str],
) -> dict[str, Any]:
    """The answer's result and checks as the critic reads them (JSON data, so a result of a
    query that reads the clock can be stored and replayed)."""

    def run() -> dict[str, Any]:
        r = verify.fetch(toolbox.sql.guard, toolbox.executor, sql, limits)
        cell = toolbox.cfg["agent"]["max_cell_chars"]
        return {
            "ok": r.ok,
            "error": r.error,
            "columns": r.columns,
            "rows": len(r.rows),
            "truncated": r.truncated,
            "preview": [[display_value(v, cell) for v in row] for row in r.rows[:rows_shown]],
            "checks": verify.checks(sql, r, max_rows),
        }

    if clock is None:
        return run()
    return clock.through(toolbox.db, toolbox.sql.guard.tables, sql, at, run)


def _checks_text(checks: dict[str, Any], max_rows: int) -> str:
    def yes(b: bool) -> str:
        return "yes" if b else "no"

    ranges = checks.get("out_of_range") or []
    found = (
        "; ".join(f"column {r['column']} ({r['kind']}) holds {r['value']}" for r in ranges)
        or "none"
    )
    return (
        f"empty result: {yes(checks['empty'])}; "
        f"repeated identical rows: {yes(checks['repeated_rows'])}; "
        f"values outside their possible range: {found}; "
        f"more than {max_rows:,} rows: {yes(checks['too_many_rows'])}."
    )


def review_text(question: str, answer: dict, evidence: dict, max_rows: int) -> str:
    """What the critic reads after the schema: the question as the analyst saw it, the answer,
    and its evidence."""
    lines = [question, "", "The analyst's SQL:", answer["final_sql"], ""]
    lines.append(f"The analyst's answer: {answer['answer'] or '(no answer in words)'}")
    assumptions = answer.get("assumptions") or []
    lines.append("Assumptions stated: " + ("none" if not assumptions else ""))
    lines += [f"- {a}" for a in assumptions]
    shown = len(evidence["preview"])
    total = f"{evidence['rows']:,} rows" + (
        " (more were not fetched)" if evidence["truncated"] else ""
    )
    lines += [
        "",
        f"The query's result: {total}; the first {shown} shown."
        if shown < evidence["rows"]
        else f"The query's result: {total}.",
        _json({"columns": evidence["columns"], "rows": evidence["preview"]}),
        "",
        "Automatic checks: " + _checks_text(evidence["checks"], max_rows),
    ]
    return "\n".join(lines)


def schema_context(tools: AgentTools, db: str) -> str:
    """The schema exactly as design 1 reads it."""
    return "Database: " + db + "\n\nSchema:\n" + _json(tools.full_schema())


# ------------------------------------------------------------------------------ one review


class CriticRun:
    """The review of one answer: one conversation of one call, or none if the answer has no
    result. Driven like a question run (src/agent/driver.py)."""

    def __init__(
        self,
        q: Question,
        answer: dict,
        answer_run: str,
        toolbox: Toolbox,
        crit: dict[str, Any],
        system: str,
        agent_cfg: dict[str, Any],
        cost: Callable[[str, TokenUsage, bool], float],
        tracer: Tracer | None = None,
        clock: ClockStore | None = None,
        evidence: bool = True,
    ):
        """`evidence`: whether the answer was given the benchmark's hint (the critic reads the
        question as the analyst saw it)."""
        self.q, self.answer, self.answer_run = q, answer, answer_run
        self.crit, self.cost, self.tracer = crit, cost, tracer
        self.model = crit["model"]
        self.max_rows = agent_cfg["checks"]["max_rows"]
        self.evidence: dict | None = None
        self.not_reviewed: str | None = None
        self.conversation: Conversation | None = None
        self._span = tracer.start_span("critic.run") if tracer else None
        if answer["declined"]:
            self.not_reviewed = "declined"
        elif not answer["final_sql"]:
            self.not_reviewed = "no_sql"
        else:
            limits = Limits(
                max_rows=toolbox.cfg["evaluation"]["max_rows"],
                timeout_s=float(agent_cfg["checks"].get("timeout_s", 30)),
                count_total=False,
            )
            at = (STAGE, f"{answer_run}#{q.question_id}")
            self.evidence = fetch_evidence(
                toolbox,
                answer["final_sql"],
                limits,
                self.max_rows,
                crit["rows_shown"],
                clock,
                at,
            )
            if not self.evidence["ok"]:
                self.not_reviewed = f"final_sql_{(self.evidence['error'] or {}).get('kind')}"
        if self.not_reviewed is None:
            s = crit["settings"]
            settings = Settings(
                backend=s["backend"],
                model=self.model,
                max_tokens=s["max_tokens"],
                params=dict(s["params"]),
                prompt_cache=s["prompt_cache"],
                force_tool=VERDICT,
                final_tool=VERDICT,
            )
            tools = AgentTools(toolbox, [], crit["rows_shown"])
            self.conversation = Conversation(
                settings,
                system,
                schema_context(tools, q.db_id),
                review_text(question_text(q, evidence), answer, self.evidence, self.max_rows),
                [VERDICT_TOOL],
                None,
            )

    @property
    def done(self) -> bool:
        return self.conversation is None or self.conversation.done

    def pending(self) -> list[tuple[Conversation, LLMRequest]]:
        c = self.conversation
        return [] if c is None or c.done else [(c, c.request())]

    def feed(self, conversation: Conversation, key: str, response: LLMResponse) -> None:
        conversation.feed(key, response)
        self._llm_span(response)

    def stop(self, conversation: Conversation, kind: str, message: str) -> None:
        conversation.stop(kind, message)

    def _call_cost(self, response: LLMResponse) -> float:
        return self.cost(self.model, response.tokens, response.extra.get("service") == "batch")

    def _llm_span(self, response: LLMResponse) -> None:
        if self.tracer is None or self._span is None:
            return
        now = time.time_ns()
        t = response.tokens
        span = self.tracer.start_span(
            "llm.call",
            context=ot.set_span_in_context(self._span),
            start_time=now - int(response.latency_ms * 1e6),
        )
        span.set_attributes(
            {
                "model": self.model,
                "stop_reason": response.stop_reason or "",
                "input_tokens": t.input,
                "output_tokens": t.output,
                "cache_read_tokens": t.cache_read,
                "cache_write_tokens": t.cache_write_5m + t.cache_write_1h,
                "latency_ms": response.latency_ms,
                "batch": response.extra.get("service") == "batch",
                "cost_usd": self._call_cost(response),
            }
        )
        span.end(end_time=now)

    def finish(self) -> tuple[dict, dict]:
        """(the record without its run name, trace path and time; the trace)."""
        if not self.done:
            raise RuntimeError("the review is not finished")
        c = self.conversation
        verdict = Verdict(None, None, ())
        errors: list[dict] = []
        usage = TokenUsage()
        cost = latency = 0.0
        requests = []
        if c is not None:
            if c.submitted is not None:
                verdict = parse_verdict(c.submitted)
            errors += c.errors
            errors += [{"kind": "verdict_format", "message": p} for p in verdict.format_problems]
            for turn in c.turns:
                t = turn.response.tokens
                usage = TokenUsage(
                    usage.input + t.input,
                    usage.output + t.output,
                    usage.cache_write_5m + t.cache_write_5m,
                    usage.cache_write_1h + t.cache_write_1h,
                    usage.cache_read + t.cache_read,
                )
                cost += self._call_cost(turn.response)
                latency += turn.response.latency_ms / 1000
                requests.append(
                    {
                        "cache_key": turn.cache_key,
                        "stop_reason": turn.response.stop_reason,
                        "usage": turn.response.usage,
                        "batch": turn.response.extra.get("service") == "batch",
                    }
                )
        reviewed = self.not_reviewed is None
        confidence = verdict.confidence if reviewed else 0.0
        record = {
            "question_id": self.q.question_id,
            "source": self.q.source,
            "db_id": self.q.db_id,
            "difficulty": self.q.difficulty,
            "answer_run": self.answer_run,
            "model": self.model,
            "reviewed": reviewed,
            "not_reviewed": self.not_reviewed,
            "verdict": verdict.verdict,
            "confidence": confidence,
            "problems": list(verdict.problems),
            "result_rows": None if self.evidence is None else self.evidence["rows"],
            "checks_failed": None
            if self.evidence is None
            else bool(self.evidence["checks"].get("failed")),
            "tokens": {
                "input": usage.input,
                "output": usage.output,
                "cache_write_5m": usage.cache_write_5m,
                "cache_write_1h": usage.cache_write_1h,
                "cache_read": usage.cache_read,
            },
            "cost_usd": round(cost, 8),
            "latency_s": round(latency, 4),
            "errors": [{"kind": e["kind"], "message": str(e["message"])} for e in errors],
        }
        trace = {
            "question": {
                "question_id": self.q.question_id,
                "db_id": self.q.db_id,
                "question": self.q.question,
                "evidence": self.q.evidence,
            },
            "answer_run": self.answer_run,
            "answer": {
                k: self.answer[k] for k in ("final_sql", "answer", "assumptions", "declined")
            },
            "model": self.model,
            "model_settings": self.crit["settings"],
            "evidence": self.evidence,
            "not_reviewed": self.not_reviewed,
            "messages": None if c is None else c.messages,
            "submitted": None if c is None else c.submitted,
            "requests": requests,
            "errors": record["errors"],
        }
        if self._span is not None:
            self._span.set_attributes(
                {
                    "question_id": str(self.q.question_id),
                    "db_id": self.q.db_id,
                    "answer_run": self.answer_run,
                    "reviewed": reviewed,
                    "not_reviewed": self.not_reviewed or "",
                    "verdict": verdict.verdict or "",
                    "confidence": -1.0 if confidence is None else confidence,
                    "cost_usd": cost,
                }
            )
            self._span.end()
        return record, trace


# ------------------------------------------------------------------------------ runs


@dataclass
class CriticSpec:
    name: str  # e.g. "critic/ablation-critic-claude-sonnet-5"
    items: list[tuple[Question, dict]]  # each question with the answer it reviews
    answer_run: str
    evidence: bool  # whether the answers were given the benchmark's hint
    phase: str
    cache_dir: Path
    mode: str = "batch"  # or "direct"
    records: Path | None = None
    traces_dir: Path | None = None
    spans: Path | None = None
    ledger_path: Path | None = None  # None: the project's ledger (tests: a copy)
    batch_dir: Path = BATCH_DIR
    poll_seconds: float | None = None


def items_for(set_name: str, answers: Sequence[dict], limit: int | None = None) -> list:
    """The questions of one split with their answers (never their gold)."""
    by_id = {r["question_id"]: r for r in answers}
    out = [(q, by_id[q.question_id]) for q, _gold in benchmark_set(set_name)]
    return out[:limit] if limit else out


def make_spec(
    set_name: str,
    limit: int | None,
    model: str,
    items: list,
    answer_run: str,
    phase: str,
    mode: str,
    evidence: bool = True,
) -> CriticSpec:
    rid = f"{set_name}-critic-{model}" + (f"-first{limit}" if limit else "")
    return CriticSpec(
        name=f"{STAGE}/{rid}",
        items=items,
        answer_run=answer_run,
        evidence=evidence,
        phase=phase,
        cache_dir=ROOT / f"data/cache/llm_{STAGE}",
        mode=mode,
        records=ROOT / f"results/runs/{STAGE}/{rid}.jsonl",
        traces_dir=ROOT / f"data/traces/{STAGE}/{rid}",
        spans=ROOT / f"data/spans/{STAGE}/{rid}.jsonl",
    )


def execute(
    specs: list[CriticSpec],
    crit: dict[str, Any],
    log: Callable[[str], None] = print,
    backend: Backend | None = None,
) -> list[list[dict]]:
    """Review every answer of the given runs and write their records and traces. Batched runs
    go together (one set of batches); direct runs one review per worker. Returns each run's
    records."""
    shared = {(s.mode, s.cache_dir, s.phase, s.ledger_path, s.batch_dir) for s in specs}
    if len(shared) != 1:
        raise ValueError("runs executed together must share mode, cache, phase and ledger")
    agent_cfg = config()
    spec = specs[0]
    system, _ = load_prompt(ROOT / crit["prompt"])
    backend = backend or make_backend(crit["settings"]["backend"])
    ledger = load_ledger(spec.phase, ledger_path=spec.ledger_path)
    cache = LayeredCache(ResponseCache(spec.cache_dir))
    caller = Caller(
        backend,
        cache,
        ledger,
        agent_cfg["run"]["batch_poll_seconds"] if spec.poll_seconds is None else spec.poll_seconds,
        spec.batch_dir,
        log=log,
    )
    tracers = [jsonl_tracer(s.spans) if s.spans else (None, None) for s in specs]
    pool = ToolboxPool()
    clock = ClockStore(spec.cache_dir)

    def make(s: CriticSpec, tracer, q: Question, answer: dict) -> CriticRun:
        box = pool.get(q.db_id)
        return CriticRun(
            q,
            answer,
            s.answer_run,
            box,
            crit,
            system,
            agent_cfg,
            ledger.cost,
            tracer,
            clock,
            s.evidence,
        )

    jobs = [
        (k, i)
        for k, s in enumerate(specs)
        for i in sorted(range(len(s.items)), key=lambda i, s=s: (s.items[i][0].db_id, i))
    ]
    makers = [partial(make, specs[k], tracers[k][0], *specs[k].items[i]) for k, i in jobs]
    try:
        if spec.mode == "batch":
            done = drive_batch(makers, caller, log)
        else:
            done = drive_direct(makers, caller, agent_cfg["run"]["workers"])
    finally:
        pool.close()
        for _, provider in tracers:
            if provider is not None:
                provider.shutdown()

    by_run: list[dict] = [{} for _ in specs]
    for (k, i), (_, (record, trace)) in zip(jobs, done, strict=True):
        by_run[k][i] = (record, trace)
    out = []
    for s, results in zip(specs, by_run, strict=True):
        records = []
        when = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        for i in range(len(s.items)):
            record, trace = results[i]
            trace_rel = None
            if s.traces_dir is not None:
                path = s.traces_dir / f"{record['question_id']}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(trace, ensure_ascii=False, indent=1, default=str),
                    encoding="utf-8",
                    newline="\n",
                )
                trace_rel = (
                    path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
                )
            records.append({"run": s.name, **record, "trace": trace_rel, "evaluated_at": when})
        if s.records is not None:
            write_records(s.records, records)
        out.append(records)
    return out
