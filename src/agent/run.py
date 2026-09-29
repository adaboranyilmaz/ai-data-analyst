"""One question under one design and model: its conversations, its final answer and its trace.

A `QuestionRun` owns the conversations of one question (design 5's narrowing call, then one
conversation per sample), hands out their pending requests and takes their responses, like a
single conversation does, so a driver can run many questions at once, directly or in batched
rounds (src/agent/driver.py). When every conversation is done, `finish()` executes each sample's
final SQL (through the same guard and read-only execution the agent's tools use), runs the
automatic checks, applies design 4's vote where there are several samples, validates the chart,
and returns the result and the trace.

The trace holds everything needed to audit or replay the run: the question as the model saw it,
every request's cache key, the full conversation, every tool result, the step lines, the final
result's preview and the checks. It never holds a gold query.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from opentelemetry import trace as ot
from opentelemetry.trace import Tracer

from src.agent import verify
from src.agent.answer import (
    SELECT_SCHEMA,
    SELECT_SCHEMA_TOOL,
    SUBMIT,
    SUBMIT_TOOL,
    Answer,
    parse_selection,
)
from src.agent.clock import ClockStore
from src.agent.conversation import Conversation, Settings
from src.agent.tools import AgentTools
from src.db.execute import Limits
from src.llm.types import LLMRequest, LLMResponse, TokenUsage
from src.tools.chart import validate_chart
from src.tools.definitions import TOOLS
from src.tools.sql import display_value
from src.tools.toolbox import Toolbox

ROOT = Path(__file__).resolve().parent.parent.parent
AGENT_CONFIG = ROOT / "configs/agent.yaml"
CostFn = Callable[[str, TokenUsage, bool], float]


def config() -> dict[str, Any]:
    return yaml.safe_load(AGENT_CONFIG.read_text(encoding="utf-8"))


def prompt(cfg: dict[str, Any], name: str) -> tuple[str, str]:
    """A prompt's text and the sha256 of its file."""
    data = (ROOT / cfg["prompts"][name]).read_bytes().replace(b"\r\n", b"\n")
    return data.decode("utf-8"), hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Question:
    question_id: int | str
    source: str  # "bird" | "own"
    db_id: str
    question: str
    evidence: str | None  # the benchmark's hint
    difficulty: str | None = None
    category: str | None = None


def question_text(q: Question, evidence: bool) -> str:
    text = f"Question: {q.question.strip()}"
    if evidence and q.evidence and q.evidence.strip():
        text += f"\nHint: {q.evidence.strip()}"
    return text


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


@dataclass
class Finished:
    answer: Answer
    confidence: float
    final_sql: str | None
    result: verify.FinalResult
    checks: dict
    trace: dict
    usage: TokenUsage
    cost_usd: float
    latency_s: float
    steps: int
    tool_calls: int
    errors: list[dict] = field(default_factory=list)


class QuestionRun:
    def __init__(
        self,
        q: Question,
        design: str,
        model: str,
        evidence: bool,
        toolbox: Toolbox,
        cost: CostFn,
        tracer: Tracer | None = None,
        cfg: dict[str, Any] | None = None,
        clock: ClockStore | None = None,
    ):
        """`clock`: where the results of the agent's queries that read the clock are stored,
        so a replay sees what the first run saw (src/agent/clock.py)."""
        self.cfg = cfg = cfg or config()
        self.q, self.design, self.model, self.evidence = q, design, model, evidence
        self.d = cfg["designs"][design]
        self.m = cfg["models"][model]
        self.toolbox = toolbox
        self.cost = cost
        self.tracer = tracer
        self.clock = clock
        self.rows_shown = cfg["presentation"]["rows_shown"]
        self.base_tools = AgentTools(toolbox, self.d["tools"], self.rows_shown, clock=clock)
        self.prompts: dict[str, str] = {}
        self.narrowing: Conversation | None = None
        self.selection: dict[str, list[str]] | None = None
        self.samples: list[Conversation] = []
        self.sample_tools: list[AgentTools] = []
        self.problems: list[dict] = []
        self._span = tracer.start_span("agent.run") if tracer else None
        if self._span is not None:
            self._span.set_attributes(
                {
                    "question_id": str(q.question_id),
                    "db_id": q.db_id,
                    "design": design,
                    "model": model,
                    "evidence": evidence,
                }
            )
        if self.d["narrow"]:
            self.narrowing = self._narrowing_conversation()
        else:
            self._start_samples(self.base_tools)

    # ---------------------------------------------------------------- building conversations

    def _settings(self, **kw) -> Settings:
        return Settings(
            backend=self.m["backend"],
            model=self.model,
            max_tokens=self.m["max_tokens"],
            params=dict(self.m["params"]),
            prompt_cache=self.m["prompt_cache"],
            max_model_calls=self.cfg["max_model_calls"],
            **kw,
        )

    def _prompt(self, name: str) -> str:
        text, sha = prompt(self.cfg, name)
        self.prompts[name] = sha
        return text

    def _narrowing_conversation(self) -> Conversation:
        return Conversation(
            self._settings(force_tool=SELECT_SCHEMA, final_tool=SELECT_SCHEMA),
            self._prompt("narrow"),
            "Database: " + self.q.db_id + "\n\nSchema:\n" + _json(self.base_tools.full_schema()),
            question_text(self.q, self.evidence),
            [SELECT_SCHEMA_TOOL],
            None,
        )

    def _start_samples(self, tools: AgentTools) -> None:
        single_shot = not self.d["tools"]
        for i in range(self.d["samples"]):
            if single_shot:
                settings = self._settings(force_tool=SUBMIT)
                system = self._prompt("single_shot")
                context = "Database: " + self.q.db_id + "\n\nSchema:\n" + _json(tools.full_schema())
                defs = [SUBMIT_TOOL]
            else:
                settings = self._settings(
                    max_tool_calls=self.d["max_tool_calls"],
                    resubmits=self.d["resubmits"],
                    reminders=self.cfg["reminders"],
                    sample=i,
                )
                system = self._prompt("agent")
                context = "Database: " + self.q.db_id + "\n\nTables:\n" + _json(tools.table_list())
                defs = [t for t in TOOLS if t["name"] in self.d["tools"]] + [SUBMIT_TOOL]
            if single_shot and i:
                settings.sample = i
            self.samples.append(
                Conversation(
                    settings,
                    system,
                    context,
                    question_text(self.q, self.evidence),
                    defs,
                    None if single_shot else tools,
                )
            )
            self.sample_tools.append(tools)

    # ---------------------------------------------------------------- driving

    @property
    def conversations(self) -> list[Conversation]:
        return ([self.narrowing] if self.narrowing else []) + self.samples

    @property
    def done(self) -> bool:
        return bool(self.samples) and all(c.done for c in self.samples)

    def pending(self) -> list[tuple[Conversation, LLMRequest]]:
        return [(c, c.request()) for c in self.conversations if not c.done]

    def feed(self, conversation: Conversation, key: str, response: LLMResponse) -> None:
        before = len(conversation.outcomes)
        conversation.feed(key, response)
        self._spans(response, conversation.outcomes[before:])
        if conversation is self.narrowing and conversation.done:
            self._after_narrowing()

    def stop(self, conversation: Conversation, kind: str, message: str) -> None:
        """End one conversation without an answer; a stopped narrowing call leaves the samples
        the full schema."""
        conversation.stop(kind, message)
        if conversation is self.narrowing:
            self._after_narrowing()

    def _after_narrowing(self) -> None:
        known = {
            t: [c["name"] for c in self.toolbox.schema.describe_table(t)["columns"]]
            for t in self.toolbox.schema.tables
        }
        chosen, problems = parse_selection(self.narrowing.submitted, known)
        for p in problems:
            self.problems.append({"kind": "narrowing", "message": p})
        if not chosen:
            self.problems.append(
                {"kind": "narrowing_failed", "message": "no tables chosen; the full schema is used"}
            )
            chosen = None
        self.selection = chosen
        tools = AgentTools(self.toolbox, self.d["tools"], self.rows_shown, chosen, self.clock)
        self._start_samples(tools)

    def _spans(self, response: LLMResponse, outcomes: list) -> None:
        if self.tracer is None or self._span is None:
            return
        parent = ot.set_span_in_context(self._span)
        now = time.time_ns()
        t = response.tokens
        span = self.tracer.start_span(
            "llm.call", context=parent, start_time=now - int(response.latency_ms * 1e6)
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
        for o in outcomes:
            secs = float(o.result.get("seconds") or 0.0)
            tool = self.tracer.start_span(
                "tool.call", context=parent, start_time=now - int(secs * 1e9)
            )
            attrs: dict[str, Any] = {"tool": o.name, "ok": not o.is_error, "seconds": secs}
            if o.name == "run_sql" and isinstance(o.input, dict):
                attrs["sql"] = str(o.input.get("sql", ""))[:2000]
            if o.result.get("ok") and "row_count" in o.result:
                attrs["row_count"] = o.result["row_count"]
            if o.is_error:
                attrs["error"] = str((o.result.get("error") or {}).get("kind", "error"))
            tool.set_attributes(attrs)
            tool.end(end_time=now)

    def _call_cost(self, response: LLMResponse) -> float:
        if self.m["backend"] != "anthropic":
            return 0.0
        return self.cost(self.model, response.tokens, response.extra.get("service") == "batch")

    # ---------------------------------------------------------------- the final answer

    def finish(self) -> Finished:
        if not self.done:
            raise RuntimeError("the run is not finished")
        limits = Limits(
            max_rows=self.toolbox.cfg["evaluation"]["max_rows"],
            timeout_s=float(self.cfg["checks"].get("timeout_s", 30)),
            count_total=False,
        )
        max_rows = self.cfg["checks"]["max_rows"]
        per_sample = []
        for conv, tools in zip(self.samples, self.sample_tools, strict=True):
            a = conv.answer or Answer.none()
            sql = None if a.declined else a.sql
            result = verify.fetch(tools._sql.guard, self.toolbox.executor, sql, limits)
            per_sample.append((a, result, verify.checks(sql, result, max_rows)))

        if len(per_sample) > 1:
            v = verify.vote(
                [(a.declined, r) for a, r, _ in per_sample], [c["failed"] for _, _, c in per_sample]
            )
            chosen, confidence = v.chosen, v.confidence
            vote_info: dict | None = {"chosen": v.chosen, "group": v.group, "groups": v.groups}
        else:
            chosen, confidence, vote_info = 0, per_sample[0][0].confidence or 0.0, None
        answer, result, found = per_sample[chosen]

        errors = list(self.problems)
        for conv in self.conversations:
            errors += conv.errors
        for a, _, _ in per_sample:
            errors += [{"kind": "answer_format", "message": p} for p in a.problems]
        if not answer.declined and answer.sql and not result.ok:
            errors.append(
                {"kind": f"final_sql_{result.error['kind']}", "message": result.error["message"]}
            )

        chart = None
        if answer.chart_spec is not None:
            check = validate_chart(
                answer.chart_spec, result.columns, self.toolbox.cfg["chart"]["max_spec_chars"]
            )
            if check.get("ok"):
                chart = answer.chart_spec
            else:
                errors.append({"kind": "chart_invalid", "message": _json(check.get("error"))})

        usage = TokenUsage()
        cost = latency = 0.0
        requests = []
        for conv in self.conversations:
            for turn in conv.turns:
                t = turn.response.tokens
                usage = TokenUsage(
                    usage.input + t.input,
                    usage.output + t.output,
                    usage.cache_write_5m + t.cache_write_5m,
                    usage.cache_write_1h + t.cache_write_1h,
                    usage.cache_read + t.cache_read,
                )
                c = self._call_cost(turn.response)
                cost += c
                latency += turn.response.latency_ms / 1000
                requests.append(
                    {
                        "cache_key": turn.cache_key,
                        "stop_reason": turn.response.stop_reason,
                        "usage": turn.response.usage,
                        "cost_usd": c,
                        "latency_ms": turn.response.latency_ms,
                        "batch": turn.response.extra.get("service") == "batch",
                    }
                )
        tool_calls = sum(len(c.outcomes) for c in self.conversations)
        steps = sum(len(c.turns) for c in self.conversations)
        latency += sum(
            float(o.result.get("seconds") or 0.0) for c in self.conversations for o in c.outcomes
        )

        trace = {
            "question": {
                "question_id": self.q.question_id,
                "source": self.q.source,
                "db_id": self.q.db_id,
                "question": self.q.question,
                "evidence": self.q.evidence if self.evidence else None,
            },
            "design": self.design,
            "model": self.model,
            "model_settings": self.m,
            "evidence": self.evidence,
            "prompts_sha256": self.prompts,
            "narrowing": None
            if self.narrowing is None
            else {"selection": self.selection, "messages": self.narrowing.messages},
            "samples": [
                {
                    "sample": i,
                    "messages": conv.messages,
                    "tool_results": [
                        {"tool": o.name, "input": o.input, "result": o.result}
                        for o in conv.outcomes
                    ],
                    "steps": conv.step_lines,
                    "answer": a.to_dict(),
                    "result_ok": r.ok,
                    "result_error": r.error,
                    "checks": c,
                    "errors": conv.errors,
                }
                for i, (conv, (a, r, c)) in enumerate(zip(self.samples, per_sample, strict=True))
            ],
            "vote": vote_info,
            "requests": requests,
            "final": {
                "sql": answer.sql,
                "answer": answer.to_dict(),
                "confidence": confidence,
                "result": {
                    "ok": result.ok,
                    "error": result.error,
                    "columns": result.columns,
                    "rows": len(result.rows),
                    "truncated": result.truncated,
                    "preview": [
                        [display_value(v, 300) for v in row]
                        for row in result.rows[: self.rows_shown]
                    ],
                },
                "checks": found,
                "chart_spec": chart,
                "steps": self.samples[chosen].step_lines,
            },
            "errors": errors,
        }
        if self._span is not None:
            self._span.set_attributes(
                {
                    "steps": steps,
                    "tool_calls": tool_calls,
                    "cost_usd": cost,
                    "confidence": confidence,
                    "declined": answer.declined,
                    "errors": len(errors),
                }
            )
            self._span.end()
        return Finished(
            answer=answer,
            confidence=float(confidence),
            final_sql=None if answer.declined else answer.sql,
            result=result,
            checks=found,
            trace=trace,
            usage=usage,
            cost_usd=cost,
            latency_s=latency,
            steps=steps,
            tool_calls=tool_calls,
            errors=errors,
        )
