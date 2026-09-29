"""One conversation with a model: a request at a time, until it submits an answer or runs out.

A conversation never calls a model itself. It hands out its next request (`request()`), and is
fed the response (`feed()`), which runs the requested tools and builds the next request. So the
same loop runs with direct calls, one conversation at a time, or in rounds where every open
conversation's next request goes out in one Message Batch. Requests are pure functions of the
responses before them and of the tools' results, which are deterministic, so a replayed run
rebuilds every request exactly and finds each response in the cache.

The loop, for one sample of one design:
- The model may call its design's data tools, up to the design's budget; a call past it, or to
  a tool it does not have, gets an error result telling it to submit. Several calls in one turn
  are all answered in one message.
- It finishes by calling `submit_answer`. Where the design allows resubmitting (design 3 and
  after), a submitted query that the guard refuses, that fails, that times out or that returns no
  rows goes back to the model with what happened, at most `resubmits` times; a declined answer
  is accepted as it is.
- A turn without any tool call gets a reminder to submit, at most `reminders` times, then the
  conversation ends without an answer. So does a refusal, a reply cut off at `max_tokens`, a
  local prompt or reply that does not fit the model's context, or reaching `max_model_calls`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent import steps
from src.agent.answer import SUBMIT, Answer, parse_answer
from src.agent.tools import AgentTools, ToolOutcome
from src.llm.types import LLMRequest, LLMResponse

REMINDER = "Call `submit_answer` now to give your answer."
BUDGET_SPENT = "The tool budget for this question is spent: call `submit_answer` now."
_KEEP = {  # what an assistant turn sends back: the fields the API accepts as input
    "text": ("type", "text"),
    "tool_use": ("type", "id", "name", "input"),
    "thinking": ("type", "thinking", "signature"),
    "redacted_thinking": ("type", "data"),
}


@dataclass
class Settings:
    backend: str
    model: str
    max_tokens: int
    params: dict[str, Any]
    prompt_cache: bool
    max_tool_calls: int = 0
    resubmits: int = 0
    reminders: int = 0
    max_model_calls: int = 24
    force_tool: str | None = None  # a single-turn call that must use this tool
    final_tool: str = SUBMIT  # the call that ends the conversation
    sample: int = 0  # the sample index; 0 adds nothing to the request, so sample 0 of design 4
    # is the same request as design 3's run


@dataclass
class Turn:
    """One model call: the request's cache key and what came back."""

    cache_key: str
    response: LLMResponse


@dataclass
class Conversation:
    settings: Settings
    system: str
    context: str  # what the model reads first: the schema or the table list
    question: str
    tool_defs: list[dict]
    tools: AgentTools | None
    messages: list[dict] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    outcomes: list[ToolOutcome] = field(default_factory=list)
    step_lines: list[str] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    submitted: dict | None = None  # the raw input of the accepted submit call
    answer: Answer | None = None
    done: bool = False
    tool_calls: int = 0
    resubmits_used: int = 0
    reminders_used: int = 0

    def __post_init__(self) -> None:
        s = self.settings
        context = {"type": "text", "text": self.context.rstrip() + "\n\n"}
        if s.prompt_cache:  # the schema is shared by every question on this database
            context["cache_control"] = {"type": "ephemeral"}
        self.messages = [
            {"role": "user", "content": [context, {"type": "text", "text": self.question}]}
        ]

    # ---------------------------------------------------------------- the next request

    def request(self) -> LLMRequest:
        s = self.settings
        system: str | list[dict] = self.system
        params = dict(s.params)
        if s.backend == "anthropic":
            if s.prompt_cache:
                system = [
                    {"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}
                ]
                if s.force_tool is None:  # multi-turn: also cache the conversation as it grows
                    params["cache_control"] = {"type": "ephemeral"}
            if s.force_tool is not None:
                params["tool_choice"] = {"type": "tool", "name": s.force_tool}
        if s.sample:
            params["_sample"] = s.sample
        return LLMRequest(
            s.backend, s.model, system, list(self.messages), s.max_tokens, self.tool_defs, params
        )

    # ---------------------------------------------------------------- a response

    def feed(self, cache_key: str, response: LLMResponse) -> None:
        if self.done:
            raise RuntimeError("the conversation is already finished")
        self.turns.append(Turn(cache_key, response))
        s = self.settings
        if response.stop_reason == "model_error":  # a failed call, stored as a response
            return self._finish(None, "model_error", response.extra.get("error", "model error"))
        if response.stop_reason == "refusal":
            return self._finish(None, "refusal", "the model refused")
        if response.stop_reason == "max_tokens":
            return self._finish(None, "max_tokens", "the reply was cut off at max_tokens")
        if response.stop_reason == "context_overflow":  # the local model's context is full
            return self._finish(
                None, "context_overflow", response.extra.get("error", "the context is full")
            )

        calls = response.tool_calls()
        self.messages.append({"role": "assistant", "content": _as_input(response.content)})
        if not calls:
            if self.reminders_used < s.reminders and self._calls_left():
                self.reminders_used += 1
                self.messages.append({"role": "user", "content": REMINDER})
                return None
            return self._finish(None, "no_answer", "no answer was submitted")

        results: list[dict] = []
        submit: dict | None = None
        over_budget = []
        for call in calls:
            if call["name"] == s.final_tool:
                if submit is None:
                    submit = call
                else:
                    results.append(_result(call["id"], "Only one answer is taken.", True))
                continue
            self.tool_calls += 1
            if self.tools is None or self.tool_calls > s.max_tool_calls:
                results.append(_result(call["id"], BUDGET_SPENT, True))
                over_budget.append(call["name"])
                continue
            outcome = self.tools.call(call["name"], call.get("input"), at=(cache_key, call["id"]))
            self.outcomes.append(outcome)
            self.step_lines.append(steps.tool_line(outcome.name, outcome.input, outcome.result))
            if outcome.is_error:
                kind = (outcome.result.get("error") or {}).get("kind", "error")
                self.errors.append({"kind": f"tool_{kind}", "message": outcome.name})
            results.append(_result(call["id"], outcome.shown, outcome.is_error))
        if over_budget:  # one error per turn, however many calls it made past the budget
            self.errors.append(
                {"kind": "tool_budget", "message": f"{len(over_budget)} calls past the budget"}
            )

        if submit is not None and s.final_tool != SUBMIT:  # e.g. design 5's narrowing call
            self.submitted = submit.get("input")
            self.done = True
            return None
        if submit is not None:
            answer = parse_answer(submit.get("input"))
            self.step_lines.append(steps.submit_line(answer.declined, answer.confidence))
            problem = self._resubmission_problem(answer, at=(cache_key, submit["id"]))
            if problem is None:
                self.submitted = submit.get("input")
                self.answer = answer
                self.done = True
                return None
            self.resubmits_used += 1
            self.step_lines.append(steps.resubmit_line(problem))
            self.errors.append({"kind": "resubmitted", "message": problem})
            results.append(
                _result(submit["id"], f"Your query {problem}. Fix it and submit again.", True)
            )
        if not self._calls_left():
            return self._finish(None, "model_call_limit", "the model call limit was reached")
        self.messages.append({"role": "user", "content": results})
        return None

    def _resubmission_problem(self, answer: Answer, at: tuple[str, str]) -> str | None:
        """Why a submitted answer goes back to the model, or None to accept it. `at`: the submit
        call (the request whose response made it, and its id)."""
        if self.resubmits_used >= self.settings.resubmits or self.tools is None or answer.declined:
            return None
        if not answer.sql:
            return "is missing (sql is null and the answer is not declined)"
        r = self.tools.run_final(answer.sql, at=at)
        if r.get("ok"):
            return "returned no rows" if r.get("total_rows") == 0 else None
        err = r.get("error") or {}
        if err.get("kind") == "refused":
            return "was refused: " + "; ".join(err.get("reasons") or [])
        if err.get("kind") == "timeout":
            return "timed out"
        return f"failed: {err.get('message', err.get('kind'))}"

    def stop(self, kind: str, message: str) -> None:
        """End the conversation without an answer, for a reason found outside it (e.g. its next
        request would not fit the local model's context)."""
        if not self.done:
            self._finish(None, kind, message)

    def _calls_left(self) -> bool:
        return len(self.turns) < self.settings.max_model_calls

    def _finish(self, answer: Answer | None, kind: str, message: str) -> None:
        self.errors.append({"kind": kind, "message": message})
        self.answer = answer or Answer.none()
        self.done = True


def _result(tool_use_id: str, content: str, is_error: bool) -> dict:
    out = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        out["is_error"] = True
    return out


def _as_input(content: list[dict]) -> list[dict]:
    """An assistant turn as it is sent back: only the fields the API accepts as input."""
    out = []
    for block in content:
        keys = _KEEP.get(block.get("type"))
        if keys is None or (block.get("type") == "text" and not block.get("text", "").strip()):
            continue  # the API refuses an empty text block
        out.append({k: block[k] for k in keys if k in block})
    return out
