"""The guardrail's model calls: one forced tool call per item, through the project's response
cache and spend ledger.

Three kinds, each with its own prompt, tool and model (configs/guardrail.yaml `calls`):
`classify` (is the question statistical?), `plan` (the analysis plan, written from the schema
as design 1 reads it) and `answer` (the answer written from the analysis's result). Each item is
one conversation of one call (src/agent/conversation.py), driven like the agent's runs
(src/agent/driver.py): directly, a few at a time, or in batched rounds at half price. Every
response lands in the stage's response cache, so a finished run replays at $0, and every paid
call is reserved against the phase's cap first.

A prompt is frozen by its sha256 in the config once its format has been checked; a run whose
prompt file does not match refuses to start (a probe may run before the hash is set).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from src.agent.conversation import Conversation, Settings
from src.agent.driver import Caller, LayeredCache, drive_batch, drive_direct
from src.llm.backends import Backend, make_backend
from src.llm.batch import BATCH_DIR
from src.llm.cache import ResponseCache
from src.llm.ledger import SpendLedger, load_ledger
from src.llm.types import LLMRequest, LLMResponse, TokenUsage

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG = ROOT / "configs/guardrail.yaml"
_TEXT_OR_NULL = {"anyOf": [{"type": "string"}, {"type": "null"}]}

CLASSIFY_TOOL: dict[str, Any] = {
    "name": "classify_question",
    "description": "Classify the question as statistical or descriptive.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "statistical": {"type": "boolean"},
            "kind": {
                "type": "string",
                "enum": ["descriptive", "comparison", "trend", "association", "causal"],
            },
            "reason": {"type": "string"},
        },
        "required": ["statistical", "kind", "reason"],
        "additionalProperties": False,
    },
}

PLAN_TOOL: dict[str, Any] = {
    "name": "submit_analysis",
    "description": (
        "Submit the analysis plan: the query returning one row per unit, and the analysis the "
        "program runs on its rows."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "unit": {"type": "string", "description": "What one row of the query is."},
            "sql": {**_TEXT_OR_NULL, "description": "One PostgreSQL SELECT query, or null."},
            "analysis": {"type": "string", "enum": ["compare_groups", "trend"]},
            "outcome": {"type": "string", "description": "The outcome column."},
            "outcome_type": {"type": "string", "enum": ["binary", "numeric"]},
            "group": {**_TEXT_OR_NULL, "description": "The groups column (compare_groups)."},
            "reference_group": {
                **_TEXT_OR_NULL,
                "description": "The group the others are compared with, as in the result.",
            },
            "x": {**_TEXT_OR_NULL, "description": "The x column (trend)."},
            "x_describes": {"type": "string", "enum": ["unit", "area"]},
            "strata": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Up to two columns to compare within.",
            },
            "strata_reason": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "declined": {"type": "boolean"},
            "decline_reason": {**_TEXT_OR_NULL},
        },
        "required": [
            "unit",
            "sql",
            "analysis",
            "outcome",
            "outcome_type",
            "group",
            "reference_group",
            "x",
            "x_describes",
            "strata",
            "strata_reason",
            "assumptions",
            "declined",
            "decline_reason",
        ],
        "additionalProperties": False,
    },
}

ANSWER_TOOL: dict[str, Any] = {
    "name": "submit_finding",
    "description": "Submit the answer to the question, and what it claims.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "claims_effect": {"type": "string", "enum": ["yes", "no", "unclear"]},
            "higher": {**_TEXT_OR_NULL},
        },
        "required": ["answer", "claims_effect", "higher"],
        "additionalProperties": False,
    },
}

TOOLS = {"classify": CLASSIFY_TOOL, "plan": PLAN_TOOL, "answer": ANSWER_TOOL}


def config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_prompt(path: Path) -> tuple[str, str]:
    """A prompt's text and the sha256 of its file (LF line endings)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return data.decode("utf-8"), hashlib.sha256(data).hexdigest()


def prompt_for(kind: str, cfg: dict, probe: bool = False) -> tuple[str, str]:
    """The kind's prompt, refused unless it matches its frozen hash (a probe may run before the
    hash is set, never against a different one)."""
    spec = cfg["calls"][kind]
    text, sha = load_prompt(ROOT / spec["prompt"])
    frozen = spec.get("prompt_sha256")
    if frozen is None and not probe:
        raise RuntimeError(f"the {kind} prompt is not frozen yet (calls.{kind}.prompt_sha256)")
    if frozen is not None and frozen != sha:
        raise RuntimeError(f"{spec['prompt']} does not match its frozen sha256")
    return text, sha


@dataclass(frozen=True)
class Item:
    id: str
    context: str  # what the model reads first (the schema, or the database's name)
    text: str  # the question, or the question with its results


class OneCall:
    """One item: one conversation of one forced tool call."""

    def __init__(
        self,
        item: Item,
        kind: str,
        system: str,
        settings: dict,
        cost: Callable[[str, TokenUsage, bool], float],
    ):
        self.item, self.kind, self.cost = item, kind, cost
        self.model = settings["model"]
        tool = TOOLS[kind]
        s = Settings(
            backend=settings["backend"],
            model=self.model,
            max_tokens=settings["max_tokens"],
            params=dict(settings["params"]),
            prompt_cache=settings["prompt_cache"],
            force_tool=tool["name"],
            final_tool=tool["name"],
        )
        self.conversation = Conversation(s, system, item.context, item.text, [tool], None)

    @property
    def done(self) -> bool:
        return self.conversation.done

    def pending(self) -> list[tuple[Conversation, LLMRequest]]:
        c = self.conversation
        return [] if c.done else [(c, c.request())]

    def feed(self, conversation: Conversation, key: str, response: LLMResponse) -> None:
        conversation.feed(key, response)

    def stop(self, conversation: Conversation, kind: str, message: str) -> None:
        conversation.stop(kind, message)

    def finish(self) -> dict[str, Any]:
        c = self.conversation
        usage = TokenUsage()
        cost = 0.0
        keys = []
        for turn in c.turns:
            t = turn.response.tokens
            usage = TokenUsage(
                usage.input + t.input,
                usage.output + t.output,
                usage.cache_write_5m + t.cache_write_5m,
                usage.cache_write_1h + t.cache_write_1h,
                usage.cache_read + t.cache_read,
            )
            cost += self.cost(self.model, t, turn.response.extra.get("service") == "batch")
            keys.append(turn.cache_key)
        return {
            "id": self.item.id,
            "kind": self.kind,
            "model": self.model,
            "submitted": c.submitted,
            "errors": [{"kind": e["kind"], "message": str(e["message"])} for e in c.errors],
            "tokens": {
                "input": usage.input,
                "output": usage.output,
                "cache_write_5m": usage.cache_write_5m,
                "cache_write_1h": usage.cache_write_1h,
                "cache_read": usage.cache_read,
            },
            "cost_usd": round(cost, 8),
            "cache_keys": keys,
            "messages": c.messages,
        }


def requests_for(items: list[Item], kind: str, cfg: dict, probe: bool = False) -> list[LLMRequest]:
    """The first request of each item, as it would be sent (for counting tokens)."""
    system, _ = prompt_for(kind, cfg, probe)
    settings = cfg["calls"][kind]["settings"]
    return [OneCall(i, kind, system, settings, lambda *a: 0.0).pending()[0][1] for i in items]


def run_calls(
    items: list[Item],
    kind: str,
    cfg: dict,
    mode: str,
    cache_dir: Path,
    phase: str,
    log: Callable[[str], None] = print,
    probe: bool = False,
    backend: Backend | None = None,
    ledger: SpendLedger | None = None,
    batch_dir: Path = BATCH_DIR,
    poll_seconds: float = 30.0,
    workers: int = 4,
) -> list[dict]:
    """Run one call per item; results in the order given."""
    system, _ = prompt_for(kind, cfg, probe)
    settings = cfg["calls"][kind]["settings"]
    backend = backend or make_backend(settings["backend"])
    ledger = ledger or load_ledger(phase)
    caller = Caller(
        backend, LayeredCache(ResponseCache(cache_dir)), ledger, poll_seconds, batch_dir, log=log
    )
    makers = [partial(OneCall, item, kind, system, settings, ledger.cost) for item in items]
    if mode == "batch":
        done = drive_batch(makers, caller, log)
    else:
        done = drive_direct(makers, caller, workers)
    return [finished for _, finished in done]


def require_protocol(cfg: dict) -> str:
    """The pre-registration's hash, refused unless it is frozen and the file matches it."""
    spec = cfg["protocol"]
    _, sha = load_prompt(ROOT / spec["path"])
    if spec.get("sha256") is None:
        raise RuntimeError(f"{spec['path']} is not frozen yet (protocol.sha256)")
    if spec["sha256"] != sha:
        raise RuntimeError(f"{spec['path']} does not match its frozen sha256")
    return sha
