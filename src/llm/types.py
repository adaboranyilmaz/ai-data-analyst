"""What is sent to a model and what comes back, as plain JSON data.

A request's `cache_key` is a SHA-256 of the whole request in canonical JSON, so any change
to the system prompt, the conversation, the tools, the model or a parameter is a cache
miss, and an identical request is never paid for twice. A request must therefore be pure
JSON data: SDK objects, such as the content blocks of an earlier assistant turn, are
converted with `model_dump()` before they go into `messages`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMRequest:
    backend: str  # "anthropic" | "ollama"
    model: str
    system: str | list[dict[str, Any]]  # text, or content blocks (which may carry cache_control)
    messages: list[dict[str, Any]]  # the whole conversation, in the Messages API shape
    max_tokens: int
    tools: list[dict[str, Any]] = field(default_factory=list)  # Messages API tool definitions
    params: dict[str, Any] = field(default_factory=dict)  # backend-specific, all recorded

    def __post_init__(self) -> None:
        try:
            self.canonical()
        except (TypeError, ValueError) as e:
            raise TypeError(f"an LLMRequest must be JSON data: {e}") from e

    @classmethod
    def single(
        cls,
        backend: str,
        model: str,
        system: str | list[dict[str, Any]],
        user: str,
        max_tokens: int,
        params: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMRequest:
        """A one-turn request: a system prompt and one user message."""
        return cls(
            backend,
            model,
            system,
            [{"role": "user", "content": user}],
            max_tokens,
            list(tools or []),
            dict(params or {}),
        )

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    @property
    def cache_key(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    def prompt_text(self) -> str:
        """Everything the model reads, as one string. For pessimistic size bounds only."""
        return json.dumps([self.system, self.messages, self.tools], ensure_ascii=False)

    def uses_prompt_cache(self) -> bool:
        """Whether any part of the request carries a prompt-cache marker."""
        return '"cache_control"' in self.canonical()


@dataclass(frozen=True)
class TokenUsage:
    input: int = 0  # uncached input tokens
    output: int = 0  # output tokens, thinking included
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    cache_read: int = 0

    @classmethod
    def from_usage(cls, usage: dict[str, Any]) -> TokenUsage:
        """From a Messages API `usage` object as JSON. `input_tokens` counts only the
        uncached remainder of the prompt. Cache writes are split by their time to live when
        the API reports the split; otherwise they are counted at the 1-hour price, the
        higher of the two, so the ledger never records less than was spent."""
        written = usage.get("cache_creation_input_tokens") or 0
        split = usage.get("cache_creation") or {}
        if split:
            write_1h = split.get("ephemeral_1h_input_tokens") or 0
            write_5m = split.get("ephemeral_5m_input_tokens") or 0
        else:
            write_1h, write_5m = written, 0
        return cls(
            input=usage.get("input_tokens") or 0,
            output=usage.get("output_tokens") or 0,
            cache_write_5m=write_5m,
            cache_write_1h=write_1h,
            cache_read=usage.get("cache_read_input_tokens") or 0,
        )

    @property
    def prompt_tokens(self) -> int:
        """The whole prompt: uncached input, cache writes and cache reads."""
        return self.input + self.cache_write_5m + self.cache_write_1h + self.cache_read


@dataclass
class LLMResponse:
    text: str  # the text blocks, joined
    content: list[dict[str, Any]]  # every content block (text, tool_use, thinking) as JSON
    model_reported: str  # the model string the server says it ran
    stop_reason: str | None
    usage: dict[str, Any]  # token counts as the backend reported them
    latency_ms: float
    created_utc: str
    request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> TokenUsage:
        return TokenUsage.from_usage(self.usage)

    def tool_calls(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]
