"""Model backends behind one interface: `check(request)` and `generate(request)`.

Anthropic (Messages API):
- Parameters a model rejects with a 400 are refused by `check`, before a request is
  reserved, sent or batched, so a misconfigured arm fails at once and costs nothing.
  Claude Opus 5.5 cannot run with thinking disabled: effort is its only control, and it
  must be stated in the request, because its default (`medium`) would otherwise go
  unrecorded. Claude Sonnet 5 and Claude Opus 5.5 reject sampling parameters. Claude
  Opus 5.5 rejects a forced `tool_choice` (`any` or `tool`).
- Request params whose names start with "_" are metadata: part of the cache key, never
  sent (e.g. `_sample`, the sample index of a self-consistency arm).
- The response keeps every content block as JSON (text, tool_use, thinking), so a
  tool-using turn replays exactly, and the full `usage`, so cache writes and reads are
  priced.

Ollama (local model): runs at temperature 0 with a fixed seed. Ollama silently drops the
start of a prompt that exceeds `num_ctx`, so a request whose pessimistic size estimate would
not fit alongside `max_tokens` of output is refused beforehand. Requests use the Messages
API shape for both backends: tools are converted to Ollama's function schema, and the
model's tool calls come back as `tool_use` blocks.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any, Protocol

from src.llm.ledger import estimate_tokens_upper
from src.llm.types import LLMRequest, LLMResponse

# Removed from the SDK's `messages.create()` signature; sent via `extra_body` on the models
# that still accept them.
SAMPLING_PARAMS = ("temperature", "top_p", "top_k")
THINKING_ALWAYS_ON = frozenset({"claude-opus-5-5"})
REJECTS_SAMPLING = frozenset({"claude-sonnet-5", "claude-opus-5-5"})
REJECTS_FORCED_TOOL_CHOICE = frozenset({"claude-opus-5-5"})


class Backend(Protocol):
    name: str

    def check(self, request: LLMRequest) -> None: ...

    def generate(self, request: LLMRequest) -> LLMResponse: ...


class UnsupportedParams(ValueError):
    """A request parameter the model is known to reject."""


class ContextOverflow(ValueError):
    """A local-model prompt that could exceed the model's context window."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _to_json(obj: Any) -> Any:
    """An SDK object (pydantic model or plain attributes) as JSON data, dropping None."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list | tuple):
        return [_to_json(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {
            k: _to_json(v) for k, v in vars(obj).items() if v is not None and not k.startswith("_")
        }
    return obj


# --------------------------------------------------------------------------------------
# Anthropic


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, client: Any = None, max_retries: int = 5):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=max_retries)
        self.client = client

    @staticmethod
    def check(request: LLMRequest) -> None:
        model, params = request.model, request.params
        thinking = (params.get("thinking") or {}).get("type")
        if model in THINKING_ALWAYS_ON:
            if thinking in ("disabled", "enabled"):
                raise UnsupportedParams(
                    f"{model} rejects thinking {thinking!r}; set output_config.effort instead"
                )
            if "effort" not in (params.get("output_config") or {}):
                raise UnsupportedParams(
                    f"{model}: state output_config.effort explicitly, so the effort level is "
                    "recorded in the request rather than left to the model's default"
                )
        if model in REJECTS_SAMPLING and any(k in params for k in SAMPLING_PARAMS):
            raise UnsupportedParams(f"{model} rejects sampling parameters {SAMPLING_PARAMS}")
        tool_choice = (params.get("tool_choice") or {}).get("type")
        if model in REJECTS_FORCED_TOOL_CHOICE and tool_choice in ("any", "tool"):
            raise UnsupportedParams(f"{model} rejects a forced tool_choice ({tool_choice!r})")

    @staticmethod
    def call_params(request: LLMRequest, direct: bool = False) -> dict[str, Any]:
        """The Messages API parameters for a request, shared by direct and batch calls.

        For a direct call, sampling parameters go in `extra_body`, which is merged into the
        request JSON as it is; a batch request's params are sent as JSON already, so they
        stay where they are. The request, and so its cache key, is the same either way."""
        params = {k: v for k, v in request.params.items() if not k.startswith("_")}
        if direct:
            sampling = {k: params.pop(k) for k in SAMPLING_PARAMS if k in params}
            if sampling:
                params["extra_body"] = {**params.get("extra_body", {}), **sampling}
        out: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": request.messages,
        }
        if request.system:
            out["system"] = request.system
        if request.tools:
            out["tools"] = request.tools
        return {**out, **params}

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.check(request)
        t0 = time.perf_counter()
        msg = self.client.messages.create(**self.call_params(request, direct=True))
        latency = (time.perf_counter() - t0) * 1000
        return response_from_message(msg, latency, getattr(msg, "_request_id", None))


def response_from_message(
    msg: Any, latency_ms: float, request_id: str | None, extra: dict[str, Any] | None = None
) -> LLMResponse:
    content = [_to_json(b) for b in msg.content]
    extra = dict(extra or {})
    if msg.stop_reason == "refusal" and getattr(msg, "stop_details", None) is not None:
        extra["stop_details"] = _to_json(msg.stop_details)
    return LLMResponse(
        text="".join(b.get("text", "") for b in content if b.get("type") == "text"),
        content=content,
        model_reported=msg.model,
        stop_reason=msg.stop_reason,
        usage=_to_json(msg.usage),
        latency_ms=latency_ms,
        created_utc=_now(),
        request_id=request_id,
        extra=extra,
    )


# --------------------------------------------------------------------------------------
# Ollama


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if b.get("type") == "text")


def ollama_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """A Messages API tool definition in Ollama's function schema."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["input_schema"],
        },
    }


def ollama_messages(request: LLMRequest) -> list[dict[str, Any]]:
    """The system prompt and the Messages API conversation in Ollama's chat format:
    `tool_use` blocks become the assistant's `tool_calls`, and each `tool_result` block
    becomes a `tool` message naming the tool it answers."""
    out: list[dict[str, Any]] = []
    system = _text_of(request.system)
    if system:
        out.append({"role": "system", "content": system})
    tool_names: dict[str, str] = {}
    for m in request.messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            calls = []
            for b in content:
                if b.get("type") == "tool_use":
                    tool_names[b["id"]] = b["name"]
                    calls.append({"function": {"name": b["name"], "arguments": b["input"]}})
            msg: dict[str, Any] = {"role": "assistant", "content": _text_of(content)}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        for b in content:
            if b.get("type") == "tool_result":
                out.append(
                    {
                        "role": "tool",
                        "content": _text_of(b.get("content", "")),
                        "tool_name": tool_names.get(b["tool_use_id"], ""),
                    }
                )
        text = _text_of(content)
        if text:
            out.append({"role": m["role"], "content": text})
    return out


class OllamaBackend:
    name = "ollama"

    def __init__(self, client: Any = None, host: str | None = None):
        if client is None:
            import ollama

            client = ollama.Client(host=host or os.environ.get("OLLAMA_HOST") or None)
        self.client = client
        self._digests: dict[str, str | None] = {}

    @staticmethod
    def request_params(temperature: float, seed: int, num_ctx: int) -> dict[str, Any]:
        return {"options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}}

    @staticmethod
    def check(request: LLMRequest) -> None:
        num_ctx = request.params["options"]["num_ctx"]
        needed = estimate_tokens_upper(request.prompt_text()) + request.max_tokens
        if needed > num_ctx:
            raise ContextOverflow(
                f"prompt (pessimistic estimate) + max_tokens = {needed} tokens exceeds "
                f"num_ctx={num_ctx}; Ollama would silently truncate the prompt"
            )

    def digest(self, model: str) -> str | None:
        """Content digest of the local weights: the pinned version of a local model, since a
        tag like `qwen2.5:3b-instruct` can be re-pointed upstream."""
        if model not in self._digests:
            listed = {m.model: m.digest for m in self.client.list().models}
            self._digests[model] = listed.get(model)
        return self._digests[model]

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.check(request)
        options = {**request.params["options"], "num_predict": request.max_tokens}
        t0 = time.perf_counter()
        resp = self.client.chat(
            model=request.model,
            messages=ollama_messages(request),
            tools=[ollama_tool(t) for t in request.tools] or None,
            options=options,
            stream=False,
        )
        latency = (time.perf_counter() - t0) * 1000
        content: list[dict[str, Any]] = []
        if getattr(resp.message, "thinking", None):
            content.append({"type": "thinking", "thinking": resp.message.thinking})
        if resp.message.content:
            content.append({"type": "text", "text": resp.message.content})
        for i, call in enumerate(resp.message.tool_calls or []):
            content.append(
                {
                    "type": "tool_use",
                    "id": f"call_{i}",
                    "name": call.function.name,
                    "input": dict(call.function.arguments),
                }
            )
        return LLMResponse(
            text=resp.message.content or "",
            content=content,
            model_reported=resp.model,
            stop_reason=resp.done_reason,
            usage={"input_tokens": resp.prompt_eval_count, "output_tokens": resp.eval_count},
            latency_ms=latency,
            created_utc=_now(),
            extra={
                "digest": self.digest(request.model),
                "load_ms": (resp.load_duration or 0) / 1e6,
                "prompt_eval_ms": (resp.prompt_eval_duration or 0) / 1e6,
                "eval_ms": (resp.eval_duration or 0) / 1e6,
            },
        )


def make_backend(backend: str) -> Backend:
    if backend == "anthropic":
        return AnthropicBackend()
    if backend == "ollama":
        return OllamaBackend()
    raise ValueError(f"unknown backend {backend!r}")
