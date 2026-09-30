"""A LangChain chat model whose calls go through the project's response cache and spend ledger.

The graph's sub-agents speak LangChain: system and human messages, tools bound with a tool
choice. This adapter turns their messages into the project's request (src/llm/types.py
`LLMRequest`) with the content blocks, cache markers and parameters passed through unchanged,
sends it through the same `Caller` as the own loop (the response cache, then the spend ledger,
then the API), and returns the reply as an `AIMessage` carrying its tool calls. The same messages
therefore give the same request as the own loop builds, and so the same cache key: a graph run
replays the own loop's stored responses, and in replay-only mode it cannot call a model at all.

Each call is appended to `calls` (the request and the response), for the graph's record and
spans. The sub-agents make single calls, so only system, human and assistant messages are
translated.
"""

from __future__ import annotations

import copy
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from src.llm.types import LLMRequest


class CachedChatModel(BaseChatModel):
    model: str
    backend: str = "anthropic"
    max_tokens: int
    params: dict[str, Any] = Field(default_factory=dict)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: dict[str, Any] | None = None
    caller: Any = Field(default=None, exclude=True)  # src/agent/driver.py `Caller`
    calls: list = Field(default_factory=list, exclude=True)  # (request, response) per call

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "cached-messages-api"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "max_tokens": self.max_tokens, "params": self.params}

    def bind_tools(
        self, tools: list[dict[str, Any]], *, tool_choice: dict[str, Any] | None = None, **_: Any
    ) -> CachedChatModel:
        """The model with these tools (Messages API definitions, sent as they are) and this
        tool choice; the list of calls is shared with the unbound model."""
        return self.model_copy(
            update={"tools": [copy.deepcopy(t) for t in tools], "tool_choice": tool_choice}
        )

    def request(self, messages: list[BaseMessage]) -> LLMRequest:
        system: Any = ""
        out = []
        for m in messages:
            if isinstance(m, SystemMessage):
                system = copy.deepcopy(m.content)
            elif isinstance(m, HumanMessage):
                out.append({"role": "user", "content": copy.deepcopy(m.content)})
            elif isinstance(m, AIMessage):
                out.append({"role": "assistant", "content": copy.deepcopy(m.content)})
            else:
                raise TypeError(f"no translation for a {type(m).__name__}")
        params = copy.deepcopy(self.params)
        if self.tool_choice is not None:
            params["tool_choice"] = copy.deepcopy(self.tool_choice)
        return LLMRequest(
            self.backend, self.model, system, out, self.max_tokens, list(self.tools), params
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop:
            raise ValueError("stop sequences are not part of the project's requests")
        request = self.request(messages)
        response = self.caller.one(request)
        self.calls.append((request, response))
        tool_calls = [
            {"name": b["name"], "args": b["input"], "id": b["id"], "type": "tool_call"}
            for b in response.tool_calls()
        ]
        message = AIMessage(
            content=copy.deepcopy(response.content),
            tool_calls=tool_calls,
            response_metadata={
                "stop_reason": response.stop_reason,
                "cache_key": request.cache_key,
                "model": response.model_reported,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
