"""The router's larger model, as the service runs it.

The winning design forces its one `submit_answer` call, and Claude Opus 5.5 refuses a forced tool
choice, so on such a model the call is asked for by the prompt and the tool list alone, with the
loop's one reminder. This is the class and the settings merge the escalation arm was measured
with (src/agent/escalation.py), kept here so the service imports no evaluation code; a test holds
the two to the same requests, byte for byte.
"""

from __future__ import annotations

import copy
from typing import Any

from src.agent.answer import SUBMIT
from src.agent.conversation import Settings
from src.agent.run import QuestionRun, config
from src.llm.backends import REJECTS_FORCED_TOOL_CHOICE


class AutoToolRun(QuestionRun):
    """A question run whose single-shot call is not forced on models that refuse it."""

    def _settings(self, **kw) -> Settings:
        if kw.get("force_tool") == SUBMIT and self.model in REJECTS_FORCED_TOOL_CHOICE:
            kw = {**kw, "force_tool": None, "reminders": self.cfg["reminders"]}
        return super()._settings(**kw)


def agent_config_with(model: str, settings: dict[str, Any]) -> dict[str, Any]:
    """The agent's configuration with one more model's settings."""
    cfg = copy.deepcopy(config())
    cfg["models"][model] = copy.deepcopy(settings)
    return cfg
