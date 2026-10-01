"""The agent the service runs: the registry's champion.

The deployable unit is a registered agent configuration (src/tracking/registry.py). The service
asks the registry which one holds the `champion` alias and runs exactly that: its design, model,
hint setting, router and guardrail, with the calibration and decline threshold it was evaluated
with. Promoting another configuration through the evaluation gate changes what the service runs
with no change to the service's code or its serving settings.

Without a registry server the committed state decides (the container image carries only that);
with `ANALYST_REGISTRY_URI` the alias is read from MLflow's model registry and must agree with it.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.serving.meter import ROOT, Meter
from src.tracking import registry

REGISTRY_URI = "ANALYST_REGISTRY_URI"


@dataclass(frozen=True)
class Champion:
    config: dict[str, Any]  # the resolved configuration
    config_sha256: str
    evaluation: dict[str, Any]  # its held-out evaluation file

    @property
    def name(self) -> str:
        return self.config["name"]

    @property
    def router(self) -> dict[str, Any] | None:
        return self.config["router"]

    def meter(self, cfg: dict[str, Any]) -> Meter:
        return Meter.from_registry(self.config, self.evaluation, cfg)

    def info(self) -> dict[str, Any]:
        """What the page and the metrics say about the running agent."""
        return {
            "name": self.name,
            "config_sha256": self.config_sha256,
            "design": self.config["design"],
            "model": self.config["model"],
            "router": None if self.router is None else {"model": self.router["model"]},
            "guardrail": self.config["guardrail"],
        }

    def apply(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """The serving settings with the live section set to what this agent runs."""
        out = copy.deepcopy(cfg)
        live = out["live"]
        live["design"] = self.config["design"]
        live["model"] = self.config["model"]
        live["evidence"] = self.config["evidence"]
        live["router"] = self.router is not None
        live["guardrail"] = bool(live.get("guardrail")) and self.config["guardrail"]
        return out


def load(root: Path = ROOT, tracking_uri: str | None = None) -> Champion:
    uri = tracking_uri if tracking_uri is not None else os.environ.get(REGISTRY_URI) or None
    state = registry.read_state(root)
    resolved = registry.load_champion(root, uri)
    version = state["versions"][registry.champion_name(state)]
    evaluation = json.loads((root / version["evaluation"]).read_text(encoding="utf-8"))
    return Champion(resolved, version["config_sha256"], evaluation)
