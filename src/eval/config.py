"""The harness settings (configs/eval.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_CONFIG = ROOT / "configs/eval.yaml"


def config() -> dict[str, Any]:
    return yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8"))


def official_evaluator_dir(cfg: dict[str, Any] | None = None) -> Path:
    return ROOT / (cfg or config())["official_evaluator"]["dir"]
