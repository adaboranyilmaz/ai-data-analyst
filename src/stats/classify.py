"""Which questions need statistics: keyword rules and a model, combined by OR.

A question is statistical when an honest answer needs an inference about the population the
data describes (whether a difference, trend or relationship holds, or one thing affects
another); a count, a lookup or arithmetic on stored numbers is descriptive, even when it compares
two of them. The rules (configs/guardrail.yaml `classify.rules`) look for the forms such a
question usually takes; the model (prompts/classify_v1.md) reads the question text only, no
schema and no hint. The guardrail takes a question down the statistical path if either says so,
since a missed statistical question risks an unguarded claim while a descriptive one sent down the
path costs one analysis. Each is also reported alone.
"""

from __future__ import annotations

import re
from typing import Any

KINDS = ("descriptive", "comparison", "trend", "association", "causal")


def rules(cfg: dict) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in cfg["classify"]["rules"]]


def rule_hits(question: str, patterns: list[re.Pattern]) -> list[str]:
    text = " ".join(question.strip().split())
    return [p.pattern for p in patterns if p.search(text)]


def parse(submitted: Any) -> dict[str, Any]:
    """The model's classification, checked; a malformed one counts as not statistical and is
    recorded as a format problem."""
    problems = []
    if not isinstance(submitted, dict):
        return {
            "statistical": None,
            "kind": None,
            "reason": None,
            "format_problems": ["no classification was submitted"],
        }
    stat = submitted.get("statistical")
    kind = submitted.get("kind")
    if not isinstance(stat, bool):
        problems.append(f"statistical {stat!r} is not true or false")
        stat = None
    if kind not in KINDS:
        problems.append(f"kind {kind!r} is not one of {KINDS}")
        kind = None
    if stat is not None and kind is not None and stat != (kind != "descriptive"):
        problems.append(f"statistical {stat} disagrees with kind {kind}")
    return {
        "statistical": stat,
        "kind": kind,
        "reason": submitted.get("reason"),
        "format_problems": problems,
    }


def combine(rule_flag: bool, model_flag: bool | None) -> bool:
    return bool(rule_flag or model_flag)
