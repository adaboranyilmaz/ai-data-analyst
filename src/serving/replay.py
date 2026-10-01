"""A recorded run as a stream of events with their delays.

The same events a live run sends: `start`, one `step` per step, then the SQL, rows, checks,
statistics, chart, answer and confidence, then `done`. A step keeps the duration it was recorded
with, capped; a step with no recorded duration (a batched call has none) takes the configured
default. Delays are returned, not slept, so a test or the serving check can run a replay at once
and the service can sleep them.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from src.serving.evidence import final_events

# the events after the steps arrive this soon after one another: they are computed, not waited for
FINAL_GAP_MS = 250


def start_event(ev: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "type": "start",
        "id": ev["id"],
        "mode": mode,
        "question": ev["question"],
        "hint": ev.get("hint"),
        "db_id": ev["db_id"],
        "kind": ev["kind"],
    }


def done_event(ev: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "done",
        "id": ev["id"],
        "status": ev["status"],
        "evaluation": ev["evaluation"],
        "explanation": ev.get("explanation"),
    }


def step_delay_ms(step: dict[str, Any], cfg: dict[str, Any]) -> float:
    ms = step.get("recorded_ms") or cfg["default_ms"]
    return min(float(ms), float(cfg["cap_ms"]))


def events(ev: dict[str, Any], cfg: dict[str, Any]) -> Iterator[tuple[dict[str, Any], float]]:
    """(event, seconds to wait before sending it), in order."""
    speed = float(cfg.get("speed") or 1.0)
    yield start_event(ev, "replay"), 0.0
    for step in ev["steps"]:
        yield step, step_delay_ms(step, cfg) / 1000 / speed
    for event in final_events(ev):
        yield event, FINAL_GAP_MS / 1000 / speed
    yield done_event(ev), 0.0


def assemble(stream: list[dict[str, Any]]) -> dict[str, Any]:
    """What a client reads off a stream: the steps and the final answer, as the page shows them.
    The serving check compares this with the evaluated record."""
    out: dict[str, Any] = {"steps": [], "sql": None, "answer": None, "confidence": None}
    for e in stream:
        if e["type"] == "step":
            out["steps"].append(e["text"])
        elif e["type"] == "sql":
            out["sql"] = e["sql"]
        elif e["type"] == "rows":
            out["columns"], out["rows"] = e["columns"], e["rows"]
        elif e["type"] == "answer":
            out["answer"] = {k: v for k, v in e.items() if k != "type"}
        elif e["type"] == "confidence":
            out["confidence"] = {k: v for k, v in e.items() if k != "type"}
        elif e["type"] == "done":
            out["status"] = e["status"]
    return out
