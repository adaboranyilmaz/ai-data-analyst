"""Does what the service serves equal what was evaluated?

Three comparisons per curated run, each exact:
1. the served evidence record against the evaluated record it came from (answer, SQL, stated
   confidence, the verdict, and the calibrated confidence recomputed from the stated one);
2. the replayed event stream, assembled the way a client reads it, against the evidence record;
3. the same stream as the API sends it, against the evidence record.

A run that differs on any field is listed with the field, so a regression in the replay or in
the meter shows up as a named difference, never as a changed number somewhere else.
"""

from __future__ import annotations

from typing import Any

from src.eval.summary import own_success
from src.serving import replay
from src.serving.meter import Meter


def _verdict(record: dict[str, Any]) -> bool | None:
    if record["source"] == "own":
        v = own_success(record)
        return None if v is None else bool(v)
    return bool(record["correct"])


def against_record(
    ev: dict[str, Any], record: dict[str, Any] | None, meter: Meter
) -> dict[str, tuple[Any, Any]]:
    """Fields where the served evidence record differs from the evaluated record: name ->
    (served, evaluated). Guardrail runs have their own record shape."""
    if ev["kind"] == "guardrail":
        return {} if record is None else _against_guardrail(ev, record)
    if record is None:
        return {"record": (ev["id"], None)}
    served = {
        "answer_text": ev["answer"]["text"],
        "sql": ev["answer"]["sql"],
        "declined": ev["answer"]["declined"],
        "stated_confidence": ev["confidence"]["stated"],
        "calibrated": ev["confidence"]["calibrated"],
        "verdict": ev["evaluation"]["correct"],
    }
    declined = bool(record["declined"])
    expected_conf = meter.describe(record["confidence"], declined)
    evaluated = {
        "answer_text": record["answer"],
        "sql": None if declined else record["final_sql"],
        "declined": declined,
        "stated_confidence": expected_conf["stated"],
        "calibrated": expected_conf["calibrated"],
        "verdict": _verdict(record),
    }
    return {k: (served[k], evaluated[k]) for k in served if served[k] != evaluated[k]}


def _against_guardrail(ev: dict[str, Any], rec: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    served = {"answer_text": ev["answer"]["text"], "statistics": ev["statistics"]}
    evaluated = {"answer_text": rec["guarded"]["delivered"], "statistics": rec["analysis"]}
    return {k: (served[k], evaluated[k]) for k in served if served[k] != evaluated[k]}


def stream_against_record(stream: list[dict[str, Any]], ev: dict[str, Any]) -> dict[str, tuple]:
    """Fields where the assembled stream differs from the evidence record."""
    got = replay.assemble(stream)
    answer = got["answer"] or {}
    served = {
        "answer_text": answer.get("text"),
        "sql": got["sql"],
        "declined": answer.get("declined"),
        "confidence": got["confidence"],
        "status": got.get("status"),
        "steps": got["steps"],
    }
    want_conf = {k: v for k, v in ev["confidence"].items()}
    want = {
        "answer_text": ev["answer"]["text"],
        "sql": ev["answer"]["sql"],
        "declined": ev["answer"]["declined"],
        "confidence": want_conf,
        "status": ev["status"],
        "steps": [s["text"] for s in ev["steps"]],
    }
    return {k: (served[k], want[k]) for k in served if served[k] != want[k]}
