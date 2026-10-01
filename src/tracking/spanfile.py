"""Reading a run's recorded spans back, to send them to a tracing server.

An evaluation writes the spans of every run to a JSON-lines file (src/tracking/otel.py). A span
read back here keeps its trace and span ids, its timing and its attributes, so a tracing server
shows the recorded run as it happened, not a new one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags

from src.tracking.otel import SERVICE_NAME

RESOURCE = Resource.create({"service.name": SERVICE_NAME})
SCOPE = InstrumentationScope("src.agent")


def _context(trace_id: str, span_id: str) -> SpanContext:
    return SpanContext(
        int(trace_id, 16), int(span_id, 16), is_remote=False, trace_flags=TraceFlags(1)
    )


def to_span(d: dict[str, Any]) -> ReadableSpan:
    """A recorded span (one line of the file) as a finished span."""
    parent = _context(d["trace_id"], d["parent_id"]) if d.get("parent_id") else None
    return ReadableSpan(
        name=d["name"],
        context=_context(d["trace_id"], d["span_id"]),
        parent=parent,
        resource=RESOURCE,
        attributes=d.get("attributes") or {},
        kind=SpanKind.INTERNAL,
        instrumentation_scope=SCOPE,
        status=Status(StatusCode[d.get("status", "UNSET")]),
        start_time=d["start_ns"],
        end_time=d["end_ns"],
    )


def read(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_trace(records: Iterable[dict[str, Any]], question_id: str) -> list[dict[str, Any]]:
    """The spans of the one run that answered `question_id`: its `agent.run` root and
    everything in the same trace, in the order they ended."""
    records = list(records)
    roots = [
        r
        for r in records
        if r["name"] == "agent.run" and str(r["attributes"].get("question_id")) == str(question_id)
    ]
    if not roots:
        raise KeyError(f"no recorded run for question {question_id!r}")
    trace_ids = {r["trace_id"] for r in roots}
    return [r for r in records if r["trace_id"] in trace_ids]


def question_ids(records: Iterable[dict[str, Any]]) -> list[str]:
    return [
        str(r["attributes"]["question_id"])
        for r in records
        if r["name"] == "agent.run" and "question_id" in r["attributes"]
    ]
