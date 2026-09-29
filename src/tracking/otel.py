"""OpenTelemetry for the agent: one instrumentation, exported as JSON lines for now.

The agent opens a span per run, per sample, per model call and per tool call (src/agent/). Here
they are written, one JSON object per line, to a local file; other exporters (a tracing server)
attach to the same provider without touching the instrumentation.

Spans are observability, not results: their timings differ on every run, and no reported number
is read from them.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Tracer

SERVICE_NAME = "ai-data-analyst"


def span_dict(span: ReadableSpan) -> dict[str, Any]:
    ctx = span.get_span_context()
    return {
        "name": span.name,
        "trace_id": f"{ctx.trace_id:032x}",
        "span_id": f"{ctx.span_id:016x}",
        "parent_id": f"{span.parent.span_id:016x}" if span.parent else None,
        "start_ns": span.start_time,
        "end_ns": span.end_time,
        "status": span.status.status_code.name,
        "attributes": dict(span.attributes or {}),
    }


class JsonLinesExporter(SpanExporter):
    """Appends each finished span to a JSON-lines file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        lines = "".join(json.dumps(span_dict(s), default=str) + "\n" for s in spans)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(lines)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


class MemoryExporter(SpanExporter):
    """Keeps finished spans in a list (tests)."""

    def __init__(self):
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def make_tracer(exporter: SpanExporter, name: str = "src.agent") -> tuple[Tracer, TracerProvider]:
    """A tracer on its own provider (not the global one, so runs and tests do not share state).
    Spans are exported as each ends."""
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(name), provider


def jsonl_tracer(path: Path) -> tuple[Tracer, TracerProvider]:
    return make_tracer(JsonLinesExporter(path))
