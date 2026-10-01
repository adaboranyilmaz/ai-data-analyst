"""Sending the agent's spans to tracing servers over OTLP/HTTP, from the one instrumentation.

The agent opens its spans once (src/agent/); where they go is decided here. A provider built
with `provider(exporters)` fans each finished span out to every exporter: the local JSON-lines
file, an MLflow tracking server (its OTLP endpoint) and a Langfuse server, whichever are set.
Nothing in the agent knows which tools are watching.

Langfuse reads OpenTelemetry attributes with its own names; `LangfuseExporter` adds them to a
copy of each span on the way out (a model call becomes a generation with its token usage and
cost), so the instrumentation itself stays tool-neutral.

Spans are observability, not results: no reported number is read from them, and a tracing server
that is down never stops a run.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Mapping, Sequence

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from src.tracking.otel import SERVICE_NAME

log = logging.getLogger(__name__)

MLFLOW_URI = "ANALYST_TRACE_MLFLOW_URI"
MLFLOW_EXPERIMENT_ID = "ANALYST_TRACE_MLFLOW_EXPERIMENT_ID"
LANGFUSE_HOST = "LANGFUSE_HOST"
LANGFUSE_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
LANGFUSE_SECRET_KEY = "LANGFUSE_SECRET_KEY"
EXPORT_TIMEOUT_S = 10


class FanOutExporter(SpanExporter):
    """Gives every finished span to each exporter; one that fails does not stop the others."""

    def __init__(self, exporters: Sequence[SpanExporter]):
        self.exporters = list(exporters)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        ok = True
        for exporter in self.exporters:
            try:
                if exporter.export(spans) is not SpanExportResult.SUCCESS:
                    ok = False
            except Exception:  # a tracing server down must never stop a run
                log.warning("span export failed for %s", type(exporter).__name__, exc_info=True)
                ok = False
        return SpanExportResult.SUCCESS if ok else SpanExportResult.FAILURE

    def shutdown(self) -> None:
        for exporter in self.exporters:
            exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return all(e.force_flush(timeout_millis) for e in self.exporters)


def mlflow_exporter(tracking_uri: str, experiment_id: str) -> SpanExporter:
    """MLflow's OTLP endpoint (`/v1/traces` on the tracking server); the header names the
    experiment the traces belong to."""
    return OTLPSpanExporter(
        endpoint=tracking_uri.rstrip("/") + "/v1/traces",
        headers={"x-mlflow-experiment-id": str(experiment_id)},
        timeout=EXPORT_TIMEOUT_S,
    )


def langfuse_auth(public_key: str, secret_key: str) -> str:
    return "Basic " + base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()


def with_attributes(span: ReadableSpan, extra: Mapping[str, object]) -> ReadableSpan:
    """A copy of a finished span with more attributes (a span's own are read-only)."""
    return ReadableSpan(
        name=span.name,
        context=span.get_span_context(),
        parent=span.parent,
        resource=span.resource,
        attributes={**dict(span.attributes or {}), **extra},
        events=span.events,
        links=span.links,
        kind=span.kind,
        instrumentation_scope=span.instrumentation_scope,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
    )


def langfuse_attributes(span: ReadableSpan) -> dict[str, object]:
    """The attributes Langfuse reads, derived from the agent's own: a model call is a
    generation (model, token usage, cost); a run or a tool call is a plain observation."""
    a = span.attributes or {}
    if span.name == "llm.call":
        usage = {
            "input": a.get("input_tokens", 0),
            "output": a.get("output_tokens", 0),
            "cache_read_input_tokens": a.get("cache_read_tokens", 0),
            "cache_creation_input_tokens": a.get("cache_write_tokens", 0),
        }
        out: dict[str, object] = {
            "langfuse.observation.type": "generation",
            "langfuse.observation.model.name": a.get("model", ""),
            "langfuse.observation.usage_details": json.dumps(usage),
        }
        if "cost_usd" in a:
            out["langfuse.observation.cost_details"] = json.dumps({"total": a["cost_usd"]})
        return out
    out = {"langfuse.observation.type": "span"}
    if span.name == "tool.call" and "sql" in a:
        out["langfuse.observation.input"] = str(a["sql"])
    if span.parent is None:
        out["langfuse.trace.name"] = span.name
    return out


class LangfuseExporter(OTLPSpanExporter):
    """OTLP to a Langfuse server, with Langfuse's attribute names added to each span."""

    def __init__(self, host: str, public_key: str, secret_key: str):
        super().__init__(
            endpoint=host.rstrip("/") + "/api/public/otel/v1/traces",
            headers={"Authorization": langfuse_auth(public_key, secret_key)},
            timeout=EXPORT_TIMEOUT_S,
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return super().export([with_attributes(s, langfuse_attributes(s)) for s in spans])


def exporters_from_env(env: Mapping[str, str]) -> list[SpanExporter]:
    """The tracing servers the environment names; none when it names none (the default)."""
    out: list[SpanExporter] = []
    if env.get(MLFLOW_URI) and env.get(MLFLOW_EXPERIMENT_ID):
        out.append(mlflow_exporter(env[MLFLOW_URI], env[MLFLOW_EXPERIMENT_ID]))
    if env.get(LANGFUSE_HOST) and env.get(LANGFUSE_PUBLIC_KEY) and env.get(LANGFUSE_SECRET_KEY):
        out.append(
            LangfuseExporter(env[LANGFUSE_HOST], env[LANGFUSE_PUBLIC_KEY], env[LANGFUSE_SECRET_KEY])
        )
    return out


def provider(exporters: Sequence[SpanExporter], batch: bool = False) -> TracerProvider:
    """A tracer provider of its own (not the global one) that sends every span to all of
    `exporters`. `batch`: export on a background thread, as a service does; scripts export
    each span as it ends and so can read the result straight away."""
    tp = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    fan = FanOutExporter(exporters)
    tp.add_span_processor(BatchSpanProcessor(fan) if batch else SimpleSpanProcessor(fan))
    return tp
