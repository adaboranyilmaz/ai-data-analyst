"""OTLP export: fan-out, Langfuse's attribute names, and spans arriving at a real HTTP endpoint."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from src.tracking import otlp
from src.tracking.otel import MemoryExporter


class Receiver:
    """A local OTLP/HTTP endpoint that keeps what it is sent."""

    def __init__(self):
        self.requests: list[tuple[str, dict, ExportTraceServiceRequest]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                msg = ExportTraceServiceRequest()
                msg.ParseFromString(body)
                outer.requests.append((self.path, dict(self.headers), msg))
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.end_headers()
                self.wfile.write(b"")

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def span_names(self) -> list[str]:
        return [
            s.name
            for _, _, msg in self.requests
            for rs in msg.resource_spans
            for ss in rs.scope_spans
            for s in ss.spans
        ]

    def attributes(self, span_name: str) -> dict[str, str]:
        for _, _, msg in self.requests:
            for rs in msg.resource_spans:
                for ss in rs.scope_spans:
                    for s in ss.spans:
                        if s.name == span_name:
                            return {
                                kv.key: kv.value.WhichOneof("value")
                                and getattr(kv.value, kv.value.WhichOneof("value"))
                                for kv in s.attributes
                            }
        return {}


def agent_spans(tp):
    tracer = tp.get_tracer("t")
    root = tracer.start_span("agent.run")
    from opentelemetry import trace as ot

    parent = ot.set_span_in_context(root)
    call = tracer.start_span("llm.call", context=parent)
    call.set_attributes(
        {
            "model": "claude-sonnet-5",
            "input_tokens": 45,
            "output_tokens": 553,
            "cache_read_tokens": 11378,
            "cache_write_tokens": 0,
            "cost_usd": 0.0079,
        }
    )
    call.end()
    tool = tracer.start_span("tool.call", context=parent)
    tool.set_attributes({"tool": "run_sql", "sql": "SELECT 1", "ok": True})
    tool.end()
    root.end()


def test_exporters_from_env_none_by_default():
    assert otlp.exporters_from_env({}) == []
    # half a Langfuse configuration is no configuration
    assert otlp.exporters_from_env({otlp.LANGFUSE_HOST: "http://x"}) == []
    assert otlp.exporters_from_env({otlp.MLFLOW_URI: "http://x"}) == []


def test_mlflow_endpoint_and_experiment_header():
    with Receiver() as rx:
        env = {otlp.MLFLOW_URI: rx.url + "/", otlp.MLFLOW_EXPERIMENT_ID: "7"}
        (exporter,) = otlp.exporters_from_env(env)
        agent_spans(otlp.provider([exporter]))
        path, headers, _ = rx.requests[0]
    assert path == "/v1/traces"
    assert {k.lower(): v for k, v in headers.items()}["x-mlflow-experiment-id"] == "7"
    assert sorted(rx.span_names()) == ["agent.run", "llm.call", "tool.call"]


def test_langfuse_endpoint_auth_and_attribute_names():
    with Receiver() as rx:
        env = {
            otlp.LANGFUSE_HOST: rx.url,
            otlp.LANGFUSE_PUBLIC_KEY: "pk-lf-1",
            otlp.LANGFUSE_SECRET_KEY: "sk-lf-2",
        }
        (exporter,) = otlp.exporters_from_env(env)
        agent_spans(otlp.provider([exporter]))
        path, headers, _ = rx.requests[0]
        call = rx.attributes("llm.call")
        root = rx.attributes("agent.run")
        tool = rx.attributes("tool.call")
    assert path == "/api/public/otel/v1/traces"
    sent = {k.lower(): v for k, v in headers.items()}
    assert sent["authorization"] == otlp.langfuse_auth("pk-lf-1", "sk-lf-2")
    assert call["langfuse.observation.type"] == "generation"
    assert call["langfuse.observation.model.name"] == "claude-sonnet-5"
    usage = json.loads(call["langfuse.observation.usage_details"])
    assert usage["input"] == 45 and usage["output"] == 553 and usage["cache_read_input_tokens"]
    assert json.loads(call["langfuse.observation.cost_details"]) == {"total": 0.0079}
    assert root["langfuse.trace.name"] == "agent.run"
    assert tool["langfuse.observation.input"] == "SELECT 1"
    # the agent's own attributes are kept alongside
    assert call["input_tokens"] == 45


def test_with_attributes_leaves_the_original_span_alone():
    mem = MemoryExporter()
    agent_spans(otlp.provider([mem]))
    span = next(s for s in mem.spans if s.name == "llm.call")
    copy = otlp.with_attributes(span, {"extra": 1})
    assert copy.attributes["extra"] == 1 and "extra" not in span.attributes
    assert copy.get_span_context() == span.get_span_context()
    assert copy.parent == span.parent and copy.end_time == span.end_time


class Failing(SpanExporter):
    def export(self, spans):
        raise ConnectionError("tracing server down")

    def shutdown(self):
        pass


def test_a_failing_exporter_does_not_stop_the_others_or_the_run():
    mem = MemoryExporter()
    agent_spans(otlp.provider([Failing(), mem]))  # no exception reaches the caller
    assert sorted(s.name for s in mem.spans) == ["agent.run", "llm.call", "tool.call"]
    assert otlp.FanOutExporter([Failing()]).export([]) is SpanExportResult.FAILURE
