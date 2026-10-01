"""Recorded spans read back keep their ids, timing and attributes, and reach a server intact."""

from __future__ import annotations

import pytest
from opentelemetry import trace as ot

from src.tracking import otlp, spanfile
from src.tracking.otel import JsonLinesExporter, make_tracer
from tests.test_otlp import Receiver


@pytest.fixture
def recorded(tmp_path):
    path = tmp_path / "spans.jsonl"
    tracer, _ = make_tracer(JsonLinesExporter(path))
    for qid in ("7", "8"):
        root = tracer.start_span("agent.run")
        root.set_attributes({"question_id": qid, "model": "claude-sonnet-5"})
        parent = ot.set_span_in_context(root)
        call = tracer.start_span("llm.call", context=parent)
        call.set_attributes({"model": "claude-sonnet-5", "input_tokens": 10, "output_tokens": 3})
        call.end()
        tool = tracer.start_span("tool.call", context=parent)
        tool.set_attributes({"tool": "run_sql", "sql": "SELECT 1", "ok": True})
        tool.end()
        root.end()
    return path


def test_a_run_is_selected_by_its_question(recorded):
    records = spanfile.read(recorded)
    assert spanfile.question_ids(records) == ["7", "8"]
    spans = spanfile.run_trace(records, "8")
    assert [s["name"] for s in spans] == ["llm.call", "tool.call", "agent.run"]
    assert len({s["trace_id"] for s in spans}) == 1
    with pytest.raises(KeyError):
        spanfile.run_trace(records, "9")


def test_spans_keep_ids_timing_and_parents(recorded):
    records = spanfile.run_trace(spanfile.read(recorded), "7")
    spans = [spanfile.to_span(r) for r in records]
    for r, s in zip(records, spans, strict=True):
        ctx = s.get_span_context()
        assert f"{ctx.trace_id:032x}" == r["trace_id"] and f"{ctx.span_id:016x}" == r["span_id"]
        assert s.start_time == r["start_ns"] and s.end_time == r["end_ns"]
        assert dict(s.attributes) == r["attributes"]
        assert (s.parent is None) == (r["parent_id"] is None)
    root = spans[-1]
    assert all(s.parent.span_id == root.get_span_context().span_id for s in spans[:-1])


def test_recorded_spans_reach_an_otlp_endpoint(recorded):
    spans = [spanfile.to_span(r) for r in spanfile.run_trace(spanfile.read(recorded), "7")]
    with Receiver() as rx:
        exporter = otlp.mlflow_exporter(rx.url, "3")
        assert exporter.export(spans).name == "SUCCESS"
    assert sorted(rx.span_names()) == ["agent.run", "llm.call", "tool.call"]
    sent = [
        s
        for _, _, m in rx.requests
        for rs in m.resource_spans
        for ss in rs.scope_spans
        for s in ss.spans
    ]
    trace_id = spanfile.run_trace(spanfile.read(recorded), "7")[0]["trace_id"]
    assert {s.trace_id.hex() for s in sent} == {trace_id}
