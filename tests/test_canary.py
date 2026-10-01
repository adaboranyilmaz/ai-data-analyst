"""The canary: its fixed questions, how it reads a re-sent response, what it flags, its history."""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from src.llm.types import LLMResponse
from src.tracking import canary, gate

ROOT = canary.ROOT


@pytest.fixture(scope="module")
def cfg():
    return canary.config()


@pytest.fixture(scope="module")
def bundle():
    data = json.loads((ROOT / gate.BUNDLE).read_text(encoding="utf-8"))
    return {e["id"]: e for e in data["entries"]}


def test_the_fixed_set_is_twenty_questions_and_three_larger_model_calls(cfg):
    items = canary.items(cfg)
    assert len(items) == 23 and len({i["id"] for i in items}) == 23
    by_model = [i["model"] for i in items]
    assert by_model.count("claude-sonnet-5") == 20 and by_model.count("claude-opus-5-5") == 3
    larger = [i["id"] for i in items if i["model"] == "claude-opus-5-5"]
    assert larger == [f"bench-{q}/claude-opus-5-5" for q in cfg["benchmark"][:3]]
    assert [i["evidence"] for i in items if i["id"].startswith("bank-")] == [False] * 10


def test_no_expert_query_can_reach_a_request(cfg):
    """A banking question's request is built from its text alone; the file's gold query is never
    read into it."""
    own = yaml.safe_load((ROOT / "own_set/questions.yaml").read_text(encoding="utf-8"))
    gold = {q["id"]: " ".join(q["gold_sql"].split()) for q in own["questions"] if q.get("gold_sql")}
    for entry in canary.items(cfg):
        if not entry["id"].startswith("bank-"):
            continue
        run, box = gate._run(entry, ROOT)
        try:
            ((_, request),) = run.pending()
        finally:
            box.close()
        text = " ".join(json.dumps([request.system, request.messages]).split())
        if entry["question"]["question_id"] in gold:  # categories c, d and e have no expert query
            assert gold[entry["question"]["question_id"]] not in text


def test_the_recorded_requests_are_the_requests_the_canary_rebuilds(cfg, bundle):
    for entry in canary.items(cfg):
        if entry["id"] in bundle:
            run, box = gate._run(entry, ROOT)
            try:
                ((_, request),) = run.pending()
            finally:
                box.close()
            assert request.cache_key == bundle[entry["id"]]["request_key"], entry["id"]


def respond(bundle_entry, **changes) -> LLMResponse:
    r = copy.deepcopy(bundle_entry["response"])
    call = next(c for c in r["content"] if c["type"] == "tool_use")
    call["input"].update(changes)
    return LLMResponse(**r)


def test_an_unchanged_response_reads_as_unchanged(cfg, bundle):
    entry = next(i for i in canary.items(cfg) if i["id"] == "bench-782/claude-sonnet-5")
    out = canary.run_item(entry, lambda request: respond(bundle[entry["id"]]))
    assert out["valid_answer"] and out["sql_unchanged"] and not out["declined"]
    assert out["confidence"] == out["recorded_confidence"]
    assert out["prompt_tokens"] == out["recorded_prompt_tokens"]


def test_a_changed_response_is_noticed(cfg, bundle):
    entry = next(i for i in canary.items(cfg) if i["id"] == "bench-782/claude-sonnet-5")
    out = canary.run_item(
        entry, lambda request: respond(bundle[entry["id"]], sql="SELECT 1", confidence=0.31)
    )
    assert out["valid_answer"] and out["sql_unchanged"] is False
    assert abs(out["confidence"] - out["recorded_confidence"]) > 0.1


def test_a_response_that_does_not_parse_is_not_a_valid_answer(cfg, bundle):
    entry = next(i for i in canary.items(cfg) if i["id"] == "bench-782/claude-sonnet-5")
    bad = respond(bundle[entry["id"]])
    bad.content[:] = [{"type": "text", "text": "I cannot help with that."}]
    out = canary.run_item(entry, lambda request: bad)
    assert out["valid_answer"] is False


def result(**kw):
    base = {
        "id": "x",
        "model": "m",
        "stop_reason": "tool_use",
        "valid_answer": True,
        "declined": False,
        "recorded_declined": False,
        "sql_unchanged": True,
        "confidence": 0.9,
        "recorded_confidence": 0.9,
        "prompt_tokens": 1000,
        "recorded_prompt_tokens": 1000,
    }
    return base | kw


def test_a_quiet_run_raises_no_flag(cfg):
    s = canary.summarize([result(), result()], cfg["flags"])
    assert s["flags"] == [] and s["valid_answers"] == 1.0 and s["declined_changed"] == 0


@pytest.mark.parametrize(
    "changes, flag",
    [
        ({"valid_answer": False}, "did not parse"),
        ({"sql_unchanged": False}, "recorded SQL"),
        ({"confidence": 0.5}, "confidence moved"),
        ({"prompt_tokens": 1100}, "different tokens"),
    ],
)
def test_each_flag_is_raised_by_its_own_change(cfg, changes, flag):
    s = canary.summarize([result(**changes)], cfg["flags"])
    assert any(flag in f for f in s["flags"])


def test_the_history_appends_and_refuses_a_run_twice(tmp_path):
    path = tmp_path / "history.json"
    run = {"utc": "2026-10-01T10:00:00+00:00", "summary": {}, "results": []}
    assert len(canary.append(path, run)["runs"]) == 1
    assert len(canary.append(path, run | {"utc": "2026-10-02T10:00:00+00:00"})["runs"]) == 2
    with pytest.raises(ValueError):
        canary.append(path, run)


def test_the_runs_span_is_closed_so_a_trace_shows_the_whole_run(cfg, bundle):
    from src.tracking.otel import MemoryExporter, make_tracer

    entry = next(i for i in canary.items(cfg) if i["id"] == "bench-782/claude-sonnet-5")
    exporter = MemoryExporter()
    tracer, _ = make_tracer(exporter)
    canary.run_item(entry, lambda request: respond(bundle[entry["id"]]), tracer=tracer)
    names = [s.name for s in exporter.spans]
    assert names == ["llm.call", "agent.run"]
    root = exporter.spans[-1]
    assert root.attributes["question_id"] == "782" and "confidence" in root.attributes
