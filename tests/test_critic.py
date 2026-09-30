"""The critic: its verdict parsing and review text, and (on the loaded benchmark, with a scripted
model in place of an API) its requests, the answers it does not review, its records, replay at
$0 and a result that reads the clock."""

from __future__ import annotations

import json

import pytest

from src.agent.confidence import confidence_config
from src.agent.critic import (
    VERDICT,
    VERDICT_TOOL,
    CriticRun,
    CriticSpec,
    _checks_text,
    execute,
    items_for,
    load_prompt,
    parse_verdict,
    review_text,
)
from src.agent.evaluate import benchmark_set
from src.agent.run import QuestionRun, config
from src.llm.cache import CacheMiss
from src.llm.types import LLMResponse
from src.tools.toolbox import Toolbox

CHECKS_OK = {
    "applicable": True,
    "empty": False,
    "repeated_rows": False,
    "out_of_range": [],
    "too_many_rows": False,
    "failed": False,
}


class TestVerdict:
    def test_a_valid_verdict(self):
        v = parse_verdict({"verdict": "incorrect", "confidence": 0.2, "problems": ["extra column"]})
        assert (v.verdict, v.confidence, v.problems, v.format_problems) == (
            "incorrect",
            0.2,
            ("extra column",),
            (),
        )

    def test_a_confidence_outside_0_1_is_clipped_and_recorded(self):
        v = parse_verdict({"verdict": "correct", "confidence": 1.4, "problems": []})
        assert v.confidence == 1.0 and "clipped" in v.format_problems[0]

    def test_malformed_fields_are_recorded_not_repaired(self):
        v = parse_verdict({"verdict": "maybe", "confidence": "high", "problems": "none"})
        assert v.verdict is None and v.confidence is None and v.problems == ()
        assert len(v.format_problems) == 3
        assert parse_verdict("text").format_problems == ("the verdict is not an object",)

    def test_the_tool_is_strict(self):
        assert VERDICT_TOOL["strict"] and VERDICT_TOOL["name"] == VERDICT
        assert VERDICT_TOOL["input_schema"]["additionalProperties"] is False


class TestReviewText:
    answer = {"final_sql": "SELECT 1", "answer": "One.", "assumptions": ["a is b"]}

    def test_shows_the_sql_answer_assumptions_result_and_checks(self):
        ev = {
            "columns": ["n"],
            "rows": 1,
            "truncated": False,
            "preview": [[1]],
            "checks": CHECKS_OK,
        }
        text = review_text("Question: Q?\nHint: H.", self.answer, ev, 1000)
        assert text.startswith("Question: Q?\nHint: H.")
        assert "SELECT 1" in text and "One." in text and "- a is b" in text
        assert '{"columns":["n"],"rows":[[1]]}' in text
        assert "The query's result: 1 rows." in text
        assert "more than 1,000 rows: no." in text

    def test_says_when_only_the_first_rows_are_shown(self):
        ev = {
            "columns": ["n"],
            "rows": 50_000,
            "truncated": True,
            "preview": [[i] for i in range(20)],
            "checks": {**CHECKS_OK, "too_many_rows": True, "failed": True},
        }
        text = review_text("Question: Q?", {**self.answer, "assumptions": []}, ev, 1000)
        assert "50,000 rows (more were not fetched); the first 20 shown." in text
        assert "Assumptions stated: none" in text
        assert "more than 1,000 rows: yes." in text

    def test_names_values_out_of_range(self):
        checks = {**CHECKS_OK, "out_of_range": [{"column": "n", "kind": "count", "value": -2.0}]}
        assert "column n (count) holds -2.0" in _checks_text(checks, 1000)


def test_the_critic_config():
    crit = confidence_config()["critic"]
    assert crit["model"] == "claude-sonnet-5"
    assert crit["settings"]["params"] == {"thinking": {"type": "disabled"}}
    assert crit["rows_shown"] == config()["presentation"]["rows_shown"]
    text, sha = load_prompt(crit["prompt"])
    assert "submit_verdict" in text and len(sha) == 64


# ------------------------------------------------------------------------------ on the benchmark

GOOD = "SELECT COUNT(*) FROM loan"
STATED = 0.123456  # a stated confidence no request may carry


def answer_record(sql=GOOD, declined=False):
    return {
        "final_sql": None if declined else sql,
        "answer": "The number of loans.",
        "confidence": STATED,
        "declined": declined,
        "assumptions": ["loans are rows of loan"],
    }


def verdict_response(confidence=0.8, stop="tool_use"):
    args = {"verdict": "correct", "confidence": confidence, "problems": []}
    return LLMResponse(
        text="",
        content=[{"type": "tool_use", "id": "v1", "name": VERDICT, "input": args}],
        model_reported="m",
        stop_reason=stop,
        usage={"input_tokens": 900, "output_tokens": 40, "cache_read_input_tokens": 4000},
        latency_ms=5.0,
        created_utc="2026-09-29T00:00:00+00:00",
    )


class Critic:
    name = "fake"

    def __init__(self):
        self.calls = []

    def check(self, request):
        pass

    def generate(self, request):
        self.calls.append(request)
        return verdict_response()


class NoCalls:
    name = "fake"

    def check(self, request):
        pass

    def generate(self, request):
        raise AssertionError("replay must not call the model")


@pytest.fixture
def financial(bird_ready):
    q = next(q for q, _ in benchmark_set("pilot") if q.db_id == "financial")
    box = Toolbox("financial")
    yield q, box
    box.close()


def critic_run(q, box, answer, clock=None):
    crit = confidence_config()["critic"]
    system, _ = load_prompt(crit["prompt"])
    return CriticRun(q, answer, "main/x", box, crit, system, config(), lambda *a: 0.0, None, clock)


@pytest.mark.bird
class TestCriticRun:
    def test_the_request(self, financial):
        q, box = financial
        run = critic_run(q, box, answer_record())
        ((_, req),) = run.pending()
        assert req.params["tool_choice"] == {"type": "tool", "name": VERDICT}
        assert req.params["thinking"] == {"type": "disabled"}
        assert [t["name"] for t in req.tools] == [VERDICT]
        assert req.system[0]["text"].startswith("You review the work of a data analyst")
        context, body = req.messages[0]["content"]
        # the schema block is exactly the one design 1 reads, and is cached
        d1 = QuestionRun(q, "d1", "claude-sonnet-5", True, box, lambda *a: 0.0)
        assert context == d1.pending()[0][1].messages[0]["content"][0]
        assert body["text"].startswith("Question: ") and GOOD in body["text"]
        assert "Automatic checks: empty result: no" in body["text"]
        assert str(STATED) not in req.canonical()  # never the stated confidence

    @pytest.mark.parametrize(
        ("answer", "why"),
        [
            (answer_record(declined=True), "declined"),
            (answer_record(sql=None), "no_sql"),
            (answer_record(sql="SELECT nope FROM loan"), "final_sql_sql_error"),
            (answer_record(sql="DELETE FROM loan"), "final_sql_refused"),
        ],
    )
    def test_an_answer_without_a_result_is_not_reviewed(self, financial, answer, why):
        q, box = financial
        run = critic_run(q, box, answer)
        assert run.done and run.pending() == []
        record, trace = run.finish()
        assert (record["reviewed"], record["not_reviewed"], record["confidence"]) == (
            False,
            why,
            0.0,
        )
        assert record["cost_usd"] == 0 and trace["messages"] is None

    def test_a_review_without_a_verdict_has_no_confidence(self, financial):
        q, box = financial
        run = critic_run(q, box, answer_record())
        ((conv, req),) = run.pending()
        run.feed(conv, req.cache_key, verdict_response(stop="max_tokens"))
        record, _ = run.finish()
        assert record["reviewed"] and record["confidence"] is None
        assert record["errors"][0]["kind"] == "max_tokens"


def spec(tmp_path, items, mode="direct"):
    return CriticSpec(
        name="critic/test",
        items=items,
        answer_run="main/x",
        evidence=True,
        phase="phase5",
        cache_dir=tmp_path / "cache",
        mode=mode,
        records=tmp_path / "records.jsonl",
        traces_dir=tmp_path / "traces",
        spans=tmp_path / "spans.jsonl",
        ledger_path=tmp_path / "spend.json",
        batch_dir=tmp_path / "batches",
        poll_seconds=0,
    )


def stable(records):
    return [{k: v for k, v in r.items() if k not in ("latency_s", "evaluated_at")} for r in records]


@pytest.mark.bird
def test_records_traces_spans_and_replay_at_zero_cost(tmp_path, bird_ready, monkeypatch):
    qs = [q for q, _ in benchmark_set("ablation") if q.db_id == "financial"][:2]
    items = [(qs[0], answer_record()), (qs[1], answer_record(declined=True))]
    crit = confidence_config()["critic"]
    backend = Critic()
    (first,) = execute([spec(tmp_path, items)], crit, lambda _: None, backend)
    assert len(backend.calls) == 1  # the declined answer is not reviewed
    assert [r["confidence"] for r in first] == [0.8, 0.0]
    assert first[0]["verdict"] == "correct" and first[0]["cost_usd"] > 0
    trace = json.loads((tmp_path / "traces" / f"{qs[0].question_id}.json").read_text("utf-8"))
    assert trace["evidence"]["preview"] and "confidence" not in trace["answer"]
    spans = [json.loads(line) for line in (tmp_path / "spans.jsonl").read_text().splitlines()]
    assert sorted(s["name"] for s in spans) == ["critic.run", "critic.run", "llm.call"]
    monkeypatch.setenv("ANALYST_REPLAY_ONLY", "1")
    (again,) = execute([spec(tmp_path, items)], crit, lambda _: None, NoCalls())
    assert stable(again) == stable(first)


@pytest.mark.bird
def test_a_result_that_reads_the_clock_replays_exactly(tmp_path, bird_ready, monkeypatch):
    """`now()` differs on every run: without its first result stored, the replayed review would
    show a different result, and its request would miss the cache."""
    q = next(q for q, _ in benchmark_set("pilot") if q.db_id == "financial")
    items = [(q, answer_record(sql="SELECT now() AS t, COUNT(*) AS n FROM loan"))]
    crit = confidence_config()["critic"]
    execute([spec(tmp_path, items)], crit, lambda _: None, Critic())
    assert len(list((tmp_path / "cache" / "clock").glob("*/*.json"))) == 1
    monkeypatch.setenv("ANALYST_REPLAY_ONLY", "1")
    execute([spec(tmp_path, items)], crit, lambda _: None, NoCalls())  # no CacheMiss
    (tmp_path / "cache" / "clock").rename(tmp_path / "moved")
    with pytest.raises(CacheMiss):  # the control: without the stored result it misses
        execute([spec(tmp_path, items)], crit, lambda _: None, NoCalls())


@pytest.mark.bird
def test_items_pair_questions_with_answers_never_gold(bird_ready):
    answers = [{"question_id": q.question_id, "x": 1} for q, _ in benchmark_set("pilot")]
    items = items_for("pilot", answers, limit=3)
    assert len(items) == 3 and all(a["question_id"] == q.question_id for q, a in items)
    assert all(not hasattr(q, "queries") for q, _ in items)
