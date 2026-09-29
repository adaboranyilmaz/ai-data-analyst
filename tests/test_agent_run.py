"""Whole runs on the loaded benchmark, with a scripted model in place of an API: the designs'
requests, design 5's narrowing, the span tree, scoring, batched rounds, and replay at $0. Also
that no gold query ever reaches a request."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.agent.answer import SELECT_SCHEMA, SUBMIT
from src.agent.driver import Caller, LayeredCache, drive_batch
from src.agent.evaluate import RunSpec, banking_set, benchmark_set, execute
from src.agent.run import QuestionRun, config
from src.llm.cache import CacheMiss, ResponseCache
from src.llm.types import LLMResponse
from src.tools.toolbox import Toolbox
from src.tracking.otel import MemoryExporter, make_tracer

pytestmark = pytest.mark.bird

SONNET = "claude-sonnet-5"
LOCAL = "qwen2.5:3b-instruct"
GOOD = "SELECT COUNT(*) FROM loan"


def response(*blocks, stop="tool_use") -> LLMResponse:
    return LLMResponse(
        text="",
        content=list(blocks),
        model_reported="m",
        stop_reason=stop,
        usage={"input_tokens": 1000, "output_tokens": 50, "cache_read_input_tokens": 500},
        latency_ms=10.0,
        created_utc="2026-09-28T00:00:00+00:00",
    )


def tool_use(name, args, id_):
    return {"type": "tool_use", "id": id_, "name": name, "input": args}


def submit_args(sql, confidence=0.7):
    return {
        "sql": sql,
        "answer": "An answer.",
        "confidence": confidence,
        "declined": False,
        "decline_reason": None,
        "clarifying_question": None,
        "assumptions": [],
        "premise_correction": None,
        "chart_spec": None,
    }


class Scripted:
    """A model that describes a table, runs its query, then submits it; or narrows to `loan`."""

    name = "fake"

    def __init__(self, sql=GOOD, table="loan"):
        self.sql, self.table, self.calls = sql, table, []

    def check(self, request):
        pass

    def generate(self, request):
        self.calls.append(request)
        names = {t["name"] for t in request.tools}
        turn = sum(m["role"] == "assistant" for m in request.messages)
        if SELECT_SCHEMA in names:
            args = {"tables": [{"table": self.table, "columns": ["loan_id", "amount"]}]}
            return response(tool_use(SELECT_SCHEMA, args, "n1"))
        if "describe_table" in names and turn == 0:
            return response(tool_use("describe_table", {"table": self.table}, f"d{turn}"))
        if "run_sql" in names and turn == 1:
            return response(tool_use("run_sql", {"sql": self.sql}, f"r{turn}"))
        return response(tool_use(SUBMIT, submit_args(self.sql), f"s{turn}"))


class NoCalls:
    name = "fake"

    def check(self, request):
        pass

    def generate(self, request):
        raise AssertionError("replay must not call the model")


def cost(model, usage, batch=False):
    return (
        (usage.input * 2 + usage.cache_read * 0.2 + usage.output * 10) / 1e6 * (0.5 if batch else 1)
    )


def one_question(db="financial"):
    return next(q for q, _ in benchmark_set("pilot") if q.db_id == db)


def drive(run, backend):
    while not run.done:
        for conv, req in run.pending():
            run.feed(conv, req.cache_key, backend.generate(req))
    return run.finish()


@pytest.fixture
def financial_box(bird_ready):
    with Toolbox("financial") as box:
        yield box


class TestDesigns:
    def test_design1_sees_the_full_schema_and_must_submit(self, financial_box):
        backend = Scripted()
        run = QuestionRun(one_question(), "d1", SONNET, True, financial_box, cost)
        fin = drive(run, backend)
        (req,) = backend.calls
        assert [t["name"] for t in req.tools] == [SUBMIT]
        assert req.params["tool_choice"] == {"type": "tool", "name": SUBMIT}
        context = req.messages[0]["content"][0]["text"]
        assert '"table":"trans"' in context and '"columns"' in context  # every table described
        assert fin.final_sql == GOOD and fin.confidence == 0.7 and fin.result.ok
        assert fin.cost_usd == pytest.approx(cost(SONNET, fin.usage))

    def test_design3_first_request_is_design4_sample0(self, financial_box):
        q = one_question()
        d3 = QuestionRun(q, "d3", SONNET, True, financial_box, cost)
        d4 = QuestionRun(q, "d4", SONNET, True, financial_box, cost)
        ((_, r3),) = d3.pending()
        r4 = [r for _, r in d4.pending()]
        assert len(r4) == 3 and r4[0].cache_key == r3.cache_key
        assert [r.params.get("_sample") for r in r4] == [None, 1, 2]

    def test_design3_runs_tools_and_scores_its_checks(self, financial_box):
        run = QuestionRun(one_question(), "d3", SONNET, True, financial_box, cost)
        fin = drive(run, Scripted())
        assert fin.tool_calls == 2 and fin.steps == 3
        assert fin.checks["applicable"] and not fin.checks["failed"]
        steps = fin.trace["final"]["steps"]
        assert steps[0].startswith("Looked up the table loan") and steps[-1].startswith("Submitted")

    def test_design4_votes(self, financial_box):
        run = QuestionRun(one_question(), "d4", SONNET, True, financial_box, cost)
        fin = drive(run, Scripted())
        assert fin.confidence == 1.0 and fin.trace["vote"]["group"] == [0, 1, 2]

    def test_design5_narrows_what_the_samples_see_and_query(self, financial_box):
        backend = Scripted(sql="SELECT COUNT(*) FROM account")
        run = QuestionRun(one_question(), "d5", SONNET, True, financial_box, cost)
        fin = drive(run, backend)
        assert run.selection == {"loan": ["loan_id", "amount"]}
        sample_first = next(
            r for r in backend.calls if SELECT_SCHEMA not in {t["name"] for t in r.tools}
        )
        assert '"table":"account"' not in sample_first.messages[0]["content"][0]["text"]
        described = fin.trace["samples"][0]["tool_results"][0]["result"]
        assert [c["name"] for c in described["columns"]] == ["loan_id", "amount"]
        ran = fin.trace["samples"][0]["tool_results"][1]["result"]
        assert ran["ok"] is False and ran["error"]["kind"] == "refused"  # account was not chosen
        assert not fin.result.ok  # and the final query is checked through the same guard


class TestSpans:
    def test_span_tree(self, financial_box):
        exporter = MemoryExporter()
        tracer, provider = make_tracer(exporter)
        run = QuestionRun(one_question(), "d3", SONNET, True, financial_box, cost, tracer)
        drive(run, Scripted())
        provider.shutdown()
        spans = {s.name: [] for s in exporter.spans}
        for s in exporter.spans:
            spans[s.name].append(s)
        (root,) = spans["agent.run"]
        assert root.parent is None and len(spans["llm.call"]) == 3 and len(spans["tool.call"]) == 2
        rid = root.get_span_context().span_id
        assert all(s.parent.span_id == rid for s in spans["llm.call"] + spans["tool.call"])
        sql_span = next(s for s in spans["tool.call"] if s.attributes["tool"] == "run_sql")
        assert sql_span.attributes["sql"] == GOOD and sql_span.attributes["row_count"] == 1
        assert root.attributes["design"] == "d3" and root.attributes["steps"] == 3


def spec(tmp_path, items, design="d3", model=SONNET, **kw):
    return RunSpec(
        name=f"test/{design}",
        items=items,
        design=design,
        model=model,
        evidence=True,
        phase="phase4",
        cache_dir=tmp_path / "cache",
        records=tmp_path / "records.jsonl",
        traces_dir=tmp_path / "traces",
        spans=tmp_path / "spans.jsonl",
        ledger_path=tmp_path / "spend.json",
        batch_dir=tmp_path / "batches",
        poll_seconds=0,
        **kw,
    )


def volatile_free(records):
    return [{k: v for k, v in r.items() if k not in ("latency_s", "evaluated_at")} for r in records]


@pytest.mark.usefixtures("bird_ready")  # the questions come from the benchmark files
class TestExecute:
    def test_scores_records_and_replays_at_zero_cost(self, tmp_path, monkeypatch):
        items = [x for x in benchmark_set("pilot") if x[0].db_id == "financial"]
        first = execute(spec(tmp_path, items), log=lambda _: None, backend=Scripted())
        assert first[0]["score_outcome"] == "ok" and first[0]["cost_usd"] > 0
        assert first[0]["trace"] and (tmp_path / "traces").exists()
        assert (tmp_path / "spend.json").exists()  # the test's ledger, not the project's
        monkeypatch.setenv("ANALYST_REPLAY_ONLY", "1")
        again = execute(spec(tmp_path, items), log=lambda _: None, backend=NoCalls())
        assert volatile_free(again) == volatile_free(first)

    def test_replay_misses_raise(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANALYST_REPLAY_ONLY", "1")
        items = [x for x in benchmark_set("pilot") if x[0].db_id == "financial"]
        with pytest.raises(CacheMiss):
            execute(spec(tmp_path, items), log=lambda _: None, backend=Scripted())

    def test_banking_set_scoring(self, tmp_path):
        items = banking_set()
        by_cat = {}
        for q, g in items:
            by_cat.setdefault(q.category, (q, g))
        picked = [by_cat[c] for c in "acde"]
        records = execute(
            spec(tmp_path, picked, design="d1"), log=lambda _: None, backend=Scripted()
        )
        outcome = {r["category"]: r["score_outcome"] for r in records}
        assert outcome["d"] == outcome["e"] == "not_scored" and outcome["a"] == "ok"
        assert all(r["evidence"] is False for r in records)


class FakeBatches:
    """Message Batches that answer from a scripted model at once."""

    def __init__(self, model):
        self.model, self.batches = model, {}

    def create(self, requests):
        bid = f"b{len(self.batches)}"
        self.batches[bid] = requests
        return SimpleNamespace(id=bid)

    def retrieve(self, bid):
        return SimpleNamespace(processing_status="ended", request_counts=None)

    def results(self, bid):
        from src.llm.types import LLMRequest

        for r in self.batches[bid]:
            p = r["params"]
            req = LLMRequest(
                "anthropic",
                p["model"],
                p.get("system", ""),
                p["messages"],
                p["max_tokens"],
                p.get("tools", []),
            )
            resp = self.model.generate(req)
            msg = SimpleNamespace(
                content=[SimpleNamespace(**b) for b in resp.content],
                model="m",
                stop_reason=resp.stop_reason,
                usage=SimpleNamespace(**resp.usage),
            )
            yield SimpleNamespace(
                custom_id=r["custom_id"], result=SimpleNamespace(type="succeeded", message=msg)
            )


def test_batched_rounds_match_direct_answers(tmp_path, bird_ready):
    from src.llm.ledger import load_ledger

    q = one_question()
    backend = SimpleNamespace(
        name="anthropic",
        client=SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches(Scripted()))),
    )
    ledger = load_ledger("phase4", ledger_path=tmp_path / "spend.json")
    caller = Caller(
        backend,
        LayeredCache(ResponseCache(tmp_path / "c")),
        ledger,
        0,
        tmp_path / "b",
        log=lambda _: None,
    )
    with Toolbox("financial") as box:
        ((run, fin),) = drive_batch(
            [lambda: QuestionRun(q, "d3", SONNET, True, box, cost)], caller, log=lambda _: None
        )
    assert fin.final_sql == GOOD and fin.steps == 3
    assert all(r["batch"] for r in fin.trace["requests"])
    assert ledger.state["by_phase"]["phase4"]["n_batch_calls"] == 3


def test_no_gold_query_reaches_any_request(tmp_path, bird_ready):
    """Run every design on questions from each source and look for any gold query in what was
    sent. The scripted model submits a fixed query, never a gold one."""
    items = [x for x in benchmark_set("pilot")][:4] + banking_set()[:6]
    golds = {g for _, gold in items for g in gold.queries}
    backend = Scripted()
    for design in ("d1", "d3", "d5"):
        execute(spec(tmp_path / design, items, design=design), log=lambda _: None, backend=backend)
    sent = "\n".join(r.canonical() for r in backend.calls)
    leaked = [g for g in golds if g.strip() in sent or json.dumps(g.strip())[1:-1] in sent]
    assert backend.calls and leaked == []


def test_a_run_whose_queries_read_the_clock_replays_exactly(tmp_path, bird_ready, monkeypatch):
    """`now()` differs on every run: without its first result stored, the replayed request
    after the tool call would differ from the cached one and miss the cache."""
    items = [x for x in benchmark_set("pilot") if x[0].db_id == "financial"][:1]
    clock_sql = "SELECT now() AS t, COUNT(*) AS n FROM loan"
    (first,) = execute(spec(tmp_path, items), lambda _: None, Scripted(sql=clock_sql))
    stored = list((tmp_path / "cache" / "clock").glob("*/*.json"))
    assert len(stored) == 2  # the run_sql call and the check of the submitted query
    monkeypatch.setenv("ANALYST_REPLAY_ONLY", "1")
    (again,) = execute(spec(tmp_path, items), lambda _: None, NoCalls())
    assert volatile_free([again]) == volatile_free([first])


def test_a_final_query_the_guard_refuses_is_wrong(tmp_path, bird_ready):
    """The gold query with a comment: the database would run it and it would score correct,
    but the guard refuses comments, so the system would never run it."""
    items = [x for x in benchmark_set("pilot") if x[0].db_id == "financial"][:1]
    gold_sql = items[0][1].queries[0]
    (plain,) = execute(
        spec(tmp_path / "plain", items, design="d1"), lambda _: None, Scripted(sql=gold_sql)
    )
    assert plain["correct"] == 1  # the control: the same query without the comment
    commented = Scripted(sql=f"{gold_sql} -- checked")
    (r,) = execute(spec(tmp_path / "refused", items, design="d1"), lambda _: None, commented)
    assert (r["correct"], r["soft_f1"], r["score_outcome"]) == (0, 0.0, "refused")
    assert any(e["kind"] == "final_sql_refused" for e in r["errors"])


def test_config_models_and_designs():
    cfg = config()
    assert cfg["models"][SONNET]["params"] == {"thinking": {"type": "disabled"}}
    assert cfg["designs"]["d4"]["samples"] == cfg["designs"]["d5"]["samples"] == 3
    assert (
        cfg["designs"]["d3"]["max_tool_calls"] == 15
        and cfg["designs"]["d2"]["max_tool_calls"] == 10
    )


class Overflowing(Scripted):
    """A local model whose second prompt does not fit its context (as Ollama reports it)."""

    name = "ollama"

    def generate(self, request):
        if self.calls:
            from src.llm.backends import overflow_response

            self.calls.append(request)
            return overflow_response(request.model, "Ollama refused the prompt: too long")
        return super().generate(request)


def test_context_overflow_ends_only_that_conversation(tmp_path, bird_ready):
    items = [x for x in benchmark_set("pilot") if x[0].db_id == "financial"]
    records = execute(spec(tmp_path, items, model=LOCAL), log=lambda _: None, backend=Overflowing())
    (r,) = records
    assert r["final_sql"] is None and r["errors"][-1]["kind"] == "context_overflow"
    assert "too long" in r["errors"][-1]["message"]
    assert r["steps"] == 2 and r["tool_calls"] == 1  # the refused call is a step, like a failure


def test_batched_runs_go_together_and_share_requests(tmp_path, bird_ready):
    from src.agent.evaluate import execute_many

    items = [x for x in benchmark_set("pilot") if x[0].db_id == "financial"]
    batches = FakeBatches(Scripted())
    backend = SimpleNamespace(
        name="anthropic", client=SimpleNamespace(messages=SimpleNamespace(batches=batches))
    )
    specs = [spec(tmp_path / d, items, design=d, mode="batch") for d in ("d3", "d4")]
    for s in specs:  # one stage: one cache, one ledger
        s.cache_dir, s.ledger_path = tmp_path / "cache", tmp_path / "spend.json"
        s.batch_dir = tmp_path / "batches"
    d3, d4 = execute_many(specs, log=lambda _: None, backend=backend)
    sent = [r["custom_id"] for b in batches.batches.values() for r in b]
    assert len(sent) == len(set(sent))  # nothing sent twice
    # 3 turns each: design 3's conversation (shared with design 4's first sample) + 2 new samples
    assert len(sent) == 3 * 3
    assert d3[0]["final_sql"] == d4[0]["final_sql"] == GOOD


class RejectingBatches(FakeBatches):
    """Batches whose API rejects every request after a conversation's first (as it would a
    prompt longer than the model's context), or fails them for a passing reason."""

    def __init__(self, model, error="invalid_request_error"):
        super().__init__(model)
        self.error = error

    def results(self, bid):
        for r, answered in zip(self.batches[bid], super().results(bid), strict=True):
            if len(r["params"]["messages"]) == 1:
                yield answered
                continue
            error = SimpleNamespace(type=self.error, message="prompt is too long")
            yield SimpleNamespace(
                custom_id=r["custom_id"],
                result=SimpleNamespace(
                    type="errored", error=SimpleNamespace(type="error", error=error)
                ),
            )


def batch_backend(batches):
    return SimpleNamespace(
        name="anthropic", client=SimpleNamespace(messages=SimpleNamespace(batches=batches))
    )


def test_a_rejected_batch_request_ends_only_its_conversation(tmp_path, bird_ready, monkeypatch):
    items = [x for x in benchmark_set("pilot") if x[0].db_id == "financial"]
    batches = RejectingBatches(Scripted())
    (r,) = execute(spec(tmp_path, items, mode="batch"), lambda _: None, batch_backend(batches))
    assert r["final_sql"] is None and r["errors"][-1]["kind"] == "model_error"
    assert "prompt is too long" in r["errors"][-1]["message"]
    assert len(batches.batches) == 2  # the rejected request was not sent again
    # the rejection is stored: a replay meets it without calling the API
    monkeypatch.setenv("ANALYST_REPLAY_ONLY", "1")
    (again,) = execute(spec(tmp_path, items, mode="batch"), lambda _: None, batch_backend(None))
    assert volatile_free([again]) == volatile_free([r])


def test_a_rejection_for_credit_is_not_stored_as_an_answer():
    from src.agent.driver import permanent

    too_long = {"error": "invalid_request_error", "message": "prompt is too long: 250000 tokens"}
    credit = {
        "error": "invalid_request_error",
        "message": "Your credit balance is too low to access the Anthropic API.",
    }
    assert permanent(too_long) and not permanent(credit)
    assert not permanent({"error": "overloaded_error", "message": "Overloaded"})


def test_a_passing_batch_failure_is_retried_then_stops(tmp_path, bird_ready):
    items = [x for x in benchmark_set("pilot") if x[0].db_id == "financial"]
    batches = RejectingBatches(Scripted(), error="overloaded_error")
    with pytest.raises(RuntimeError, match="no request succeeded in 3 rounds"):
        execute(spec(tmp_path, items, mode="batch"), lambda _: None, batch_backend(batches))
    assert len(batches.batches) == 1 + 3  # the first turn, then three tries of the second


class ResponseError(Exception):
    """Named like the Ollama client's error."""


class FailingLocal(Scripted):
    """A local model whose first call fails as Ollama's does."""

    name = "ollama"

    def generate(self, request):
        self.calls.append(request)
        raise ResponseError("prediction aborted, token repeat limit reached (status code: 500)")


class NoLocalCalls(NoCalls):
    name = "ollama"


def test_local_model_failure_is_recorded_and_replayed(tmp_path, bird_ready, monkeypatch):
    items = [x for x in benchmark_set("pilot") if x[0].db_id == "financial"]
    (first,) = execute(
        spec(tmp_path, items, model=LOCAL), log=lambda _: None, backend=FailingLocal()
    )
    assert first["final_sql"] is None and first["errors"][-1]["kind"] == "model_error"
    monkeypatch.setenv("ANALYST_REPLAY_ONLY", "1")
    (again,) = execute(
        spec(tmp_path, items, model=LOCAL), log=lambda _: None, backend=NoLocalCalls()
    )
    assert volatile_free([again]) == volatile_free([first])


def test_api_errors_are_not_swallowed(tmp_path, bird_ready):
    class Failing(Scripted):
        def generate(self, request):
            raise ResponseError("boom")

    items = [x for x in benchmark_set("pilot") if x[0].db_id == "financial"]
    with pytest.raises(ResponseError):
        execute(spec(tmp_path, items), log=lambda _: None, backend=Failing())
