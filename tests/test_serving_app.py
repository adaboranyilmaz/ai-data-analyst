"""The service: its endpoints in replay mode (no database, no model), and live mode with a scripted
model on the loaded benchmark."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.llm.types import LLMResponse
from src.serving.app import Settings, create_app
from src.serving.live import LiveRunner
from src.serving.meter import Meter, config

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "results/demo"


def events(response) -> list[dict]:
    """The events of a Server-Sent Events body."""
    out = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        data = json.loads(lines["data"])
        assert lines["event"] == data["type"]
        out.append(data)
    return out


@pytest.fixture
def replay_client(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYST_REPLAY_SPEED", "1000")
    log = tmp_path / "requests.jsonl"
    app = create_app(Settings(mode="replay", request_log=log))
    return TestClient(app), log


class TestReplayMode:
    def test_health_and_ready(self, replay_client):
        client, _ = replay_client
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready")
        assert (
            ready.status_code == 200 and ready.json()["ready"] and ready.json()["mode"] == "replay"
        )

    def test_meta_has_the_meter_and_the_suggestions(self, replay_client):
        client, _ = replay_client
        meta = client.get("/api/meta").json()
        assert meta["mode"] == "replay" and meta["connection"] is None
        assert len(meta["suggested"]) == 20 and meta["meter"]["held_out"]["questions"] == 320

    def test_ask_by_run_id_streams_the_recorded_run(self, replay_client):
        client, _ = replay_client
        r = client.post("/ask", json={"run_id": "bench-782"})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        evs = events(r)
        types = [e["type"] for e in evs]
        assert types[0] == "start" and types[-1] == "done"
        assert evs[0]["mode"] == "replay" and "answer" in types and "confidence" in types
        recorded = client.get("/runs/bench-782").json()
        answer = next(e for e in evs if e["type"] == "answer")
        assert answer["text"] == recorded["answer"]["text"]
        assert answer["sql"] == recorded["answer"]["sql"]

    def test_ask_by_question_text_matches_a_recorded_question(self, replay_client):
        client, _ = replay_client
        question = client.get("/runs/bench-782").json()["question"]
        r = client.post("/ask", json={"question": "  " + question.upper() + " "})
        assert r.status_code == 200 and events(r)[0]["id"] == "bench-782"

    def test_a_question_that_was_not_recorded_is_refused_not_invented(self, replay_client):
        client, _ = replay_client
        r = client.post("/ask", json={"question": "How many loans were granted in 1995?"})
        assert r.status_code == 404 and "recorded" in r.json()["detail"]

    def test_a_bad_run_id_is_a_client_error(self, replay_client):
        client, _ = replay_client
        assert client.post("/ask", json={"run_id": "../index"}).status_code == 400
        assert client.get("/runs/..%2findex").status_code == 404
        assert client.get("/runs/nope").status_code == 404

    def test_every_recorded_run_streams_through_the_api(self, replay_client):
        client, _ = replay_client
        for entry in client.get("/api/meta").json()["suggested"]:
            evs = events(client.post("/ask", json={"run_id": entry["id"]}))
            assert evs[-1]["type"] == "done" and evs[-1]["status"] == entry["status"]

    def test_connections_are_not_available_in_replay_mode(self, replay_client):
        client, _ = replay_client
        body = {"host": "h", "dbname": "d", "user": "u", "password": "p"}
        assert client.post("/connections", json=body).status_code == 403
        assert client.get("/connections").status_code == 403

    def test_the_request_log_has_no_bodies(self, replay_client):
        client, log = replay_client
        client.post("/ask", json={"question": "a secret question about client 12345"})
        client.get("/health")
        text = log.read_text(encoding="utf-8")
        lines = [json.loads(x) for x in text.splitlines()]
        assert {"/health", "/ask"} <= {x["path"] for x in lines}
        assert all({"ts", "method", "path", "status", "ms", "mode"} == set(x) for x in lines)
        assert "secret" not in text and "12345" not in text

    def test_metrics_count_requests_and_questions(self, replay_client):
        client, _ = replay_client
        client.post("/ask", json={"run_id": "bench-782"})
        text = client.get("/metrics").text
        assert 'analyst_questions_total{mode="replay",outcome="answered"} 1.0' in text
        assert 'analyst_http_requests_total{method="POST",path="/ask",status="200"} 1.0' in text

    def test_two_services_in_one_process_do_not_share_counters(self, tmp_path):
        a = TestClient(create_app(Settings(request_log=None)))
        b = TestClient(create_app(Settings(request_log=None)))
        a.get("/health")
        assert 'path="/health"' not in b.get("/metrics").text


# --------------------------------------------------------------------------- live mode

SQL = "SELECT COUNT(*) AS n FROM loan"
SUBMIT = {
    "sql": SQL,
    "answer": "There are 682 loans.",
    "confidence": 0.95,
    "declined": False,
    "decline_reason": None,
    "clarifying_question": None,
    "assumptions": [],
    "premise_correction": None,
    "chart_spec": None,
}


class Scripted:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def check(self, request):
        pass

    def generate(self, request):
        self.calls += 1
        return LLMResponse(
            text="",
            content=[{"type": "tool_use", "id": "t1", "name": "submit_answer", "input": SUBMIT}],
            model_reported="m",
            stop_reason="tool_use",
            usage={"input_tokens": 3000, "output_tokens": 200},
            latency_ms=12.0,
            created_utc="2026-10-01T00:00:00+00:00",
        )


class Refusing(Scripted):
    def generate(self, request):
        raise AssertionError("no model call is allowed")


def live_settings(tmp_path, backend, cap=1.0):
    cfg = copy.deepcopy(config())
    cfg["live"]["cache_dir"] = str(tmp_path / "cache")
    cfg["live"]["ledger"] = str(tmp_path / "ledger.json")
    cfg["live"]["runs_dir"] = str(tmp_path / "runs")
    cfg["live"]["spend_cap_usd"] = cap
    meter = Meter.load(cfg)
    runner = LiveRunner(cfg, meter, backend=backend)
    return Settings(
        mode="live", cfg=cfg, runner=runner, meter=meter, request_log=tmp_path / "log.jsonl"
    )


@pytest.mark.bird
class TestLiveMode:
    def test_a_live_question_streams_saves_and_is_charged(self, tmp_path, bird_ready):
        backend = Scripted()
        client = TestClient(create_app(live_settings(tmp_path, backend)))
        r = client.post("/ask", json={"question": "How many loans are there?"})
        assert r.status_code == 200
        evs = events(r)
        types = [e["type"] for e in evs]
        assert types[0] == "start" and evs[0]["mode"] == "live" and types[-1] == "done"
        assert (
            types.index("step") < types.index("sql") < types.index("rows") < types.index("answer")
        )
        rows = next(e for e in evs if e["type"] == "rows")
        assert rows["rows"] == [[682]]
        confidence = next(e for e in evs if e["type"] == "confidence")
        assert confidence["stated"] == 0.95 and confidence["calibrated"] is not None
        done = evs[-1]
        assert done["evaluation"] is None and done["cost_usd"] > 0
        saved = client.get(f"/runs/{done['id']}").json()
        assert saved["answer"]["text"] == "There are 682 loans." and saved["kind"] == "live"
        ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
        assert ledger["total_usd"] == pytest.approx(done["cost_usd"])

    def test_the_same_question_twice_costs_once(self, tmp_path, bird_ready):
        backend = Scripted()
        client = TestClient(create_app(live_settings(tmp_path, backend)))
        client.post("/ask", json={"question": "How many loans are there?"})
        client.post("/ask", json={"question": "How many loans are there?"})
        assert backend.calls == 1

    def test_an_empty_or_long_question_is_refused_before_any_call(self, tmp_path, bird_ready):
        backend = Refusing()
        client = TestClient(create_app(live_settings(tmp_path, backend)))
        assert client.post("/ask", json={"question": "   "}).status_code == 400
        assert client.post("/ask", json={"question": "x" * 601}).status_code == 400

    def test_a_reached_spend_cap_refuses_new_questions_but_not_recorded_runs(
        self, tmp_path, bird_ready
    ):
        client = TestClient(create_app(live_settings(tmp_path, Refusing(), cap=0.05)))
        r = client.post("/ask", json={"question": "How many loans are there?"})
        assert r.status_code == 429 and "spend cap" in r.json()["detail"]
        assert client.post("/ask", json={"run_id": "bench-782"}).status_code == 200

    def test_a_model_failure_ends_the_stream_with_an_error_event(self, tmp_path, bird_ready):
        client = TestClient(create_app(live_settings(tmp_path, Refusing())))
        evs = events(client.post("/ask", json={"question": "How many loans are there?"}))
        assert evs[-1]["type"] == "error" and "no model call" in evs[-1]["message"]

    def test_the_deployment_ledger_is_not_the_projects(self, tmp_path, bird_ready):
        client = TestClient(create_app(live_settings(tmp_path, Scripted())))
        before = (ROOT / "results/metrics/api_spend.json").read_bytes()
        client.post("/ask", json={"question": "How many loans are there?"})
        assert (ROOT / "results/metrics/api_spend.json").read_bytes() == before


# --------------------------------------------------------------------------- the live guardrail

PLAN_SQL = (
    "SELECT a.account_id, CASE WHEN l.loan_id IS NOT NULL THEN 1 ELSE 0 END AS has_loan, "
    "CASE WHEN c.card_id IS NOT NULL THEN 'true' ELSE 'false' END AS has_card "
    "FROM account a LEFT JOIN loan l ON l.account_id = a.account_id "
    "LEFT JOIN disp dp ON dp.account_id = a.account_id AND dp.type = 'OWNER' "
    "LEFT JOIN card c ON c.disp_id = dp.disp_id"
)
INPUTS = {
    "submit_answer": SUBMIT,
    "classify_question": {"statistical": True, "kind": "causal", "reason": "asks for a cause"},
    "submit_analysis": {
        "unit": "an account",
        "sql": PLAN_SQL,
        "analysis": "compare_groups",
        "outcome": "has_loan",
        "outcome_type": "binary",
        "group": "has_card",
        "reference_group": "false",
        "x": None,
        "x_describes": "unit",
        "strata": [],
        "strata_reason": "",
        "assumptions": [],
        "declined": False,
        "decline_reason": None,
    },
    "submit_finding": {
        "answer": "Accounts with a card more often have a loan, an association only.",
        "claims_effect": "yes",
        "higher": "true",
    },
}


class ScriptedByTool(Scripted):
    """Answers each forced tool call with its scripted input; counts calls by tool."""

    def __init__(self, statistical=True):
        super().__init__()
        self.by_tool = {}
        self.statistical = statistical

    def generate(self, request):
        name = request.tools[0]["name"]
        self.by_tool[name] = self.by_tool.get(name, 0) + 1
        data = INPUTS[name]
        if name == "classify_question" and not self.statistical:
            data = {"statistical": False, "kind": "descriptive", "reason": "a count"}
        return LLMResponse(
            text="",
            content=[{"type": "tool_use", "id": "t1", "name": name, "input": data}],
            model_reported="m",
            stop_reason="tool_use",
            usage={"input_tokens": 3000, "output_tokens": 200},
            latency_ms=12.0,
            created_utc="2026-10-01T00:00:00+00:00",
        )


def guarded_client(tmp_path, backend, sandbox_ready):
    from src.serving.live_guardrail import LiveGuardrail

    cfg = copy.deepcopy(config())
    cfg["live"]["cache_dir"] = str(tmp_path / "cache")
    cfg["live"]["ledger"] = str(tmp_path / "ledger.json")
    cfg["live"]["runs_dir"] = str(tmp_path / "runs")
    cfg["live"]["guardrail_cache_dir"] = str(tmp_path / "gcache")
    meter = Meter.load(cfg)
    from src.serving.live import deployment_ledger

    ledger = deployment_ledger(cfg)
    guardrail = LiveGuardrail(cfg["live"], ledger, backend, sandbox_ready=sandbox_ready)
    runner = LiveRunner(cfg, meter, backend=backend, ledger=ledger, guardrail=guardrail)
    return TestClient(
        create_app(Settings(mode="live", cfg=cfg, runner=runner, meter=meter, request_log=None))
    )


QUESTION = "Did giving clients a card make them more likely to take out a loan?"


@pytest.mark.bird
class TestLiveGuardrail:
    def test_a_descriptive_question_costs_one_classification_and_gets_no_statistics(
        self, tmp_path, bird_ready
    ):
        backend = ScriptedByTool(statistical=False)
        client = guarded_client(tmp_path, backend, sandbox_ready=True)
        evs = events(client.post("/ask", json={"question": "How many loans are there?"}))
        assert not any(e["type"] == "statistics" for e in evs)
        assert backend.by_tool == {"submit_answer": 1, "classify_question": 1}
        assert next(e for e in evs if e["type"] == "answer")["text"] == "There are 682 loans."

    def test_without_the_sandbox_a_flagged_question_is_answered_with_a_notice(
        self, tmp_path, bird_ready
    ):
        backend = ScriptedByTool()
        client = guarded_client(tmp_path, backend, sandbox_ready=False)
        evs = events(client.post("/ask", json={"question": QUESTION}))
        answer = next(e for e in evs if e["type"] == "answer")
        assert "SQL answer alone" in answer["notice"] and "interval" in answer["notice"]
        assert "submit_analysis" not in backend.by_tool and evs[-1]["type"] == "done"

    @pytest.mark.sandbox
    def test_a_comparative_question_is_analyzed_and_answered_from_the_result(
        self, tmp_path, bird_ready, sandbox_ready
    ):
        backend = ScriptedByTool()
        client = guarded_client(tmp_path, backend, sandbox_ready=True)
        evs = events(client.post("/ask", json={"question": QUESTION}))
        types = [e["type"] for e in evs]
        assert "statistics" in types and types[-1] == "done"
        stats = next(e for e in evs if e["type"] == "statistics")["statistics"]
        assert stats["n"] == 4500 and stats["groups"] and stats["comparisons"][0]["ci"]
        answer = next(e for e in evs if e["type"] == "answer")
        assert "95% intervals" in answer["text"] and "association" in answer["text"]
        confidence = next(e for e in evs if e["type"] == "confidence")
        assert confidence["calibrated"] is None and confidence["not_calibrated"] is True
        saved = client.get(f"/runs/{evs[-1]['id']}").json()
        assert saved["before"]["text"] == "There are 682 loans." and saved["kind"] == "live"
        assert backend.by_tool["submit_analysis"] == 1


class TestReadiness:
    def test_live_readiness_reports_a_database_that_cannot_be_reached(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANALYST_DB_PORT", "5999")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        client = TestClient(create_app(live_settings(tmp_path, Scripted())))
        r = client.get("/ready")
        assert r.status_code == 503 and "not reachable" in " ".join(r.json()["problems"])

    @pytest.mark.bird
    def test_live_readiness_passes_with_the_database_up(self, tmp_path, bird_ready):
        client = TestClient(create_app(live_settings(tmp_path, Scripted())))
        assert client.get("/ready").json() == {"ready": True, "mode": "live", "problems": []}


class TestHousekeeping:
    def test_old_live_runs_are_pruned_and_the_newest_kept(self, tmp_path):
        import os

        from src.serving.store import RunStore

        store = RunStore(DEMO, tmp_path)
        ids = []
        for i in range(5):
            ev = {"id": store.new_live_id(), "question": str(i)}
            store.save_live(ev)
            os.utime(tmp_path / f"{ev['id']}.json", (1000 + i, 1000 + i))
            ids.append(ev["id"])
        assert store.prune_live(2) == 3
        assert [store.get(i) is not None for i in ids] == [False, False, False, True, True]

    def test_the_request_log_rolls_over_beyond_its_size(self, tmp_path, monkeypatch):
        cfg = copy.deepcopy(config())
        cfg["api"]["request_log_max_mb"] = 0.001
        log = tmp_path / "requests.jsonl"
        client = TestClient(create_app(Settings(cfg=cfg, request_log=log)))
        for _ in range(40):
            client.get("/health")
        assert log.exists() and log.with_suffix(".jsonl.1").exists()
        assert log.stat().st_size < 4000
