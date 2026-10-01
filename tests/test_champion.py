"""The service runs the registry's champion: its meter, its router and its settings."""

from __future__ import annotations

import copy
import json

import pytest
from fastapi.testclient import TestClient

from src.llm.types import LLMResponse
from src.serving import champion as champion_mod
from src.serving.app import Settings, create_app
from src.serving.live import LiveRunner, deployment_ledger
from src.serving.meter import ROOT, Meter, config
from src.tracking import registry

BASELINE = "d1-sonnet-5"
ROUTER = "d1-sonnet-5-opus-router"


def champion_of(name: str) -> champion_mod.Champion:
    state = registry.read_state()
    v = state["versions"][name]
    evaluation = json.loads((ROOT / v["evaluation"]).read_text(encoding="utf-8"))
    return champion_mod.Champion(v["config"], v["config_sha256"], evaluation)


def test_the_baseline_meter_from_the_registry_is_the_phase_5_meter():
    cfg = config()
    from_registry = champion_of(BASELINE).meter(cfg)
    original = Meter.load(cfg)
    assert from_registry.summary() == original.summary()
    for stated in (0.0, 0.5, 0.8, 0.9, 0.95, 1.0):
        assert from_registry.describe(stated, False) == original.describe(stated, False)


def test_a_router_meter_calibrates_each_model_by_its_own_curve():
    c = champion_of(ROUTER)
    meter = c.meter(config())
    larger = c.config["router"]["calibrator"]
    primary = c.config["calibration"]["calibrator"]
    assert meter.calibrate(0.9) != meter.calibrate(0.9, larger=True)
    assert meter.calibrate(0.9, larger=True) == pytest.approx(
        1 / (1 + 2.718281828459045 ** -(larger["slope"] * 0.9 + larger["intercept"]))
    )
    assert meter.calibrate(0.9) == pytest.approx(
        1 / (1 + 2.718281828459045 ** -(primary["slope"] * 0.9 + primary["intercept"]))
    )
    # the router's own threshold is the system's, not the primary model's
    assert meter.threshold == c.config["decline_threshold"] != primary and meter.larger
    with pytest.raises(ValueError):
        champion_of(BASELINE).meter(config()).calibrate(0.9, larger=True)


def test_the_champion_sets_what_the_live_section_runs():
    cfg = config()
    base = champion_of(BASELINE).apply(cfg)
    assert base["live"]["design"] == "d1" and base["live"]["model"] == "claude-sonnet-5"
    assert base["live"]["router"] is False and base["live"]["evidence"] is False
    assert champion_of(ROUTER).apply(cfg)["live"]["router"] is True
    assert cfg["live"]["router"] is False  # the settings handed in are not changed


def test_load_reads_the_committed_champion():
    c = champion_mod.load()
    state = registry.read_state()
    assert c.name == registry.champion_name(state)
    assert c.config_sha256 == registry.sha256_of(c.config)
    assert c.info()["name"] == c.name


def test_the_service_reports_its_agent(tmp_path):
    app = create_app(Settings(request_log=None))
    meta = TestClient(app).get("/api/meta").json()
    assert meta["agent"]["name"] == champion_mod.load().name
    assert meta["agent"]["config_sha256"] == champion_mod.load().config_sha256


def test_the_serving_larger_model_run_asks_exactly_what_the_escalation_arm_asked():
    """The service's copy of the escalation arm's run class builds the same requests."""
    from src.agent import escalation
    from src.agent.run import Question
    from src.serving import larger
    from src.tools.toolbox import Toolbox

    router = champion_of(ROUTER).config["router"]
    q = Question(1, "bird", "financial", "How many loans are there?", None)
    cfg = escalation.agent_config_with(router["model"], router["request_settings"])
    assert cfg == larger.agent_config_with(router["model"], router["request_settings"])
    keys = []
    for cls in (escalation.AutoToolRun, larger.AutoToolRun):
        with Toolbox("financial") as box:
            run = cls(q, "d1", router["model"], True, box, lambda *a: 0.0, None, cfg, None)
            keys.append([r.cache_key for _, r in run.pending()])
    assert keys[0] == keys[1] and len(keys[0]) == 1


# --------------------------------------------------------------------------- live routing

SQL = "SELECT COUNT(*) AS n FROM loan"


def submit(confidence: float) -> dict:
    return {
        "sql": SQL,
        "answer": "There are 682 loans.",
        "confidence": confidence,
        "declined": False,
        "decline_reason": None,
        "clarifying_question": None,
        "assumptions": [],
        "premise_correction": None,
        "chart_spec": None,
    }


class ByModel:
    """Answers with a confidence per model and counts the calls of each."""

    name = "fake"

    def __init__(self, confidences: dict[str, float]):
        self.confidences = confidences
        self.calls: list[str] = []

    def check(self, request):
        pass

    def generate(self, request):
        self.calls.append(request.model)
        return LLMResponse(
            text="",
            content=[
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "submit_answer",
                    "input": submit(self.confidences[request.model]),
                }
            ],
            model_reported="m",
            stop_reason="tool_use",
            usage={"input_tokens": 3000, "output_tokens": 200},
            latency_ms=12.0,
            created_utc="2026-10-01T00:00:00+00:00",
        )


def events(response) -> list[dict]:
    out = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append(json.loads(lines["data"]))
    return out


def router_client(tmp_path, backend):
    c = champion_of(ROUTER)
    cfg = c.apply(copy.deepcopy(config()))
    cfg["live"]["cache_dir"] = str(tmp_path / "cache")
    cfg["live"]["ledger"] = str(tmp_path / "ledger.json")
    cfg["live"]["runs_dir"] = str(tmp_path / "runs")
    cfg["live"]["guardrail"] = False
    meter = c.meter(cfg)
    runner = LiveRunner(cfg, meter, backend=backend, ledger=deployment_ledger(cfg), champion=c)
    app = create_app(
        Settings(mode="live", cfg=cfg, runner=runner, meter=meter, champion=c, request_log=None)
    )
    return TestClient(app), meter, c


@pytest.mark.bird
class TestRouter:
    def test_a_low_confidence_answer_goes_to_the_larger_model(self, tmp_path, bird_ready):
        backend = ByModel({"claude-sonnet-5": 0.5, "claude-opus-5-5": 0.9})
        client, meter, c = router_client(tmp_path, backend)
        evs = events(client.post("/ask", json={"question": "How many loans are there?"}))
        assert backend.calls == ["claude-sonnet-5", "claude-opus-5-5"]
        steps = [e["text"] for e in evs if e["type"] == "step"]
        assert any("asking claude-opus-5-5" in t for t in steps)
        confidence = next(e for e in evs if e["type"] == "confidence")
        assert confidence["stated"] == 0.9
        assert confidence["calibrated"] == pytest.approx(
            meter.calibrate(0.9, larger=True), abs=1e-4
        )
        done = evs[-1]
        saved = client.get(f"/runs/{done['id']}").json()
        assert saved["routed"]["to"] == "claude-opus-5-5" and saved["model"] == "claude-opus-5-5"
        assert saved["steps"] == [e for e in evs if e["type"] == "step"]
        # both calls are charged
        ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
        assert ledger["total_usd"] == pytest.approx(done["cost_usd"])
        assert set(ledger["by_model"]) == {"claude-sonnet-5", "claude-opus-5-5"}

    def test_a_confident_answer_is_not_routed(self, tmp_path, bird_ready):
        backend = ByModel({"claude-sonnet-5": 0.97, "claude-opus-5-5": 0.9})
        client, meter, c = router_client(tmp_path, backend)
        assert meter.calibrate(0.97) >= c.config["calibration"]["decline_threshold"]
        evs = events(client.post("/ask", json={"question": "How many loans are there?"}))
        assert backend.calls == ["claude-sonnet-5"]
        done = evs[-1]
        assert "routed" not in client.get(f"/runs/{done['id']}").json()
        confidence = next(e for e in evs if e["type"] == "confidence")
        assert confidence["calibrated"] == pytest.approx(meter.calibrate(0.97), abs=1e-4)
