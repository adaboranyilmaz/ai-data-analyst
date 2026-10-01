"""Drift of the confidence distribution, and the metrics that carry it."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from src.serving.app import Settings, create_app
from src.serving.drift import DriftMonitor, bin_of, psi
from src.serving.meter import Meter


def test_psi_of_identical_distributions_is_zero():
    assert psi([10, 20, 30], [10, 20, 30]) == pytest.approx(0.0, abs=1e-12)
    assert psi([10, 20, 30], [1, 2, 3], smoothing=0) == pytest.approx(0.0, abs=1e-12)  # same shape


def test_psi_matches_the_formula_on_a_known_case():
    # reference 50/50, live 90/10, no smoothing: 0.4 ln(1.8) + 0.4 ln(5)
    expected = 0.4 * math.log(0.9 / 0.5) + (-0.4) * math.log(0.1 / 0.5)
    assert psi([50, 50], [90, 10], smoothing=0) == pytest.approx(expected)
    assert expected == pytest.approx(0.8789, abs=1e-4)


def test_an_empty_bin_does_not_make_psi_infinite():
    assert math.isfinite(psi([50, 50], [100, 0]))
    with pytest.raises(ValueError):
        psi([1, 2], [1, 2, 3])


def test_bins_are_equal_width_with_one_in_the_last():
    assert [bin_of(c, 10) for c in (0.0, 0.05, 0.1, 0.55, 0.999, 1.0)] == [0, 0, 1, 5, 9, 9]
    assert bin_of(-0.2, 10) == 0


def reference_monitor(**kw) -> DriftMonitor:
    return DriftMonitor([10, 10, 20, 30, 30, 50, 60, 70, 60, 20], threshold=0.8, **kw)


def test_a_window_like_the_reference_shows_no_shift():
    m = reference_monitor(window=400, min_window=50)
    for i, n in enumerate(m.reference):  # the reference itself, bin by bin
        for _ in range(int(n)):
            m.observe((i + 0.5) / 10)
    snap = m.snapshot()
    assert snap["ready"] and snap["psi"] < 0.02 and m.reading(snap["psi"]) == "no shift"
    assert snap["below_psi"] < 0.02


def test_a_window_of_low_confidence_answers_is_a_major_shift():
    m = reference_monitor(window=200, min_window=50)
    for _ in range(200):
        m.observe(0.05)
    snap = m.snapshot()
    assert snap["psi"] > 0.25 and m.reading(snap["psi"]) == "major shift"
    assert snap["below_share"] == 1.0 and snap["below_psi"] > 0.25


def test_the_monitor_says_nothing_before_it_has_enough_answers():
    m = reference_monitor(min_window=50)
    for _ in range(49):
        m.observe(0.9)
    snap = m.snapshot()
    assert snap["ready"] is False and snap["psi"] is None
    assert m.reading(snap["psi"]) == "not enough answers yet"
    m.observe(0.9)
    assert m.snapshot()["ready"] is True


def test_the_window_forgets_the_oldest_answers():
    m = reference_monitor(window=100, min_window=50)
    for _ in range(100):
        m.observe(0.05)
    for _ in range(100):
        m.observe(0.95)
    assert max(m.recent) == min(m.recent) == 0.95 and len(m.recent) == 100


def test_a_declined_answer_counts_at_zero():
    m = reference_monitor()
    m.observe(None)
    assert list(m.recent) == [0.0]


def test_the_reference_share_below_the_threshold_splits_the_bin_it_falls_in():
    m = DriftMonitor([10, 10, 10, 10], threshold=0.375)  # bins of 0.25: 1 full + 0.5 of the next
    assert m.reference_below == pytest.approx(15.0)


def test_the_reference_comes_from_the_meters_held_out_table():
    meter = Meter.load()
    m = DriftMonitor.from_meter(meter)
    assert m.bins == 10 and m.reference_total == sum(b.n for b in meter.bands)
    assert m.threshold == meter.threshold


# --------------------------------------------------------------------------- through the API


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ANALYST_REPLAY_SPEED", "1000")
    return TestClient(create_app(Settings(request_log=None)))


def ask_all(client, times):
    ids = [e["id"] for e in client.get("/api/meta").json()["suggested"]]
    for i in range(times):
        client.post("/ask", json={"run_id": ids[i % len(ids)]})


def test_the_metrics_carry_what_the_dashboards_need(client):
    ask_all(client, 20)
    text = client.get("/metrics").text
    for series in (
        "analyst_agent_info{",
        "analyst_calibrated_confidence_bucket",
        'analyst_sql_results_total{outcome="ok"}',
        "analyst_questions_total{",
        "analyst_drift_window_answers 16.0",
    ):
        assert series in text, series


def test_psi_appears_once_the_window_is_ready_and_the_api_reads_it(client):
    ask_all(client, 20)  # 16 of the 20 recorded runs carry a confidence (4 are guarded comparisons)
    early = client.get("/api/drift").json()
    assert early["ready"] is False and early["psi"] is None
    ask_all(client, 60)
    snap = client.get("/api/drift").json()
    assert snap["ready"] and snap["window"] >= 50 and snap["psi"] is not None
    assert snap["psi_reading"] in ("no shift", "moderate shift", "major shift")
    assert snap["warn"] == 0.10 and snap["alert"] == 0.25
    assert "analyst_confidence_psi " in client.get("/metrics").text


def test_blocked_and_failed_queries_are_counted_by_kind(client):
    from src.serving.metrics import Metrics

    m = Metrics()
    m.observe({"type": "rows", "ok": True})
    m.observe({"type": "rows", "ok": False, "error": {"kind": "refused"}})
    m.observe({"type": "rows", "ok": False, "error": {"kind": "timeout"}})
    text = m.render().decode()
    for outcome in ("ok", "refused", "timeout"):
        assert f'analyst_sql_results_total{{outcome="{outcome}"}} 1.0' in text


def test_the_agent_the_service_runs_is_a_series(client):
    from src.serving import champion

    c = champion.load()
    text = client.get("/metrics").text
    assert f'name="{c.name}"' in text and f'config_sha256="{c.config_sha256[:12]}"' in text


def test_a_live_runs_step_durations_are_measured_by_kind():
    from src.serving.metrics import Metrics

    m = Metrics()
    m.observe({"type": "step", "kind": "model", "recorded_ms": 2500})
    m.observe({"type": "step", "kind": "tool", "recorded_ms": 40})
    m.observe({"type": "step", "kind": "model", "recorded_ms": None})  # a batched call: no timing
    text = m.render().decode()
    assert 'analyst_step_seconds_count{kind="model"} 1.0' in text
    assert 'analyst_step_seconds_count{kind="tool"} 1.0' in text
