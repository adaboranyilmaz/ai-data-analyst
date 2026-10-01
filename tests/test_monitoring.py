"""The monitoring files are consistent with what the service exposes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from src.serving.metrics import Metrics

ROOT = Path(__file__).resolve().parent.parent
MON = ROOT / "monitoring"


def alerts() -> list[dict]:
    doc = yaml.safe_load((MON / "alerts.yml").read_text(encoding="utf-8"))
    return [r for g in doc["groups"] for r in g["rules"]]


def exercised_metrics() -> str:
    """The service's metrics after every kind of event."""
    m = Metrics()
    m.request("GET", "/health", 200, 0.01)
    m.ask("live", "answered")
    m.run_seconds("live", 3.0)
    m.add_spend(0.01)
    m.agent("a", "0" * 64, True)
    m.observe({"type": "step", "kind": "model", "recorded_ms": 500})
    m.observe({"type": "rows", "ok": True})
    m.observe({"type": "confidence", "calibrated": 0.7})
    m.drift({"window": 60, "ready": True, "psi": 0.1, "below_psi": 0.1, "below_share": 0.5})
    return m.render().decode()


def metric_names(expression: str) -> set[str]:
    return set(re.findall(r"\banalyst_[a-z_]+", expression))


def base(name: str) -> str:
    """A histogram's series are exposed as <name>_bucket, _sum and _count."""
    return re.sub(r"_(bucket|sum|count)$", "", name)


def test_every_alert_says_what_it_means_and_how_serious_it_is():
    rules = alerts()
    assert len(rules) == 9 and len({r["alert"] for r in rules}) == 9
    for r in rules:
        assert r["labels"]["severity"] in ("info", "warning", "critical"), r["alert"]
        assert r["annotations"]["summary"].strip(), r["alert"]


def test_every_alert_has_a_test_that_fires_it():
    tests = yaml.safe_load((MON / "tests/alerts_test.yml").read_text(encoding="utf-8"))["tests"]
    fired = {t["alertname"] for case in tests for t in case["alert_rule_test"] if t["exp_alerts"]}
    assert fired == {r["alert"] for r in alerts()}


def test_alerts_and_dashboard_use_only_series_the_service_exposes():
    text = exercised_metrics()
    dashboard = json.loads((MON / "grafana/dashboards/analyst.json").read_text(encoding="utf-8"))
    exprs = [r["expr"] for r in alerts()]
    exprs += [t["expr"] for p in dashboard["panels"] for t in p.get("targets", [])]
    missing = {base(n) for e in exprs for n in metric_names(e) if f"{base(n)}" not in text}
    assert missing == set()


def test_the_dashboard_is_well_formed_and_covers_what_was_asked_for():
    d = json.loads((MON / "grafana/dashboards/analyst.json").read_text(encoding="utf-8"))
    ids = [p["id"] for p in d["panels"]]
    assert d["uid"] == "analyst-service" and len(ids) == len(set(ids))
    exprs = " ".join(t["expr"] for p in d["panels"] for t in p.get("targets", []))
    for needed in (
        "analyst_http_requests_total",  # requests
        "analyst_step_seconds",  # latency by step
        "analyst_sql_results_total",  # tool errors and blocked queries
        "analyst_live_spend_usd_total",  # spend per hour
        "analyst_questions_total",  # decline rate
        "analyst_calibrated_confidence",  # confidence distribution
        "analyst_confidence_psi",  # drift
    ):
        assert needed in exprs, needed


def test_prometheus_scrapes_the_api_and_loads_the_rules():
    cfg = yaml.safe_load((MON / "prometheus.yml").read_text(encoding="utf-8"))
    assert cfg["rule_files"] == ["/etc/prometheus/alerts.yml"]
    (job,) = cfg["scrape_configs"]
    assert job["metrics_path"] == "/metrics" and job["static_configs"][0]["targets"] == ["api:8000"]
