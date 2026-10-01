"""Prometheus metrics for one service, in its own registry so a second app in one process (the
tests) does not collide with the first.

What is measured, for the dashboards and the alert rules in monitoring/:

- requests and their latency, by route;
- questions by mode and outcome (answered, withheld, declined, clarify, error, ...): the decline
  rate is the share that is not `answered`;
- each run's duration, and each step's by kind (a model call, a tool call, the submission);
- what became of the final query: `ok`, `refused` (the guard or the database blocked it), or
  another failure: the tool error rate and the blocked-query count;
- the calibrated confidence of every answer, in ten bins, and the drift of that distribution from
  the evaluation's (src/serving/drift.py);
- live model spend;
- which agent configuration is running.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

CONFIDENCE_BUCKETS = tuple(round(i / 10, 1) for i in range(1, 11))


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self._requests = Counter(
            "analyst_http_requests_total",
            "HTTP requests",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self._latency = Histogram(
            "analyst_http_request_seconds",
            "HTTP request duration",
            ["path"],
            registry=self.registry,
        )
        self._asks = Counter(
            "analyst_questions_total",
            "Questions by mode and outcome",
            ["mode", "outcome"],
            registry=self.registry,
        )
        self._run = Histogram(
            "analyst_run_seconds",
            "A run's stream, from start to done",
            ["mode"],
            buckets=(0.5, 1, 2, 4, 8, 16, 32, 64),
            registry=self.registry,
        )
        self._step = Histogram(
            "analyst_step_seconds",
            "One step of a run, by kind (model, tool, submit)",
            ["kind"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
            registry=self.registry,
        )
        self._sql = Counter(
            "analyst_sql_results_total",
            "What became of a run's final query: ok, refused (blocked), or the failure's kind",
            ["outcome"],
            registry=self.registry,
        )
        self._confidence = Histogram(
            "analyst_calibrated_confidence",
            "Calibrated confidence of each answer (a declined answer counts at 0)",
            buckets=CONFIDENCE_BUCKETS,
            registry=self.registry,
        )
        self._spend = Counter(
            "analyst_live_spend_usd_total", "Live model spend, USD", registry=self.registry
        )
        self._agent = Gauge(
            "analyst_agent_info",
            "The agent configuration the service runs (value 1)",
            ["name", "config_sha256", "router"],
            registry=self.registry,
        )
        self._psi = Gauge(
            "analyst_confidence_psi",
            "Population stability index of the recent calibrated confidences against the "
            "held-out reference",
            registry=self.registry,
        )
        self._below_psi = Gauge(
            "analyst_withheld_share_psi",
            "Population stability index of the share of answers withheld or declined against "
            "the held-out share",
            registry=self.registry,
        )
        self._below_share = Gauge(
            "analyst_withheld_share",
            "Share of recent answers withheld or declined",
            registry=self.registry,
        )
        self._window = Gauge(
            "analyst_drift_window_answers",
            "Answers in the drift window",
            registry=self.registry,
        )
        self.content_type = CONTENT_TYPE_LATEST

    def request(self, method: str, path: str, status: int, seconds: float) -> None:
        self._requests.labels(method, path, str(status)).inc()
        self._latency.labels(path).observe(seconds)

    def ask(self, mode: str, outcome: str) -> None:
        self._asks.labels(mode, outcome).inc()

    def run_seconds(self, mode: str, seconds: float) -> None:
        self._run.labels(mode).observe(seconds)

    def add_spend(self, usd: float) -> None:
        self._spend.inc(max(usd, 0.0))

    def agent(self, name: str, config_sha256: str, router: bool) -> None:
        self._agent.labels(name, config_sha256[:12], str(router).lower()).set(1)

    def observe(self, event: dict[str, Any]) -> None:
        """What one stream event says: a step's duration, a final query's outcome, an answer's
        calibrated confidence."""
        kind = event.get("type")
        if kind == "step" and event.get("recorded_ms"):
            self._step.labels(event.get("kind", "tool")).observe(event["recorded_ms"] / 1000)
        elif kind == "rows":
            if event.get("ok"):
                self._sql.labels("ok").inc()
            else:
                self._sql.labels(str((event.get("error") or {}).get("kind", "error"))).inc()
        elif kind == "confidence" and not event.get("not_calibrated"):
            calibrated = event.get("calibrated")
            self._confidence.observe(0.0 if calibrated is None else float(calibrated))

    def drift(self, snapshot: dict[str, Any]) -> None:
        self._window.set(snapshot["window"])
        if "below_share" in snapshot:
            self._below_share.set(snapshot["below_share"])
        if snapshot["psi"] is not None:
            self._psi.set(snapshot["psi"])
            self._below_psi.set(snapshot["below_psi"])

    def render(self) -> bytes:
        return generate_latest(self.registry)
