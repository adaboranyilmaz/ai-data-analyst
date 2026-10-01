"""Prometheus metrics for one service, in its own registry so a second app in one process (the
tests) does not collide with the first."""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)


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
        self._spend = Counter(
            "analyst_live_spend_usd_total", "Live model spend, USD", registry=self.registry
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

    def render(self) -> bytes:
        return generate_latest(self.registry)
