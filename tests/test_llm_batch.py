"""The Message Batches runner (src/llm/batch.py) against a fake client: only uncached requests
are sent, interrupted runs re-attach instead of resubmitting, batches are packed under the
caps, and misconfigured or failed requests are caught and reported."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from src.llm.backends import AnthropicBackend, UnsupportedParams
from src.llm.batch import pending_records, run_batch_cached
from src.llm.cache import CacheMiss, ResponseCache
from src.llm.ledger import BudgetExceeded, ModelPrice, SpendLedger
from src.llm.types import LLMRequest, LLMResponse

PRICE = ModelPrice(input=2.0, output=10.0, cache_write_5m=2.5, cache_write_1h=4.0, cache_read=0.2)
# Output tokens only, so a request's worst case is exactly its max_tokens of output:
# 100 x $10 / 1e6 x 0.5 = $0.0005, and a real (5 in, 50 out) call costs $0.00025.
OUTPUT_ONLY = ModelPrice(
    input=0.0, output=10.0, cache_write_5m=0.0, cache_write_1h=0.0, cache_read=0.0
)
MODEL = "claude-sonnet-5"


def ledger(tmp_path, phase_cap=10.0, price=PRICE) -> SpendLedger:
    return SpendLedger(tmp_path / "spend.json", {MODEL: price}, 60.0, "p", phase_cap)


def req(i: int, **params) -> LLMRequest:
    return LLMRequest.single("anthropic", MODEL, "sys", f"user {i}", 100, params)


class FakeBatches:
    def __init__(self, fail=(), usage=(1000, 100)):
        self.created: dict[str, list[dict]] = {}
        self.fail = set(fail)
        self.usage = usage

    def create(self, requests):
        bid = f"batch_{len(self.created)}"
        self.created[bid] = list(requests)
        return SimpleNamespace(id=bid)

    def retrieve(self, bid):
        return SimpleNamespace(processing_status="ended", request_counts=None)

    def results(self, bid):
        for r in self.created[bid]:
            user = r["params"]["messages"][0]["content"]
            if user in self.fail:
                yield SimpleNamespace(
                    custom_id=r["custom_id"],
                    result=SimpleNamespace(type="errored", error=SimpleNamespace(type="api_error")),
                )
                continue
            msg = SimpleNamespace(
                content=[SimpleNamespace(type="text", text=f"echo {user}")],
                model=r["params"]["model"],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=self.usage[0], output_tokens=self.usage[1]),
            )
            yield SimpleNamespace(
                custom_id=r["custom_id"], result=SimpleNamespace(type="succeeded", message=msg)
            )


def fake_client(**kw):
    return SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches(**kw)))


def run(requests, cache, led, client, batch_dir, **kw):
    return run_batch_cached(
        requests,
        cache,
        led,
        client,
        poll_seconds=0,
        batch_dir=batch_dir,
        log=lambda s: None,
        sleep=lambda s: None,
        **kw,
    )


def sizes(client) -> list[int]:
    return [len(v) for v in client.messages.batches.created.values()]


class TestBatchRunner:
    def test_submits_only_uncached_and_caches_results(self, tmp_path):
        cache, led, client = ResponseCache(tmp_path / "c"), ledger(tmp_path), fake_client()
        cached = req(0)
        cache.put(
            cached,
            LLMResponse("old", [], "m", "end_turn", {"input_tokens": 1}, 1.0, "t"),
        )
        out = run([cached, req(1), req(2), req(1)], cache, led, client, tmp_path / "b")
        assert (out.n_requested, out.n_cached_before, out.n_submitted, out.n_succeeded) == (
            3,
            1,
            2,
            2,
        )
        sent = [
            r["params"]["messages"][0]["content"]
            for rs in client.messages.batches.created.values()
            for r in rs
        ]
        assert sorted(sent) == ["user 1", "user 2"]
        resp = cache.get(req(1).cache_key)
        assert resp.text == "echo user 1" and resp.extra["service"] == "batch"
        # 2 calls x (1000 in x $2 + 100 out x $10) / 1e6, at half price
        assert out.new_cost_usd == pytest.approx(2 * 0.003 * 0.5)
        assert led.state["total_usd"] == pytest.approx(out.new_cost_usd)
        assert led.state["by_phase"]["p"]["n_batch_calls"] == 2
        assert led._reserved == pytest.approx(0.0)
        assert pending_records(tmp_path / "b") == []

    def test_failures_are_reported_and_left_uncached(self, tmp_path):
        cache, led = ResponseCache(tmp_path / "c"), ledger(tmp_path)
        out = run([req(1), req(2)], cache, led, fake_client(fail={"user 2"}), tmp_path / "b")
        assert out.n_succeeded == 1 and len(out.failures) == 1
        assert out.failures[0]["type"] == "errored" and not cache.has(req(2).cache_key)

    def test_reattaches_to_a_pending_batch_instead_of_resubmitting(self, tmp_path):
        cache, led, client = ResponseCache(tmp_path / "c"), ledger(tmp_path), fake_client()
        batches = client.messages.batches
        pending = [req(1), req(2)]
        batches.created["batch_old"] = [
            {"custom_id": r.cache_key, "params": AnthropicBackend.call_params(r)} for r in pending
        ]
        rec = {
            "batch_id": "batch_old",
            "status": "submitted",
            "reserved_usd": 0.01,
            "requests": {r.cache_key: asdict(r) for r in pending},
        }
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "batch_old.json").write_text(json.dumps(rec))
        out = run(pending, cache, led, client, tmp_path / "b")
        assert out.n_submitted == 0 and list(batches.created) == ["batch_old"]
        assert all(cache.has(r.cache_key) for r in pending)
        assert led._reserved == pytest.approx(0.0)

    def test_packs_batches_under_the_cap(self, tmp_path):
        # Cap $0.0019: 3 worst cases ($0.0015) fit in the first batch, a 4th would not. After
        # it settles at $0.00075, the remaining $0.00115 fits the other 2.
        cache, client = ResponseCache(tmp_path / "c"), fake_client(usage=(5, 50))
        led = ledger(tmp_path, phase_cap=0.0019, price=OUTPUT_ONLY)
        out = run([req(i) for i in range(5)], cache, led, client, tmp_path / "b")
        assert out.n_succeeded == 5 and sizes(client) == [3, 2]
        assert led.phase_spent() <= 0.0019

    def test_waits_instead_of_sending_slivers(self, tmp_path):
        # Cap $0.00165, at most 2 per batch. After the first batch of 2 ($0.001 reserved), the
        # headroom fits 1 more: with a batch in flight and 3 still queued, that sliver waits.
        # The last request goes alone only when it is all that is left.
        cache, client = ResponseCache(tmp_path / "c"), fake_client(usage=(5, 50))
        led = ledger(tmp_path, phase_cap=0.00165, price=OUTPUT_ONLY)
        out = run(
            [req(i) for i in range(5)],
            cache,
            led,
            client,
            tmp_path / "b",
            max_requests_per_batch=2,
            min_requests_per_batch=2,
        )
        assert out.n_succeeded == 5 and sizes(client) == [2, 2, 1]

    def test_refuses_when_nothing_fits(self, tmp_path):
        cache, led = ResponseCache(tmp_path / "c"), ledger(tmp_path, phase_cap=0.0)
        with pytest.raises(BudgetExceeded):
            run([req(1)], cache, led, fake_client(), tmp_path / "b")

    def test_misconfigured_request_stops_the_run_before_any_submission(self, tmp_path):
        client = fake_client()
        bad = LLMRequest.single(
            "anthropic", "claude-opus-5-5", "sys", "u", 100, {"thinking": {"type": "disabled"}}
        )
        with pytest.raises(UnsupportedParams):
            run([req(1), bad], ResponseCache(tmp_path / "c"), ledger(tmp_path), client, tmp_path)
        assert client.messages.batches.created == {}

    def test_replay_only_miss_raises_before_submitting(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANALYST_REPLAY_ONLY", "1")
        client = SimpleNamespace()  # any attribute access would raise
        with pytest.raises(CacheMiss, match="1 batch requests"):
            run([req(1)], ResponseCache(tmp_path / "c"), ledger(tmp_path), client, tmp_path / "b")
