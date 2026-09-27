"""src/llm: request cache keys, the response cache, token usage, the spend ledger, replay-only
mode, and the Anthropic and Ollama backends against fake clients."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.llm.backends import (
    AnthropicBackend,
    ContextOverflow,
    OllamaBackend,
    UnsupportedParams,
    ollama_messages,
)
from src.llm.cache import CacheMiss, ResponseCache
from src.llm.generate import generate_cached
from src.llm.ledger import (
    TOOL_PROMPT_ALLOWANCE_TOKENS,
    BudgetExceeded,
    ModelPrice,
    SpendLedger,
    estimate_tokens_upper,
    load_ledger,
)
from src.llm.types import LLMRequest, LLMResponse, TokenUsage

ROOT = Path(__file__).resolve().parent.parent
PRICE = ModelPrice(input=2.0, output=10.0, cache_write_5m=2.5, cache_write_1h=4.0, cache_read=0.2)
TOOL = {
    "name": "run_sql",
    "description": "Run one read-only query.",
    "input_schema": {
        "type": "object",
        "properties": {"sql": {"type": "string"}},
        "required": ["sql"],
    },
}
CACHED_SYSTEM = [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}]


def req(**kw) -> LLMRequest:
    base = {
        "backend": "anthropic",
        "model": "m",
        "system": "s",
        "messages": [{"role": "user", "content": "u"}],
        "max_tokens": 100,
        "tools": [],
        "params": {},
    }
    return LLMRequest(**{**base, **kw})


def resp(text: str = "ok", usage: dict | None = None) -> LLMResponse:
    return LLMResponse(
        text=text,
        content=[{"type": "text", "text": text}],
        model_reported="m",
        stop_reason="end_turn",
        usage=usage or {"input_tokens": 10, "output_tokens": 5},
        latency_ms=1.0,
        created_utc="t",
    )


def ledger(tmp_path, project=1.0, phase=0.5) -> SpendLedger:
    return SpendLedger(tmp_path / "spend.json", {"m": PRICE}, project, "p", phase)


class FakeBackend:
    """A backend that records its calls and returns a fixed response (or raises)."""

    def __init__(self, name="anthropic", error: BaseException | None = None):
        self.name, self.error, self.calls = name, error, []

    def check(self, request):
        pass

    def generate(self, request):
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return resp()


# --------------------------------------------------------------------------------------
# Requests and the cache


class TestRequest:
    def test_param_order_does_not_matter(self):
        a = req(params={"options": {"seed": 0, "temperature": 0}})
        b = req(params={"options": {"temperature": 0, "seed": 0}})
        assert a.cache_key == b.cache_key

    @pytest.mark.parametrize(
        "change",
        [
            {"model": "m2"},
            {"system": "s2"},
            {"system": CACHED_SYSTEM},
            {"messages": [{"role": "user", "content": "u2"}]},
            {"max_tokens": 101},
            {"tools": [TOOL]},
            {"params": {"_sample": 1}},
        ],
    )
    def test_any_change_is_a_miss(self, change):
        assert req().cache_key != req(**change).cache_key

    def test_single_is_one_user_turn(self):
        r = LLMRequest.single("anthropic", "m", "s", "u", 100)
        assert r == req()

    def test_non_json_request_is_refused(self):
        with pytest.raises(TypeError, match="JSON data"):
            req(params={"client": object()})

    def test_cache_marker_is_detected_anywhere(self):
        assert not req().uses_prompt_cache()
        assert req(system=CACHED_SYSTEM).uses_prompt_cache()
        assert req(params={"cache_control": {"type": "ephemeral"}}).uses_prompt_cache()


class TestCache:
    def test_roundtrip(self, tmp_path):
        cache = ResponseCache(tmp_path)
        assert cache.get(req().cache_key) is None
        cache.put(req(), resp())
        assert cache.get(req().cache_key) == resp()

    def test_entries_are_written_with_lf(self, tmp_path):
        ResponseCache(tmp_path).put(req(system="a\nb"), resp())
        (entry,) = tmp_path.rglob("*.json")
        assert b"\r\n" not in entry.read_bytes()


# --------------------------------------------------------------------------------------
# Token usage and the ledger


class TestTokenUsage:
    def test_cache_writes_split_by_lifetime(self):
        u = TokenUsage.from_usage(
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_creation_input_tokens": 300,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 200,
                    "ephemeral_1h_input_tokens": 100,
                },
                "cache_read_input_tokens": 1000,
            }
        )
        assert u == TokenUsage(100, 10, 200, 100, 1000)
        assert u.prompt_tokens == 1400

    def test_unsplit_cache_writes_count_at_the_dearer_price(self):
        u = TokenUsage.from_usage({"input_tokens": 1, "cache_creation_input_tokens": 300})
        assert (u.cache_write_5m, u.cache_write_1h) == (0, 300)

    def test_missing_counts_are_zero(self):
        assert TokenUsage.from_usage({"input_tokens": None, "output_tokens": None}) == TokenUsage()


class TestLedger:
    def test_settle_prices_every_token_kind_and_persists(self, tmp_path):
        led = ledger(tmp_path)
        reserved = led.reserve(req())
        cost = led.settle(reserved, "m", TokenUsage(1000, 100, 200, 100, 1000))
        # 1000 x 2 + 200 x 2.5 + 100 x 4 + 1000 x 0.2 + 100 x 10, per million tokens
        assert cost == pytest.approx(4100 / 1e6)
        again = ledger(tmp_path)
        assert again.state["total_usd"] == pytest.approx(cost)
        bucket = again.state["by_phase"]["p"]
        assert (bucket["n_calls"], bucket["cache_write_tokens"], bucket["cache_read_tokens"]) == (
            1,
            300,
            1000,
        )

    def test_batch_is_half_of_every_price(self, tmp_path):
        u = TokenUsage(1000, 100, 200, 100, 1000)
        led = ledger(tmp_path)
        assert led.cost("m", u, batch=True) == pytest.approx(led.cost("m", u) / 2)

    def test_worst_case(self, tmp_path):
        led = ledger(tmp_path)
        plain = req()
        n = estimate_tokens_upper(plain.prompt_text())
        assert led.worst_case(plain) == pytest.approx((n * 2.0 + 100 * 10.0) / 1e6)
        cached = req(system=CACHED_SYSTEM)  # priced at the 1-hour cache-write rate
        n = estimate_tokens_upper(cached.prompt_text())
        assert led.worst_case(cached) == pytest.approx((n * 4.0 + 100 * 10.0) / 1e6)
        tools = req(tools=[TOOL])  # plus the tool-use system prompt
        n = estimate_tokens_upper(tools.prompt_text()) + TOOL_PROMPT_ALLOWANCE_TOKENS
        assert led.worst_case(tools) == pytest.approx((n * 2.0 + 100 * 10.0) / 1e6)

    def test_refuses_call_that_could_exceed_phase_cap(self, tmp_path):
        led = ledger(tmp_path, phase=0.0005)
        with pytest.raises(BudgetExceeded, match="p cap"):
            led.reserve(req())  # 100 output tokens alone cost $0.001

    def test_in_flight_reservations_count(self, tmp_path):
        led = ledger(tmp_path, phase=0.0015)
        led.reserve(req())
        with pytest.raises(BudgetExceeded):
            led.reserve(req())

    def test_unpriced_model_refused(self, tmp_path):
        with pytest.raises(BudgetExceeded, match="no price"):
            ledger(tmp_path).reserve(req(model="unknown"))

    def test_reserve_amount_respects_caps_unless_forced(self, tmp_path):
        led = ledger(tmp_path, phase=1.0)
        with pytest.raises(BudgetExceeded):
            led.reserve_amount(1.5)
        led.reserve_amount(1.5, force=True)
        assert led.headroom() < 0

    def test_earlier_phase_caps_are_kept(self, tmp_path):
        path = tmp_path / "spend.json"
        path.write_text(
            json.dumps(
                {
                    "total_usd": 0,
                    "n_calls": 0,
                    "by_phase": {},
                    "by_model": {},
                    "caps": {"project_usd": 60.0, "phase3_usd": 1.0},
                }
            )
        )
        led = SpendLedger(path, {"m": PRICE}, 60.0, "phase4", 20.0)
        led.settle(0.0, "m", TokenUsage(10, 10))
        caps = json.loads(path.read_text())["caps"]
        assert caps == {"project_usd": 60.0, "phase3_usd": 1.0, "phase4_usd": 20.0}

    def test_api_call_without_ledger_refused(self, tmp_path):
        backend = FakeBackend()
        with pytest.raises(BudgetExceeded):
            generate_cached(backend, req(), ResponseCache(tmp_path))
        assert backend.calls == []

    def test_cached_call_costs_nothing(self, tmp_path):
        backend = FakeBackend()
        cache, led = ResponseCache(tmp_path / "c"), ledger(tmp_path)
        _, cached1, cost1 = generate_cached(backend, req(), cache, led)
        _, cached2, cost2 = generate_cached(backend, req(), cache, led)
        assert (cached1, cached2, len(backend.calls)) == (False, True, 1)
        assert cost1 > 0 and cost2 == 0.0
        assert led.state["n_calls"] == 1

    def test_failed_call_releases_reservation(self, tmp_path):
        led = ledger(tmp_path, phase=0.0015)
        backend = FakeBackend(error=ConnectionError("network"))
        with pytest.raises(ConnectionError):
            generate_cached(backend, req(), ResponseCache(tmp_path), led)
        led.reserve(req())  # would raise if the failed reservation had leaked


class TestBudgetConfig:
    def test_phase0_cannot_spend(self, tmp_path):
        led = load_ledger("phase0", ROOT / "configs/budget.yaml", tmp_path / "spend.json")
        assert led.phase_cap_usd == 0.0
        with pytest.raises(BudgetExceeded):
            led.reserve(LLMRequest.single("anthropic", "claude-sonnet-5", "s", "u", 1))

    def test_phase_without_a_cap_is_refused(self, tmp_path):
        with pytest.raises(BudgetExceeded, match="no spending cap"):
            load_ledger("phase99", ROOT / "configs/budget.yaml", tmp_path / "spend.json")

    def test_prices_follow_the_published_multipliers(self, tmp_path):
        prices = load_ledger("phase0", ROOT / "configs/budget.yaml", tmp_path / "s.json").prices
        for model, p in prices.items():
            assert p.cache_write_5m == pytest.approx(1.25 * p.input), model
            assert p.cache_write_1h == pytest.approx(2.0 * p.input), model
            read = 0.05 if model == "claude-opus-5-5" else 0.1
            assert p.cache_read == pytest.approx(read * p.input), model


# --------------------------------------------------------------------------------------
# Replay-only mode


class TestReplayOnly:
    def test_miss_raises_without_calling_backend(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANALYST_REPLAY_ONLY", "1")
        backend = FakeBackend(name="ollama")
        with pytest.raises(CacheMiss):
            generate_cached(backend, req(), ResponseCache(tmp_path))
        assert backend.calls == []

    def test_hit_is_served(self, tmp_path, monkeypatch):
        cache = ResponseCache(tmp_path)
        cache.put(req(), resp())
        monkeypatch.setenv("ANALYST_REPLAY_ONLY", "1")
        backend = FakeBackend(name="ollama")
        _, was_cached, cost = generate_cached(backend, req(), cache)
        assert (was_cached, cost, backend.calls) == (True, 0.0, [])


# --------------------------------------------------------------------------------------
# Anthropic backend


class FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.kwargs = None
        self.stop_reason = stop_reason

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Let me query."),
                SimpleNamespace(
                    type="tool_use", id="t1", name="run_sql", input={"sql": "SELECT 1"}
                ),
            ],
            model="claude-sonnet-5",
            stop_reason=self.stop_reason,
            stop_details=SimpleNamespace(type="refusal", category="cyber", explanation=None),
            usage=SimpleNamespace(
                input_tokens=900,
                output_tokens=40,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=500,
                cache_creation=SimpleNamespace(
                    ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=0
                ),
            ),
            _request_id="req_1",
        )


def anthropic(stop_reason="end_turn") -> tuple[AnthropicBackend, FakeMessages]:
    fake = FakeMessages(stop_reason)
    return AnthropicBackend(client=SimpleNamespace(messages=fake)), fake


class TestAnthropicBackend:
    def test_request_shape(self):
        backend, fake = anthropic()
        r = req(
            model="claude-sonnet-5",
            system=CACHED_SYSTEM,
            tools=[TOOL],
            params={"thinking": {"type": "disabled"}, "_sample": 2},
        )
        backend.generate(r)
        assert fake.kwargs["system"] == CACHED_SYSTEM
        assert fake.kwargs["tools"] == [TOOL]
        assert fake.kwargs["messages"] == [{"role": "user", "content": "u"}]
        assert fake.kwargs["thinking"] == {"type": "disabled"}
        assert "_sample" not in fake.kwargs  # metadata: in the cache key, never sent

    def test_empty_system_and_tools_are_not_sent(self):
        backend, fake = anthropic()
        backend.generate(req(model="claude-sonnet-5", system=""))
        assert "system" not in fake.kwargs and "tools" not in fake.kwargs

    def test_sampling_goes_in_extra_body_for_direct_calls_only(self):
        r = req(model="claude-haiku-4-5", params={"temperature": 0})
        direct = AnthropicBackend.call_params(r, direct=True)
        batch = AnthropicBackend.call_params(r)
        assert direct["extra_body"] == {"temperature": 0} and "temperature" not in direct
        assert batch["temperature"] == 0 and "extra_body" not in batch

    @pytest.mark.parametrize(
        "model, params, message",
        [
            ("claude-opus-5-5", {"thinking": {"type": "disabled"}}, "rejects thinking"),
            ("claude-opus-5-5", {"output_config": {}}, "effort explicitly"),
            ("claude-sonnet-5", {"temperature": 0}, "sampling"),
            ("claude-opus-5-5", {"output_config": {"effort": "low"}, "top_p": 0.9}, "sampling"),
            (
                "claude-opus-5-5",
                {"output_config": {"effort": "low"}, "tool_choice": {"type": "any"}},
                "forced tool_choice",
            ),
        ],
    )
    def test_rejected_parameters_are_refused(self, model, params, message):
        with pytest.raises(UnsupportedParams, match=message):
            AnthropicBackend.check(req(model=model, params=params))

    @pytest.mark.parametrize(
        "model, params",
        [
            ("claude-opus-5-5", {"output_config": {"effort": "medium"}}),
            ("claude-sonnet-5", {"thinking": {"type": "disabled"}, "tool_choice": {"type": "any"}}),
            ("claude-haiku-4-5", {"temperature": 0}),
        ],
    )
    def test_accepted_parameters_pass(self, model, params):
        AnthropicBackend.check(req(model=model, params=params))

    def test_refused_before_any_reservation_or_call(self, tmp_path):
        backend, fake = anthropic()
        led = ledger(tmp_path)
        r = req(model="claude-opus-5-5", params={"thinking": {"type": "disabled"}})
        with pytest.raises(UnsupportedParams):
            generate_cached(backend, r, ResponseCache(tmp_path), led)
        assert fake.kwargs is None and led.headroom() == pytest.approx(0.5)

    def test_response_keeps_blocks_and_cache_usage(self):
        backend, _ = anthropic()
        out = backend.generate(req(model="claude-sonnet-5"))
        assert out.text == "Let me query."
        assert out.tool_calls() == [
            {"type": "tool_use", "id": "t1", "name": "run_sql", "input": {"sql": "SELECT 1"}}
        ]
        assert out.tokens == TokenUsage(input=900, output=40, cache_read=500)
        assert (out.model_reported, out.request_id) == ("claude-sonnet-5", "req_1")
        assert "stop_details" not in out.extra

    def test_refusal_details_are_kept(self):
        backend, _ = anthropic(stop_reason="refusal")
        out = backend.generate(req(model="claude-sonnet-5"))
        assert out.extra["stop_details"]["category"] == "cyber"


# --------------------------------------------------------------------------------------
# Ollama backend


class FakeOllama:
    def __init__(self, tool_calls=None):
        self.kwargs = None
        self.tool_calls = tool_calls

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            message=SimpleNamespace(content="done", thinking=None, tool_calls=self.tool_calls),
            model="qwen2.5:3b-instruct",
            done_reason="stop",
            prompt_eval_count=800,
            eval_count=30,
            load_duration=0,
            prompt_eval_duration=1e6,
            eval_duration=2e6,
        )

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model="qwen2.5:3b-instruct", digest="abc")])


def local(**kw) -> LLMRequest:
    base = {
        "backend": "ollama",
        "model": "qwen2.5:3b-instruct",
        "params": OllamaBackend.request_params(0, 0, 8192),
    }
    return req(**{**base, **kw})


class TestOllamaBackend:
    def test_options_and_digest(self):
        fake = FakeOllama()
        out = OllamaBackend(client=fake).generate(local())
        assert fake.kwargs["options"] == {
            "temperature": 0,
            "seed": 0,
            "num_ctx": 8192,
            "num_predict": 100,
        }
        assert fake.kwargs["tools"] is None
        assert out.extra["digest"] == "abc" and out.text == "done"
        assert out.tokens == TokenUsage(input=800, output=30)

    def test_prompt_that_would_be_truncated_is_refused(self):
        fake = FakeOllama()
        with pytest.raises(ContextOverflow):
            OllamaBackend(client=fake).generate(local(system="x" * 20_000, max_tokens=2048))
        assert fake.kwargs is None  # never sent

    def test_tools_are_converted_and_calls_come_back_as_tool_use(self):
        call = SimpleNamespace(
            function=SimpleNamespace(name="run_sql", arguments={"sql": "SELECT 1"})
        )
        fake = FakeOllama(tool_calls=[call])
        out = OllamaBackend(client=fake).generate(local(tools=[TOOL]))
        (sent,) = fake.kwargs["tools"]
        assert sent["function"]["name"] == "run_sql"
        assert sent["function"]["parameters"] == TOOL["input_schema"]
        assert out.tool_calls() == [
            {"type": "tool_use", "id": "call_0", "name": "run_sql", "input": {"sql": "SELECT 1"}}
        ]

    def test_conversation_with_tool_results_is_converted(self):
        r = local(
            messages=[
                {"role": "user", "content": "How many?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Querying."},
                        {"type": "tool_use", "id": "c1", "name": "run_sql", "input": {"sql": "Q"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "c1",
                            "content": [{"type": "text", "text": "42"}],
                        }
                    ],
                },
            ]
        )
        assert ollama_messages(r) == [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "How many?"},
            {
                "role": "assistant",
                "content": "Querying.",
                "tool_calls": [{"function": {"name": "run_sql", "arguments": {"sql": "Q"}}}],
            },
            {"role": "tool", "content": "42", "tool_name": "run_sql"},
        ]
