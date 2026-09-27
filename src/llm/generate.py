"""One model call through the response cache and, for paid backends, the spend ledger."""

from __future__ import annotations

from src.llm.backends import Backend
from src.llm.cache import CacheMiss, ResponseCache, replay_only
from src.llm.ledger import BudgetExceeded, SpendLedger
from src.llm.types import LLMRequest, LLMResponse

PAID_BACKENDS = frozenset({"anthropic"})


def generate_cached(
    backend: Backend,
    request: LLMRequest,
    cache: ResponseCache,
    ledger: SpendLedger | None = None,
) -> tuple[LLMResponse, bool, float]:
    """Return (response, was_cached, new_cost_usd).

    A cached response is served without a call. Otherwise, in order: replay-only mode turns
    the miss into an error; the backend rejects parameters its model does not accept; a
    paid backend must come with a ledger, which reserves the call's worst case; the call is
    made; the reservation is settled at the real cost; the response is cached."""
    cached = cache.get(request.cache_key)
    if cached is not None:
        return cached, True, 0.0
    if replay_only():
        raise CacheMiss(
            f"{request.model} request {request.cache_key[:12]} is not in the response cache "
            "(ANALYST_REPLAY_ONLY=1)"
        )
    backend.check(request)
    if backend.name in PAID_BACKENDS and ledger is None:
        raise BudgetExceeded(f"{backend.name} calls require a SpendLedger")
    reserved = ledger.reserve(request) if ledger is not None else 0.0
    try:
        response = backend.generate(request)
    except BaseException:
        if ledger is not None:
            ledger.release(reserved)
        raise
    cost = 0.0
    if ledger is not None:
        cost = ledger.settle(reserved, request.model, response.tokens)
    cache.put(request, response)
    return response, False, cost
