"""The spend ledger: enforces the API cost caps and records every dollar spent.

Before an uncached API call the ledger reserves the call's worst case: the prompt's size,
estimated pessimistically from its characters and priced at the dearest input rate the
request could incur (a 1-hour cache write, if it carries a cache marker), plus the full
`max_tokens` of output. After the call it settles the reservation at the real cost from the
response's token usage. A call that could take either the phase cap or the project cap past
its limit is refused before it is sent. Reservations are thread-safe, so concurrent calls
cannot overshoot. A phase without a configured cap cannot spend anything.

Prices are explicit per model and token kind, not multiples of the input price: cache reads
cost a tenth of the input price on most models but a twentieth on Claude Opus 5.5. Message
Batches calls are billed at `batch_discount` of every price.

The ledger file changes with every paid call. Pipeline stages must not declare it as a DVC
dependency: they read it only for the caps, and a dependency would mark them stale whenever
anything else spends.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.llm.types import LLMRequest, TokenUsage

BUDGET_CONFIG = Path("configs/budget.yaml")

# Pessimistic characters per token, used only for the pre-flight bound of the budget
# reservation, never for reported numbers. English prose runs at about
# 4 characters per token and code or JSON at about 3, while figure-dense text can fall to
# about 2.4; 1.5 leaves a wide margin. The counted dry run before any paid run checks it
# against the token-counting endpoint on the real prompts.
PESSIMISTIC_CHARS_PER_TOKEN = 1.5
# With tools present the API adds a tool-use system prompt: 286 to 588 tokens for the models
# priced in configs/budget.yaml, by the published per-model counts. 1,000 covers all of them.
TOOL_PROMPT_ALLOWANCE_TOKENS = 1_000


def estimate_tokens_upper(text: str) -> int:
    return int(len(text) / PESSIMISTIC_CHARS_PER_TOKEN) + 1


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens."""

    input: float
    output: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float


class SpendLedger:
    """Cumulative API spend across runs and phases, persisted as JSON."""

    def __init__(
        self,
        path: Path,
        prices: dict[str, ModelPrice],
        project_cap_usd: float,
        phase: str,
        phase_cap_usd: float,
        batch_discount: float = 0.5,
    ):
        self.path = Path(path)
        self.prices = prices
        self.project_cap_usd = project_cap_usd
        self.phase = phase
        self.phase_cap_usd = phase_cap_usd
        self.batch_discount = batch_discount
        self._lock = threading.Lock()
        self._reserved = 0.0
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.state = {"total_usd": 0.0, "n_calls": 0, "by_phase": {}, "by_model": {}}

    def price(self, model: str) -> ModelPrice:
        if model not in self.prices:
            raise BudgetExceeded(f"no price configured for {model!r}; refusing to call it")
        return self.prices[model]

    def cost(self, model: str, usage: TokenUsage, batch: bool = False) -> float:
        p = self.price(model)
        usd = (
            usage.input * p.input
            + usage.cache_write_5m * p.cache_write_5m
            + usage.cache_write_1h * p.cache_write_1h
            + usage.cache_read * p.cache_read
            + usage.output * p.output
        ) / 1_000_000
        return usd * self.batch_discount if batch else usd

    def phase_spent(self) -> float:
        return self.state["by_phase"].get(self.phase, {}).get("usd", 0.0)

    def worst_case(self, request: LLMRequest, batch: bool = False) -> float:
        p = self.price(request.model)
        prompt = estimate_tokens_upper(request.prompt_text())
        if request.tools:
            prompt += TOOL_PROMPT_ALLOWANCE_TOKENS
        rate = p.input
        if request.uses_prompt_cache():
            rate = max(p.input, p.cache_write_5m, p.cache_write_1h)
        usd = (prompt * rate + request.max_tokens * p.output) / 1_000_000
        return usd * self.batch_discount if batch else usd

    def headroom(self) -> float:
        """How much more can be reserved before either cap would be exceeded."""
        with self._lock:
            return self._headroom()

    def _headroom(self) -> float:
        return min(
            self.project_cap_usd - self.state["total_usd"] - self._reserved,
            self.phase_cap_usd - self.phase_spent() - self._reserved,
        )

    def reserve_amount(self, amount: float, force: bool = False) -> None:
        """Reserve a known amount (a whole batch's worst case). `force` skips the cap check:
        used only when re-attaching to a batch already submitted, whose cost is committed."""
        with self._lock:
            if not force and amount > self._headroom():
                raise BudgetExceeded(
                    f"reserving ${amount:.4f} would exceed a cap (headroom ${self._headroom():.4f})"
                )
            self._reserved += amount

    def reserve(self, request: LLMRequest) -> float:
        worst = self.worst_case(request)
        with self._lock:
            project_after = self.state["total_usd"] + self._reserved + worst
            phase_after = self.phase_spent() + self._reserved + worst
            if project_after > self.project_cap_usd:
                raise BudgetExceeded(
                    f"project cap ${self.project_cap_usd:.2f} would be exceeded "
                    f"(spent ${self.state['total_usd']:.4f}, in flight ${self._reserved:.4f}, "
                    f"this call up to ${worst:.4f})"
                )
            if phase_after > self.phase_cap_usd:
                raise BudgetExceeded(
                    f"{self.phase} cap ${self.phase_cap_usd:.2f} would be exceeded "
                    f"(spent ${self.phase_spent():.4f}, in flight ${self._reserved:.4f}, "
                    f"this call up to ${worst:.4f})"
                )
            self._reserved += worst
        return worst

    def settle(self, reserved: float, model: str, usage: TokenUsage, batch: bool = False) -> float:
        actual = self.cost(model, usage, batch)
        with self._lock:
            self._reserved -= reserved
            self.state["total_usd"] += actual
            self.state["n_calls"] += 1
            for bucket, name in (("by_phase", self.phase), ("by_model", model)):
                b = self.state[bucket].setdefault(
                    name,
                    {
                        "usd": 0.0,
                        "n_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_write_tokens": 0,
                        "cache_read_tokens": 0,
                    },
                )
                b["usd"] += actual
                b["n_calls"] += 1
                b["input_tokens"] += usage.input
                b["output_tokens"] += usage.output
                b["cache_write_tokens"] += usage.cache_write_5m + usage.cache_write_1h
                b["cache_read_tokens"] += usage.cache_read
                if batch:
                    b["n_batch_calls"] = b.get("n_batch_calls", 0) + 1
            self.state["updated_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
            caps = self.state.setdefault("caps", {})  # earlier phases' caps stay on record
            caps["project_usd"] = self.project_cap_usd
            caps[f"{self.phase}_usd"] = self.phase_cap_usd
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8", newline="\n")
        return actual

    def release(self, reserved: float) -> None:
        """Drop a reservation for a call that failed before any tokens were billed."""
        with self._lock:
            self._reserved -= reserved


def load_ledger(
    phase: str, config_path: Path = BUDGET_CONFIG, ledger_path: Path | None = None
) -> SpendLedger:
    """The ledger for one phase, with the caps and prices from the budget config. A phase
    missing from `phase_caps_usd` is refused: every phase's cap is set before it spends."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    caps = cfg["phase_caps_usd"] or {}
    if phase not in caps:
        raise BudgetExceeded(f"no spending cap configured for {phase!r} in {config_path}")
    return SpendLedger(
        Path(ledger_path or cfg["ledger"]),
        {m: ModelPrice(**p) for m, p in cfg["prices_usd_per_mtok"].items()},
        cfg["project_cap_usd"],
        phase,
        caps[phase],
        cfg["batch_discount"],
    )
