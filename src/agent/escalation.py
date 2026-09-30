"""The escalation arm: the winning design on a model that refuses a forced tool choice.

Design 1 forces its one `submit_answer` call (`tool_choice`), and Claude Opus 5.5 refuses a forced
tool choice with an error. On such a model the call is asked for by the prompt and the tool list
alone (`tool_choice` auto, the same prompt and tool), and a reply without it gets the loop's one
reminder, as in the multi-turn designs. Everything else is design 1 unchanged. The model's own
settings (configs/confidence.yaml: its effort level, since its thinking cannot be switched off)
are recorded in every trace.

Runs go through the same response cache, spend ledger, scoring and records as the agent's stages
(src/agent/evaluate.py), so a finished run replays at $0; only the run class differs.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from functools import partial
from typing import Any

from src.agent.answer import SUBMIT
from src.agent.clock import ClockStore
from src.agent.conversation import Settings
from src.agent.driver import Caller, LayeredCache, ToolboxPool, drive_batch, drive_direct
from src.agent.evaluate import RunSpec, _score_and_write
from src.agent.run import Question, QuestionRun, config
from src.llm.backends import REJECTS_FORCED_TOOL_CHOICE, Backend, make_backend
from src.llm.cache import ResponseCache
from src.llm.ledger import load_ledger
from src.tracking.otel import jsonl_tracer


def agent_config_with(model: str, settings: dict[str, Any]) -> dict[str, Any]:
    """The agent's configuration with one more model's settings."""
    cfg = copy.deepcopy(config())
    cfg["models"][model] = copy.deepcopy(settings)
    return cfg


class AutoToolRun(QuestionRun):
    """A question run whose single-shot call is not forced on models that refuse it."""

    def _settings(self, **kw) -> Settings:
        if kw.get("force_tool") == SUBMIT and self.model in REJECTS_FORCED_TOOL_CHOICE:
            kw = {**kw, "force_tool": None, "reminders": self.cfg["reminders"]}
        return super()._settings(**kw)


def execute(
    specs: list[RunSpec],
    cfg: dict[str, Any],
    log: Callable[[str], None] = print,
    backend: Backend | None = None,
) -> list[list[dict]]:
    """Run, score and write escalation runs, as src/agent/evaluate.py `execute_many` does:
    batched runs together (one set of batches per round), direct runs one question per worker.
    `cfg`: the agent's configuration with the escalation model added. Returns each run's
    records."""
    shared = {
        (s.mode, s.cache_dir, s.read_caches, s.phase, s.ledger_path, s.batch_dir) for s in specs
    }
    if len(shared) != 1:
        raise ValueError("runs executed together must share mode, caches, phase and ledger")
    if {cfg["models"][s.model]["backend"] for s in specs} != {"anthropic"}:
        raise ValueError("the escalation arm runs on the Anthropic backend")
    spec = specs[0]
    backend = backend or make_backend("anthropic")
    ledger = load_ledger(spec.phase, ledger_path=spec.ledger_path)
    cache = LayeredCache(
        ResponseCache(spec.cache_dir), [ResponseCache(p) for p in spec.read_caches]
    )
    caller = Caller(
        backend,
        cache,
        ledger,
        cfg["run"]["batch_poll_seconds"] if spec.poll_seconds is None else spec.poll_seconds,
        spec.batch_dir,
        log=log,
    )
    tracers = [jsonl_tracer(s.spans) if s.spans else (None, None) for s in specs]
    pool = ToolboxPool()
    clock = ClockStore(spec.cache_dir, spec.read_caches)

    def make(s: RunSpec, tracer, q: Question) -> QuestionRun:
        box = pool.get(q.db_id)
        return AutoToolRun(q, s.design, s.model, s.evidence, box, ledger.cost, tracer, cfg, clock)

    jobs = [
        (k, i)
        for k, s in enumerate(specs)
        for i in sorted(range(len(s.items)), key=lambda i, s=s: (s.items[i][0].db_id, i))
    ]
    makers = [partial(make, specs[k], tracers[k][0], specs[k].items[i][0]) for k, i in jobs]
    try:
        if spec.mode == "batch":
            done = drive_batch(makers, caller, log)
        else:
            done = drive_direct(makers, caller, cfg["run"]["workers"])
    finally:
        pool.close()
        for _, provider in tracers:
            if provider is not None:
                provider.shutdown()

    finished: list[dict] = [{} for _ in specs]
    for (k, i), (_, fin) in zip(jobs, done, strict=True):
        finished[k][specs[k].items[i][0].question_id] = fin
    return [_score_and_write(s, f) for s, f in zip(specs, finished, strict=True)]


def router_spec(conf: dict[str, Any], design: str) -> RunSpec:
    """The escalation model on the router's whole set (the held-out questions), with the winning
    design, exactly as in the escalation arm. The router's choice (which questions go to it) is
    applied afterwards, from the calibrated confidence and the decline threshold
    (src/agent/confidence.py `routed_ids`): its answers do not depend on the choice, and having
    them on every question also gives the escalation model alone on the set, for comparison. Its
    own stage, `router`, after the escalation decision."""
    from src.agent.stages import make_spec

    esc = conf["escalation"]
    return make_spec(
        "router",
        esc["router"]["set"],
        design,
        esc["model"],
        esc["evidence"],
        esc["router"].get("mode", esc["mode"]),
        None,
        phase=conf["phase"],
    )
