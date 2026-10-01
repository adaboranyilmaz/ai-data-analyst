"""A live question: the winning design run once on a model, streamed as it happens.

The run is the same `QuestionRun` the evaluation used, driven one model call at a time through
the response cache and a spend ledger, so a question asked twice costs once. Nothing in the
agent, the cache or the ledger is changed: the runner emits an event before each model call and
after each, from the steps the conversation records.

The deployment's ledger is its own file with its own cap (configs/serving.yaml), so a public or
shared service can never spend past what its operator allowed, and never touches the project's
evaluation ledger.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import yaml

from src.agent.clock import ClockStore
from src.agent.driver import Caller
from src.agent.run import Question, QuestionRun
from src.agent.run import config as agent_config
from src.llm.backends import Backend, make_backend
from src.llm.cache import ResponseCache
from src.llm.ledger import ModelPrice, SpendLedger
from src.serving import evidence
from src.serving.connections import ConnectionToolbox, Validated
from src.serving.live_guardrail import LiveGuardrail
from src.serving.meter import ROOT, Meter
from src.serving.replay import done_event
from src.tools.toolbox import Toolbox

BUDGET = ROOT / "configs/budget.yaml"


def deployment_ledger(cfg: dict[str, Any], budget_path: Path = BUDGET) -> SpendLedger:
    """The service's ledger: its own file, capped at the operator's spend cap, priced from the
    project's price table."""
    live = cfg["live"]
    budget = yaml.safe_load(budget_path.read_text(encoding="utf-8"))
    cap = float(live["spend_cap_usd"])
    return SpendLedger(
        ROOT / live["ledger"],
        {m: ModelPrice(**p) for m, p in budget["prices_usd_per_mtok"].items()},
        cap,
        live["phase"],
        cap,
        budget["batch_discount"],
    )


class LiveRunner:
    def __init__(
        self,
        cfg: dict[str, Any],
        meter: Meter,
        backend: Backend | None = None,
        ledger: SpendLedger | None = None,
        toolbox: Callable[[str], Toolbox] = Toolbox,
        guardrail: LiveGuardrail | None = None,
    ):
        self.guardrail = guardrail
        self.cfg = cfg
        self.live = cfg["live"]
        self.meter = meter
        self.backend = backend
        self.ledger = ledger
        self.toolbox_factory = toolbox
        self.agent_cfg = agent_config()

    @classmethod
    def default(cls, cfg: dict[str, Any], meter: Meter) -> LiveRunner:
        """The service's runner: the statistical guardrail on, unless the operator turned it off
        (ANALYST_GUARDRAIL=0, as the container does: its sandbox needs Docker)."""
        runner = cls(cfg, meter)
        if cfg["live"]["guardrail"] and os.environ.get("ANALYST_GUARDRAIL") != "0":
            runner.guardrail = LiveGuardrail(cfg["live"], runner._ledger())
        return runner

    def guardrail_status(self) -> dict[str, Any]:
        if self.guardrail is None:
            return {"enabled": False, "available": False}
        return {"enabled": True, "available": self.guardrail.available() is None}

    def _backend(self) -> Backend:
        if self.backend is None:
            model = self.agent_cfg["models"][self.live["model"]]
            self.backend = make_backend(model["backend"])
        return self.backend

    def _ledger(self) -> SpendLedger:
        if self.ledger is None:
            self.ledger = deployment_ledger(self.cfg)
        return self.ledger

    def problems(self) -> list[str]:
        """What stops live mode from running a question, for the readiness check."""
        out = []
        if self.backend is None and not os.environ.get("ANTHROPIC_API_KEY"):
            out.append("no ANTHROPIC_API_KEY in the environment")
        box = None
        try:
            box = self.toolbox_factory(self.live["database"])
            probe = box.call("run_sql", {"sql": f"SELECT 1 FROM {box.schema.tables[0]} LIMIT 1"})
            if not probe.get("ok"):
                raise RuntimeError(str(probe.get("error"))[:120])
        except Exception as e:  # the database is down or the role is missing
            out.append(f"the {self.live['database']} database is not reachable: {e}"[:200])
        finally:
            if box is not None:
                box.close()
        return out

    def refusal(self, question: str) -> dict[str, Any] | None:
        """Why this question cannot be run, if so: an error event, before anything is spent."""
        if not question.strip():
            return {"type": "error", "kind": "empty_question", "message": "Ask a question."}
        limit = self.live["max_question_chars"]
        if len(question) > limit:
            return {
                "type": "error",
                "kind": "question_too_long",
                "message": f"A question can be at most {limit} characters.",
            }
        headroom = self._ledger().headroom()
        if headroom < self.live["worst_case_question_usd"]:
            return {
                "type": "error",
                "kind": "spend_cap",
                "message": "This deployment's spend cap is reached, so it cannot run new "
                "questions. Recorded runs still work.",
            }
        return None

    def run(
        self,
        run_id: str,
        question: str,
        sink: list[dict[str, Any]],
        connection: Validated | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Events as the run goes; the last event is `done`, or an `error`. The finished evidence
        record is appended to `sink` just before the final events. `connection`: a person's own
        database, validated read-only (src/serving/connections.py); its answers carry no
        calibrated confidence and its cache is kept apart."""
        if (why := self.refusal(question)) is not None:
            yield why
            return
        db = self.live["database"] if connection is None else connection.spec.dbname
        design, model = self.live["design"], self.live["model"]
        ledger = self._ledger()
        cache_dir = ROOT / self.live["cache_dir"]
        if connection is not None:
            cache_dir = ROOT / self.live["connection_cache_dir"]
        cache = ResponseCache(cache_dir)
        caller = Caller(self._backend(), cache, ledger)
        q = Question(run_id, "live", db, question.strip(), None)
        yield {
            "type": "start",
            "id": run_id,
            "mode": "live",
            "question": q.question,
            "hint": None,
            "db_id": db,
            "kind": "live",
        }
        box = None
        try:
            box = self.toolbox_factory(db) if connection is None else ConnectionToolbox(connection)
            run = QuestionRun(
                q,
                design,
                model,
                self.live["evidence"],
                box,
                ledger.cost,
                None,
                self.agent_cfg,
                ClockStore(cache_dir) if connection is None else None,
            )
            seen: dict[int, int] = {}
            first_call = True
            while not run.done:
                for conv, request in run.pending():
                    if first_call:
                        yield {
                            "type": "step",
                            "kind": "model",
                            "text": evidence.model_step_text(db),
                            "recorded_ms": None,
                        }
                        first_call = False
                    started = time.perf_counter()
                    response = caller.one(request)
                    run.feed(conv, request.cache_key, response)
                    took = round((time.perf_counter() - started) * 1000)
                    for line in conv.step_lines[seen.get(id(conv), 0) :]:
                        kind = "submit" if line.startswith(evidence.SUBMIT_PREFIXES) else "tool"
                        yield {"type": "step", "kind": kind, "text": line, "recorded_ms": took}
                    seen[id(conv)] = len(conv.step_lines)
            fin = run.finish()
        except Exception as e:  # the model or the database failed: the run ends, the page says so
            yield {"type": "error", "kind": type(e).__name__, "message": str(e)[:300]}
            return
        finally:
            if box is not None:
                box.close()
        record = {
            "source": "live",
            "category": None,
            "difficulty": None,
            "correct": None,
            "score_outcome": None,
            "cost_usd": fin.cost_usd,
        }
        ev = evidence.from_run(
            run_id=run_id,
            kind="live",
            record=record,
            trace=fin.trace,
            meter=self.meter if connection is None else None,
            source_run="live",
        )
        if self.guardrail is not None and connection is None:
            self.guardrail.backend = self.guardrail.backend or self._backend()
            extra: list[dict[str, Any]] = []
            found: list[dict[str, Any] | None] = []
            for event in self.guardrail.run(run_id, q.question, found):
                extra.append(event)
                yield event
            ev["steps"] = ev["steps"] + extra
            ev["cost_usd"] += self.guardrail.last_cost
            outcome = found[0] if found else None
            if outcome is not None and "notice" in outcome:
                ev["answer"]["notice"] = outcome["notice"]
            elif outcome is not None:
                merged = evidence.from_guardrail(
                    run_id=run_id, rec=outcome, before=ev, review=None, source_run="live"
                )
                merged["kind"], merged["evaluation"] = "live", None
                merged["cost_usd"] = ev["cost_usd"]
                merged["steps"] = ev["steps"] + merged["steps"][1:]
                yield from merged["steps"][len(ev["steps"]) :]
                ev = merged
        sink.append(ev)
        yield from evidence.final_events(ev)
        yield done_event(ev) | {"cost_usd": ev["cost_usd"]}
