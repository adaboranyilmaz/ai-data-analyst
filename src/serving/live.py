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
from src.serving.champion import Champion
from src.serving.connections import ConnectionToolbox, Validated
from src.serving.larger import AutoToolRun, agent_config_with
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
        champion: Champion | None = None,
    ):
        self.guardrail = guardrail
        self.champion = champion
        self.router = champion.router if champion is not None else None
        self.cfg = cfg
        self.live = cfg["live"]
        self.meter = meter
        self.backend = backend
        self.ledger = ledger
        self.toolbox_factory = toolbox
        self.agent_cfg = agent_config()

    @classmethod
    def default(
        cls, cfg: dict[str, Any], meter: Meter, champion: Champion | None = None
    ) -> LiveRunner:
        """The service's runner: the statistical guardrail on, unless the operator turned it off
        (ANALYST_GUARDRAIL=0, as the container does: its sandbox needs Docker)."""
        runner = cls(cfg, meter, champion=champion)
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

    def _drive(self, run: QuestionRun, caller: Caller, first_step: str) -> Iterator[dict[str, Any]]:
        """A run's model calls, one at a time: a step before the first call and one per step line
        after each."""
        seen: dict[int, int] = {}
        first: str | None = first_step
        while not run.done:
            for conv, request in run.pending():
                if first is not None:
                    yield {"type": "step", "kind": "model", "text": first, "recorded_ms": None}
                    first = None
                started = time.perf_counter()
                response = caller.one(request)
                run.feed(conv, request.cache_key, response)
                took = round((time.perf_counter() - started) * 1000)
                for line in conv.step_lines[seen.get(id(conv), 0) :]:
                    kind = "submit" if line.startswith(evidence.SUBMIT_PREFIXES) else "tool"
                    yield {"type": "step", "kind": kind, "text": line, "recorded_ms": took}
                seen[id(conv)] = len(conv.step_lines)

    def _routing(self, fin) -> dict[str, Any] | None:
        """Whether the first model's answer goes to the larger model: its calibrated confidence
        is under the threshold chosen for routing (a declined answer has confidence 0)."""
        if self.router is None or self.champion is None:
            return None
        threshold = self.champion.config["calibration"]["decline_threshold"]
        declined = bool(fin.answer.declined)
        calibrated = 0.0 if declined else self.meter.calibrate(fin.confidence)
        if calibrated < threshold:
            model = self.router["model"]
            return {
                "calibrated": round(calibrated, 4),
                "threshold": round(threshold, 4),
                "text": f"Confidence {calibrated:.0%} is under {threshold:.0%}: asking {model}.",
            }
        return None

    def _larger_run(self, q: Question, box: Toolbox, ledger: SpendLedger, cache_dir: Path):
        cfg = agent_config_with(self.router["model"], self.router["request_settings"])
        return AutoToolRun(
            q,
            self.live["design"],
            self.router["model"],
            self.live["evidence"],
            box,
            ledger.cost,
            None,
            cfg,
            ClockStore(cache_dir),
        )

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
        streamed: list[dict[str, Any]] = []
        routing = None
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
            for event in self._drive(run, caller, evidence.model_step_text(db)):
                streamed.append(event)
                yield event
            fin = run.finish()
            final, routing = fin, self._routing(fin) if connection is None else None
            if routing is not None:
                note = {
                    "type": "step",
                    "kind": "model",
                    "text": routing["text"],
                    "recorded_ms": None,
                }
                streamed.append(note)
                yield note
                second = self._larger_run(q, box, ledger, cache_dir)
                again = evidence.model_step_text(db).replace("Read", "Read again", 1)
                for event in self._drive(second, caller, again):
                    streamed.append(event)
                    yield event
                final = second.finish()
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
            "cost_usd": fin.cost_usd + (final.cost_usd if final is not fin else 0.0),
        }
        ev = evidence.from_run(
            run_id=run_id,
            kind="live",
            record=record,
            trace=final.trace,
            meter=self.meter if connection is None else None,
            source_run="live",
            larger=final is not fin,
        )
        if final is not fin:  # both models' steps, in the order they happened
            ev["steps"] = streamed
            ev["routed"] = {
                "from": self.live["model"],
                "to": self.router["model"],
                "calibrated_confidence": routing["calibrated"],
                "threshold": routing["threshold"],
            }
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
