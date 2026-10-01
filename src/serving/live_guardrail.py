"""The statistical guardrail for a live question.

After the analyst has answered with SQL, a question that asks for a comparison, a trend or a
cause (the classifier, src/stats/classify.py) is also analyzed: the analyst plans the analysis,
the plan's query pulls the rows through the two guards, a locked-down container computes the
interval and the checks, and the answer is rewritten from the computed result. The SQL answer
stays beside it, unchanged.

It uses the recorded evaluation's own code and frozen prompts (src/stats/), with its caches and
the deployment's ledger redirected, so a live service never writes into the evaluation's
caches or spends from its ledger. The container needs Docker; where it is not available (the
service itself runs in a container) a flagged question is answered with SQL alone and says so.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

from src.llm.backends import Backend
from src.llm.ledger import SpendLedger
from src.serving.meter import ROOT
from src.stats import guardrail as gr
from src.stats.calls import config as stats_config

NOTICE_NO_SANDBOX = (
    "This question asks for a comparison or a cause. The statistical analysis is not available "
    "in this deployment (it needs Docker to run its sandbox), so this is the SQL answer alone: it "
    "carries no interval and no check against reading an association as a cause."
)
NOTICE_FAILED = (
    "This question asks for a comparison or a cause, but the statistical analysis could not be "
    "completed ({why}). This is the SQL answer alone: it carries no interval and no check "
    "against reading an association as a cause."
)


def step(text: str, kind: str = "tool") -> dict[str, Any]:
    return {"type": "step", "kind": kind, "text": text, "recorded_ms": None}


class LiveGuardrail:
    def __init__(
        self,
        live_cfg: dict[str, Any],
        ledger: SpendLedger,
        backend: Backend | None = None,
        sandbox_ready: bool | None = None,
    ):
        self.ledger = ledger
        self.backend = backend
        self._sandbox_ready = sandbox_ready
        cfg = copy.deepcopy(stats_config())
        base = str(ROOT / live_cfg["guardrail_cache_dir"])
        for stage in ("classify", "plans", "own"):
            cfg["stages"][stage]["cache"] = f"{base}/{stage}"
        for kind in ("classify", "plan", "answer"):
            cfg["calls"][kind]["mode"] = "direct"
        self.cfg = cfg
        self.last_cost = 0.0  # what the last question's guardrail calls cost, all of them

    def available(self) -> str | None:
        """Why the analysis cannot run here, or None."""
        if self._sandbox_ready is None:
            from src.stats.sandbox import SandboxUnavailable, docker, image_present

            try:
                docker()
                self._sandbox_ready = image_present()
            except SandboxUnavailable:
                self._sandbox_ready = False
        return None if self._sandbox_ready else NOTICE_NO_SANDBOX

    def _kw(self) -> dict[str, Any]:
        return {"backend": self.backend, "ledger": self.ledger}

    def run(
        self, run_id: str, question: str, sink: list[dict[str, Any] | None]
    ) -> Iterator[dict[str, Any]]:
        """Step events; then one item in `sink`: None (not a statistical question), a record with
        the guarded answer, or {"notice": why} when the analysis could not be done."""
        q = {
            "id": f"live:{run_id}",
            "source": "live",
            "db_id": gr.DB,
            "category": None,
            "question": question,
        }
        self.last_cost = 0.0
        self.last_cost = 0.0
        yield step("Checked whether the question asks for a comparison, a trend or a cause.")
        try:
            classified = gr.run_classify(
                [q], self.cfg, log=lambda *_: None, workers=1, **self._kw()
            )[0]
        except Exception as e:  # a failed classification must not lose the SQL answer
            sink.append({"notice": NOTICE_FAILED.format(why=_why(e))})
            return
        self.last_cost += classified["cost_usd"]
        self.last_cost += classified["cost_usd"]
        if not classified["statistical"]:
            sink.append(None)
            return
        yield step("It does: the answer needs an interval, not only numbers.")
        if (why := self.available()) is not None:
            sink.append({"notice": why})
            return
        try:
            plans = gr.run_plans([q], self.cfg, log=lambda *_: None, **self._kw())
            self.last_cost += plans[0]["cost_usd"]
            self.last_cost += plans[0]["cost_usd"]
            rec = gr.run_own(
                plans, [classified], self.cfg, "own", log=lambda *_: None, **self._kw()
            )[0]
        except Exception as e:  # the sandbox, the database or the model failed
            sink.append({"notice": NOTICE_FAILED.format(why=_why(e))})
            return
        self.last_cost += rec.get("cost_usd") or 0.0
        self.last_cost += rec.get("cost_usd") or 0.0
        if "guarded" not in rec:
            reason = rec.get("analysis_error") or rec.get("plan_error") or {}
            text = reason.get("message") if isinstance(reason, dict) else None
            sink.append({"notice": NOTICE_FAILED.format(why=text or "no analysis was planned")})
            return
        sink.append(rec)


def _why(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"[:160]
