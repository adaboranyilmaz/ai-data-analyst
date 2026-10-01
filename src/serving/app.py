"""The service: FastAPI endpoints, one event stream per question, recorded or live.

- `POST /ask` streams a question's run as Server-Sent Events: `start`, a `step` per step, the
  `sql`, `rows`, `checks`, `statistics`, `chart`, `answer`, `confidence`, then `done` (or an
  `error`). In replay mode it serves a recorded run with the timing it was recorded with; it
  holds no key and makes no model call. In live mode it runs the analyst.
- `GET /runs/{id}` is a run's full evidence record, the target of a permalink.
- `GET /health`, `GET /ready`, `GET /metrics` (Prometheus), and a JSON-lines request log that
  records the method, path, status and duration of each request, never a body.
- `POST /connections` (local mode only) points live mode at a person's own PostgreSQL, after
  checking that its role can only read.

The built UI, if present, is served at `/`.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from src.serving import champion as champion_mod
from src.serving import connections, replay
from src.serving.drift import DriftMonitor
from src.serving.live import LiveRunner
from src.serving.meter import ROOT, Meter, config
from src.serving.metrics import Metrics
from src.serving.store import RunStore, valid_id

UI_DIST = ROOT / "ui/dist"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")


@dataclass
class Settings:
    mode: str = "replay"  # "replay" | "live"
    local_mode: bool = False  # the service is bound to this machine: connections are allowed
    cfg: dict[str, Any] = field(default_factory=config)
    request_log: Path | None = None
    runner: LiveRunner | None = None
    store: RunStore | None = None
    meter: Meter | None = None
    champion: champion_mod.Champion | None = None

    @classmethod
    def from_env(cls) -> Settings:
        mode = os.environ.get("ANALYST_SERVING_MODE", "replay")
        if mode not in ("replay", "live"):
            raise ValueError("ANALYST_SERVING_MODE must be 'replay' or 'live'")
        return cls(
            mode=mode,
            local_mode=os.environ.get("ANALYST_LOCAL_MODE") == "1",
            request_log=ROOT / "data/serving/requests.jsonl",
        )


class AskBody(BaseModel):
    question: str | None = None
    run_id: str | None = None


class ConnectionBody(BaseModel):
    host: str = ""
    port: int = 5432
    dbname: str = ""
    user: str = ""
    password: str = ""
    schema_: str | None = Field(default=None, alias="schema")

    model_config = {"populate_by_name": True}


def sse(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def normalize(question: str) -> str:
    return " ".join(question.lower().split())


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings.from_env()
    champion = s.champion or champion_mod.load()
    cfg = champion.apply(s.cfg)  # the live section follows the registry's champion
    # Recorded runs were described by the evaluated system's meter; a live question is described
    # by the champion's own.
    meter = s.meter or (Meter.load(cfg) if s.mode == "replay" else champion.meter(cfg))
    store = s.store or RunStore(
        ROOT / cfg["curated"]["out_dir"],
        ROOT / cfg["live"]["runs_dir"] if s.mode == "live" else None,
    )
    if s.mode == "live" and s.runner is None:
        s.runner = LiveRunner.default(cfg, meter, champion)
    metrics = Metrics()
    metrics.agent(champion.name, champion.config_sha256, champion.router is not None)
    drift = DriftMonitor.from_meter(meter, **cfg["drift"])
    busy = asyncio.Semaphore(cfg["live"]["max_concurrent"])
    state: dict[str, Any] = {"connection": None}

    app = FastAPI(title="AI data analyst", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def local_hosts_only(request: Request, call_next):
        """A service meant for this machine answers only to this machine's names. A web page
        that points its own domain at 127.0.0.1 (DNS rebinding) arrives with that domain in the
        Host header, and is refused here before it can reach the connection form or a live run."""
        if s.local_mode:
            host = request.headers.get("host", "")
            name = host.rsplit(":", 1)[0] if not host.endswith("]") else host
            if name not in LOCAL_HOSTS:
                return JSONResponse({"detail": "This service answers on localhost only."}, 403)
        return await call_next(request)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            ms = round((time.perf_counter() - start) * 1000, 1)
            route = request.scope.get("route")
            path = route.path if route is not None else "unmatched"
            metrics.request(request.method, path, status, ms / 1000)
            if s.request_log is not None:
                s.request_log.parent.mkdir(parents=True, exist_ok=True)
                limit = cfg["api"]["request_log_max_mb"] * 1_000_000
                if s.request_log.exists() and s.request_log.stat().st_size > limit:
                    s.request_log.replace(s.request_log.with_suffix(".jsonl.1"))  # keep one old
                line = {
                    "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                    "method": request.method,
                    "path": path,
                    "status": status,
                    "ms": ms,
                    "mode": s.mode,
                }
                with s.request_log.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(line) + "\n")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> JSONResponse:
        problems = []
        if not store.index():
            problems.append("no recorded runs")
        if s.mode == "live" and s.runner is not None:
            problems += s.runner.problems()
        return JSONResponse(
            {"ready": not problems, "mode": s.mode, "problems": problems},
            status_code=200 if not problems else 503,
        )

    @app.get("/api/drift")
    def drift_status() -> dict[str, Any]:
        snap = drift.snapshot()
        return {
            **snap,
            "psi_reading": drift.reading(snap["psi"]),
            "withheld_share_reading": drift.reading(snap["below_psi"]),
            "warn": drift.warn,
            "alert": drift.alert,
        }

    @app.get("/metrics")
    def prometheus() -> Response:
        return Response(metrics.render(), media_type=metrics.content_type)

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {
            "mode": s.mode,
            "local_mode": s.local_mode,
            "agent": champion.info(),
            "database": cfg["live"]["database"],
            "meter": meter.summary(),
            "connection": state["connection"].spec.public() if state["connection"] else None,
            "guardrail": s.runner.guardrail_status() if s.runner else {"enabled": False},
            "suggested": store.index(),
            "max_question_chars": cfg["live"]["max_question_chars"],
        }

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        ev = store.get(run_id)
        if ev is None:
            raise HTTPException(404, "no such run")
        return ev

    def find_recorded(body: AskBody) -> dict[str, Any] | None:
        if body.run_id:
            return store.get(body.run_id)
        if body.question:
            want = normalize(body.question)
            for entry in store.index():
                if normalize(entry["question"]) == want:
                    return store.get(entry["id"])
        return None

    @app.post("/ask")
    async def ask(body: AskBody) -> StreamingResponse:
        if body.run_id is not None and not valid_id(body.run_id):
            raise HTTPException(400, "bad run id")
        if s.mode == "replay" or body.run_id:
            ev = find_recorded(body)
            if ev is None:
                metrics.ask("replay", "not_recorded")
                raise HTTPException(
                    404,
                    "This demo serves recorded runs only. Live questions work when the service "
                    "runs locally with an API key.",
                )
            return StreamingResponse(
                replayed(ev, "replay"), media_type="text/event-stream", headers=_SSE_HEADERS
            )
        question = (body.question or "").strip()
        refusal = s.runner.refusal(question)
        if refusal is not None:
            code = 429 if refusal["kind"] == "spend_cap" else 400
            metrics.ask("live", refusal["kind"])
            raise HTTPException(code, refusal["message"])
        if busy.locked():
            metrics.ask("live", "busy")
            raise HTTPException(429, "The analyst is busy with other questions; try again soon.")
        run_id = store.new_live_id()
        return StreamingResponse(
            live_stream(run_id, question), media_type="text/event-stream", headers=_SSE_HEADERS
        )

    def observe(event: dict[str, Any]) -> None:
        """What an event says about the service: step latency, the final query's outcome, the
        calibrated confidence and its drift from the evaluation's."""
        metrics.observe(event)
        if event.get("type") == "confidence" and not event.get("not_calibrated"):
            drift.observe(event.get("calibrated"))
            metrics.drift(drift.snapshot())

    async def replayed(ev: dict[str, Any], mode: str):
        started = time.perf_counter()
        status = ev["status"]
        for event, delay in replay.events(ev, cfg["replay"] | _speed()):
            if delay:
                await asyncio.sleep(delay)
            observe(event)
            yield sse(event)
        metrics.ask(mode, status)
        metrics.run_seconds(mode, time.perf_counter() - started)

    def _speed() -> dict[str, float]:
        env = os.environ.get("ANALYST_REPLAY_SPEED")
        return {"speed": float(env)} if env else {}

    async def live_stream(run_id: str, question: str):
        async with busy:
            started = time.perf_counter()
            sink: list[dict[str, Any]] = []
            saved = False
            outcome = "error"
            conn = state["connection"]
            async for event in iterate_in_threadpool(s.runner.run(run_id, question, sink, conn)):
                if sink and not saved:
                    store.save_live(sink[0])
                    store.prune_live(cfg["live"]["keep_runs"])
                    saved = True
                if event["type"] == "done":
                    outcome = event["status"]
                    metrics.add_spend(event.get("cost_usd") or 0.0)
                observe(event)
                yield sse(event)
            metrics.ask("live", outcome)
            metrics.run_seconds("live", time.perf_counter() - started)

    # --- a person's own database, local mode only ----------------------------------------

    def require_local() -> None:
        if not s.local_mode:
            raise HTTPException(
                403, "Connections are available only when the service runs on this machine."
            )
        if s.mode != "live":
            raise HTTPException(409, "Connections need live mode.")

    @app.post("/connections")
    def connect(body: ConnectionBody) -> dict[str, Any]:
        require_local()
        try:
            spec = connections.parse(
                {**body.model_dump(exclude={"schema_"}), "schema": body.schema_}
            )
            checked = connections.validate(spec)
        except connections.ConnectionRefused as e:
            raise HTTPException(422, {"refused": e.reasons}) from None
        state["connection"] = checked
        return {
            "connection": spec.public(),
            "tables": len(checked.tables),
            "warnings": list(checked.warnings),
            "calibrated": False,
        }

    @app.get("/connections")
    def current_connection() -> dict[str, Any]:
        require_local()
        c = state["connection"]
        return {"connection": c.spec.public() if c else None}

    @app.delete("/connections")
    def disconnect() -> dict[str, Any]:
        require_local()
        state["connection"] = None
        return {"connection": None}

    # --- the built UI --------------------------------------------------------------------

    if UI_DIST.is_dir():
        assets = UI_DIST / "assets"

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(UI_DIST / "index.html")

        if assets.is_dir():
            from fastapi.staticfiles import StaticFiles

            app.mount("/assets", StaticFiles(directory=assets), name="assets")
    else:

        @app.get("/")
        def no_ui() -> dict[str, str]:
            return {"message": "The API is running; the UI is not built (see ui/README)."}

    return app


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
