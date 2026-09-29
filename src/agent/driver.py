"""Running many questions: direct calls, or every open conversation's next turn in one batch.

- `direct`: each worker takes a question and runs it to the end, one model call at a time
  (`generate_cached`: the response cache, then the spend ledger, then the call).
- `batch`: rounds. Each round collects the next request of every open conversation of every
  question and sends them as Message Batches at half price (`run_batch_cached`); the responses
  land in the response cache, and each conversation is fed its own. A request that failed in a
  batch is simply asked again in the next round; a round in which nothing advances stops the
  run, so a persistent failure cannot loop.

Either way every response goes through the one response cache, so a finished run replays at $0
in either mode. A `LayeredCache` lets a stage read the responses an earlier stage stored (e.g.
the winner's run on all questions reuses the design comparison's answers) while writing only to
its own directory.

The tools of a question run on a connection of the thread that feeds it (`ToolboxPool`), so a
batched run of hundreds of questions holds one connection per database, not one per question.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from src.agent.run import Finished, QuestionRun
from src.llm.backends import Backend
from src.llm.batch import BATCH_DIR, run_batch_cached
from src.llm.cache import ResponseCache
from src.llm.generate import generate_cached
from src.llm.ledger import SpendLedger
from src.llm.types import LLMRequest, LLMResponse
from src.tools.toolbox import Toolbox

MODEL_ERROR = "model_error"
INVALID_REQUEST = "invalid_request_error"  # the API's error type for a request it cannot serve
BILLING_MARKERS = ("credit balance", "billing", "credit")  # the same type, about the account
STALLED_ROUNDS = 3  # batched rounds in a row in which nothing advances before a run stops


def permanent(failure: dict) -> bool:
    """Whether a failed batch request would fail again if sent again: the API rejected the
    request itself. A rejection for the account's credit is of the same type, but says nothing
    about the request: it is asked again once there is credit."""
    message = (failure.get("message") or "").lower()
    return failure.get("error") == INVALID_REQUEST and not any(
        m in message for m in BILLING_MARKERS
    )


def error_response(message: str) -> LLMResponse:
    """A model call that failed, in the shape of a response (no content, no tokens)."""
    return LLMResponse(
        text="",
        content=[],
        model_reported="",
        stop_reason=MODEL_ERROR,
        usage={},
        latency_ms=0.0,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        extra={"error": message},
    )


class LayeredCache:
    """Writes to one cache directory; reads it first, then the others in order."""

    def __init__(self, write: ResponseCache, read_also: Sequence[ResponseCache] = ()):
        self.write = write
        self.layers = [write, *read_also]
        self.root = write.root

    def has(self, key: str) -> bool:
        return any(c.has(key) for c in self.layers)

    def get(self, key: str) -> LLMResponse | None:
        for c in self.layers:
            r = c.get(key)
            if r is not None:
                return r
        return None

    def put(self, request: LLMRequest, response: LLMResponse) -> None:
        self.write.put(request, response)


class ToolboxPool:
    """One Toolbox per (database, thread), closed together at the end.

    The agent's database role has a connection limit (20), so a thread keeps at most one
    connection open when it starts a question: taking a toolbox closes the thread's toolboxes
    for other databases (a closed toolbox reconnects when used again). Direct runs, one question
    per worker at a time, so hold one connection per worker; a batched run holds its runs on
    every database at once, one connection per database.
    """

    def __init__(self, factory: Callable[[str], Toolbox] = Toolbox):
        self.factory = factory
        self._boxes: dict[tuple[str, int], Toolbox] = {}
        self._lock = threading.Lock()

    def get(self, db: str) -> Toolbox:
        thread = threading.get_ident()
        with self._lock:
            for (other, t), box in self._boxes.items():
                if t == thread and other != db:
                    box.close()
            key = (db, thread)
            if key not in self._boxes:
                self._boxes[key] = self.factory(db)
            return self._boxes[key]

    def close(self) -> None:
        with self._lock:
            for box in self._boxes.values():
                box.close()
            self._boxes.clear()


class Caller:
    """Model calls through the response cache and the spend ledger."""

    def __init__(
        self,
        backend: Backend,
        cache: ResponseCache | LayeredCache,
        ledger: SpendLedger | None,
        poll_seconds: float = 30.0,
        batch_dir: Path = BATCH_DIR,
        log: Callable[[str], None] = print,
    ):
        self.backend, self.cache, self.ledger = backend, cache, ledger
        self.poll_seconds, self.batch_dir, self.log = poll_seconds, batch_dir, log

    def one(self, request: LLMRequest) -> LLMResponse:
        try:
            response, _, _ = generate_cached(self.backend, request, self.cache, self.ledger)
        except Exception as e:
            if self.backend.name != "ollama" or type(e).__name__ != "ResponseError":
                raise
            # The local model failed on this request (e.g. Ollama stops a generation stuck
            # repeating itself). It runs at temperature 0 with a fixed seed, so the failure
            # repeats: it is stored like a response, and a replay meets it again instead of
            # calling the model. The conversation ends without an answer.
            response = error_response(str(e))
            self.cache.put(request, response)
        return response

    def many(self, requests: list[LLMRequest]) -> dict[str, LLMResponse]:
        """The responses that arrived (failed requests are missing: asked again next round)."""
        todo = [r for r in requests if not self.cache.has(r.cache_key)]
        if todo:
            if self.ledger is None:
                raise ValueError("a batch needs a spend ledger")
            outcome = run_batch_cached(
                todo,
                self.cache,
                self.ledger,
                self.backend.client,
                poll_seconds=self.poll_seconds,
                batch_dir=self.batch_dir,
                log=self.log,
            )
            by_key = {r.cache_key: r for r in todo}
            for f in outcome.failures:
                self.log(f"  batch request failed: {f}")
                # The API rejected the request itself (e.g. a prompt longer than the model's
                # context): sent again it would fail again, so it is stored like a response and
                # that conversation ends without an answer, as a failed local call does. Other
                # failures (overloaded, expired) are asked again in the next round.
                if permanent(f) and f["cache_key"] in by_key:
                    message = f.get("message") or "invalid request"
                    self.cache.put(
                        by_key[f["cache_key"]], error_response(f"the API rejected it: {message}")
                    )
        out = {}
        for r in requests:
            got = self.cache.get(r.cache_key)
            if got is not None:
                out[r.cache_key] = got
        return out


def drive_direct(
    make_runs: Sequence[Callable[[], QuestionRun]], caller: Caller, workers: int = 1
) -> list[tuple[QuestionRun, Finished]]:
    """Each question is built, run and finished in one worker thread, so its tools use that
    thread's connections only. Results come back in the order given."""

    def work(make: Callable[[], QuestionRun]) -> tuple[QuestionRun, Finished]:
        run = make()
        while not run.done:
            for conv, request in run.pending():
                run.feed(conv, request.cache_key, caller.one(request))
        return run, run.finish()

    if workers <= 1:
        return [work(m) for m in make_runs]
    with ThreadPoolExecutor(workers) as pool:
        return [f.result() for f in [pool.submit(work, m) for m in make_runs]]


def drive_batch(
    make_runs: Sequence[Callable[[], QuestionRun]],
    caller: Caller,
    log: Callable[[str], None] = print,
) -> list[tuple[QuestionRun, Finished]]:
    """Every question in this thread; model calls in batched rounds."""
    runs = [m() for m in make_runs]
    round_no = stalled = 0
    while open_runs := [r for r in runs if not r.done]:
        round_no += 1
        pending = [(r, conv, req) for r in open_runs for conv, req in r.pending()]
        log(f"round {round_no}: {len(pending)} requests from {len(open_runs)} questions")
        responses = caller.many([req for _, _, req in pending])
        fed = 0
        for run, conv, req in pending:
            if req.cache_key in responses:
                run.feed(conv, req.cache_key, responses[req.cache_key])
                fed += 1
        # a request that failed for a passing reason is asked again; a run in which nothing
        # advances for several rounds stops (every response so far is cached for a re-run)
        stalled = 0 if fed else stalled + 1
        if stalled >= STALLED_ROUNDS:
            raise RuntimeError(f"round {round_no}: no request succeeded in {stalled} rounds")
    return [(r, r.finish()) for r in runs]
