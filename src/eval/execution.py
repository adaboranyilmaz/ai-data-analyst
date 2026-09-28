"""Scoring a prediction: the predicted and the gold query in one read-only transaction.

The official evaluator runs both queries on one connection in one transaction, so both see the
same `now()` (eleven gold queries compute ages from the current date). So does `PairExecutor`,
through the agent's execution layer (src/db/execute.py): the question's schema role, a
read-only transaction that is always rolled back, a server-side cursor (one query, nothing
else), the fixed session settings, and a time limit enforced by the client, here for the pair
as a whole (the official evaluator's 30 s). No query guard: the official evaluator has none,
and the database layer contains any query on its own, as the security suite shows. Among the
fixed settings, every sequential scan starts at the table's first block: Soft-F1 pairs rows by
position, so it needs the row order not to depend on the queries run before.

Two differences from the agent's execution:
- `cursor_tuple_fraction = 1`. A cursor is otherwise planned to return its first tenth of rows
  quickly, which can change the plan and with it the row order and, for a LIMIT over tied
  rows, which rows come back. At 1 it is planned as the plain query the official evaluator
  runs.
- No row limit. The prediction streams through `PredictionStream` (src/eval/ex.py) in batches;
  memory stays bounded by the gold result for EX.

The gold runs first (the official evaluator runs the prediction first; the order changes
nothing but how long a failing pair takes).

By construction, three kinds of prediction score 0 here where the official evaluator, which
connects as a superuser through psycopg2, may score 1: more than one statement (psycopg2 runs
them all and compares the last one's rows), a statement that is not a query (e.g. `SET` before
the query), and a query that reads another database's tables (the schema role cannot). The
agent's own SQL tool refuses all three before they run, so no answer it gives can meet them.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

import psycopg

from src.db.execute import Limits, ReadOnlyExecutor, Target, _classify, _message
from src.db.values import set_hash
from src.eval.config import config
from src.eval.ex import PredictionStream

GOLD_CURSOR = "gold_query"
PREDICTED_CURSOR = "predicted_query"
# Functions whose value depends on when a query runs. A gold query using one (the benchmark
# has eleven, all computing ages) returns a different result once the date moves on, so its
# stored result is only a snapshot; scoring always runs it again beside the prediction.
_CLOCK = re.compile(
    r"\b(?:now\s*\(|current_date\b|current_timestamp\b|current_time\b|localtime(?:stamp)?\b)",
    re.IGNORECASE,
)


def reads_the_clock(sql: str) -> bool:
    return _CLOCK.search(sql) is not None


@dataclass(frozen=True)
class ScoreSettings:
    timeout_s: float
    fetch_rows: int
    soft_f1_max_distinct_rows: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> ScoreSettings:
        e = (cfg or config())["execution_accuracy"]
        return cls(e["timeout_s"], e["fetch_rows"], e["soft_f1_max_distinct_rows"])


@dataclass
class PairScore:
    """outcome: `ok`; `no_prediction` (no SQL to score); `timeout`; `error` (the prediction
    failed: `error_kind` as in src/db/execute.py); `comparison_error` (the rows could not be
    compared, e.g. an array value, which cannot be hashed); `gold_error` (the gold query
    failed, which the gold runs rule out: a sign of a broken environment).
    soft_f1 is None when the prediction had too many distinct rows to compute it;
    soft_f1_error says why Soft-F1 alone scored 0 when EX could still be computed."""

    ex: int
    soft_f1: float | None
    outcome: str
    error_kind: str | None = None
    message: str | None = None
    soft_f1_error: str | None = None
    gold_rows: int | None = None
    gold_set_hash: str | None = None
    predicted_rows: int | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _TimeUp(Exception):
    pass


class PairExecutor(ReadOnlyExecutor):
    """Scores predictions against gold queries on one benchmark schema, one connection."""

    def __init__(self, target: Target, settings: ScoreSettings | None = None):
        super().__init__(target)
        self.settings = settings or ScoreSettings.from_config()

    def score(self, predicted_sql: str | None, gold_sql: str) -> PairScore:
        start = time.perf_counter()

        def done(**kw) -> PairScore:
            return PairScore(seconds=round(time.perf_counter() - start, 4), **kw)

        if predicted_sql is None:
            return done(ex=0, soft_f1=0.0, outcome="no_prediction")
        try:
            conn = self._connection()
        except psycopg.Error as e:
            return done(
                ex=0, soft_f1=0.0, outcome="error", error_kind="connection", message=_message(e)
            )

        s = self.settings
        cancelled = threading.Event()

        def cancel() -> None:
            cancelled.set()
            try:
                conn.cancel_safe()
            except psycopg.Error:
                pass

        def check_time() -> None:
            # a cancel sent while Python, not the server, was busy reaches no query
            if cancelled.is_set():
                raise _TimeUp

        timer = threading.Timer(s.timeout_s, cancel)
        stage = "gold"
        gold: list[tuple] = []
        stream: PredictionStream | None = None
        try:
            with conn.transaction():
                self._set_local(conn, Limits(max_rows=0, timeout_s=s.timeout_s))
                conn.execute("SET LOCAL cursor_tuple_fraction = 1")
                timer.start()
                try:
                    with conn.cursor(name=GOLD_CURSOR) as cur:
                        cur.execute(gold_sql)
                        gold = cur.fetchall()
                    check_time()
                    stage = "comparison"
                    stream = PredictionStream(gold, s.soft_f1_max_distinct_rows)
                    stage = "prediction"
                    with conn.cursor(name=PREDICTED_CURSOR) as cur:
                        cur.execute(predicted_sql)
                        while not stream.settled and (rows := cur.fetchmany(s.fetch_rows)):
                            check_time()
                            stage = "comparison"
                            stream.feed(rows)
                            stage = "prediction"
                    check_time()
                finally:
                    timer.cancel()
                    timer.join()
                raise psycopg.Rollback
        except _TimeUp:
            return done(ex=0, soft_f1=0.0, outcome="timeout", gold_rows=len(gold))
        except psycopg.Error as e:
            if cancelled.is_set():
                return done(ex=0, soft_f1=0.0, outcome="timeout", message=_message(e))
            kind = _classify(e)
            outcome = "gold_error" if stage == "gold" else "error"
            return done(
                ex=0,
                soft_f1=0.0,
                outcome=outcome,
                error_kind=kind,
                message=_message(e),
                gold_rows=None if stage == "gold" else len(gold),
            )
        except Exception as e:
            # the official evaluator scores any exception 0, a failed comparison included
            if stage != "comparison":
                raise
            return done(
                ex=0,
                soft_f1=0.0,
                outcome="comparison_error",
                message=f"{type(e).__name__}: {e}",
                gold_rows=len(gold),
            )

        assert stream is not None
        soft_f1_error = None
        try:
            f1 = stream.soft_f1()
        except Exception as e:  # e.g. a gold row with no columns: Soft-F1 alone fails
            f1, soft_f1_error = 0.0, f"{type(e).__name__}: {e}"
        return done(
            ex=stream.ex(),
            soft_f1=f1,
            outcome="ok",
            soft_f1_error=soft_f1_error,
            gold_rows=len(gold),
            gold_set_hash=set_hash(gold),
            predicted_rows=stream.rows,
        )


class Scorer:
    """One `PairExecutor` per benchmark schema, opened on first use."""

    def __init__(self, targets: dict[str, Target], settings: ScoreSettings | None = None):
        self.targets = targets
        self.settings = settings or ScoreSettings.from_config()
        self._executors: dict[str, PairExecutor] = {}

    def __enter__(self) -> Scorer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        for ex in self._executors.values():
            ex.close()
        self._executors.clear()

    def score(self, db: str, predicted_sql: str | None, gold_sql: str) -> PairScore:
        if db not in self._executors:
            self._executors[db] = PairExecutor(self.targets[db], self.settings)
        return self._executors[db].score(predicted_sql, gold_sql)
