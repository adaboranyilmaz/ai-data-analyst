"""Read-only execution of one query as the agent's role, with limits it cannot lift.

This module runs whatever query it is given; checking the query is the job of the caller
(src/db/guard.py, through src/tools/sql.py). What it enforces on its own, so that each of these
holds even for a query no checker has seen:

- **One query, no side effects.** The query runs through a server-side cursor: PostgreSQL wraps
  it in `DECLARE ... CURSOR FOR`, which accepts a single `SELECT`, `VALUES` or `WITH` query and
  nothing else, and sends it over the extended protocol, which refuses a second statement.
  The transaction is `READ ONLY` and always rolled back.
- **The schema's own privileges.** Each transaction switches to the target's schema role,
  which can read that schema and nothing else (src/db/hardening.py).
- **Settings fixed per call.** Each transaction sets its own `search_path`, time zone,
  statement timeout, parallel-worker limit, scan start and string-literal rules with
  `SET LOCAL`, so nothing a previous query did to the session carries over. Results depend on
  three of them: the benchmark's timestamps were written at UTC+8; parallel workers make
  floating-point sums and averages add their terms in a different order on every run; and a
  synchronized sequential scan of a large table starts where the previous scan of it stopped,
  so without a complete ORDER BY the rows come back in an order (and, under a LIMIT, a
  selection) that depends on the queries run before. Every scan here starts at the table's
  first block, so a replayed run gets the same rows in the same order. Statements are never
  prepared, so each is planned under these settings.
- **A time limit enforced by the client.** The role can switch off its own statement timeout,
  so a timer cancels the query from the client side when the limit runs out.
- **A row limit.** At most `max_rows + 1` rows leave the server. When there are more, the rest
  are counted on the server (`MOVE ALL`) without being sent, if the time limit allows.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg import errors, sql

from src.db.connection import AGENT_ROLE, connect

CURSOR_NAME = "analyst_query"
# Sessions opened here carry this name, so a server-side check can find the agent's queries.
APPLICATION_NAME = "analyst_agent"
# A cancelled query needs a moment to stop; the server-side timeout is a backstop only.
SERVER_TIMEOUT_MARGIN_S = 1.0


@dataclass(frozen=True)
class Target:
    """Where a query runs: a database, the schema its tables are in, its time zone, and the
    role whose privileges the query runs with (the schema's own role, src/db/hardening.py;
    None keeps the privileges of the role that connected)."""

    dbname: str
    schema: str
    time_zone: str
    role: str | None = None


@dataclass(frozen=True)
class Limits:
    max_rows: int
    timeout_s: float
    count_total: bool = True  # count the rows beyond max_rows on the server


@dataclass
class QueryResult:
    columns: list[tuple[str, str]]  # (name, PostgreSQL type)
    rows: list[tuple[Any, ...]]
    truncated: bool
    total_rows: int | None  # None: truncated, and counting ran out of time
    seconds: float


@dataclass
class QueryError(Exception):
    """A query that did not return a result.

    kind: `timeout`, `resource_limit` (temp_file_limit and similar), `permission_denied`,
    `read_only` (the read-only transaction refused a write), `multiple_statements`,
    `syntax_error` (including a statement the cursor does not accept: anything but a query),
    `not_supported` (e.g. a data-modifying WITH in a cursor), `sql_error` (anything else
    PostgreSQL reported), `connection`.
    """

    kind: str
    message: str
    sqlstate: str | None = None
    seconds: float = field(default=0.0, compare=False)

    def __str__(self) -> str:
        return f"{self.kind}: {self.message}"


def _classify(e: psycopg.Error) -> str:
    if isinstance(e, errors.QueryCanceled):
        return "timeout"
    if isinstance(e, errors.ConfigurationLimitExceeded | errors.ProgramLimitExceeded):
        return "resource_limit"
    if isinstance(e, errors.InsufficientPrivilege):
        return "permission_denied"
    if isinstance(e, errors.ReadOnlySqlTransaction):
        return "read_only"
    if isinstance(e, errors.SyntaxError):
        return "multiple_statements" if "multiple commands" in _message(e) else "syntax_error"
    if isinstance(e, errors.FeatureNotSupported):
        return "not_supported"
    if e.sqlstate is None:
        return "connection"
    return "sql_error"


def _message(e: psycopg.Error) -> str:
    """The error's first line, without the statement text psycopg appends."""
    text = str(e).strip()
    return text.splitlines()[0] if text else type(e).__name__


class ReadOnlyExecutor:
    """Runs queries against one target on one connection, opened on first use.

    `transaction_read_only=False` exists for the security suite alone: it removes the
    read-only transaction so that a test can show the role's privileges block writes on
    their own. Nothing else passes it.
    """

    def __init__(
        self,
        target: Target,
        role: str = AGENT_ROLE,
        transaction_read_only: bool = True,
    ):
        self.target = target
        self.role = role
        self.transaction_read_only = transaction_read_only
        self._conn: psycopg.Connection | None = None

    def __enter__(self) -> ReadOnlyExecutor:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _connection(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed or self._conn.broken:
            self._conn = connect(
                self.role,
                self.target.dbname,
                prepare_threshold=None,
                application_name=APPLICATION_NAME,
            )
        self._conn.read_only = self.transaction_read_only
        return self._conn

    def execute(self, query: str, limits: Limits) -> QueryResult:
        start = time.perf_counter()
        try:
            conn = self._connection()
        except psycopg.Error as e:
            raise QueryError("connection", _message(e)) from None

        cancelled = threading.Event()

        def cancel() -> None:
            cancelled.set()
            try:
                conn.cancel_safe()
            except psycopg.Error:  # the query ended and the connection closed meanwhile
                pass

        timer = threading.Timer(limits.timeout_s, cancel)
        rows: list[tuple] = []
        columns: list[tuple[str, str]] = []
        total: int | None = None
        try:
            with conn.transaction():
                self._set_local(conn, limits)
                timer.start()
                try:
                    with conn.cursor(name=CURSOR_NAME) as cur:
                        cur.execute(query)
                        types = conn.adapters.types
                        columns = [
                            (d.name, t.name if (t := types.get(d.type_code)) else str(d.type_code))
                            for d in cur.description or []
                        ]
                        rows = cur.fetchmany(limits.max_rows + 1)
                        if len(rows) <= limits.max_rows:
                            total = len(rows)
                        elif limits.count_total:
                            total = self._count_rest(conn, cur.name, len(rows))
                finally:
                    timer.cancel()
                    timer.join()
                raise psycopg.Rollback  # nothing a query does is ever kept
        except psycopg.Error as e:
            kind = "timeout" if cancelled.is_set() else _classify(e)
            raise QueryError(
                kind,
                _message(e),
                getattr(e, "sqlstate", None),
                round(time.perf_counter() - start, 4),
            ) from None
        truncated = len(rows) > limits.max_rows
        return QueryResult(
            columns=columns,
            rows=rows[: limits.max_rows],
            truncated=truncated,
            total_rows=total,
            seconds=round(time.perf_counter() - start, 4),
        )

    def _set_local(self, conn: psycopg.Connection, limits: Limits) -> None:
        if self.target.role is not None:
            conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(self.target.role)))
        server_ms = int((limits.timeout_s + SERVER_TIMEOUT_MARGIN_S) * 1000)
        for name, value in (
            ("statement_timeout", f"{server_ms}ms"),
            ("search_path", self.target.schema),
            ("TimeZone", self.target.time_zone),
            ("max_parallel_workers_per_gather", "0"),
            ("synchronize_seqscans", "off"),
            # a backslash in '...' is an ordinary character, as the query guard assumes
            ("standard_conforming_strings", "on"),
        ):
            conn.execute(
                sql.SQL("SET LOCAL {} = {}").format(sql.Identifier(name), sql.Literal(value))
            )

    @staticmethod
    def _count_rest(conn: psycopg.Connection, cursor: str, fetched: int) -> int | None:
        """Rows fetched plus the rows left, counted on the server; None if time ran out."""
        try:
            with conn.transaction():  # a savepoint: a cancelled count leaves the rows usable
                moved = conn.execute(
                    sql.SQL("MOVE FORWARD ALL IN {}").format(sql.Identifier(cursor))
                ).rowcount
            return fetched + moved
        except errors.QueryCanceled:
            return None
