"""`run_sql` and `sample_rows`: the agent's two ways to read data, both through both layers.

Every query passes the guard (src/db/guard.py) and then runs through the read-only executor
(src/db/execute.py) as the agent's role. A refused query never reaches the database.

Results are returned as plain JSON for the model: every value is a JSON number, string,
boolean or null (dates as ISO strings, decimals as strings, so no digit is lost), and long
text is cut to `max_cell_chars` with a marker. The values are data from the database: a text
value that reads like an instruction is returned exactly as stored, like any other value, in
the `rows` field, never mixed into the tool's own messages.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from decimal import Decimal
from typing import Any

from src.db.execute import Limits, QueryError, ReadOnlyExecutor
from src.db.guard import QueryGuard

JSONValue = Any


def display_value(v: Any, max_chars: int) -> JSONValue:
    if v is None or isinstance(v, bool | int):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else repr(v)
    if isinstance(v, str):
        if len(v) > max_chars:
            return f"{v[:max_chars]}...[{len(v) - max_chars} more characters]"
        return v
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, dt.date | dt.time):  # datetime is a date
        return v.isoformat()
    if isinstance(v, dt.timedelta):
        return str(v)
    if isinstance(v, bytes | bytearray | memoryview):
        return f"<{len(bytes(v))} bytes>"
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, list | tuple):
        return [display_value(x, max_chars) for x in v]
    return display_value(str(v), max_chars)


def refused(reasons: tuple[str, ...] | list[str]) -> dict:
    return {"ok": False, "error": {"kind": "refused", "reasons": list(reasons)}}


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class SqlTools:
    """`run_sql` and `sample_rows` for one database, with fixed limits."""

    def __init__(
        self,
        guard: QueryGuard,
        executor: ReadOnlyExecutor,
        limits: Limits,
        max_cell_chars: int,
        sample_default_rows: int = 5,
        sample_max_rows: int = 20,
    ):
        self.guard = guard
        self.executor = executor
        self.limits = limits
        self.max_cell_chars = max_cell_chars
        self.sample_default_rows = sample_default_rows
        self.sample_max_rows = sample_max_rows

    def run_sql(self, sql: str) -> dict:
        verdict = self.guard.check(sql)
        if not verdict.allowed:
            return refused(verdict.reasons)
        return self._execute(verdict.query, self.limits)

    def sample_rows(self, table: str, n: int | None = None) -> dict:
        """A few whole rows of one table, the same rows every time for the same data.

        Ordered by a hash of each row's text: the choice depends only on the rows' contents
        (not on where they sit on disk), so a replayed run sees the same sample, and it spreads
        over the table rather than showing its first rows.
        """
        n = self.sample_default_rows if n is None else n
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= self.sample_max_rows:
            return refused([f"n must be a whole number from 1 to {self.sample_max_rows}"])
        if table not in self.guard.tables:
            return refused([f"unknown table {table!r}"])
        query = f"SELECT * FROM {quote_ident(table)} AS t ORDER BY md5(t::text) LIMIT {n}"
        verdict = self.guard.check(query)
        if not verdict.allowed:  # cannot happen for a known table; both layers still apply
            return refused(verdict.reasons)
        return self._execute(verdict.query, Limits(n, self.limits.timeout_s, count_total=False))

    def _execute(self, query: str, limits: Limits) -> dict:
        try:
            r = self.executor.execute(query, limits)
        except QueryError as e:
            return {
                "ok": False,
                "error": {"kind": e.kind, "message": e.message, "sqlstate": e.sqlstate},
                "seconds": e.seconds,
            }
        return {
            "ok": True,
            "columns": [{"name": n, "type": t} for n, t in r.columns],
            "rows": [[display_value(v, self.max_cell_chars) for v in row] for row in r.rows],
            "row_count": len(r.rows),
            "total_rows": r.total_rows,
            "truncated": r.truncated,
            "seconds": r.seconds,
        }
