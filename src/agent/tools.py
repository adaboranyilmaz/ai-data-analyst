"""The tools as one design's agent sees them.

Wraps the guarded tools (src/tools/) with three things the agent needs and the tools themselves
do not decide:

- **Which tools a design may use.** A call to any other tool is an error result for the model.
- **Design 5's narrowed schema.** After the narrowing call, the agent sees only the chosen tables
  and columns (`list_tables`, `describe_table`, `sample_rows`), and `run_sql` accepts only the
  chosen tables: the query guard is built on them, so any other table is refused before it
  reaches the database.
- **What the model is shown.** A tool's result reaches the model as compact JSON with the first
  `rows_shown` rows and the total count, and never a timing: a timing differs on every run, so it
  would change the next request and its cache key, and a replayed run would miss the cache. The
  full result (timings included) stays in the trace. The one result that is not a function of
  the database and the query, that of a query reading the clock, is stored at its first run and
  returned on replays (src/agent/clock.py).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from src.agent.clock import ClockStore
from src.db.guard import QueryGuard
from src.tools.sql import SqlTools, refused
from src.tools.toolbox import Toolbox, _check_input, agent_limits

DATA_TOOLS = ("list_tables", "describe_table", "sample_rows", "run_sql")
_VOLATILE = ("seconds",)


@dataclass
class ToolOutcome:
    name: str
    input: Any
    result: dict  # the tool's full result, as the trace records it
    shown: str  # what the model reads
    is_error: bool


def _shown_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


NO_DESCRIPTION = "no description available (the column holds data; its meaning is not documented)"


def readable_description(result: dict) -> dict:
    """A table description as the model reads it. The dictionary marks a column whose source
    documentation is empty with `status: missing`, which a model can read as "no data"; here it
    becomes an explicit description saying the meaning is undocumented."""
    if not result.get("ok") or "columns" not in result:
        return result
    columns = []
    for c in result["columns"]:
        if c.get("status") == "missing" and not c.get("description"):
            c = {k: v for k, v in c.items() if k != "status"} | {"description": NO_DESCRIPTION}
        columns.append(c)
    return {**result, "columns": columns}


def present(result: dict, rows_shown: int) -> dict:
    """The part of a tool result the model sees: no timings, at most `rows_shown` rows."""
    out = {k: v for k, v in result.items() if k not in _VOLATILE}
    rows = out.get("rows")
    if isinstance(rows, list) and len(rows) > rows_shown:
        out["rows"] = rows[:rows_shown]
        out["rows_shown"] = rows_shown
    return out


class AgentTools:
    """The tools of one run on one database."""

    def __init__(
        self,
        toolbox: Toolbox,
        allowed: tuple[str, ...] | list[str],
        rows_shown: int,
        selection: dict[str, list[str]] | None = None,
        clock: ClockStore | None = None,
    ):
        self.toolbox = toolbox
        self.allowed = tuple(allowed)
        self.rows_shown = rows_shown
        self.selection = selection
        self.clock = clock
        self._sql = toolbox.sql
        if selection is not None:
            cfg = toolbox.cfg
            self._sql = SqlTools(
                QueryGuard(toolbox.db, set(selection)),
                toolbox.executor,
                agent_limits(cfg),
                cfg["agent"]["max_cell_chars"],
                cfg["sample_rows"]["default_rows"],
                cfg["sample_rows"]["max_rows"],
            )

    # ---------------------------------------------------------------- what the model reads first

    def table_list(self) -> dict:
        out = self.toolbox.schema.list_tables()
        if self.selection is not None:
            out = {**out, "tables": [t for t in out["tables"] if t["table"] in self.selection]}
        return out

    def full_schema(self) -> list[dict]:
        """The table list and every table described: what design 1 and the narrowing call read."""
        return [self.table_list()] + [self._describe(t) for t in self._tables()]

    def _tables(self) -> list[str]:
        tables = self.toolbox.schema.tables
        return [t for t in tables if self.selection is None or t in self.selection]

    def _describe(self, table: str) -> dict:
        if self.selection is not None and table not in self.selection:
            return refused([f"unknown table {table!r}"])
        out = readable_description(self.toolbox.schema.describe_table(table))
        if self.selection is not None and out.get("ok"):
            keep = set(self.selection[table])
            out = {**out, "columns": [c for c in out["columns"] if c["name"] in keep]}
        return out

    # ---------------------------------------------------------------- tool calls

    def call(self, name: str, args: Any, at: tuple[str, str] | None = None) -> ToolOutcome:
        """`at`: the call this answers (the request whose response made it, and its id), which
        keys a stored result of a query that reads the clock (src/agent/clock.py)."""
        start = time.perf_counter()
        if name not in self.allowed:
            result = refused([f"the tool {name!r} is not available"])
        elif name == "run_sql" and isinstance(args, dict) and isinstance(args.get("sql"), str):
            result = self._clocked(args["sql"], at, lambda: self._call(name, args))
        else:
            result = self._call(name, args)
        result.setdefault("seconds", round(time.perf_counter() - start, 6))
        shown = _shown_json(present(result, self.rows_shown))
        return ToolOutcome(name, args, result, shown, not result.get("ok", False))

    def _call(self, name: str, args: Any) -> dict:
        if self.selection is None:
            result = self.toolbox.call(name, args)
            return readable_description(result) if name == "describe_table" else result
        return self._narrowed_call(name, args)

    def _clocked(self, sql: str, at: tuple[str, str] | None, run) -> dict:
        if self.clock is None:
            return run()
        return self.clock.through(self.toolbox.db, self._sql.guard.tables, sql, at, run)

    def _narrowed_call(self, name: str, args: Any) -> dict:
        if problems := _check_input(name, args):  # the same input checks as the full toolbox
            return refused(problems)
        if name == "list_tables":
            return self.table_list()
        if name == "describe_table":
            return self._describe(args["table"])
        if name == "sample_rows":
            table = args["table"]
            if table not in (self.selection or {}):
                return refused([f"unknown table {table!r}"])
            out = self._sql.sample_rows(table, args.get("n"))
            return self._project(out, self.selection[table])
        return self._sql.run_sql(args["sql"])

    @staticmethod
    def _project(result: dict, columns: list[str]) -> dict:
        """Only the chosen columns of a whole-row sample."""
        if not result.get("ok"):
            return result
        idx = [i for i, c in enumerate(result["columns"]) if c["name"] in set(columns)]
        return {
            **result,
            "columns": [result["columns"][i] for i in idx],
            "rows": [[row[i] for i in idx] for row in result["rows"]],
        }

    def run_final(self, sql: str, at: tuple[str, str] | None = None) -> dict:
        """The submitted SQL through the (narrowed) guard and the agent's limits, for design 3's
        resubmission rule; `at`: the submit call, as for `call`."""
        return self._clocked(sql, at, lambda: self._sql.run_sql(sql))
