"""BIRD's official evaluator, loaded from its downloaded files to check the project's scoring.

Needs the `bird-eval` dependency group: the evaluator imports psycopg2, func_timeout and
PyMySQL. Its files are used unmodified; the one thing replaced is `connect_postgresql`, which
holds the upstream authors' own connection string (their `postgres` superuser, database
`bird`, everything in one schema). Here it connects as the read-only `analyst_ro` to this
project's `bird` database, with `search_path` listing every benchmark schema: table names are
unique across the benchmark's databases, so every name resolves as it did in the single schema
of BIRD's dump. The database's default time zone applies, as for every other connection.

Synchronized sequential scans are turned off for the session, as in the project's scoring:
otherwise a scan of a large table starts where the previous one stopped, and the row order of a
query without a complete ORDER BY (which Soft-F1 depends on) changes with the queries run
before it. `parallel_plans=False` also turns parallel query off, as the project's own execution
does: with parallel workers, floating-point sums and averages add their terms in a different
order on each run. A server-side statement timeout a little above the evaluator's
own time limit stops a query that `func_timeout` gave up on (it abandons the thread, not the
query).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from src.db.connection import AGENT_ROLE, BIRD_DB, conninfo
from src.db.hardening import benchmark_schemas

MODULES = ("evaluation_utils", "evaluation_ex", "evaluation_f1")
STATEMENT_TIMEOUT_MARGIN_S = 5


def load(directory: Path) -> SimpleNamespace:
    """The evaluator's modules: `.utils`, `.ex`, `.f1`."""
    if not all((directory / f"{m}.py").exists() for m in MODULES):
        raise FileNotFoundError(f"the official evaluator is not in {directory}")
    # no __pycache__ in the downloaded directory: DVC tracks it, and would see it change
    write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(directory))
    try:
        mods: dict[str, ModuleType] = {m: importlib.import_module(m) for m in MODULES}
    finally:
        sys.path.remove(str(directory))
        sys.dont_write_bytecode = write_bytecode
    return SimpleNamespace(
        utils=mods["evaluation_utils"], ex=mods["evaluation_ex"], f1=mods["evaluation_f1"]
    )


def connect_postgresql_to_project(timeout_s: float, parallel_plans: bool):
    """A replacement for the evaluator's `connect_postgresql`."""
    import psycopg2

    settings = [
        f"search_path={','.join(benchmark_schemas())}",
        f"statement_timeout={int((timeout_s + STATEMENT_TIMEOUT_MARGIN_S) * 1000)}",
        "synchronize_seqscans=off",
    ]
    if not parallel_plans:
        settings.append("max_parallel_workers_per_gather=0")
    options = " ".join(f"-c {s}" for s in settings)

    def connect_postgresql() -> Any:
        return psycopg2.connect(conninfo(AGENT_ROLE, BIRD_DB), options=options)

    return connect_postgresql


def use_project_database(official: SimpleNamespace, timeout_s: float, parallel_plans: bool):
    # evaluation_utils.execute_sql looks connect_postgresql up in its module at call time
    official.utils.connect_postgresql = connect_postgresql_to_project(timeout_s, parallel_plans)


def execute_pair(
    official: SimpleNamespace, metric: str, predicted_sql: str, gold_sql: str, timeout_s: float
) -> float:
    """One pair through the evaluator's own per-question function (its time limit, its
    exception handling, its comparison): EX (0 or 1) or Soft-F1."""
    module = official.ex if metric == "ex" else official.f1
    return module.execute_model(predicted_sql, gold_sql, "", 0, timeout_s, "PostgreSQL")["res"]
