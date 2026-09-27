"""The security suite: every attack in configs/security_attacks.yaml, against each layer alone.

Configurations (the attack file explains them): `parser` (the query guard's verdict; a
resource attack it accepts runs as the admin role, with no database layer, under the
execution limits), `database` (the guard bypassed; the agent's role through the read-only
executor) and `privileges` (as `database`, without the read-only transaction).

Outcomes. Blocked: `rejected` (the guard refused it), `denied` (PostgreSQL refused it),
`contained` (stopped by, or finished within, the time and row limits: resource attacks only)
and `no_effect` (it ran, and did nothing its goal needed: e.g. the statement a comment hid never
ran). Not blocked: `breach`. `disclosed` is the documented exception: object names listed in
the database-only configurations.

After every attack, whatever its outcome, its effects are checked: the canary table, the
relations in the schemas the agent can see, large objects, grants on the secret table,
advisory locks, agent queries still running on the server, the secret's token anywhere in
the result, and (in the database configurations) a probe on the same session, which must see
the target's time zone, the agent's role and its own schema. Any effect is a breach.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import psycopg
import yaml

from src.data import bird
from src.db.connection import ADMIN_ROLE, AGENT_ROLE, BIRD_DB, DB_NAME, connect
from src.db.execute import (
    APPLICATION_NAME,
    Limits,
    QueryError,
    QueryResult,
    ReadOnlyExecutor,
    Target,
)
from src.db.guard import QueryGuard
from src.db.hardening import SECURITY_SCHEMA, schema_role

ROOT = Path(__file__).resolve().parent.parent.parent
ATTACKS_FILE = ROOT / "configs/security_attacks.yaml"
MODES = ("parser", "database", "privileges")
BLOCKED = frozenset({"rejected", "denied", "contained", "no_effect"})
DOCUMENTED_EXCEPTION = "disclosed"
SECRET_TOKEN = "SECRET-TOKEN-4d1f9c"  # src/db/security_fixtures.sql
# how long a cancelled query may take to stop, beyond the time limit
STOP_MARGIN_S = 2.0
CONTAINED_KINDS = frozenset({"timeout", "resource_limit"})


@dataclass(frozen=True)
class SuiteTarget:
    target: Target
    tables: frozenset[str]
    probe_table: str
    probe_rows: int
    utc_offset_hours: int


def targets(include_bird: bool) -> dict[str, SuiteTarget]:
    out = {
        "analyst": SuiteTarget(
            Target(DB_NAME, SECURITY_SCHEMA, "UTC", schema_role(SECURITY_SCHEMA)),
            frozenset({"notes", "numbers"}),
            "numbers",
            10_000,
            0,
        )
    }
    if include_bird:
        zone = bird.config()["bird_minidev"]["time_zone"]
        guard = QueryGuard.for_benchmark_db("financial")
        out["bird"] = SuiteTarget(
            Target(BIRD_DB, "financial", zone, schema_role("financial")),
            guard.tables,
            "district",
            77,
            8,
        )
    return out


def load_attacks(path: Path = ATTACKS_FILE) -> tuple[dict, list[dict]]:
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    return d["settings"], d["attacks"]


def bird_available() -> bool:
    try:
        with connect(ADMIN_ROLE, BIRD_DB) as conn:
            return bool(conn.execute("SELECT to_regnamespace('financial')").fetchone()[0])
    except psycopg.OperationalError:
        return False


def _state(dbnames: tuple[str, ...]) -> dict[str, Any]:
    """What an attack could change, read as the admin role."""
    state: dict[str, Any] = {}
    with connect(ADMIN_ROLE, DB_NAME) as conn:
        state["canary"] = conn.execute(
            "SELECT id, note FROM role_check.canary ORDER BY id"
        ).fetchall()
        state["secret_grant"] = [
            conn.execute(
                "SELECT has_table_privilege(%s, 'security_check.secret', 'SELECT')", (role,)
            ).fetchone()[0]
            for role in (AGENT_ROLE, schema_role(SECURITY_SCHEMA))
        ]
        state["advisory_locks"] = conn.execute(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory'"
        ).fetchone()[0]
    for db in dbnames:
        with connect(ADMIN_ROLE, db) as conn:
            state[f"{db}.relations"] = conn.execute(
                r"""
                SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
                  AND n.nspname NOT LIKE 'pg\_temp%' AND n.nspname NOT LIKE 'pg\_toast\_temp%'
                """
            ).fetchone()[0]
            state[f"{db}.temp_relations"] = conn.execute(
                "SELECT count(*) FROM pg_class WHERE relpersistence = 't'"
            ).fetchone()[0]
            state[f"{db}.large_objects"] = conn.execute(
                "SELECT count(*) FROM pg_largeobject_metadata"
            ).fetchone()[0]
    return state


def _running_agent_queries(timeout_s: float) -> int:
    """Agent queries still active on the server, after waiting up to `timeout_s` for them."""
    deadline = time.monotonic() + timeout_s
    with connect(ADMIN_ROLE, DB_NAME, autocommit=True) as conn:
        while True:
            n = conn.execute(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE application_name = %s AND state = 'active'",
                (APPLICATION_NAME,),
            ).fetchone()[0]
            if n == 0 or time.monotonic() > deadline:
                return n
            time.sleep(0.1)


def _probe(executor: ReadOnlyExecutor, st: SuiteTarget, limits: Limits) -> list[str]:
    """The next query on the attacked session sees the settings every query must see."""
    try:
        r = executor.execute(
            "SELECT extract(hour FROM timestamptz '2000-01-01 00:00:00+00')::int, "
            f"current_user::text, (SELECT count(*) FROM {st.probe_table})",
            limits,
        )
    except QueryError as e:
        return [f"probe failed: {e}"]
    want = (st.utc_offset_hours, st.target.role, st.probe_rows)
    got = tuple(r.rows[0]) if r.rows else None
    return [] if got == want else [f"probe saw {got}, expected {want}"]


def _effects(before: dict, after: dict, result_text: str, running: int) -> list[str]:
    problems = [
        f"{k} changed: {before[k]!r} -> {after[k]!r}" for k in before if before[k] != after[k]
    ]
    if SECRET_TOKEN in result_text:
        problems.append("the secret token is in the result")
    if running:
        problems.append(f"{running} agent quer{'y' if running == 1 else 'ies'} still running")
    return problems


def _result_text(r: QueryResult | None) -> str:
    return "" if r is None else json.dumps([list(map(str, row)) for row in r.rows])


def _outcome_of_execution(
    attack: dict,
    result: QueryResult | None,
    error: QueryError | None,
    seconds: float,
    limit_s: float,
) -> tuple[str, str]:
    goal = attack["goal"]
    if seconds > limit_s + STOP_MARGIN_S:
        return "breach", f"ran {seconds:.1f} s, beyond the {limit_s:g} s limit"
    if error is not None:
        if error.kind in CONTAINED_KINDS:
            return "contained", f"{error.kind}: {error.message}"
        return "denied", f"{error.kind}: {error.message}"
    assert result is not None
    if goal == "resource":
        how = "row limit" if result.truncated else "finished within the limits"
        return "contained", f"{how}: {len(result.rows)} rows in {seconds:.2f} s"
    if goal == "write":
        return "no_effect", f"ran as a harmless query ({len(result.rows)} rows); nothing changed"
    if goal == "names":
        return DOCUMENTED_EXCEPTION, f"returned {len(result.rows)} rows of object names"
    return "breach", f"succeeded: {len(result.rows)} rows"


def run_attack(
    attack: dict,
    mode: str,
    st: SuiteTarget,
    limits: Limits,
    executors: dict[str, ReadOnlyExecutor],
) -> dict:
    dbnames = tuple(dict.fromkeys((DB_NAME, st.target.dbname)))
    before = _state(dbnames)
    start = time.perf_counter()
    result: QueryResult | None = None
    error: QueryError | None = None
    executed = mode != "parser"
    if mode == "parser":
        verdict = QueryGuard(st.target.schema, st.tables).check(attack["sql"])
        if not verdict.allowed:
            outcome, detail = "rejected", "; ".join(verdict.reasons)
        elif attack["goal"] == "resource":
            executed = True
        elif attack["goal"] == "names":
            outcome, detail = DOCUMENTED_EXCEPTION, "accepted by the guard"
        else:
            outcome, detail = "breach", "accepted by the guard"
    if executed:
        try:
            result = executors[mode].execute(attack["sql"], limits)
        except QueryError as e:
            error = e
        except (psycopg.Error, ValueError) as e:  # e.g. a NUL byte, refused by the driver
            error = QueryError("driver", str(e).strip().splitlines()[0])
        seconds = time.perf_counter() - start
        outcome, detail = _outcome_of_execution(attack, result, error, seconds, limits.timeout_s)
    seconds = time.perf_counter() - start
    running = _running_agent_queries(STOP_MARGIN_S)
    problems = _effects(before, _state(dbnames), _result_text(result), running)
    if mode != "parser":
        problems += _probe(executors[mode], st, limits)
    if problems:
        outcome, detail = "breach", "; ".join(problems) + f" (was: {outcome}: {detail})"
    return {"outcome": outcome, "detail": detail, "seconds": round(seconds, 3)}


def injection_row_check(st: SuiteTarget, limits: Limits) -> dict:
    """The planted instruction comes back through run_sql as one value among the rows."""
    from src.tools.sql import SqlTools

    with ReadOnlyExecutor(st.target) as ex:
        tools = SqlTools(QueryGuard(st.target.schema, st.tables), ex, limits, 10_000)
        out = tools.run_sql("SELECT id, note FROM notes ORDER BY id")
    with connect(ADMIN_ROLE, DB_NAME) as conn:
        stored = conn.execute("SELECT note FROM security_check.notes WHERE id = 2").fetchone()[0]
    returned = out.get("rows", [])
    return {
        "returned_as_a_row_value": any(row == [2, stored] for row in returned),
        "tool_fields": sorted(out),
    }


def run_suite(include_bird: bool | None = None, attacks_file: Path = ATTACKS_FILE) -> dict:
    settings, attacks = load_attacks(attacks_file)
    include_bird = bird_available() if include_bird is None else include_bird
    sts = targets(include_bird)
    limits = Limits(max_rows=settings["max_rows"], timeout_s=settings["timeout_s"])

    records = []
    for target_name, st in sts.items():
        executors = {
            # no database layer: the admin role, its own privileges, no read-only transaction
            "parser": ReadOnlyExecutor(
                replace(st.target, role=None), ADMIN_ROLE, transaction_read_only=False
            ),
            "database": ReadOnlyExecutor(st.target),
            "privileges": ReadOnlyExecutor(st.target, transaction_read_only=False),
        }
        try:
            for attack in attacks:
                if attack.get("target", "analyst") != target_name:
                    continue
                modes = {m: run_attack(attack, m, st, limits, executors) for m in MODES}
                records.append(
                    {
                        "id": attack["id"],
                        "category": attack["category"],
                        "goal": attack["goal"],
                        "target": f"{st.target.dbname}.{st.target.schema}",
                        "sql": attack["sql"],
                        "modes": modes,
                    }
                )
        finally:
            for ex in executors.values():
                ex.close()
    skipped = [a["id"] for a in attacks if a.get("target", "analyst") not in sts]

    def ok(rec: dict, mode: str) -> bool:
        o = rec["modes"][mode]["outcome"]
        return o in BLOCKED or (
            o == DOCUMENTED_EXCEPTION and rec["goal"] == "names" and mode != "parser"
        )

    summary = {
        mode: {
            "attacks": len(records),
            "blocked": sum(r["modes"][mode]["outcome"] in BLOCKED for r in records),
            "documented_exception": sum(
                r["modes"][mode]["outcome"] == DOCUMENTED_EXCEPTION for r in records
            ),
            "breach": sum(r["modes"][mode]["outcome"] == "breach" for r in records),
            "outcomes": {
                o: sum(r["modes"][mode]["outcome"] == o for r in records)
                for o in sorted({r["modes"][mode]["outcome"] for r in records})
            },
        }
        for mode in MODES
    }
    return {
        "settings": {**settings, "stop_margin_s": STOP_MARGIN_S},
        "attacks": len(records),
        "skipped_without_bird": skipped,
        "categories": {
            c: sum(r["category"] == c for r in records)
            for c in dict.fromkeys(r["category"] for r in records)
        },
        "summary": summary,
        "every_attack_stopped_by_each_layer": all(ok(r, m) for r in records for m in MODES),
        "documented_exceptions": [
            {"id": r["id"], "modes": [m for m in MODES if r["modes"][m]["outcome"] == "disclosed"]}
            for r in records
            if any(r["modes"][m]["outcome"] == DOCUMENTED_EXCEPTION for m in MODES)
        ],
        "injection_row": injection_row_check(sts["analyst"], limits),
        "records": records,
    }
