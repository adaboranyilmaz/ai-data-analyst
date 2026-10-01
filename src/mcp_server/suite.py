"""The security suite through the MCP path.

The Phase 2 attacks (configs/security_attacks.yaml) are sent as `run_sql` calls from a real MCP
client to a real server process over stdio, and each is followed by the same checks of its
effects: the canary table, relations, large objects, grants, advisory locks, lingering queries,
the secret's token in any result, and a probe on the same session that must still see the
target's time zone and row counts. Then attacks that exist only at this boundary: arguments of
the wrong type, table names carrying SQL, out-of-range counts, an unknown tool, a huge search
term.

Here the whole stack stands between the client and the data, so an attack counts as blocked if
the server refused it (`rejected`: the guard, or the tool's own argument check), PostgreSQL did
(`denied`), its limits stopped it (`contained`) or it ran harmlessly (`no_effect`). The layers
are tested independently by the Phase 2 suite; this one shows the path to them is not a way
around them.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from src.db import security_suite as ps
from src.db.connection import DB_NAME
from src.db.execute import ReadOnlyExecutor, Target
from src.db.guard import QueryGuard
from src.db.hardening import SECURITY_SCHEMA, schema_role
from src.serving.connections import CatalogSchema, ConnectionSpec, Validated
from src.tools.sql import SqlTools
from src.tools.toolbox import Toolbox, agent_limits, config

ROOT = Path(__file__).resolve().parent.parent.parent
BLOCKED = ps.BLOCKED

_TABLES = {
    "notes": [{"name": "id", "type": "integer"}, {"name": "note", "type": "text"}],
    "numbers": [{"name": "n", "type": "integer"}],
}


class SuiteToolbox(Toolbox):
    """The suite's fixture schema, served exactly as the product serves a database: the schema's
    own role, the same guard, the same executor and limits."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or config()
        self.db = SECURITY_SCHEMA
        self.cfg = cfg
        dummy = ConnectionSpec("fixture", 0, DB_NAME, "fixture", "", SECURITY_SCHEMA)
        self.schema = CatalogSchema(Validated(dummy, _TABLES, {"notes": 3, "numbers": 10_000}, ()))
        self.executor = ReadOnlyExecutor(
            Target(DB_NAME, SECURITY_SCHEMA, "UTC", schema_role(SECURITY_SCHEMA))
        )
        self.sql = SqlTools(
            QueryGuard(SECURITY_SCHEMA, set(_TABLES)),
            self.executor,
            agent_limits(cfg),
            cfg["agent"]["max_cell_chars"],
            cfg["sample_rows"]["default_rows"],
            cfg["sample_rows"]["max_rows"],
        )


def server_params(target: str) -> StdioServerParameters:
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    module = "src.mcp_server.suite_server" if target == "analyst" else "src.mcp_server"
    return StdioServerParameters(
        command=sys.executable, args=["-m", module], env=env, cwd=str(ROOT)
    )


def payload(result: Any) -> dict[str, Any]:
    """A tool result as data; a protocol-level error becomes {"ok": False, "protocol_error"}."""
    if result.is_error:
        text = " ".join(getattr(c, "text", "") for c in result.content)
        return {"ok": False, "protocol_error": text[:300]}
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


def classify(attack: dict, out: dict[str, Any], seconds: float, limit_s: float) -> tuple[str, str]:
    """The outcome of one call, by what the server returned."""
    if seconds > limit_s + ps.STOP_MARGIN_S:
        return "breach", f"ran {seconds:.1f} s, beyond the {limit_s:g} s limit"
    if "protocol_error" in out:
        return "rejected", "the server's argument check: " + out["protocol_error"]
    if not out.get("ok"):
        err = out.get("error") or {}
        kind = err.get("kind", "error")
        why = err.get("message") or "; ".join(err.get("reasons") or []) or kind
        if kind == "refused":
            return "rejected", why
        if kind in ps.CONTAINED_KINDS:
            return "contained", f"{kind}: {why}"
        return "denied", f"{kind}: {why}"
    goal = attack["goal"]
    rows = len(out.get("rows") or [])
    if goal == "resource":
        return "contained", f"finished within the limits: {rows} rows in {seconds:.2f} s"
    if goal == "write":
        return "no_effect", f"ran as a harmless query ({rows} rows); nothing changed"
    return "breach", f"succeeded: {rows} rows"


@dataclass
class Probe:
    table: str
    rows: int
    utc_offset_hours: int


async def _probe(client: Client, p: Probe) -> list[str]:
    zone = payload(
        await client.call_tool(
            "run_sql",
            {"sql": "SELECT extract(hour FROM timestamptz '2000-01-01 00:00:00+00')::int"},
        )
    )
    count = payload(await client.call_tool("run_sql", {"sql": f"SELECT count(*) FROM {p.table}"}))
    got = (
        zone["rows"][0][0] if zone.get("ok") and zone.get("rows") else None,
        count["rows"][0][0] if count.get("ok") and count.get("rows") else None,
    )
    want = (p.utc_offset_hours, p.rows)
    return [] if got == want else [f"probe saw {got}, expected {want}"]


async def run_attack(client: Client, attack: dict, dbnames: tuple, probe: Probe, limit_s: float):
    before = ps._state(dbnames)
    start = time.perf_counter()
    try:
        out = payload(await client.call_tool("run_sql", {"sql": attack["sql"]}))
    except Exception as e:  # a transport failure is itself a finding: recorded, not hidden
        out = {"ok": False, "protocol_error": f"{type(e).__name__}: {e}"[:300]}
    seconds = time.perf_counter() - start
    outcome, detail = classify(attack, out, seconds, limit_s)
    running = ps._running_agent_queries(ps.STOP_MARGIN_S)
    text = json.dumps(out.get("rows") or [], default=str)
    problems = ps._effects(before, ps._state(dbnames), text, running)
    problems += await _probe(client, probe)
    if problems:
        outcome, detail = "breach", "; ".join(problems) + f" (was: {outcome}: {detail})"
    return {"outcome": outcome, "detail": detail, "seconds": round(seconds, 3)}


# attacks that exist only at the tool boundary: (id, tool, arguments, expected)
BOUNDARY_ATTACKS: list[tuple[str, str, dict[str, Any], str]] = [
    ("arg-sql-null", "run_sql", {"sql": None}, "refused"),
    ("arg-sql-integer", "run_sql", {"sql": 7}, "refused"),
    ("arg-sql-list", "run_sql", {"sql": ["SELECT 1", "DROP TABLE x"]}, "refused"),
    ("arg-sql-object", "run_sql", {"sql": {"$ne": ""}}, "refused"),
    ("arg-sql-missing", "run_sql", {}, "refused"),
    ("arg-sql-extra-field", "run_sql", {"sql": "SELECT 1", "role": "postgres"}, "harmless"),
    ("arg-sql-huge", "run_sql", {"sql": "SELECT '" + "a" * 2_000_000 + "'"}, "refused"),
    ("arg-table-sql", "describe_table", {"table": "notes; DROP TABLE notes"}, "refused"),
    ("arg-table-path", "describe_table", {"table": "../../etc/passwd"}, "refused"),
    ("arg-table-schema", "describe_table", {"table": "role_check.canary"}, "refused"),
    ("arg-table-quote", "sample_rows", {"table": 'notes" ; DROP TABLE notes; --'}, "refused"),
    ("arg-table-other-schema", "sample_rows", {"table": "role_check.canary"}, "refused"),
    ("arg-n-negative", "sample_rows", {"table": "notes", "n": -1}, "harmless"),
    ("arg-n-huge", "sample_rows", {"table": "numbers", "n": 10**9}, "harmless"),
    ("arg-n-string", "sample_rows", {"table": "notes", "n": "5; DROP TABLE notes"}, "refused"),
    ("arg-term-huge", "lookup_dictionary", {"term": "a" * 1_000_000}, "harmless"),
    ("arg-term-sql", "lookup_dictionary", {"term": "'; DROP TABLE notes; --"}, "harmless"),
    ("arg-unknown-tool", "run_python", {"code": "import os"}, "refused"),
    ("arg-unknown-tool-sql", "execute_sql", {"sql": "DROP TABLE notes"}, "refused"),
]


async def run_boundary(client: Client, dbnames: tuple, probe: Probe) -> list[dict[str, Any]]:
    out = []
    for attack_id, tool, args, expected in BOUNDARY_ATTACKS:
        before = ps._state(dbnames)
        start = time.perf_counter()
        try:
            result = payload(await client.call_tool(tool, args))
        except Exception as e:  # e.g. an unknown tool: the server refuses the call
            result = {"ok": False, "protocol_error": f"{type(e).__name__}: {e}"[:300]}
        seconds = time.perf_counter() - start
        refused = not result.get("ok") or "protocol_error" in result
        text = json.dumps(result, default=str)
        problems = ps._effects(before, ps._state(dbnames), text, ps._running_agent_queries(1.0))
        problems += await _probe(client, probe)
        if problems:
            outcome = "breach"
        elif refused:
            outcome = "rejected"
        elif expected == "harmless":
            outcome = "no_effect"
        else:
            outcome = "breach"
            problems.append("the call succeeded though it should have been refused")
        out.append(
            {
                "id": attack_id,
                "tool": tool,
                "expected": expected,
                "outcome": outcome,
                "detail": "; ".join(problems) or text[:160],
                "seconds": round(seconds, 3),
            }
        )
    return out


async def injection_row(client: Client) -> dict[str, Any]:
    """The planted instruction comes back as one value among the rows."""
    out = payload(
        await client.call_tool("run_sql", {"sql": "SELECT id, note FROM notes ORDER BY id"})
    )
    stored = (
        ps.connect(ps.ADMIN_ROLE, DB_NAME)
        .execute("SELECT note FROM security_check.notes WHERE id = 2")
        .fetchone()[0]
    )
    return {
        "returned_as_a_row_value": any(row == [2, stored] for row in out.get("rows", [])),
        "tool_fields": sorted(out),
    }


async def _run(include_bird: bool) -> dict[str, Any]:
    settings, attacks = ps.load_attacks()
    limit_s = float(config()["agent"]["timeout_s"])
    sts = ps.targets(include_bird)
    records: list[dict] = []
    boundary: list[dict] = []
    injection: dict[str, Any] = {}
    for name, st in sts.items():
        dbnames = tuple(dict.fromkeys((DB_NAME, st.target.dbname)))
        probe = Probe(st.probe_table, st.probe_rows, st.utc_offset_hours)
        async with Client(server_params(name)) as client:
            for attack in attacks:
                if attack.get("target", "analyst") != name:
                    continue
                result = await run_attack(client, attack, dbnames, probe, limit_s)
                records.append(
                    {
                        "id": attack["id"],
                        "category": attack["category"],
                        "goal": attack["goal"],
                        "target": f"{st.target.dbname}.{st.target.schema}",
                        "sql": attack["sql"],
                        **result,
                    }
                )
            if name == "analyst":
                boundary = await run_boundary(client, dbnames, probe)
                injection = await injection_row(client)
    outcomes = {
        o: sum(r["outcome"] == o for r in records) for o in sorted({r["outcome"] for r in records})
    }
    b_outcomes = {
        o: sum(r["outcome"] == o for r in boundary)
        for o in sorted({r["outcome"] for r in boundary})
    }
    blocked = all(r["outcome"] in BLOCKED for r in records) and all(
        r["outcome"] in BLOCKED for r in boundary
    )
    return {
        "note": "the Phase 2 attacks sent as run_sql calls from an MCP client to a server process "
        "over stdio, plus attacks at the tool boundary; one run",
        "settings": {"limit_s": limit_s, "stop_margin_s": ps.STOP_MARGIN_S},
        "attacks": len(records),
        "boundary_attacks": len(boundary),
        "skipped_without_bird": [a["id"] for a in attacks if a.get("target", "analyst") not in sts],
        "outcomes": outcomes,
        "boundary_outcomes": b_outcomes,
        "breaches": outcomes.get("breach", 0) + b_outcomes.get("breach", 0),
        "every_attack_blocked": blocked,
        "injection_row": injection,
        "records": records,
        "boundary": boundary,
    }


def run_suite(include_bird: bool | None = None) -> dict[str, Any]:
    include_bird = ps.bird_available() if include_bird is None else include_bird
    return asyncio.run(_run(include_bird))
