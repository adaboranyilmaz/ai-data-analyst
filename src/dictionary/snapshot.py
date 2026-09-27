"""A database's schema, and for profiled databases its column values, read from PostgreSQL.

The same function writes the committed snapshot (scripts/13_snapshot_schema.py) and reads
the live database in the test that checks the snapshot is still true, so the two cannot
drift apart in how they measure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parent.parent.parent
SNAPSHOT_DIR = ROOT / "dictionary/_snapshot"
MAX_VALUES = 50  # text columns with at most this many distinct values list them all
TEXT_TYPES = {"text", "character varying", "character"}
RANGE_TYPES = {"bigint", "integer", "smallint", "real", "double precision", "numeric", "date"}


def columns(conn: psycopg.Connection, schema: str) -> dict[str, list[dict[str, Any]]]:
    rows = conn.execute(
        "SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod), NOT a.attnotnull "
        "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = %s AND c.relkind = 'r' AND a.attnum > 0 AND NOT a.attisdropped "
        "ORDER BY c.relname, a.attnum",
        [schema],
    ).fetchall()
    out: dict[str, list[dict[str, Any]]] = {}
    for table, column, type_, nullable in rows:
        out.setdefault(table, []).append({"name": column, "type": type_, "nullable": nullable})
    return out


def _json(v: Any) -> Any:
    return v.isoformat() if hasattr(v, "isoformat") else v


def profile_column(conn: psycopg.Connection, schema: str, table: str, col: dict[str, Any]) -> dict:
    t = sql.Identifier(schema, table)
    c = sql.Identifier(col["name"])
    n_null, n_distinct = conn.execute(
        sql.SQL("SELECT count(*) - count({c}), count(DISTINCT {c}) FROM {t}").format(c=c, t=t)
    ).fetchone()
    out: dict[str, Any] = {"nulls": n_null, "distinct": n_distinct}
    if col["type"] in TEXT_TYPES and n_distinct <= MAX_VALUES:
        rows = conn.execute(
            sql.SQL(
                "SELECT {c}, count(*) FROM {t} WHERE {c} IS NOT NULL GROUP BY {c} ORDER BY {c}"
            ).format(c=c, t=t)
        ).fetchall()
        out["values"] = {v: n for v, n in rows}
    elif col["type"] in RANGE_TYPES:
        lo, hi = conn.execute(
            sql.SQL("SELECT min({c}), max({c}) FROM {t}").format(c=c, t=t)
        ).fetchone()
        out["min"], out["max"] = _json(lo), _json(hi)
    return out


def snapshot_database(conn: psycopg.Connection, schema: str, profile: bool) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for table, cols in columns(conn, schema).items():
        n = conn.execute(
            sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(schema, table))
        ).fetchone()[0]
        if profile:
            cols = [{**col, **profile_column(conn, schema, table, col)} for col in cols]
        tables[table] = {"rows": n, "columns": cols}
    return {"database": schema, "profiled": profile, "tables": tables}
