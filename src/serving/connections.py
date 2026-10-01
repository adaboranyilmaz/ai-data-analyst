"""Pointing the analyst at a person's own PostgreSQL database (local mode only).

The analyst reaches the data through the same three layers as everywhere else: the query guard
(one `SELECT`, the schema's own tables), the read-only transaction and the role's privileges. A
role with the power to write defeats the third, so a connection is validated before it is used,
and refused if the role could change anything: a superuser, a role that can create roles,
databases or objects, one with INSERT, UPDATE, DELETE or TRUNCATE on any table, or a member of a
role that can write or run programs on the server. Nothing is created or granted for the person;
they are told what to change.

A person's database has no data dictionary, so the analyst reads its tables and columns from the
catalog, and the confidence is not calibrated for it: the calibration was measured on the
benchmark's databases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.conninfo import make_conninfo

from src.db.execute import APPLICATION_NAME, ReadOnlyExecutor, Target
from src.db.guard import QueryGuard
from src.tools.sql import SqlTools
from src.tools.toolbox import Toolbox, agent_limits, config

MAX_TABLES = 60
CONNECT_TIMEOUT_S = 5
SYSTEM_SCHEMAS = ("pg_catalog", "information_schema")
POWERFUL_ROLES = (
    "pg_write_server_files",
    "pg_execute_server_program",
    "pg_read_server_files",
    "pg_write_all_data",
)
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]{0,62}$")


class ConnectionRefused(Exception):
    """The connection cannot be used; `reasons` say why and what to change."""

    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@dataclass(frozen=True)
class ConnectionSpec:
    host: str
    port: int
    dbname: str
    user: str
    password: str = field(repr=False, default="")
    schema: str = "public"

    def conninfo(self) -> str:
        return make_conninfo(
            host=self.host,
            port=str(self.port),
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            connect_timeout=CONNECT_TIMEOUT_S,
            application_name=APPLICATION_NAME,
        )

    def public(self) -> dict[str, Any]:
        """What may be shown or logged: never the password."""
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "schema": self.schema,
        }


def parse(body: dict[str, Any]) -> ConnectionSpec:
    reasons = []
    for key in ("host", "dbname", "user"):
        if not isinstance(body.get(key), str) or not body[key].strip():
            reasons.append(f"{key} is required")
    port = body.get("port", 5432)
    if not isinstance(port, int) or isinstance(port, bool) or not 0 < port < 65536:
        reasons.append("port must be a number from 1 to 65535")
    schema = body.get("schema") or "public"
    if not isinstance(schema, str) or not _NAME.match(schema):
        reasons.append("schema must be a plain name (letters, digits, underscore)")
    password = body.get("password") or ""
    if not isinstance(password, str):
        reasons.append("password must be text")
    if reasons:
        raise ConnectionRefused(reasons)
    return ConnectionSpec(
        body["host"].strip(), port, body["dbname"].strip(), body["user"].strip(), password, schema
    )


# --------------------------------------------------------------------------- validation

_ROLE_FLAGS = """
SELECT rolsuper, rolcreaterole, rolcreatedb, rolreplication, rolbypassrls
FROM pg_roles WHERE rolname = current_user
"""
_WRITABLE_TABLES = """
SELECT n.nspname || '.' || c.relname
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND n.nspname NOT IN ('pg_catalog', 'information_schema') AND n.nspname NOT LIKE 'pg\\_toast%'
  AND (has_table_privilege(c.oid, 'INSERT') OR has_table_privilege(c.oid, 'UPDATE')
       OR has_table_privilege(c.oid, 'DELETE') OR has_table_privilege(c.oid, 'TRUNCATE'))
ORDER BY 1 LIMIT 5
"""
_CREATABLE_SCHEMAS = """
SELECT nspname FROM pg_namespace
WHERE nspname NOT IN ('pg_catalog', 'information_schema') AND nspname NOT LIKE 'pg\\_%'
  AND has_schema_privilege(oid, 'CREATE')
ORDER BY 1 LIMIT 5
"""
_TABLES = """
SELECT c.relname, c.reltuples::bigint
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = %s AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND has_table_privilege(c.oid, 'SELECT')
ORDER BY 1
"""
_COLUMNS = """
SELECT table_name, column_name, data_type, is_nullable
FROM information_schema.columns WHERE table_schema = %s
ORDER BY table_name, ordinal_position
"""


@dataclass(frozen=True)
class Validated:
    spec: ConnectionSpec
    tables: dict[str, list[dict[str, Any]]]  # table -> columns
    rows: dict[str, int]  # the planner's row estimates
    warnings: tuple[str, ...]


def validate(spec: ConnectionSpec) -> Validated:
    """Connect and check that the role can only read. Raises ConnectionRefused with reasons."""
    try:
        conn = psycopg.connect(spec.conninfo(), prepare_threshold=None)
    except psycopg.Error as e:
        text = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        raise ConnectionRefused([f"could not connect: {text}"]) from None
    try:
        conn.autocommit = True  # a refused catalog read must not poison the rest of the checks
        conn.read_only = True
        return _validate(conn, spec)
    finally:
        conn.close()


def _validate(conn: psycopg.Connection, spec: ConnectionSpec) -> Validated:
    reasons: list[str] = []
    warnings: list[str] = []
    try:
        flags = conn.execute(_ROLE_FLAGS).fetchone()
    except psycopg.errors.InsufficientPrivilege:  # a hardened server hides pg_roles
        superuser = conn.execute("SHOW is_superuser").fetchone()[0] == "on"
        flags = (superuser, False, False, False, False)
        warnings.append(
            "the role's attributes could not be read; only its privileges on tables, schemas "
            "and the database were checked"
        )
    names = ("a superuser", "able to create roles", "able to create databases")
    for flag, what in zip(flags[:3], names, strict=True):
        if flag:
            reasons.append(f"the role is {what}: use a role that can only read")
    if flags[3]:
        reasons.append("the role can start replication: use a role that can only read")
    if flags[4]:
        reasons.append("the role bypasses row-level security: use a role that can only read")
    for r in POWERFUL_ROLES:
        member = conn.execute("SELECT pg_has_role(current_user, %s, 'USAGE')", (r,)).fetchone()
        if member and member[0]:
            reasons.append(f"the role is a member of {r}, which can reach beyond the database")
    writable = [r[0] for r in conn.execute(_WRITABLE_TABLES).fetchall()]
    if writable:
        reasons.append(
            "the role can change data (INSERT, UPDATE, DELETE or TRUNCATE) on "
            + ", ".join(writable)
            + ": grant it SELECT only"
        )
    creatable = [r[0] for r in conn.execute(_CREATABLE_SCHEMAS).fetchall()]
    if creatable:
        reasons.append(
            "the role can create objects in the schema " + ", ".join(creatable) + ": revoke CREATE"
        )
    db_create = conn.execute("SELECT has_database_privilege(current_database(), 'CREATE')")
    if db_create.fetchone()[0]:
        reasons.append("the role can create schemas in this database: revoke CREATE on it")
    if reasons:
        raise ConnectionRefused(reasons)

    tables = conn.execute(_TABLES, (spec.schema,)).fetchall()
    if not tables:
        raise ConnectionRefused(
            [f"the role can read no table or view in the schema {spec.schema!r}"]
        )
    if len(tables) > MAX_TABLES:
        raise ConnectionRefused(
            [f"the schema has {len(tables)} readable tables; at most {MAX_TABLES} fit in a prompt"]
        )
    columns: dict[str, list[dict[str, Any]]] = {t: [] for t, _ in tables}
    for table, column, data_type, nullable in conn.execute(_COLUMNS, (spec.schema,)).fetchall():
        if table in columns:
            columns[table].append(
                {"name": column, "type": data_type, "nullable": nullable == "YES"}
            )
    if conn.execute("SELECT has_database_privilege(current_database(), 'TEMPORARY')").fetchone()[0]:
        warnings.append(
            "the role may create temporary tables; the read-only transaction and the query "
            "guard still refuse them"
        )
    return Validated(spec, columns, {t: max(int(n), 0) for t, n in tables}, tuple(warnings))


# --------------------------------------------------------------------------- the analyst's tools


class CatalogSchema:
    """`list_tables` and `describe_table` from the catalog, in the shape SchemaTools returns."""

    def __init__(self, v: Validated):
        self.db = v.spec.dbname
        self._columns = v.tables
        self._rows = v.rows

    @property
    def tables(self) -> list[str]:
        return sorted(self._columns)

    def list_tables(self) -> dict:
        return {
            "ok": True,
            "database": self.db,
            "tables": [{"table": t, "rows": self._rows[t]} for t in self.tables],
        }

    def describe_table(self, table: str) -> dict:
        if table not in self._columns:
            return {
                "ok": False,
                "error": {"kind": "refused", "reasons": [f"unknown table {table!r}"]},
            }
        return {
            "ok": True,
            "database": self.db,
            "table": table,
            "rows": self._rows[table],
            "columns": self._columns[table],
        }


class ConnectionExecutor(ReadOnlyExecutor):
    """The read-only executor, connected with the person's own details."""

    def __init__(self, spec: ConnectionSpec):
        super().__init__(Target(spec.dbname, spec.schema, "UTC", None))
        self.spec = spec

    def _connection(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed or self._conn.broken:
            self._conn = psycopg.connect(self.spec.conninfo(), prepare_threshold=None)
        self._conn.read_only = True
        return self._conn


class ConnectionToolbox(Toolbox):
    """The agent's tools over a validated connection. `db` is the schema, which is what the
    query guard and the agent's tool layer call the namespace they confine queries to."""

    def __init__(self, v: Validated, cfg: dict | None = None):
        cfg = cfg or config()
        self.db = v.spec.schema
        self.cfg = cfg
        self.schema = CatalogSchema(v)
        self.executor = ConnectionExecutor(v.spec)
        self.sql = SqlTools(
            QueryGuard(v.spec.schema, set(self.schema.tables)),
            self.executor,
            agent_limits(cfg),
            cfg["agent"]["max_cell_chars"],
            cfg["sample_rows"]["default_rows"],
            cfg["sample_rows"]["max_rows"],
        )
