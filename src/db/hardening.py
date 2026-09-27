"""Database-side hardening beyond the agent's role: closed system functions and views, and
one role per schema that confines a query to its own database.

**Closed functions and views.** src/db/hardening.sql revokes them from PUBLIC. The lists here
are written independently of the SQL file's: the audit selects every system function whose
name matches, including overloads the SQL might have missed, and asks PostgreSQL whether the
agent's roles can execute it. `has_function_privilege` and `has_table_privilege` see
privileges held through PUBLIC as well as direct grants.

**Schema roles.** The login role `analyst_ro` can read every benchmark schema (the gold SQL
runs as it). The agent's queries do not run with its privileges: each transaction switches to
`analyst_ro_<schema>` (`SET LOCAL ROLE`, src/db/execute.py), a role that cannot log in and can
read that one schema and nothing else. `analyst_ro` is a member of each schema role with SET
but without INHERIT, so it gains nothing from them until it switches. A query cannot switch
back: `SET` and `RESET` are statements, which the executor's cursor refuses, and
`set_config('role', ...)` is among the closed functions.
"""

from __future__ import annotations

from psycopg import sql

from src.db.connection import ADMIN_ROLE, AGENT_ROLE, BIRD_DB, DB_NAME, connect
from src.dictionary.snapshot import SNAPSHOT_DIR

# LIKE patterns (backslash escapes the underscore) and exact names, in pg_catalog
CLOSED_FUNCTION_PATTERNS = (
    r"lo\_%",
    r"pg\_sleep%",
    r"pg\_advisory%",
    r"pg\_try\_advisory%",
    r"%\_to\_xml%",
    r"inet\_server\_%",
    r"inet\_client\_%",
)
CLOSED_FUNCTIONS = (
    "loread",
    "lowrite",
    "set_config",
    "current_setting",
    "pg_notify",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "version",
    "pg_postmaster_start_time",
    "pg_conf_load_time",
    "pg_stat_get_activity",
    "pg_show_all_settings",
)
CLOSED_VIEWS = (
    "pg_settings",
    "pg_stat_activity",
    "pg_stat_ssl",
    "pg_stat_gssapi",
    "pg_roles",
    "pg_user",
    "pg_group",
    "pg_auth_members",
    "pg_db_role_setting",
)
# The security suite's schema in the `analyst` database, and its tables:
# (table, readable by the schema's role)
SECURITY_SCHEMA = "security_check"
FIXTURES = (
    ("security_check.secret", False),
    ("security_check.notes", True),
    ("security_check.numbers", True),
)


def schema_role(schema: str) -> str:
    return f"{AGENT_ROLE}_{schema}"


def benchmark_schemas() -> list[str]:
    """The BIRD databases, one schema each, as listed by the committed schema snapshots."""
    return sorted(p.stem for p in SNAPSHOT_DIR.glob("*.json"))


def schema_roles(dbname: str) -> dict[str, str]:
    """schema -> its role, for the schemas the agent queries in a database."""
    schemas = benchmark_schemas() if dbname == BIRD_DB else [SECURITY_SCHEMA]
    return {s: schema_role(s) for s in schemas}


def ensure_schema_roles(dbname: str) -> None:
    """Create each schema's role if missing, and grant it that schema and nothing else.

    Roles are shared by all databases of the server; their privileges are per database. A
    table the schema role must not read (the suite's secret) has its grant revoked again.
    """
    with connect(ADMIN_ROLE, dbname, autocommit=True) as conn:
        for schema, role in schema_roles(dbname).items():
            r, s = sql.Identifier(role), sql.Identifier(schema)
            if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone():
                conn.execute(
                    sql.SQL(
                        "CREATE ROLE {} NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT "
                        "NOREPLICATION NOBYPASSRLS"
                    ).format(r)
                )
            conn.execute(
                sql.SQL("GRANT {} TO {} WITH INHERIT FALSE, SET TRUE").format(
                    r, sql.Identifier(AGENT_ROLE)
                )
            )
            conn.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(s, r))
            if dbname == BIRD_DB:
                conn.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(s, r))
            else:
                for table, readable in FIXTURES:
                    verb = "GRANT SELECT ON {} TO {}" if readable else "REVOKE ALL ON {} FROM {}"
                    conn.execute(sql.SQL(verb).format(sql.Identifier(*table.split(".")), r))


def audit(dbname: str) -> dict:
    roles = schema_roles(dbname) if dbname in (DB_NAME, BIRD_DB) else {}
    who = [AGENT_ROLE, *roles.values()]
    with connect(ADMIN_ROLE, dbname) as conn:
        functions = conn.execute(
            """
            SELECT p.oid::regprocedure::text,
                   bool_or(has_function_privilege(r, p.oid, 'EXECUTE'))
            FROM pg_proc p, unnest(%s::text[]) AS r
            WHERE p.pronamespace = 'pg_catalog'::regnamespace
              AND (p.proname LIKE ANY (%s) OR p.proname = ANY (%s))
              AND EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r)
            GROUP BY 1
            ORDER BY 1
            """,
            (who, list(CLOSED_FUNCTION_PATTERNS), list(CLOSED_FUNCTIONS)),
        ).fetchall()
        views = [
            (
                v,
                any(
                    conn.execute(
                        "SELECT has_table_privilege(%s, %s, 'SELECT')", (r, f"pg_catalog.{v}")
                    ).fetchone()[0]
                    for r in who
                    if _role_exists(conn, r)
                ),
            )
            for v in CLOSED_VIEWS
        ]
        result: dict = {
            "functions_closed": sum(not ok for _, ok in functions),
            "functions_open": [f for f, ok in functions if ok],
            "views_closed": sum(not ok for _, ok in views),
            "views_open": [v for v, ok in views if ok],
        }
        if roles:
            result["schema_roles"] = _audit_roles(conn, dbname, roles)
    return result


def _role_exists(conn, role: str) -> bool:
    return conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone() is not None


def _audit_roles(conn, dbname: str, roles: dict[str, str]) -> dict:
    """Each schema role reads its own schema's tables, and no other schema in the database."""
    schemas = [
        s
        for (s,) in conn.execute(
            "SELECT nspname FROM pg_namespace WHERE nspname NOT LIKE 'pg\\_%' "
            "AND nspname <> 'information_schema' ORDER BY 1"
        ).fetchall()
    ]
    problems: list[str] = []
    for schema, role in roles.items():
        attrs = conn.execute(
            "SELECT rolcanlogin, rolsuper, rolinherit, rolcreaterole, rolcreatedb, rolbypassrls "
            "FROM pg_roles WHERE rolname = %s",
            (role,),
        ).fetchone()
        if attrs is None:
            problems.append(f"{role}: missing")
            continue
        if any(attrs):
            problems.append(f"{role}: has attributes {attrs}")
        member = conn.execute(
            """
            SELECT m.inherit_option, m.set_option FROM pg_auth_members m
            WHERE m.roleid = %s::regrole AND m.member = %s::regrole
            """,
            (role, AGENT_ROLE),
        ).fetchone()
        if member != (False, True):
            problems.append(f"{AGENT_ROLE} in {role}: (inherit, set) = {member}")
        for other in schemas:
            usage = conn.execute(
                "SELECT has_schema_privilege(%s, %s, 'USAGE')", (role, other)
            ).fetchone()[0]
            if usage != (other == schema):
                problems.append(f"{role}: USAGE on {other} is {usage}")
        tables = conn.execute(
            """
            SELECT c.oid::regclass::text, has_table_privilege(%s, c.oid, 'SELECT'),
                   has_table_privilege(
                       %s, c.oid, 'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'
                   )
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relkind IN ('r', 'v', 'm', 'p')
            """,
            (role, role, schema),
        ).fetchall()
        fixtures = dict(FIXTURES)
        for table, can_select, can_write in tables:
            want = fixtures.get(table, True) if dbname == DB_NAME else True
            if can_select != want or can_write:
                problems.append(f"{role}: {table} select={can_select} write={can_write}")
    return {"roles": len(roles), "problems": problems}
