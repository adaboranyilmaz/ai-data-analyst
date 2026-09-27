"""The database hardening is in place (scripts/20_harden_db.py): system functions and views
closed to the agent's roles, and each schema role confined to its own schema."""

from __future__ import annotations

import pytest
from psycopg import errors

from src.db.connection import AGENT_ROLE, BIRD_DB, DB_NAME, connect
from src.db.hardening import audit, benchmark_schemas, schema_role

pytestmark = pytest.mark.db

HINT = "run scripts/20_harden_db.py"


def test_analyst_database_is_hardened(db_ready):
    a = audit(DB_NAME)
    assert a["functions_closed"] > 0 and not a["functions_open"], HINT
    assert a["views_closed"] > 0 and not a["views_open"], HINT
    assert a["schema_roles"] == {"roles": 1, "problems": []}, HINT


def test_template1_is_hardened(db_ready):
    """Databases created later (a reload of the benchmark) start hardened."""
    a = audit("template1")
    assert not a["functions_open"] and not a["views_open"], HINT


@pytest.mark.bird
def test_bird_database_is_hardened(bird_ready):
    a = audit(BIRD_DB)
    assert not a["functions_open"] and not a["views_open"], HINT
    assert a["schema_roles"] == {"roles": len(benchmark_schemas()), "problems": []}, HINT


def test_benchmark_schemas_are_the_eleven_databases():
    assert len(benchmark_schemas()) == 11 and "financial" in benchmark_schemas()


@pytest.mark.parametrize(
    "sql",
    ["SELECT lo_create(0)", "SELECT lo_from_bytea(0, 'x')", "SELECT lo_creat(-1)"],
)
def test_large_object_writes_fail_on_privileges_alone(db_ready, sql):
    """Large objects are not tables: before hardening, PUBLIC could create them with no
    privilege on any table, so only the read-only transaction stopped these writes."""
    with connect(AGENT_ROLE, autocommit=True) as conn:
        conn.execute("SET default_transaction_read_only = off")
        with pytest.raises(errors.InsufficientPrivilege):
            conn.execute(sql)


def test_the_login_role_does_not_inherit_the_schema_roles(db_ready):
    """analyst_ro reaches a schema role's privileges only by switching to it."""
    with connect(AGENT_ROLE, autocommit=True) as conn:
        with pytest.raises(errors.InsufficientPrivilege):
            conn.execute("SELECT count(*) FROM security_check.numbers")
        conn.execute(f"SET ROLE {schema_role('security_check')}")
        assert conn.execute("SELECT count(*) FROM security_check.numbers").fetchone()[0] == 10_000
