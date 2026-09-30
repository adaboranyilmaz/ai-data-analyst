"""The agent's database role cannot write, create objects or lift its resource limits.

`default_transaction_read_only` is only a default: the role can switch it off in its own
session. So every write is tried twice, once in a default session (where the read-only
transaction may be what blocks it) and once after the role has switched the default off
(where only the privileges can block it). Both must fail for the role to count as read-only.
The canary table the writes target is created by docker/postgres/init/01_roles.sh.
"""

from __future__ import annotations

import psycopg
import pytest
from psycopg import errors

from src.db.connection import ADMIN_ROLE, AGENT_ROLE, connect

pytestmark = pytest.mark.db

CANARY = [(1, "must never change")]
WRITES = [
    "INSERT INTO role_check.canary (id, note) VALUES (2, 'written')",
    "UPDATE role_check.canary SET note = 'changed'",
    "DELETE FROM role_check.canary",
    "TRUNCATE role_check.canary",
    "ALTER TABLE role_check.canary ADD COLUMN extra integer",
    "DROP TABLE role_check.canary",
    "CREATE TABLE role_check.new_table (id integer)",
    "CREATE TABLE public.new_table (id integer)",
    "CREATE SCHEMA new_schema",
    "CREATE TEMP TABLE scratch (id integer)",
    "CREATE ROLE new_role",
]


@pytest.fixture
def agent(db_ready):
    with connect(AGENT_ROLE, autocommit=True) as conn:
        yield conn


@pytest.fixture
def agent_read_write(agent):
    """The agent's session with the read-only default switched off by the role itself."""
    agent.execute("SET default_transaction_read_only = off")
    assert agent.execute("SHOW default_transaction_read_only").fetchone()[0] == "off"
    return agent


def canary() -> list[tuple]:
    with connect(ADMIN_ROLE) as conn:
        return conn.execute("SELECT id, note FROM role_check.canary ORDER BY id").fetchall()


def test_agent_can_read(agent):
    assert agent.execute("SELECT id, note FROM role_check.canary").fetchall() == CANARY


def test_admin_can_write(db_ready):
    """The canary is writable in principle, so the refusals below are the role's doing."""
    with connect(ADMIN_ROLE) as conn:
        conn.execute("INSERT INTO role_check.canary (id, note) VALUES (2, 'written')")
        assert conn.execute("SELECT count(*) FROM role_check.canary").fetchone()[0] == 2
        conn.rollback()
    assert canary() == CANARY


@pytest.mark.parametrize("sql", WRITES)
def test_write_fails_in_a_default_session(agent, sql):
    with pytest.raises((errors.ReadOnlySqlTransaction, errors.InsufficientPrivilege)):
        agent.execute(sql)
    assert canary() == CANARY


@pytest.mark.parametrize("sql", WRITES)
def test_write_fails_on_privileges_alone(agent_read_write, sql):
    with pytest.raises(errors.InsufficientPrivilege):
        agent_read_write.execute(sql)
    assert canary() == CANARY


def test_role_has_no_elevated_attributes(db_ready):
    with connect(ADMIN_ROLE) as conn:
        row = conn.execute(
            "SELECT rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls "
            "FROM pg_roles WHERE rolname = %s",
            (AGENT_ROLE,),
        ).fetchone()
    assert row == (False, False, False, False, False)


def test_session_defaults(agent):
    def show(name: str) -> str:
        return agent.execute(f"SHOW {name}").fetchone()[0]

    assert show("default_transaction_read_only") == "on"
    assert show("statement_timeout") == "30s"
    assert show("idle_in_transaction_session_timeout") == "1min"
    assert show("temp_file_limit") == "256MB"


def test_role_cannot_lift_its_temp_file_limit(agent):
    with pytest.raises(errors.InsufficientPrivilege):
        agent.execute("SET temp_file_limit = -1")


def test_statement_timeout_is_only_a_default(agent):
    """A known limit of the database layer, kept as a test so it cannot be forgotten: the
    role can lift its own statement timeout, so a timeout the agent's SQL cannot undo has to
    come from the client (canceling the query), not from the role."""
    agent.execute("SET statement_timeout = 0")
    assert agent.execute("SHOW statement_timeout").fetchone()[0] == "0"


def test_wrong_password_is_rejected(db_ready, monkeypatch):
    monkeypatch.setenv("ANALYST_RO_PASSWORD", "not-the-password")
    with pytest.raises(psycopg.OperationalError):
        connect(AGENT_ROLE)
