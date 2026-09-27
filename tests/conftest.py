"""Shared test set-up.

No test can reach a model API: the API key is removed from each test's environment, and
replay-only mode starts off so that the tests of the live path behave the same whatever the
shell sets; the replay tests turn it on themselves.

Database tests (marker `db`) need the compose PostgreSQL. When it is not running they are
skipped with the reason shown, unless ANALYST_REQUIRE_DB=1 (set in CI), which turns the skip
into a failure: CI cannot pass without proving the agent's role is read-only.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DB_ENV = ("POSTGRES_PASSWORD", "ANALYST_RO_PASSWORD", "ANALYST_DB_HOST", "ANALYST_DB_PORT")


def _load_db_env_from_dotenv() -> None:
    """Only the database settings from .env: never the API key."""
    from dotenv import dotenv_values

    for key, value in dotenv_values(ROOT / ".env").items():
        if key in DB_ENV and value and not os.environ.get(key):
            os.environ[key] = value


_load_db_env_from_dotenv()


@pytest.fixture(autouse=True)
def _no_api_access(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANALYST_REPLAY_ONLY", raising=False)


@pytest.fixture(scope="session")
def db_ready() -> None:
    import psycopg

    from src.db.connection import ADMIN_ROLE, connect

    try:
        with connect(ADMIN_ROLE) as conn:
            conn.execute("SELECT 1")
    except psycopg.OperationalError as e:
        reason = f"PostgreSQL is not reachable ({str(e).strip().splitlines()[0]})"
        if os.environ.get("ANALYST_REQUIRE_DB") == "1":
            pytest.fail(f"{reason}; ANALYST_REQUIRE_DB=1 requires it")
        pytest.skip(f"{reason}; start it with `docker compose up -d --wait`")
