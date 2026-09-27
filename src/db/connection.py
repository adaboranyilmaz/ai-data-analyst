"""Connections to the compose PostgreSQL.

The agent connects as `analyst_ro`, the read-only role created by
docker/postgres/init/01_roles.sh; `analyst_admin` owns the data and is used only to load it.
Passwords come from the environment (.env); the defaults are docker-compose.yml's local-only
fallbacks (the port is bound to 127.0.0.1 and the data is a public benchmark).
"""

from __future__ import annotations

import os

import psycopg
from psycopg.conninfo import make_conninfo

DB_NAME = "analyst"
# The BIRD mini-dev benchmark databases, one schema each (loaded by scripts/11_load_bird.py).
BIRD_DB = "bird"
AGENT_ROLE = "analyst_ro"
ADMIN_ROLE = "analyst_admin"
_PASSWORDS = {  # role -> (environment variable, local default)
    AGENT_ROLE: ("ANALYST_RO_PASSWORD", "analyst-local-ro"),
    ADMIN_ROLE: ("POSTGRES_PASSWORD", "analyst-local-admin"),
}


def conninfo(role: str = AGENT_ROLE, dbname: str = DB_NAME) -> str:
    var, default = _PASSWORDS[role]
    return make_conninfo(
        host=os.environ.get("ANALYST_DB_HOST") or "127.0.0.1",
        port=os.environ.get("ANALYST_DB_PORT") or "5432",
        dbname=dbname,
        user=role,
        password=os.environ.get(var) or default,
        connect_timeout=5,
    )


def connect(role: str = AGENT_ROLE, dbname: str = DB_NAME, **kwargs) -> psycopg.Connection:
    return psycopg.connect(conninfo(role, dbname), **kwargs)
