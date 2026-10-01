"""Run the MCP server over stdio.

    uv run python -m src.mcp_server

By default it serves the bundled Czech bank database (`financial`) from the local compose
PostgreSQL. To serve your own PostgreSQL instead, set these (the role must only be able to read;
it is checked first, and refused with the reasons if it can write):

    ANALYST_MCP_PG_HOST  ANALYST_MCP_PG_PORT  ANALYST_MCP_PG_DBNAME
    ANALYST_MCP_PG_USER  ANALYST_MCP_PG_PASSWORD  ANALYST_MCP_PG_SCHEMA (default: public)

`ANALYST_MCP_DATABASE` picks another bundled benchmark database. Nothing but the protocol is
written to stdout; problems go to stderr.
"""

from __future__ import annotations

import os
import sys

from src.mcp_server.server import build_server
from src.serving import connections
from src.tools.toolbox import Toolbox


def toolbox_from_env(env: dict[str, str] | None = None) -> Toolbox:
    env = os.environ if env is None else env
    if env.get("ANALYST_MCP_PG_DBNAME"):
        spec = connections.parse(
            {
                "host": env.get("ANALYST_MCP_PG_HOST", ""),
                "port": int(env.get("ANALYST_MCP_PG_PORT") or 5432),
                "dbname": env["ANALYST_MCP_PG_DBNAME"],
                "user": env.get("ANALYST_MCP_PG_USER", ""),
                "password": env.get("ANALYST_MCP_PG_PASSWORD", ""),
                "schema": env.get("ANALYST_MCP_PG_SCHEMA") or "public",
            }
        )
        return connections.ConnectionToolbox(connections.validate(spec))
    return Toolbox(env.get("ANALYST_MCP_DATABASE") or "financial")


def main() -> None:
    try:
        box = toolbox_from_env()
    except connections.ConnectionRefused as e:
        print("refused: " + "; ".join(e.reasons), file=sys.stderr)
        sys.exit(2)
    try:
        build_server(box).run("stdio")
    finally:
        box.close()


if __name__ == "__main__":
    main()
