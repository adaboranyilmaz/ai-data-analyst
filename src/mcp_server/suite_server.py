"""An MCP server over the security suite's fixture schema, for the suite alone.

    uv run python -m src.mcp_server.suite_server

It is the product's server (src/mcp_server/server.py) over a toolbox whose database is the
suite's throwaway schema, so attacks can be aimed at tables that hold a canary and a secret.
It is not a way to serve any real data.
"""

from __future__ import annotations

from src.mcp_server.server import build_server
from src.mcp_server.suite import SuiteToolbox


def main() -> None:
    box = SuiteToolbox()
    try:
        build_server(box).run("stdio")
    finally:
        box.close()


if __name__ == "__main__":
    main()
