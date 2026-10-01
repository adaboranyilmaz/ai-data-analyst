"""The security suite through the MCP path: every attack sent from an MCP client to a server
process over stdio. Writes results/metrics/mcp_security.json. Needs the compose PostgreSQL,
hardened (scripts/20_harden_db.py); the benchmark attacks run when the benchmark is loaded.

Usage:
    uv run python scripts/82_mcp_security.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.mcp_server.suite import run_suite  # noqa: E402

OUT = ROOT / "results/metrics/mcp_security.json"


def main() -> None:
    result = run_suite()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(
        f"{result['attacks']} attacks + {result['boundary_attacks']} at the tool boundary; "
        f"breaches {result['breaches']}; outcomes {result['outcomes']} / "
        f"{result['boundary_outcomes']}"
    )
    if not result["every_attack_blocked"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
