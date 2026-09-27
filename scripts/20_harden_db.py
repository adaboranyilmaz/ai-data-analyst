"""Harden the databases the agent connects to, then check what its role can still reach.

Applies src/db/hardening.sql (as the admin role) to template1, `analyst` and, when it exists,
`bird`, and src/db/security_fixtures.sql to `analyst`, then creates or refreshes the schema
roles (src/db/hardening.py): one per benchmark schema in `bird`, and one for the security
suite's schema in `analyst`. Then checks, as seen from the agent's roles in each database:
the closed system functions and views are closed, and each schema role reads its own schema's
tables and nothing else. Stops with an error if anything is still open.
Hardening lives in the databases' catalogs, not in any file, so a new data volume or a reload
of the benchmark needs this script again; tests/test_hardening.py fails until it has run.
Writes results/metrics/db_hardening.json (unless --no-results, as in CI).

Usage:
    uv run python scripts/20_harden_db.py [--no-results]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

from src.data.bird import write_json  # noqa: E402
from src.db.connection import ADMIN_ROLE, BIRD_DB, DB_NAME, connect  # noqa: E402
from src.db.hardening import audit, ensure_schema_roles  # noqa: E402

OUT = ROOT / "results/metrics/db_hardening.json"
HARDENING = ROOT / "src/db/hardening.sql"
FIXTURES = ROOT / "src/db/security_fixtures.sql"


def apply(dbname: str, *files: Path) -> None:
    with connect(ADMIN_ROLE, dbname, autocommit=True) as conn:
        for f in files:
            conn.execute(f.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--no-results", action="store_true", help="check, but write no results file")
    args = p.parse_args()

    databases = ["template1", DB_NAME]
    try:
        connect(ADMIN_ROLE, BIRD_DB).close()
        databases.append(BIRD_DB)
    except psycopg.OperationalError:
        print(f"{BIRD_DB}: not present, skipped")

    report: dict = {"databases": {}}
    for db in databases:
        apply(db, HARDENING, *([FIXTURES] if db == DB_NAME else []))
        if db in (DB_NAME, BIRD_DB):
            ensure_schema_roles(db)
        report["databases"][db] = result = audit(db)
        roles = result.get("schema_roles", {"roles": 0, "problems": []})
        print(
            f"{db}: {result['functions_closed']} functions and {result['views_closed']} views "
            f"closed; open: {result['functions_open'] + result['views_open'] or 'none'}; "
            f"{roles['roles']} schema roles, problems: {roles['problems'] or 'none'}"
        )
    open_items = {
        db: r["functions_open"] + r["views_open"] + r.get("schema_roles", {}).get("problems", [])
        for db, r in report["databases"].items()
    }
    report["all_closed"] = not any(open_items.values())
    if not args.no_results:
        write_json(OUT, report)
    if not report["all_closed"]:
        sys.exit(f"still open to the agent's role: {open_items}")


if __name__ == "__main__":
    main()
