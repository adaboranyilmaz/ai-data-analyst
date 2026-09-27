"""Record each BIRD database's schema, so the data dictionary can be checked without the data.

For every schema of the `bird` database, one file dictionary/_snapshot/<db>.json: its
tables, their row counts, and their columns (type, nullability, position). For the Czech bank
database (`financial`), which has a hand-written dictionary with code translations, also
per column: NULL count and distinct count; every value with its count for text columns with
at most MAX_VALUES distinct values; minimum and maximum for numbers and dates.
tests/test_dictionary.py checks the dictionary against these files (in CI, where the
database is not loaded); tests/test_bird_data.py checks these files against the database.

Usage:
    uv run python scripts/13_snapshot_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import bird  # noqa: E402
from src.db.connection import AGENT_ROLE, BIRD_DB, connect  # noqa: E402
from src.dictionary.snapshot import SNAPSHOT_DIR, snapshot_database  # noqa: E402

PROFILED = {"financial"}


def main() -> None:
    with connect(AGENT_ROLE, BIRD_DB) as conn:
        conn.read_only = True
        schemas = sorted(set(bird.table_databases().values()))
        for db in schemas:
            snap = snapshot_database(conn, db, profile=db in PROFILED)
            bird.write_json(SNAPSHOT_DIR / f"{db}.json", snap)
            cols = sum(len(t["columns"]) for t in snap["tables"].values())
            print(f"{db}: {len(snap['tables'])} tables, {cols} columns")


if __name__ == "__main__":
    main()
