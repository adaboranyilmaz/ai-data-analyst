"""Load the BIRD mini-dev PostgreSQL dump into the `bird` database, then give each BIRD
database its own schema.

  load   Recreate the `bird` database and load the dump into its `public` schema, exactly as
         BIRD ships it (the layout the benchmark's gold SQL was written for). The dump is
         streamed from the package through `psql` in the compose container and hashed on the
         way; its `OWNER TO` statements name its author's role, which does not exist here,
         so they are left out and the admin role owns everything. The agent's role gets
         CONNECT, and USAGE and SELECT on the tables, nothing else. The database's default
         time zone is the one the dump's timestamps were written in (configs/datasets.yaml
         `time_zone`), so timestamps compare and convert to dates as the benchmark expects.
         Records row counts.
  split  Move each table into the schema named after its BIRD database (financial,
         formula_1, ...), with the same grants. Tables, their indexes, constraints and
         sequences move together; the data is not touched. Gold SQL then runs with
         `search_path` set to its database's schema, so it needs no change.
Writes results/metrics/bird_load.json (load) and adds the schema layout to it (split).

Usage:
    docker compose up -d --wait
    uv run python scripts/11_load_bird.py load
    uv run python scripts/11_load_bird.py split
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

from psycopg import sql  # noqa: E402

from src.data import bird  # noqa: E402
from src.db.connection import ADMIN_ROLE, AGENT_ROLE, BIRD_DB, DB_NAME, connect  # noqa: E402

OUT = ROOT / "results/metrics/bird_load.json"
DUMP_OWNER = b" OWNER TO xiaolongli;\n"


def psql_stream(dump) -> tuple[str, int, int]:
    """Pipe the dump into psql in the container, without its OWNER TO statements. Returns
    the dump's sha256 (of every byte read, dropped lines included), lines read and dropped."""
    cmd = [
        "docker", "compose", "exec", "-T", "postgres",
        "psql", "-v", "ON_ERROR_STOP=1", "-q", "-U", ADMIN_ROLE, "-d", BIRD_DB,
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
    h, lines, dropped, in_copy = hashlib.sha256(), 0, 0, False
    for line in dump:
        h.update(line)
        lines += 1
        if in_copy:
            in_copy = line != b"\\.\n"
        elif line.startswith(b"COPY "):
            in_copy = True
        elif line.startswith(b"ALTER ") and line.endswith(DUMP_OWNER):
            dropped += 1
            continue
        proc.stdin.write(line)
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit(f"psql failed with exit code {proc.returncode}")
    return h.hexdigest(), lines, dropped


def row_counts(conn, schema_of: dict[str, str] | None = None) -> dict[str, int]:
    tables = conn.execute(
        "SELECT table_schema, table_name FROM information_schema.tables "
        "WHERE table_catalog = %s AND table_schema NOT IN ('pg_catalog', 'information_schema') "
        "ORDER BY 1, 2",
        [BIRD_DB],
    ).fetchall()
    out = {}
    for schema, table in tables:
        q = sql.SQL("SELECT count(*) FROM {}.{}").format(
            sql.Identifier(schema), sql.Identifier(table)
        )
        out[f"{schema}.{table}"] = conn.execute(q).fetchone()[0]
    return out


def grant_read(conn, schema: str) -> None:
    s = sql.Identifier(schema)
    ro = sql.Identifier(AGENT_ROLE)
    conn.execute(sql.SQL("REVOKE ALL ON SCHEMA {} FROM PUBLIC").format(s))
    conn.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(s, ro))
    conn.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(s, ro))


def load() -> None:
    cfg = bird.config()["bird_minidev"]
    with connect(ADMIN_ROLE, DB_NAME, autocommit=True) as conn:
        conn.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(BIRD_DB))
        )
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(BIRD_DB)))
        conn.execute(
            sql.SQL("ALTER DATABASE {} SET timezone TO {}").format(
                sql.Identifier(BIRD_DB), sql.Literal(cfg["time_zone"])
            )
        )
        server = conn.execute("SHOW server_version").fetchone()[0]

    start = time.monotonic()
    with zipfile.ZipFile(ROOT / cfg["package"]["path"]) as z, z.open(cfg["members"]["dump"]) as f:
        print(f"loading {cfg['members']['dump']} into {BIRD_DB}")
        dump_sha, lines, dropped = psql_stream(f)
    seconds = time.monotonic() - start
    if dump_sha != cfg["members"]["dump_sha256"]:
        sys.exit(
            f"dump sha256 {dump_sha} does not match the pinned {cfg['members']['dump_sha256']}"
        )

    with connect(ADMIN_ROLE, BIRD_DB, autocommit=True) as conn:
        db = sql.Identifier(BIRD_DB)
        conn.execute(sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(db))
        conn.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(db, sql.Identifier(AGENT_ROLE))
        )
        grant_read(conn, "public")
        conn.execute("ANALYZE")
        counts = row_counts(conn)

    bird.write_json(
        OUT,
        {
            "database": BIRD_DB,
            "server_version": server,
            "time_zone": cfg["time_zone"],
            "dump_sha256": dump_sha,
            "dump_lines": lines,
            "owner_statements_dropped": dropped,
            "load_seconds": round(seconds, 1),
            "layout": "public",
            "tables": len(counts),
            "rows": counts,
        },
    )
    print(f"loaded {len(counts)} tables in {seconds:.0f} s; wrote {OUT.relative_to(ROOT)}")


def split() -> None:
    record = json.loads(OUT.read_text(encoding="utf-8"))
    if record.get("layout") != "public":
        sys.exit("the database is not in the public layout: run `load` first")
    databases = bird.table_databases()
    with connect(ADMIN_ROLE, BIRD_DB, autocommit=True) as conn:
        before = row_counts(conn)
        tables = {k.split(".", 1)[1] for k in before if k.startswith("public.")}
        missing, unknown = set(databases) - tables, tables - set(databases)
        if missing or unknown or len(tables) != len(before):
            sys.exit(
                f"table lists differ: not loaded {sorted(missing)}, "
                f"not in BIRD's list {sorted(unknown)}"
            )
        with conn.transaction():
            for schema in sorted(set(databases.values())):
                conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            for table, schema in sorted(databases.items()):
                conn.execute(
                    sql.SQL("ALTER TABLE public.{} SET SCHEMA {}").format(
                        sql.Identifier(table), sql.Identifier(schema)
                    )
                )
            for schema in sorted(set(databases.values())):
                grant_read(conn, schema)
        after = row_counts(conn)
        leftover = conn.execute(
            "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public'"
        ).fetchone()[0]
    moved = {f"{databases[k.split('.', 1)[1]]}.{k.split('.', 1)[1]}": v for k, v in before.items()}
    if moved != after or leftover:
        sys.exit(f"row counts changed in the move, or {leftover} objects were left in public")

    schemas: dict[str, dict[str, int]] = {}
    for key, n in sorted(after.items()):
        schema, table = key.split(".", 1)
        schemas.setdefault(schema, {})[table] = n
    record.update(layout="schemas", rows=after, schemas=schemas)
    bird.write_json(OUT, record)
    print(f"moved {len(after)} tables into {len(schemas)} schemas; updated {OUT.relative_to(ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("step", choices=["load", "split"])
    {"load": load, "split": split}[p.parse_args().step]()


if __name__ == "__main__":
    main()
