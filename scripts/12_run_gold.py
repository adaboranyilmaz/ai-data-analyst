"""Execute the gold SQL of every benchmark question and store its result.

Each query runs as the agent's read-only role, in a read-only transaction, with a 300 s
statement timeout (generous: gold results define correctness; the agent's own limit is far
stricter), without parallel workers, and with `search_path` set to where its tables are.
Parallel workers split a table's blocks between them at run time, so a floating-point SUM or
AVG adds its terms in a different order on each run and its last digits change (seen on
`real` columns, which PostgreSQL sums in single precision); a serial plan returns the same
value every time. Statements are not prepared, so each one is planned under these settings.
The session time zone must be the database's (set by the loader), in which the benchmark's
timestamps were written; the run stops otherwise.
Layouts:
  --layout public   every table in `public`, as BIRD ships the dump (before `split`)
  --layout schemas  its BIRD database's schema (after `split`); the run is compared question
                    by question with the public-layout run, whose results it must reproduce
A query stopped by the role's temp_file_limit (a limit in this environment, not in the
query) is retried once as the admin role, still read-only, and marked so.

Every question is executed, including any in configs/gold_exclusions.yaml (they are marked),
and then executed a second time, and the two runs are compared.
Per question: outcome (SQLSTATE and message on error), seconds, row count, column names and
types, and two result hashes (src/db/values.py). The rows themselves, losslessly encoded, go
to a gzipped JSON-lines file under data/gold/ (DVC-tracked, not in git).
Writes results/metrics/gold_execution_public.json or results/metrics/gold_execution.json.

Usage:
    uv run python scripts/12_run_gold.py --layout public
    uv run python scripts/12_run_gold.py --layout schemas
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402
import yaml  # noqa: E402
from psycopg import errors, sql  # noqa: E402

from src.data import bird  # noqa: E402
from src.db.connection import ADMIN_ROLE, AGENT_ROLE, BIRD_DB, connect  # noqa: E402
from src.db.values import encode_row, ordered_hash, set_hash  # noqa: E402

TIMEOUT = "300s"
OUTPUTS = {  # layout -> (summary, rows)
    "public": ("results/metrics/gold_execution_public.json", "data/gold/gold_rows_public.jsonl.gz"),
    "schemas": ("results/metrics/gold_execution.json", "data/gold/gold_rows.jsonl.gz"),
}
EXCLUSIONS = ROOT / "configs/gold_exclusions.yaml"


def execute(conn: psycopg.Connection, query: str, schema: str) -> tuple[list, list]:
    with conn.transaction():
        conn.execute(sql.SQL("SET LOCAL statement_timeout = {}").format(sql.Literal(TIMEOUT)))
        conn.execute(sql.SQL("SET LOCAL search_path = {}").format(sql.Identifier(schema)))
        conn.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        cur = conn.execute(query)
        if cur.description is None:
            raise ValueError("the statement returned no result set")
        types = conn.adapters.types
        columns = [
            [d.name, t.name if (t := types.get(d.type_code)) else str(d.type_code)]
            for d in cur.description
        ]
        return columns, cur.fetchall()


def run_one(conns: dict[str, psycopg.Connection], q: dict, schema: str) -> tuple[dict, list]:
    rec: dict = {
        "question_id": q["question_id"],
        "db_id": q["db_id"],
        "difficulty": q["difficulty"],
    }
    rows: list = []
    for role in (AGENT_ROLE, ADMIN_ROLE):
        start = time.perf_counter()
        try:
            columns, rows = execute(conns[role], q["SQL"], schema)
        except errors.ConfigurationLimitExceeded as e:  # temp_file_limit
            if role == AGENT_ROLE:
                rec["agent_role_error"] = str(e).strip().splitlines()[0]
                continue
            rec.update(status="error", sqlstate=e.sqlstate, error=str(e).strip().splitlines()[0])
        except errors.QueryCanceled as e:
            rec.update(status="timeout", sqlstate=e.sqlstate, error=str(e).strip().splitlines()[0])
        except (psycopg.Error, ValueError) as e:
            rec.update(
                status="error",
                sqlstate=getattr(e, "sqlstate", None),
                error=str(e).strip().splitlines()[0],
            )
        else:
            rec.update(
                status="ok",
                columns=columns,
                rows=len(rows),
                set_hash=set_hash(rows),
                ordered_hash=ordered_hash(rows),
            )
        rec["executed_as"] = role
        rec["seconds"] = round(time.perf_counter() - start, 3)
        break
    return rec, rows


def compare(records: list[dict], other: list[dict]) -> dict:
    """Two runs of the gold SQL, question by question."""
    before = {r["question_id"]: r for r in other}
    verdicts: dict[int, str] = {}
    for r in records:
        b = before.get(r["question_id"])
        if b is None:
            verdicts[r["question_id"]] = "not in the other run"
        elif r["status"] != b["status"] or r.get("sqlstate") != b.get("sqlstate"):
            verdicts[r["question_id"]] = "outcome differs"
        elif r["status"] != "ok":
            verdicts[r["question_id"]] = "same error"
        elif r["set_hash"] != b["set_hash"] or r["columns"] != b["columns"]:
            verdicts[r["question_id"]] = "result differs"
        elif r["ordered_hash"] != b["ordered_hash"]:
            verdicts[r["question_id"]] = "same rows, different order"
        else:
            verdicts[r["question_id"]] = "identical"
    return {
        "counts": dict(sorted(Counter(verdicts.values()).items())),
        "not_identical": {
            str(k): v for k, v in sorted(verdicts.items()) if v not in ("identical", "same error")
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--layout", choices=list(OUTPUTS), required=True)
    layout = p.parse_args().layout
    summary_path, rows_path = (ROOT / x for x in OUTPUTS[layout])

    load = json.loads((ROOT / "results/metrics/bird_load.json").read_text(encoding="utf-8"))
    if load["layout"] != layout:
        sys.exit(f"the database is in the {load['layout']!r} layout, not {layout!r}")
    excluded = {
        e["question_id"]: e["reason"]
        for e in yaml.safe_load(EXCLUSIONS.read_text(encoding="utf-8"))["exclusions"]
    }
    questions = bird.questions()

    records = []
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (
        connect(AGENT_ROLE, BIRD_DB, prepare_threshold=None) as agent,
        connect(ADMIN_ROLE, BIRD_DB, prepare_threshold=None) as admin,
        gzip.GzipFile(rows_path, "wb", mtime=0) as gz,  # mtime 0: identical runs, identical file
    ):
        conns = {AGENT_ROLE: agent, ADMIN_ROLE: admin}
        for conn in conns.values():
            conn.read_only = True
            zone = conn.execute("SHOW timezone").fetchone()[0]
            if zone != bird.config()["bird_minidev"]["time_zone"]:
                sys.exit(f"the session time zone is {zone}, not the benchmark's")
        for i, q in enumerate(questions, 1):
            schema = "public" if layout == "public" else q["db_id"]
            rec, rows = run_one(conns, q, schema)
            if q["question_id"] in excluded:
                rec["excluded"] = excluded[q["question_id"]]
            records.append(rec)
            if rec["status"] == "ok":
                line = {
                    "question_id": q["question_id"],
                    "columns": rec["columns"],
                    "rows": [encode_row(r) for r in rows],
                }
                gz.write(
                    (json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                        "utf-8"
                    )
                )
            if rec["status"] != "ok" or i % 50 == 0:
                print(
                    f"[{i}/{len(questions)}] {q['question_id']} {q['db_id']}: "
                    f"{rec['status']} {rec.get('error', '')}"
                )
        seconds = time.monotonic() - start
        # every query once more on the same connections: a result that changes between two
        # runs cannot serve as gold
        repeat = [
            run_one(conns, q, "public" if layout == "public" else q["db_id"])[0] for q in questions
        ]

    status = Counter(r["status"] for r in records)
    slowest = sorted(records, key=lambda r: -r["seconds"])[:5]
    out = {
        "layout": layout,
        "database": BIRD_DB,
        "role": AGENT_ROLE,
        "statement_timeout": TIMEOUT,
        "parallel_workers": 0,
        "time_zone": bird.config()["bird_minidev"]["time_zone"],
        "questions_sha256": bird.config()["bird_minidev"]["questions"]["sha256"],
        "summary": {
            "questions": len(records),
            "status": dict(sorted(status.items())),
            "empty_results": sum(r.get("rows") == 0 for r in records),
            "executed_as_admin": sum(r["executed_as"] == ADMIN_ROLE for r in records),
            "excluded": len(excluded),
            "failures_per_database": dict(
                sorted(Counter(r["db_id"] for r in records if r["status"] != "ok").items())
            ),
            "total_seconds": round(seconds, 1),
            "slowest": [
                {"question_id": r["question_id"], "seconds": r["seconds"]} for r in slowest
            ],
        },
        "questions": records,
    }
    out["compared_with_repeat_run"] = compare(records, repeat)
    if layout == "schemas":
        public = json.loads((ROOT / OUTPUTS["public"][0]).read_text(encoding="utf-8"))["questions"]
        out["compared_with_public_layout"] = compare(records, public)
    bird.write_json(summary_path, out)
    print(json.dumps(out["summary"], indent=2))
    for key in ("compared_with_repeat_run", "compared_with_public_layout"):
        if key in out:
            print(key, json.dumps(out[key], indent=2))
    print(f"wrote {summary_path.relative_to(ROOT)} and {rows_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
