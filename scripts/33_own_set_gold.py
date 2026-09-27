"""Execute every SQL of the hand-written banking set and store its result.

Each query (a gold query, an accepted reading of an ambiguous question, or the query showing a
premise false) must pass the agent's query guard, since an agent could otherwise never produce
it, and then runs through the agent's execution path (schema role, read-only cursor, fixed
settings) with the evaluation limits, as the benchmark's gold does.

Checks recorded:
  - the file's format (src/eval/own_set.py) and the count per category and per review state;
  - every query accepted by the guard and executed without error, with a non-empty result;
  - the readings of each ambiguous question give different results (else it is not ambiguous);
  - no gold result equals the gold result of one of the benchmark's financial questions, and no
    query is the same, after normalisation, as a benchmark gold query (the set must not repeat
    the benchmark).
Writes results/metrics/own_set_gold.json. The rows go to data/own_set/gold_rows.jsonl.gz and a
review sheet with each question, its SQL and a preview of its result to data/own_set/review.md
(both local: the rows are derived from the benchmark's data).

Usage:
    uv run python scripts/33_own_set_gold.py
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sqlglot  # noqa: E402

from src.data import bird  # noqa: E402
from src.data.bird import write_json  # noqa: E402
from src.db.execute import QueryError, ReadOnlyExecutor  # noqa: E402
from src.db.guard import QueryGuard  # noqa: E402
from src.db.values import encode_row, set_hash  # noqa: E402
from src.eval import own_set  # noqa: E402
from src.eval.config import config  # noqa: E402
from src.eval.preregistration import file_sha256  # noqa: E402
from src.tools.sql import display_value  # noqa: E402
from src.tools.toolbox import benchmark_target, evaluation_limits  # noqa: E402

DB = "financial"
OUT = ROOT / "results/metrics/own_set_gold.json"
DATA = ROOT / "data/own_set"
GOLD_RUN = ROOT / "results/metrics/gold_execution.json"
PREVIEW_ROWS = 8


def normalised(sql: str) -> str:
    try:
        return sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres", normalize=True)
    except sqlglot.errors.SqlglotError:
        return " ".join(sql.lower().split())


def preview(columns: list, rows: list, total: int) -> str:
    head = "| " + " | ".join(c[0] for c in columns) + " |"
    sep = "|" + "---|" * len(columns)
    body = [
        "| " + " | ".join(str(display_value(v, 60)) for v in r) + " |" for r in rows[:PREVIEW_ROWS]
    ]
    more = [f"\n({total - PREVIEW_ROWS} more rows)"] if total > PREVIEW_ROWS else []
    return "\n".join([head, sep, *body, *more])


def main() -> None:
    path = ROOT / config()["own_set"]["path"]
    data = own_set.load(path)
    found = own_set.problems(data)
    if found:
        sys.exit("the banking set is malformed:\n  " + "\n  ".join(found))
    questions = data["questions"]

    bench = [q for q in bird.questions() if q["db_id"] == DB]
    bench_hashes = {
        r["set_hash"]: r["question_id"]
        for r in json.loads(GOLD_RUN.read_text(encoding="utf-8"))["questions"]
        if r["db_id"] == DB and r["status"] == "ok"
    }
    bench_sql = {normalised(q["SQL"]): q["question_id"] for q in bench}

    guard = QueryGuard.for_benchmark_db(DB)
    limits = evaluation_limits()
    records, sheet, failures = [], [], []
    DATA.mkdir(parents=True, exist_ok=True)
    with (
        ReadOnlyExecutor(benchmark_target(DB)) as ex,
        gzip.GzipFile(DATA / "gold_rows.jsonl.gz", "wb", mtime=0) as gz,
    ):
        for q in questions:
            sheet.append(f"## {q['id']} ({q['category']}): {q['question']}\n")
            for key in ("missing", "premise", "correction", "check"):
                if key in q:
                    sheet.append(f"**{key}:** {q[key]}\n")
            results = []
            for label, sql in own_set.queries(q):
                rec = {"question_id": q["id"], "query": label}
                verdict = guard.check(sql)
                if not verdict.allowed:
                    rec.update(status="refused", reasons=list(verdict.reasons))
                else:
                    try:
                        r = ex.execute(verdict.query, limits)
                    except QueryError as e:
                        rec.update(status="error", error=str(e))
                    else:
                        rows = [tuple(x) for x in r.rows]
                        rec.update(
                            status="ok",
                            columns=[list(c) for c in r.columns],
                            rows=len(rows),
                            set_hash=set_hash(rows),
                            seconds=r.seconds,
                        )
                        if r.truncated or not rows:
                            rec["status"] = "truncated" if r.truncated else "empty"
                        if rec["set_hash"] in bench_hashes:
                            rec["same_result_as_benchmark"] = bench_hashes[rec["set_hash"]]
                        if normalised(sql) in bench_sql:
                            rec["same_sql_as_benchmark"] = bench_sql[normalised(sql)]
                        line = {
                            "question_id": q["id"],
                            "query": label,
                            "columns": rec["columns"],
                            "rows": [encode_row(x) for x in rows],
                        }
                        gz.write((json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))
                        results.append(rec["set_hash"])
                        sheet.append(f"**{label}**\n\n```sql\n{sql.strip()}\n```\n")
                        sheet.append(preview(r.columns, rows, len(rows)) + "\n")
                if (
                    rec["status"] != "ok"
                    or "same_result_as_benchmark" in rec
                    or "same_sql_as_benchmark" in rec
                ):
                    failures.append(rec)
                    why = rec.get("reasons") or rec.get("error") or ""
                    sheet.append(f"**{label}: {rec['status']}** {why}\n")
                records.append(rec)
            if q["category"] == "c" and len(set(results)) < len(results):
                failures.append({"question_id": q["id"], "status": "readings give the same result"})
            sheet.append(f"*review: {q['review']}*\n\n---\n")

    (DATA / "review.md").write_text(
        "# Banking set: review sheet\n\n" + "\n".join(sheet), encoding="utf-8", newline="\n"
    )
    out = {
        "file": config()["own_set"]["path"],
        "file_sha256": file_sha256(path),
        "questions": len(questions),
        "categories": dict(sorted(Counter(q["category"] for q in questions).items())),
        "review": dict(sorted(Counter(q["review"] for q in questions).items())),
        "queries": len(records),
        "status": dict(sorted(Counter(r["status"] for r in records).items())),
        "problems": failures,
        "limits": {"max_rows": limits.max_rows, "timeout_s": limits.timeout_s},
        "results": records,
    }
    write_json(OUT, out)
    print({k: out[k] for k in ("questions", "categories", "review", "queries", "status")})
    for f in failures:
        print("PROBLEM", f)
    print(f"wrote {OUT.relative_to(ROOT)} and {(DATA / 'review.md').relative_to(ROOT)}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
