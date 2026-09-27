"""Check the agent's tools against the benchmark, and time every tool.

1. The query guard on every gold query: the guard must accept all 500 (a refusal means the
   agent could never answer that question correctly), and must pass them on unchanged.
2. Every gold query through the agent's execution path (the schema role, the read-only
   cursor, the per-call settings) with the evaluation limits: its result must be the one
   scripts/12_run_gold.py stored (same columns, same row count, same set of rows).
3. Latency of each tool, over: run_sql on every gold query with the agent's limits;
   list_tables, describe_table and sample_rows on every table of every benchmark database;
   validate_chart on a bar chart of each gold result's first two columns. One run, on this
   machine, with the database local; the numbers show relative cost, not a service level.
Writes results/metrics/tool_check.json.

Usage:
    uv run python scripts/22_tool_check.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import bird  # noqa: E402
from src.data.bird import write_json  # noqa: E402
from src.db.execute import QueryError, ReadOnlyExecutor  # noqa: E402
from src.db.guard import QueryGuard  # noqa: E402
from src.db.values import set_hash  # noqa: E402
from src.tools.toolbox import Toolbox, benchmark_target, config, evaluation_limits  # noqa: E402

OUT = ROOT / "results/metrics/tool_check.json"
GOLD = ROOT / "results/metrics/gold_execution.json"


def percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]


def main() -> None:
    questions = bird.questions()
    gold = {r["question_id"]: r for r in json.loads(GOLD.read_text(encoding="utf-8"))["questions"]}
    by_db: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_db[q["db_id"]].append(q)

    refused, changed, differs = [], [], []
    limits = evaluation_limits()
    for db, qs in sorted(by_db.items()):
        guard = QueryGuard.for_benchmark_db(db)
        with ReadOnlyExecutor(benchmark_target(db)) as ex:
            for q in qs:
                v = guard.check(q["SQL"])
                if not v.allowed:
                    refused.append({"question_id": q["question_id"], "reasons": list(v.reasons)})
                    continue
                if v.query != q["SQL"].strip():
                    changed.append(q["question_id"])
                g = gold[q["question_id"]]
                try:
                    r = ex.execute(v.query, limits)
                except QueryError as e:
                    differs.append({"question_id": q["question_id"], "error": str(e)})
                    continue
                got = ([list(c) for c in r.columns], len(r.rows), set_hash(r.rows))
                if got != (g["columns"], g["rows"], g["set_hash"]) or r.truncated:
                    differs.append({"question_id": q["question_id"], "error": "result differs"})

    seconds: dict[str, list[float]] = defaultdict(list)
    chart_ok = 0
    cfg = config()
    for db, qs in sorted(by_db.items()):
        with Toolbox(db, cfg) as tb:
            seconds["list_tables"].append(tb.call("list_tables", {})["seconds"])
            for table in tb.schema.tables:
                seconds["describe_table"].append(
                    tb.call("describe_table", {"table": table})["seconds"]
                )
                out = tb.call("sample_rows", {"table": table})
                assert out["ok"], (db, table, out)
                seconds["sample_rows"].append(out["seconds"])
            for q in qs:
                out = tb.call("run_sql", {"sql": q["SQL"]})
                assert out["ok"], (q["question_id"], out)
                seconds["run_sql"].append(out["seconds"])
                cols = [c["name"] for c in out["columns"]]
                spec = {
                    "data": {"name": "result"},
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": cols[0].replace(".", "\\."), "type": "nominal"},
                        "y": {"field": cols[-1].replace(".", "\\."), "type": "quantitative"},
                    },
                }
                chart = tb.call("validate_chart", {"spec": spec}, result_columns=cols)
                chart_ok += chart["ok"]
                seconds["validate_chart"].append(chart["seconds"])

    latency = {
        tool: {
            "calls": len(xs),
            "p50_ms": round(statistics.median(xs) * 1000, 2),
            "p95_ms": round(percentile(xs, 0.95) * 1000, 2),
            "max_ms": round(max(xs) * 1000, 2),
        }
        for tool, xs in sorted(seconds.items())
    }
    out = {
        "runs": 1,
        "guard": {
            "gold_queries": len(questions),
            "accepted": len(questions) - len(refused),
            "refused": refused,
            "changed_on_acceptance": changed,
        },
        "execution": {
            "limits": {"max_rows": limits.max_rows, "timeout_s": limits.timeout_s},
            "same_result_as_gold": len(questions) - len(refused) - len(differs),
            "different": differs,
        },
        "charts_valid": chart_ok,
        "latency": latency,
    }
    write_json(OUT, out)
    print(json.dumps({k: v for k, v in out.items() if k != "latency"}, indent=1)[:1500])
    print(json.dumps(latency, indent=1))
    if refused or differs or changed:
        sys.exit("the tools do not reproduce every gold query")


if __name__ == "__main__":
    main()
