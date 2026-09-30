"""Summarize the benchmark data, its dictionaries and the gold SQL execution in one file.

Reads only committed files: the sources and load records, the schema snapshots, the
dictionaries and the gold execution results. For every BIRD database: tables, rows, columns,
and how many columns the dictionary describes. For the Czech bank: loaded against published
row counts, and code coverage (values present in the data that the dictionary translates).
For the gold SQL: outcomes, and whether the results were stable across a repeat run and the
move into per-database schemas.
Writes results/metrics/data_stats.json.

Usage:
    uv run python scripts/15_data_stats.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import bird  # noqa: E402
from src.dictionary import model  # noqa: E402
from src.dictionary.snapshot import SNAPSHOT_DIR  # noqa: E402

METRICS = ROOT / "results/metrics"
OUT = METRICS / "data_stats.json"


def read(name: str) -> dict:
    return json.loads((METRICS / name).read_text(encoding="utf-8"))


def database_stats(db: str) -> dict:
    snap, d = model.load_snapshot(db), model.load(db)
    entries = {
        (t, c): e
        for t, table in d["tables"].items()
        for c, e in (table.get("columns") or {}).items()
    }
    cols = [(t, c["name"]) for t, table in snap["tables"].items() for c in table["columns"]]
    problems = (
        model.structure_problems(d)
        + model.coverage_problems(d, snap)
        + model.code_problems(d, snap)
    )
    return {
        "dictionary": d["origin"],
        "tables": len(snap["tables"]),
        "rows": sum(t["rows"] for t in snap["tables"].values()),
        "columns": len(cols),
        "columns_with_entry": sum(k in entries for k in cols),
        "columns_described": sum(
            k in entries and entries[k].get("status") != "missing" for k in cols
        ),
        "dictionary_problems": len(problems),
    }


def code_coverage(db: str) -> dict:
    snap, d = model.load_snapshot(db), model.load(db)
    columns = present = translated = nulls = nulls_explained = 0
    for t, table in snap["tables"].items():
        for col in table["columns"]:
            entry = d["tables"][t]["columns"][col["name"]]
            if col.get("nulls"):
                nulls += 1
                nulls_explained += bool(entry.get("null_meaning"))
            if entry.get("kind") != "code":
                continue
            columns += 1
            values = col.get("values") or {}
            present += len(values)
            translated += sum(v in (entry.get("codes") or {}) for v in values)
    return {
        "code_columns": columns,
        "code_values_present": present,
        "code_values_translated": translated,
        "code_coverage": translated / present if present else None,
        "columns_with_nulls": nulls,
        "columns_with_nulls_explained": nulls_explained,
    }


def main() -> None:
    sources, load, gold = (
        read("bird_sources.json"),
        read("bird_load.json"),
        read("gold_execution.json"),
    )
    dbs = sorted(p.stem for p in SNAPSHOT_DIR.glob("*.json"))
    per_db = {db: database_stats(db) for db in dbs}
    published = bird.config()["financial_published_rows"]
    loaded = load["schemas"]["financial"]
    s = gold["summary"]
    ok = s["status"].get("ok", 0)
    out = {
        "sources": {
            "package_sha256": sources["package"]["sha256"],
            "dump_sha256": load["dump_sha256"],
            "questions_revision": sources["questions"]["revision"],
            "questions_sha256": sources["questions"]["sha256"],
            "package_questions_vs_pinned": {
                k: v for k, v in sources["package_questions_vs_pinned"].items() if k != "changed"
            }
            | {
                "changed_ids": [
                    c["question_id"] for c in sources["package_questions_vs_pinned"]["changed"]
                ]
            },
        },
        "totals": {
            "databases": len(dbs),
            "tables": sum(v["tables"] for v in per_db.values()),
            "rows": sum(v["rows"] for v in per_db.values()),
            "columns": sum(v["columns"] for v in per_db.values()),
            "columns_with_entry": sum(v["columns_with_entry"] for v in per_db.values()),
            "columns_described": sum(v["columns_described"] for v in per_db.values()),
        },
        "databases": per_db,
        "financial": {
            "row_counts": {
                t: {"published": n, "loaded": loaded[t], "match": n == loaded[t]}
                for t, n in published.items()
            },
            "row_counts_all_match": all(n == loaded[t] for t, n in published.items()),
        }
        | code_coverage("financial"),
        "gold": {
            "questions": s["questions"],
            "executed_ok": ok,
            "execution_rate": ok / s["questions"],
            "status": s["status"],
            "excluded": s["excluded"],
            "executed_as_admin": s["executed_as_admin"],
            "empty_results": [r["question_id"] for r in gold["questions"] if r.get("rows") == 0],
            "per_database": sources["questions"]["per_database"],
            "per_difficulty": sources["questions"]["per_difficulty"],
            "repeat_run": gold["compared_with_repeat_run"]["counts"],
            "public_vs_schema_layout": gold["compared_with_public_layout"]["counts"],
            "same_result_sets_across_runs": all(
                v == "same rows, different order"
                for c in ("compared_with_repeat_run", "compared_with_public_layout")
                for v in gold[c]["not_identical"].values()
            ),
            "total_seconds": s["total_seconds"],
        },
    }
    bird.write_json(OUT, out)
    print(json.dumps({k: out[k] for k in ("totals", "gold")}, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
