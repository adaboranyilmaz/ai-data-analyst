"""The curated replay set: draw the runs by the rule in configs/serving.yaml and write each as an
evidence record under results/demo/, with an index. Reads the evaluated records and their traces;
needs no database and makes no model call.

Usage:
    uv run python scripts/80_demo_runs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import bird  # noqa: E402
from src.eval.summary import own_success  # noqa: E402
from src.serving import curated, evidence, explain  # noqa: E402
from src.serving.meter import Meter, config  # noqa: E402
from src.tools.toolbox import Toolbox  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def trace_of(record: dict) -> dict:
    return json.loads((ROOT / record["trace"]).read_text(encoding="utf-8"))


def explain_wrong(
    ev: dict, record: dict, expert_sql: str | None, category: str | None, note: str | None
) -> None:
    """Attach the expected result and why this answer was scored wrong (src/serving/explain.py)."""
    if expert_sql is None:
        return
    with Toolbox(ev["db_id"]) as box:
        expected = explain.reference(lambda sql: box.call("run_sql", {"sql": sql}), expert_sql)
    res = ev["result"]
    got = None if res is None or not res["ok"] else {**res}
    ev["explanation"] = explain.build(
        got=got,
        expected=expected,
        expected_sql=expert_sql,
        note=note,
        category=category,
        hint=ev["hint"],
    )


def main() -> None:
    cfg = config()
    c = cfg["curated"]
    seed = c["seed"]
    meter = Meter.load(cfg)
    out = ROOT / c["out_dir"]
    (out / "runs").mkdir(parents=True, exist_ok=True)

    splits = json.loads((ROOT / "results/metrics/splits.json").read_text(encoding="utf-8"))
    held_out = {int(i) for i in splits["sets"]["held_out"]}
    h = c["held_out"]
    main_records = read_jsonl(ROOT / f"results/runs/{h['stage']}/{h['run']}.jsonl")
    b = c["banking"]
    bank_records = read_jsonl(ROOT / f"results/runs/{b['stage']}/{b['run']}.jsonl")
    g = c["guardrail"]
    review = yaml.safe_load((ROOT / g["review"]).read_text(encoding="utf-8"))
    guard_records = read_jsonl(ROOT / g["records"])

    expert_sql = {q["question_id"]: q["SQL"] for q in bird.questions()}
    own_sql = {
        q["id"]: q["gold_sql"]
        for q in yaml.safe_load((ROOT / "own_set/questions.yaml").read_text(encoding="utf-8"))[
            "questions"
        ]
        if "gold_sql" in q
    }
    notes = yaml.safe_load(
        (ROOT / "results/reviews/demo_explanations.yaml").read_text(encoding="utf-8")
    )["notes"]
    categories = {
        a["question_id"]: a["category"]
        for a in json.loads(
            (ROOT / "results/metrics/error_analysis.json").read_text(encoding="utf-8")
        )["answers_detail"]
    }
    short: dict[str, int] = {}
    runs: list[dict] = []
    held_picks, s = curated.held_out_slots(
        main_records, held_out, meter.calibrate, meter.threshold, h["slots"], seed
    )
    short.update({f"held_out.{k}": v for k, v in s.items()})
    for p in held_picks:
        ev = evidence.from_run(
            run_id=f"bench-{p.key}",
            kind="benchmark",
            record=p.record,
            trace=trace_of(p.record),
            meter=meter,
            source_run=f"{h['stage']}/{h['run']}",
        )
        if not p.record["correct"] and not p.record["declined"]:
            explain_wrong(
                ev,
                p.record,
                expert_sql.get(p.record["question_id"]),
                categories.get(p.record["question_id"]),
                notes.get(ev["id"]),
            )
        runs.append({**ev, "slot": p.slot})
    bank_picks, s = curated.banking_slots(
        bank_records, b["one_per_category"], b["wrong_extra"], seed
    )
    short.update({f"banking.{k}": v for k, v in s.items()})
    for p in bank_picks:
        ev = evidence.from_run(
            run_id=f"bank-{p.key}",
            kind="banking",
            record=p.record,
            trace=trace_of(p.record),
            meter=meter,
            source_run=f"{b['stage']}/{b['run']}",
        )
        if p.record["category"] in ("a", "b") and own_success(p.record) == 0:
            explain_wrong(
                ev, p.record, own_sql.get(p.record["question_id"]), None, notes.get(ev["id"])
            )
        runs.append({**ev, "slot": p.slot})
    guard_picks, s = curated.guardrail_slots(
        guard_records,
        review["banking_f"],
        g["reviewed_success"],
        g["reviewed_failure"],
        seed,
    )
    short.update({f"guardrail.{k}": v for k, v in s.items()})
    notes = {r["id"]: r for r in review["banking_f"]}
    by_bank = {str(r["question_id"]): r for r in bank_records}
    for p in guard_picks:
        before = None
        if p.key in by_bank:
            before = evidence.from_run(
                run_id=f"before-{p.key}",
                kind="banking",
                record=by_bank[p.key],
                trace=trace_of(by_bank[p.key]),
                meter=meter,
                source_run=f"{b['stage']}/{b['run']}",
            )
        ev = evidence.from_guardrail(
            run_id=f"guard-{p.key}",
            rec=p.record,
            before=before,
            review=notes[p.key],
            source_run="guardrail/own",
        )
        runs.append({**ev, "slot": p.slot})

    for old in (out / "runs").glob("*.json"):
        old.unlink()
    for ev in runs:
        (out / "runs" / f"{ev['id']}.json").write_text(
            json.dumps(ev, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
    index = {
        "rule": "configs/serving.yaml, curated",
        "seed": seed,
        "count": len(runs),
        "short": short,
        "runs": [
            {
                "id": ev["id"],
                "kind": ev["kind"],
                "slot": ev["slot"],
                "question": ev["question"],
                "db_id": ev["db_id"],
                "status": ev["status"],
            }
            for ev in runs
        ],
    }
    (out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{len(runs)} runs written to {out.relative_to(ROOT)}; shortfalls: {short or 'none'}")


if __name__ == "__main__":
    main()
