"""Check that the project's scoring reproduces BIRD's official evaluator, verdict for verdict.

Every case is scored twice: by the project (src/eval/execution.py) and by the official
evaluator's own per-question functions, with their 30 s time limit and their exception
handling, connected to this project's database (src/eval/official.py). Cases:

  gold        each of the 500 gold queries submitted as its own prediction: both scorers
              must give all 500 a 1
  mutant      up to three altered versions of each gold query (src/eval/mutations.py,
              seeded): a missing DISTINCT, a flipped sort, a dropped condition, a cast...
  edge        hand-written pairs, each aimed at one rule of the comparison
              (configs/ex_edge_cases.yaml): value types, NaN, arrays, empty results, errors,
              a query over the time limit
  divergence  the three kinds of prediction the project refuses by construction (more than
              one statement, a statement before the query, another database's table): the
              official evaluator scores them 1, the project 0

EX verdicts must be identical on every gold, mutant and edge case; Soft-F1 is compared the
same way, to the last bit. Also recorded:
  - the official EX over the gold cases with PostgreSQL's default parallel plans (the project
    turns them off): floating-point sums and averages then vary from run to run;
  - the official aggregate (its own compute_acc_by_diff) beside the project's, for the gold
    and mutant cases;
  - drift: each gold result, as run for scoring, against the one stored by
    scripts/12_run_gold.py; the gold queries that read the clock are listed.

One run, on this machine. Writes results/metrics/ex_validation.json. The cases' SQL goes to
data/eval_validation/ (the mutants derive from the benchmark's gold SQL; not committed).

Usage:
    uv run --group bird-eval python scripts/31_validate_ex.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.data import bird  # noqa: E402
from src.data.bird import write_json  # noqa: E402
from src.db.hardening import benchmark_schemas  # noqa: E402
from src.eval import official as off  # noqa: E402
from src.eval.config import config, official_evaluator_dir  # noqa: E402
from src.eval.execution import Scorer, ScoreSettings, reads_the_clock  # noqa: E402
from src.eval.mutations import mutants  # noqa: E402
from src.tools.toolbox import benchmark_target  # noqa: E402

OUT = ROOT / "results/metrics/ex_validation.json"
CASES_DIR = ROOT / "data/eval_validation"
EDGE_CASES = ROOT / "configs/ex_edge_cases.yaml"
GOLD_RUN = ROOT / "results/metrics/gold_execution.json"
DIFFICULTIES = ("simple", "moderate", "challenging")


def build_cases(questions: list[dict], cfg: dict) -> list[dict]:
    cases = []
    for q in questions:
        cases.append(
            {
                "id": f"gold-{q['question_id']}",
                "part": "gold",
                "question_id": q["question_id"],
                "db": q["db_id"],
                "difficulty": q["difficulty"],
                "gold": q["SQL"],
                "predicted": q["SQL"],
            }
        )
    official = cfg["official_evaluator"]
    rng = random.Random(official["mutant_seed"])
    for q in questions:
        for kind, sql in mutants(q["SQL"], official["mutants_per_query"], rng):
            cases.append(
                {
                    "id": f"mutant-{q['question_id']}-{kind}",
                    "part": "mutant",
                    "kind": kind,
                    "question_id": q["question_id"],
                    "db": q["db_id"],
                    "difficulty": q["difficulty"],
                    "gold": q["SQL"],
                    "predicted": sql,
                }
            )
    edges = yaml.safe_load(EDGE_CASES.read_text(encoding="utf-8"))
    for part in ("edge", "divergence"):
        for e in edges[part]:
            cases.append({**e, "id": f"{part}-{e['id']}", "part": part})
    return cases


def f1_repr(x: float | int | None) -> str | None:
    return None if x is None else repr(float(x))


def official_accuracy(official, cases: list[dict], verdicts: list[int]) -> dict:
    """The official evaluator's own aggregation (compute_acc_by_diff) over these cases."""
    path = CASES_DIR / "difficulty.jsonl"
    path.write_text(
        "".join(json.dumps({"difficulty": c["difficulty"]}) + "\n" for c in cases),
        encoding="utf-8",
    )
    results = [{"sql_idx": i, "res": v} for i, v in enumerate(verdicts)]
    *scores, counts = official.ex.compute_acc_by_diff(results, str(path))
    return {
        "accuracy_percent": dict(zip((*DIFFICULTIES, "total"), scores, strict=True)),
        "counts": dict(zip((*DIFFICULTIES, "total"), counts, strict=True)),
    }


def project_accuracy(cases: list[dict], verdicts: list[int]) -> dict:
    by: dict[str, list[int]] = defaultdict(list)
    for c, v in zip(cases, verdicts, strict=True):
        by[c["difficulty"]].append(v)
        by["total"].append(v)
    keys = (*DIFFICULTIES, "total")
    return {
        "accuracy_percent": {k: sum(by[k]) / len(by[k]) * 100 for k in keys},
        "counts": {k: len(by[k]) for k in keys},
    }


def main() -> None:
    cfg = config()
    timeout = cfg["official_evaluator"]["meta_time_out_s"]
    settings = ScoreSettings.from_config(cfg)
    if settings.timeout_s != timeout:
        sys.exit("the project's time limit per pair must be the official evaluator's")
    questions = bird.questions()
    cases = build_cases(questions, cfg)
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    with (CASES_DIR / "cases.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(cases)} cases: {dict(Counter(c['part'] for c in cases))}")

    # the project's scoring
    targets = {db: benchmark_target(db) for db in benchmark_schemas()}
    ours = []
    with Scorer(targets, settings) as scorer:
        for i, c in enumerate(cases, 1):
            ours.append(scorer.score(c["db"], c["predicted"], c["gold"]))
            if i % 250 == 0:
                print(f"project: {i}/{len(cases)}")

    # the official evaluator, parallel query off as in the project's execution
    official = off.load(official_evaluator_dir(cfg))
    off.use_project_database(official, timeout, parallel_plans=False)
    theirs = []
    for i, c in enumerate(cases, 1):
        start = time.perf_counter()
        ex = off.execute_pair(official, "ex", c["predicted"], c["gold"], timeout)
        f1 = off.execute_pair(official, "f1", c["predicted"], c["gold"], timeout)
        theirs.append({"ex": ex, "soft_f1": f1, "seconds": time.perf_counter() - start})
        if i % 250 == 0:
            print(f"official: {i}/{len(cases)}")

    # the official EX over the gold cases with default parallel plans
    gold_cases = [c for c in cases if c["part"] == "gold"]
    off.use_project_database(official, timeout, parallel_plans=True)
    parallel = [
        off.execute_pair(official, "ex", c["predicted"], c["gold"], timeout) for c in gold_cases
    ]

    records = []
    for c, o, t in zip(cases, ours, theirs, strict=True):
        records.append(
            {
                "id": c["id"],
                "part": c["part"],
                "kind": c.get("kind"),
                "question_id": c.get("question_id"),
                "expect": c.get("expect"),
                "official_ex": t["ex"],
                "project_ex": o.ex,
                "official_soft_f1": f1_repr(t["soft_f1"]),
                "project_soft_f1": f1_repr(o.soft_f1),
                "outcome": o.outcome,
                "error_kind": o.error_kind,
                "project_seconds": o.seconds,
                "official_seconds": round(t["seconds"], 3),
                "gold_set_hash": o.gold_set_hash,
            }
        )

    parts: dict[str, dict] = {}
    for part in ("gold", "mutant", "edge", "divergence"):
        rs = [r for r in records if r["part"] == part]
        parts[part] = {
            "cases": len(rs),
            "official_ex_1": sum(r["official_ex"] for r in rs),
            "project_ex_1": sum(r["project_ex"] for r in rs),
            "same_ex": sum(r["official_ex"] == r["project_ex"] for r in rs),
            "same_soft_f1": sum(r["official_soft_f1"] == r["project_soft_f1"] for r in rs),
            "project_outcomes": dict(sorted(Counter(r["outcome"] for r in rs).items())),
        }
    kinds: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        if r["part"] == "mutant":
            kinds[r["kind"]]["cases"] += 1
            kinds[r["kind"]]["official_ex_1"] += r["official_ex"]
            kinds[r["kind"]]["same_ex"] += r["official_ex"] == r["project_ex"]
    parts["mutant"]["kinds"] = {k: dict(v) for k, v in sorted(kinds.items())}
    for part in ("gold", "mutant"):
        idx = [i for i, c in enumerate(cases) if c["part"] == part]
        sub = [cases[i] for i in idx]
        parts[part]["official_aggregate"] = official_accuracy(
            official, sub, [records[i]["official_ex"] for i in idx]
        )
        parts[part]["project_aggregate"] = project_accuracy(
            sub, [records[i]["project_ex"] for i in idx]
        )
        parts[part]["same_aggregate"] = (
            parts[part]["official_aggregate"] == parts[part]["project_aggregate"]
        )
    for part in ("edge", "divergence"):
        parts[part]["official_as_expected"] = sum(
            r["official_ex"] == r["expect"] for r in records if r["part"] == part
        )

    checked = [r for r in records if r["part"] != "divergence"]
    stored = {
        r["question_id"]: r for r in json.loads(GOLD_RUN.read_text(encoding="utf-8"))["questions"]
    }
    drift = [
        r["question_id"]
        for r in records
        if r["part"] == "gold" and r["gold_set_hash"] != stored[r["question_id"]]["set_hash"]
    ]
    out = {
        "runs": 1,
        "official_evaluator": {
            "commit": cfg["official_evaluator"]["commit"],
            "meta_time_out_s": timeout,
            "connection": "analyst_ro; search_path over the benchmark schemas; "
            "parallel query off (except the parallel-plans run)",
        },
        "project": {"timeout_s": settings.timeout_s, "fetch_rows": settings.fetch_rows},
        "identical_ex": {
            "cases": len(checked),
            "identical": sum(r["official_ex"] == r["project_ex"] for r in checked),
        },
        "identical_soft_f1": {
            "cases": len(checked),
            "identical": sum(r["official_soft_f1"] == r["project_soft_f1"] for r in checked),
        },
        "parts": parts,
        "ex_mismatches": [r for r in checked if r["official_ex"] != r["project_ex"]],
        "soft_f1_mismatches": [r for r in checked if r["official_soft_f1"] != r["project_soft_f1"]],
        "divergence_cases": [r for r in records if r["part"] == "divergence"],
        "edge_cases": [r for r in records if r["part"] == "edge"],
        "parallel_plans": {
            "gold_cases": len(parallel),
            "official_ex_1": sum(parallel),
            "scored_0": [
                c["question_id"] for c, v in zip(gold_cases, parallel, strict=True) if v != 1
            ],
        },
        "gold_drift": {
            "compared_with": "results/metrics/gold_execution.json",
            "compared": len(gold_cases),
            "same_as_stored": len(gold_cases) - len(drift),
            "differ": drift,
            "reads_the_clock": [q["question_id"] for q in questions if reads_the_clock(q["SQL"])],
        },
        "cases": [
            [
                r["id"],
                r["official_ex"],
                r["project_ex"],
                r["official_soft_f1"],
                r["project_soft_f1"],
                r["outcome"],
            ]
            for r in records
        ],
    }
    write_json(OUT, out)
    print(json.dumps({k: out[k] for k in ("identical_ex", "identical_soft_f1")}, indent=1))
    for part, p in parts.items():
        print(part, {k: v for k, v in p.items() if k not in ("kinds",)})
    print("parallel plans:", out["parallel_plans"])
    print("gold drift:", out["gold_drift"]["differ"])
    if out["ex_mismatches"] or parts["gold"]["project_ex_1"] != len(gold_cases):
        sys.exit("the project's EX does not reproduce the official evaluator")


if __name__ == "__main__":
    main()
