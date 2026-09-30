"""Where the analyst goes wrong: every wrong answer of the winning run on the held-out set,
classified by how its SQL and its rows differ from the expert's (src/eval/errors.py).

Both queries of each answer whose query ran are run again through the agent's checker, as the
question's schema role, with the evaluation limits (50,000 rows, 30 s), to relate the rows. A
sample is checked by hand in results/reviews/error_hand_check.yaml (`--hand-check-template`
writes it with the hand fields blank); the report gives the agreement once every sampled answer
has a hand label, and marks it pending until then.

Exploratory, not pre-registered: the same classification for the Claude Sonnet 5 runs of the five
designs on the ablation set, with each design's accuracy counting format-only misses as right and
the winner's paired differences from the others under both measures.

Writes results/metrics/error_analysis.json. No model is called. One run.

Usage:
    uv run python scripts/64_error_analysis.py --hand-check-template
    uv run python scripts/64_error_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import verify  # noqa: E402
from src.agent.confidence import answers_path, confidence_config  # noqa: E402
from src.data.bird import questions, write_json  # noqa: E402
from src.db.execute import Limits  # noqa: E402
from src.eval import errors  # noqa: E402
from src.eval.bootstrap import paired_difference  # noqa: E402
from src.eval.config import config as eval_config  # noqa: E402
from src.eval.near_miss import classify  # noqa: E402
from src.eval.records import read_records  # noqa: E402
from src.eval.reports import split_ids, subset  # noqa: E402
from src.tools.toolbox import Toolbox  # noqa: E402

OUT = ROOT / "results/metrics/error_analysis.json"
NEAR_MISS = ROOT / "results/metrics/near_miss.json"
REVIEW = ROOT / "results/reviews/error_hand_check.yaml"
ABLATION_RUNS = ROOT / "results/runs/ablation"
DESIGNS = ("d1", "d2", "d3", "d4", "d5")
HAND_CHECK_SIZE = 30
HAND_CHECK_SEED = 20260930
LIMITS = Limits(max_rows=50_000, timeout_s=30.0, count_total=False)
HAND_FIELDS = ("hand_category", "expert_questionable", "note")


class Runner:
    """Runs queries through the checker, one toolbox per database, the expert results cached."""

    def __init__(self) -> None:
        self.boxes: dict[str, Toolbox] = {}
        self.gold: dict[int, verify.FinalResult] = {}

    def run(self, db: str, sql: str) -> verify.FinalResult:
        box = self.boxes.setdefault(db, Toolbox(db))
        return verify.fetch(box.sql.guard, box.executor, sql, LIMITS)

    def expert(self, qid: int, db: str, sql: str) -> verify.FinalResult:
        if qid not in self.gold:
            self.gold[qid] = self.run(db, sql)
        return self.gold[qid]

    def close(self) -> None:
        for b in self.boxes.values():
            b.close()


def analyze_answer(r: dict, gold_sql: str, near_kind: str | None, runner: Runner) -> dict:
    """Category, differences and row relation of one wrong answer. `near_kind` None: computed."""
    row = {
        "question_id": r["question_id"],
        "db_id": r["db_id"],
        "difficulty": r["difficulty"],
        "relation": None,
    }
    if r["score_outcome"] == "ok" and r["final_sql"]:
        pred = runner.run(r["db_id"], r["final_sql"])
        gold = runner.expert(r["question_id"], r["db_id"], gold_sql)
        if pred.ok and gold.ok:
            if near_kind is None:
                near_kind = classify(pred.rows, gold.rows, len(gold.columns))
            row["relation"] = errors.result_relation(
                pred.rows, gold.rows, len(pred.columns), len(gold.columns)
            )
    row.update(errors.categorize(r["score_outcome"], near_kind, r["final_sql"], gold_sql))
    return row


def table(rows: list[dict], key: str) -> dict:
    """Category counts per value of `key` (database or difficulty)."""
    out: dict[str, Counter] = {}
    for r in rows:
        out.setdefault(r[key], Counter())[r["category"]] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def hand_check(rows: list[dict]) -> dict:
    if not REVIEW.exists():
        return {"status": "pending", "reason": f"{REVIEW.relative_to(ROOT)} does not exist"}
    review = yaml.safe_load(REVIEW.read_text(encoding="utf-8"))
    items = review["items"]
    by_id = {r["question_id"]: r for r in rows}
    blank = [i["question_id"] for i in items if not i.get("hand_category")]
    if blank:
        return {"status": "pending", "reason": f"{len(blank)} sampled answers have no hand label"}
    unknown = [i["hand_category"] for i in items if i["hand_category"] not in errors.CATEGORIES]
    if unknown:
        raise SystemExit(f"unknown hand categories: {sorted(set(unknown))}")
    auto = [by_id[i["question_id"]]["category"] for i in items]
    hand = [i["hand_category"] for i in items]
    confusion = Counter(f"{a} -> {h}" for a, h in zip(auto, hand, strict=True) if a != h)
    spot = review.get("spot_checked") or []
    return {
        "status": "done",
        "reviewer": review.get("reviewer"),
        "spot_checked_by": review.get("spot_checked_by"),
        "spot_checked": len(spot),
        # of the spot-checked answers, those whose expert query the spot check found questionable
        "spot_checked_questionable": sum(bool(s.get("expert_questionable")) for s in spot),
        "criteria": review.get("criteria"),
        **errors.agreement(auto, hand),
        "disagreements": dict(sorted(confusion.items())),
        "expert_questionable": sum(bool(i.get("expert_questionable")) for i in items),
        "hand_categories": dict(sorted(Counter(hand).items())),
    }


def write_template(rows: list[dict], qs: dict, records: dict) -> None:
    sample = errors.hand_check_sample(rows, HAND_CHECK_SIZE, HAND_CHECK_SEED)
    items = []
    for r in sample:
        q = qs[r["question_id"]]
        items.append(
            {
                "question_id": r["question_id"],
                "db_id": r["db_id"],
                "question": q["question"],
                "hint": q.get("evidence") or "",
                "answer_sql": records[r["question_id"]]["final_sql"],
                "expert_sql": q["SQL"],
                "outcome": records[r["question_id"]]["score_outcome"],
                "relation": r["relation"],
                "hand_category": None,
                "expert_questionable": None,
                "note": None,
            }
        )
    doc = {
        "answers": "the winning run's wrong held-out answers",
        "criteria": "hand_category: the first category, in the order and with the meanings in "
        "src/eval/errors.py, that describes a real mistake, a difference that changes the answer; "
        "differences in how the SQL is written that give the same answer do not count; "
        "format_only when only the shape of right rows and values differs. Judged from the "
        "question, the hint and both queries, without the automatic category. "
        "expert_questionable: true if the expert query itself does not answer the question as "
        "asked (counted, never re-scored)",
        "reviewer": None,
        "reviewed_on": None,
        "items": items,
    }
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8"
    )
    print(f"{len(items)} answers -> {REVIEW.relative_to(ROOT)} (hand fields blank)")


def designs(qs: dict, runner: Runner, boot: dict) -> dict:
    """Exploratory: the five designs' Claude Sonnet 5 runs on the ablation set."""
    per: dict[str, dict] = {}
    right: dict[str, np.ndarray] = {}
    right_up_to_format: dict[str, np.ndarray] = {}
    ids = None
    for d in DESIGNS:
        path = ABLATION_RUNS / f"ablation-{d}-claude-sonnet-5-evidence.jsonl"
        recs = sorted(read_records(path), key=lambda r: r["question_id"])
        if ids is None:
            ids = [r["question_id"] for r in recs]
        if [r["question_id"] for r in recs] != ids:
            raise SystemExit(f"{path.name}: not the same questions as d1")
        rows = [
            analyze_answer(r, qs[r["question_id"]]["SQL"], None, runner)
            for r in recs
            if r["correct"] != 1
        ]
        fmt = {r["question_id"] for r in rows if r["category"] == "format_only"}
        right[d] = np.array([r["correct"] == 1 for r in recs], dtype=float)
        right_up_to_format[d] = np.array(
            [r["correct"] == 1 or r["question_id"] in fmt for r in recs], dtype=float
        )
        per[d] = {
            "questions": len(recs),
            "wrong": len(rows),
            "categories": dict(sorted(Counter(r["category"] for r in rows).items())),
            "execution_accuracy": float(right[d].mean()),
            "accuracy_up_to_format": float(right_up_to_format[d].mean()),
        }
    n = len(ids)
    kw = {"count": boot["resamples"], "confidence": boot["confidence"], "seed": boot["seed"]}
    pairs = {}
    for d in DESIGNS[1:]:
        pairs[f"d1 - {d}"] = {
            "execution_accuracy": paired_difference(
                lambda i, a=right["d1"], b=right[d]: a[i].mean(),
                lambda i, b=right[d]: b[i].mean(),
                n,
                **kw,
            ).to_dict(),
            "accuracy_up_to_format": paired_difference(
                lambda i, a=right_up_to_format["d1"]: a[i].mean(),
                lambda i, b=right_up_to_format[d]: b[i].mean(),
                n,
                **kw,
            ).to_dict(),
        }
    return {
        "note": "exploratory, not pre-registered: format-only misses counted as right; one run "
        "per design; intervals are paired 95% bootstrap intervals over the ablation questions",
        "model": "claude-sonnet-5",
        "designs": per,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hand-check-template", action="store_true")
    args = parser.parse_args()

    path, run = answers_path(confidence_config())
    held = subset(read_records(path), split_ids("held_out"))
    wrong = sorted((r for r in held if r["correct"] != 1), key=lambda r: r["question_id"])
    qs = {q["question_id"]: q for q in questions()}
    near_ids = json.loads(NEAR_MISS.read_text(encoding="utf-8"))["question_ids"]
    near_kind = {q: kind for kind, qids in near_ids.items() for q in qids}

    runner = Runner()
    try:
        rows = [
            analyze_answer(
                r, qs[r["question_id"]]["SQL"], near_kind.get(r["question_id"], "other"), runner
            )
            for r in wrong
        ]
        if args.hand_check_template:
            write_template(rows, qs, {r["question_id"]: r for r in wrong})
            return
        exploratory = designs(qs, runner, eval_config()["bootstrap"])
    finally:
        runner.close()

    parts = Counter(p for r in rows for p in r["differences"])
    out = {
        "note": "one run; the winning run's wrong answers on the held-out set; categories by the "
        "first part in which the SQL differs from the expert's (src/eval/errors.py); no score "
        "is changed",
        "answers": run,
        "held_out_questions": len(held),
        "wrong": len(wrong),
        "categories": {c: sum(r["category"] == c for r in rows) for c in errors.CATEGORIES},
        "filter_kinds": dict(
            sorted(Counter(r["subkind"] for r in rows if r["category"] == "filter").items())
        ),
        "no_result_kinds": dict(
            sorted(Counter(r["subkind"] for r in rows if r["category"] == "no_result").items())
        ),
        "several_parts_differ": sum(len(r["differences"]) > 1 for r in rows),
        "parts_differing": {p: parts[p] for p in errors.PARTS},
        "relations": dict(sorted(Counter(str(r["relation"]) for r in rows).items())),
        "category_by_relation": {
            c: dict(sorted(Counter(str(r["relation"]) for r in rows if r["category"] == c).items()))
            for c in errors.CATEGORIES
            if any(r["category"] == c for r in rows)
        },
        "by_database": table(rows, "db_id"),
        "by_difficulty": table(rows, "difficulty"),
        "hand_check": hand_check(rows),
        "designs_exploratory": exploratory,
        "answers_detail": [
            {
                k: r[k]
                for k in (
                    "question_id",
                    "db_id",
                    "difficulty",
                    "category",
                    "subkind",
                    "relation",
                    "differences",
                )
            }
            for r in rows
        ],
    }
    write_json(OUT, out)
    print(f"{len(rows)} wrong answers: {out['categories']}")
    print(f"hand check: {out['hand_check']['status']}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
