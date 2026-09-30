"""Near misses: wrong answers whose result holds the right one, up to its columns or rounding.

Execution accuracy compares result rows exactly, as the benchmark's official evaluator does, so an
answer that returns the right values with one extra column, the columns in another order, or a
number with more decimal places counts as wrong. This counts such answers among the winning
design's wrong answers on the held-out set, to separate format misses from wrong logic. It
changes no score.

A wrong answer (answered, its query ran, EX 0) is a near miss when the gold result's rows, as a
set, can be recovered from the answer's result by:
- **columns:** choosing and ordering some of its columns (it has extra columns, or the right
  columns in another order);
- **rounding:** rounding every number in both results to two decimal places;
- or both.
Everything else is "other". Both queries are run again here, through the agent's guard and the
evaluation limits (50,000 rows, 30 s each); results over 5,000 rows, or with more than
`MAX_SEARCH` column choices to try, are not checked and are counted as such. One run.

Writes results/metrics/near_miss.json.

Usage:
    uv run python scripts/58_near_miss.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import verify  # noqa: E402
from src.agent.confidence import answers_path, confidence_config  # noqa: E402
from src.data.bird import questions, write_json  # noqa: E402
from src.db.execute import Limits  # noqa: E402
from src.eval.near_miss import MAX_ROWS_CHECKED, MAX_SEARCH, classify  # noqa: E402
from src.eval.records import read_records  # noqa: E402
from src.eval.reports import split_ids, subset  # noqa: E402
from src.tools.toolbox import Toolbox  # noqa: E402

OUT = ROOT / "results/metrics/near_miss.json"


def main() -> None:
    conf = confidence_config()
    path, answer_run = answers_path(conf)
    held_ids = split_ids("held_out")
    held = subset(read_records(path), held_ids)
    golds = {q["question_id"]: q["SQL"] for q in questions() if q["question_id"] in held_ids}
    wrong = [
        r
        for r in held
        if not r["declined"] and r["final_sql"] and r["correct"] == 0 and r["score_outcome"] == "ok"
    ]
    limits = Limits(max_rows=50_000, timeout_s=30.0, count_total=False)
    kinds: Counter = Counter()
    examples: dict[str, list] = {}
    boxes: dict[str, Toolbox] = {}
    try:
        for r in sorted(wrong, key=lambda r: (r["db_id"], r["question_id"])):
            box = boxes.setdefault(r["db_id"], Toolbox(r["db_id"]))
            pred = verify.fetch(box.sql.guard, box.executor, r["final_sql"], limits)
            gold = verify.fetch(box.sql.guard, box.executor, golds[r["question_id"]], limits)
            if not (pred.ok and gold.ok):
                kind = "not_checked"
            else:
                kind = classify(pred.rows, gold.rows, len(gold.columns))
            kinds[kind] += 1
            examples.setdefault(kind, []).append(r["question_id"])
    finally:
        for b in boxes.values():
            b.close()
    near = kinds["columns"] + kinds["rounding"] + kinds["columns_and_rounding"]
    out = {
        "note": "one run; the winning design's wrong answers on the held-out set whose query "
        "ran; no score is changed",
        "answers": answer_run,
        "held_out_questions": len(held),
        "wrong_with_a_result": len(wrong),
        "near_misses": near,
        "by_kind": {
            k: kinds[k]
            for k in ("columns", "rounding", "columns_and_rounding", "other", "not_checked")
        },
        "question_ids": {k: sorted(v) for k, v in sorted(examples.items())},
        "rules": {
            "columns": "the gold rows are some choice and order of the answer's columns",
            "rounding": "equal after rounding every number to two decimal places",
            "max_rows_checked": MAX_ROWS_CHECKED,
            "max_column_choices": MAX_SEARCH,
        },
    }
    write_json(OUT, out)
    print(f"{near} near misses among {len(wrong)} wrong answers with a result: {dict(kinds)}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
