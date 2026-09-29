"""The winning design on the hand-written banking set, category by category.

Refuses to run unless the pre-registration is frozen and unchanged. From the `own` stage's
record of the winner (results/runs/own/):

- the run summarised (execution accuracy over the questions with a gold result, cost, steps,
  errors);
- each category's success (src/eval/summary.py `own_set_behaviour`): standard and multi-step
  questions by execution accuracy; ambiguous ones by a clarifying question, or a stated
  assumption whose SQL matches an accepted reading; unanswerable ones by declining with a
  reason; false premises by a correction of the premise. Comparative and causal questions are
  scored by the statistical checks, later; here only their count and declines;
- how often a clear question with a true premise (standard or multi-step) was declined, met
  with a clarifying question, or given a premise correction: the same behaviours where they are
  not called for;
- beside the ambiguous and false-premise categories' success, their reviewed success: the
  answers that pass on form alone (a clarifying question, a premise correction) checked by hand
  against the frozen set's accepted readings and corrections (src/eval/own_review.py,
  results/reviews/own_set_review.yaml); null until the review is filled in;
- the pre-registered predictions this stage can check (P12: standard and multi-step EX against
  the held-out benchmark EX of results/metrics/benchmark_main.json; P13: the ambiguous,
  unanswerable and false-premise categories, by the pre-registered rules).

One run. Writes results/metrics/own_set.json; with --review-template, writes the review file to
fill in instead (never over a review that has verdicts).

Usage:
    uv run python scripts/46_own_set_report.py [--review-template]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.stages import run_id, winner  # noqa: E402
from src.data.bird import write_json  # noqa: E402
from src.eval import own_review, own_set, preregistration  # noqa: E402
from src.eval.config import config as eval_config  # noqa: E402
from src.eval.records import read_records  # noqa: E402
from src.eval.reports import NOT_MEASURED, cost_row, with_predictions  # noqa: E402
from src.eval.summary import mean_interval, own_set_behaviour, summarise  # noqa: E402

RUNS = ROOT / "results/runs/own"
MAIN_REPORT = ROOT / "results/metrics/benchmark_main.json"
OUT = ROOT / "results/metrics/own_set.json"
REVIEW = ROOT / "results/reviews/own_set_review.yaml"
MODEL = "claude-sonnet-5"
CATEGORY_NAMES = {
    "a": "standard",
    "b": "multi-step",
    "c": "ambiguous",
    "d": "unanswerable",
    "e": "false premise",
    "f": "comparative or causal",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--review-template", action="store_true", help="write the review file to fill")
    a = p.parse_args()
    preregistration.require()
    design = winner()
    rid = run_id("own", design, MODEL, False, None)
    path = RUNS / f"{rid}.jsonl"
    if not path.exists():
        sys.exit(f"missing run: {path.relative_to(ROOT)}")
    records = read_records(path)
    review = own_review.load(REVIEW)
    if a.review_template:
        if review and any(i.get("verdict") for i in review["items"]):
            sys.exit(f"{REVIEW.relative_to(ROOT)} has verdicts; not overwritten")
        frozen = own_set.load(ROOT / eval_config()["own_set"]["path"])
        questions = {q["id"]: q for q in frozen["questions"]}
        REVIEW.parent.mkdir(parents=True, exist_ok=True)
        REVIEW.write_text(
            yaml.safe_dump(
                own_review.template(records, questions, f"own/{rid}"),
                allow_unicode=True,
                sort_keys=False,
                width=100,
            ),
            encoding="utf-8",
            newline="\n",
        )
        print(f"wrote {REVIEW.relative_to(ROOT)}: fill in each verdict and note")
        return
    held_out = json.loads(MAIN_REPORT.read_text(encoding="utf-8"))["winner"]["held_out"]
    predictions = {
        p["id"]: p
        for p in preregistration.parse(
            (ROOT / "results/metrics/preregistration.md").read_text(encoding="utf-8")
        )["predictions"]
    }
    out = report(design, records, held_out["execution_accuracy"], predictions, review)
    write_json(OUT, out)
    for cat, row in out["categories"].items():
        s = row.get("success")
        shown = f"{s['estimate']:.2f}" if s else "scored later"
        print(f"{cat} ({row['name']}): {row['questions']} questions, success {shown}")
    print(f"wrote {OUT.relative_to(ROOT)}")


def report(
    design: str,
    records: list[dict],
    held_out_ex: dict,
    predictions: dict,
    review: dict | None = None,
) -> dict:
    """The banking-set report for the winner's records; `held_out_ex`: its execution accuracy
    on the benchmark's held-out set, with its interval; `review`: the filled review file, if
    any."""
    if any(r["source"] != "own" for r in records):
        raise ValueError("the banking-set report takes banking-set records only")
    run = summarise(records)
    run["latency_s"] = dict(NOT_MEASURED)  # the stage's one arm is batched
    behaviour = own_set_behaviour(records)
    categories = {
        cat: {"name": CATEGORY_NAMES[cat], **row}
        for cat, row in behaviour.items()
        if cat in CATEGORY_NAMES
    }
    answerable = [r for r in records if r["category"] in ("a", "b")]
    standard = summarise(answerable)["execution_accuracy"] if answerable else None
    reviewed = own_review.reviewed_success(records, review) if review else {}
    for cat in ("c", "e"):
        if cat in categories:
            s = reviewed.get(cat)
            categories[cat]["reviewed_success"] = mean_interval(s) if s else None
    out = {
        "note": "one run; intervals are 95% bootstrap over questions",
        "design": design,
        "model": MODEL,
        "run": run,
        "categories": categories,
        "not_called_for_ab": {
            "note": "standard and multi-step questions: clear, with a true premise",
            "declined": behaviour.get("false_decline_rate_ab"),
            "clarifying_question": behaviour.get("clarification_rate_ab"),
            "premise_correction": behaviour.get("premise_correction_rate_ab"),
        },
        "review": {
            "file": REVIEW.relative_to(ROOT).as_posix(),
            "done": bool(review),
            "criteria": own_review.CRITERIA,
            "reviewer": (review or {}).get("reviewer"),
            "reviewed_on": (review or {}).get("reviewed_on"),
            "checked": len((review or {}).get("items", [])),
        },
        "cost_per_correct": [cost_row(run, model=MODEL, design=design, set="own")],
    }

    def success(cat: str) -> dict | None:
        return categories.get(cat, {}).get("success")

    observed = {
        "P12": {
            "standard_and_multi_step_ex": standard,
            "held_out_benchmark_ex": held_out_ex,
        },
        "P13": {
            "ambiguous": success("c"),
            "unanswerable": success("d"),
            "false_premise": success("e"),
        },
    }
    out["predictions"] = with_predictions(observed, predictions)
    return out


if __name__ == "__main__":
    main()
