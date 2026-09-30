"""Calibrated confidence, the decline threshold and the critic, on held-out questions only.

Refuses to run unless the pre-registration is frozen and unchanged. From the winning design's run
on all 500 questions (results/runs/main/) and the critic's reviews (results/runs/critic/):

- **Calibration:** Platt scaling (primary) and isotonic regression (secondary) of the stated
  confidence, fitted on the calibration split (the ablation set) and applied unchanged to the
  held-out set, where ECE, the Brier score, AUROC and the reliability bins are reported, beside
  those of the raw confidence. Nothing is reported on the data a calibrator was fitted on except
  its parameters and the threshold choice.
- **The decline threshold:** chosen on the calibration split on the Platt-calibrated confidence
  (configs/confidence.yaml `decline`), then applied to the held-out set: the coverage and the
  answered questions' accuracy there, with intervals.
- **The critic:** its confidence against the stated one's calibrated confidence by AURC, paired, on
  the held-out set (pre-registered); its own Platt calibration, fitted on the calibration split.
- **Combined (exploratory, not pre-registered):** one logistic regression on the stated and the
  critic's confidence, fitted on the calibration split, evaluated on the held-out set.
- **Stated, zeroed without a result (exploratory, not pre-registered):** the stated confidence set
  to 0 where the answer's query returned no result (refused or failing). The critic gives those
  answers 0 without a call, from the execution alone; this baseline shows how much of the critic's
  ranking that alone accounts for.
- The predictions this stage checks: P07 (the calibrated confidence's ECE), P08 (accuracy and
  coverage at the threshold).

A declined answer returns no result: its calibrated and critic confidence are 0, and it ranks
below every answered question. A review that ended without a verdict counts at the configured
"cannot tell" confidence. Intervals are 95% bootstrap over the held-out questions, with the
fitted calibrators and the threshold held fixed (the variability of fitting them on 150 questions
is not in the intervals). Every result is from one run.

Writes results/metrics/calibration.json and results/metrics/risk_coverage.json.

Usage:
    uv run python scripts/53_confidence_report.py
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.agent import confidence  # noqa: E402
from src.data.bird import write_json  # noqa: E402
from src.eval import preregistration  # noqa: E402
from src.eval.bootstrap import bootstrap  # noqa: E402
from src.eval.calibrate import (  # noqa: E402
    Isotonic,
    Logistic,
    Platt,
    calibrated,
    choose_threshold,
    threshold_outcome,
)
from src.eval.config import config as eval_config  # noqa: E402
from src.eval.records import answered_correct, read_records  # noqa: E402
from src.eval.reports import split_ids, subset, with_predictions  # noqa: E402
from src.eval.selective import oracle_aurc  # noqa: E402
from src.eval.summary import answered_confidence, compare, summarise  # noqa: E402

CALIBRATION = ROOT / "results/metrics/calibration.json"
RISK_COVERAGE = ROOT / "results/metrics/risk_coverage.json"
SELECTIVE_KEYS = ("aurc", "e_aurc", "accuracy_at_80", "accuracy_at_90")


def critic_confidences(answers: list[dict], reviews: list[dict], missing: float) -> np.ndarray:
    """The critic's confidence for each answer, in the answers' order."""
    by_id = {r["question_id"]: r for r in reviews}
    if {a["question_id"] for a in answers} - by_id.keys():
        raise ValueError("the critic has not reviewed every answer")
    return np.array(
        [
            missing
            if by_id[a["question_id"]]["confidence"] is None
            else by_id[a["question_id"]]["confidence"]
            for a in answers
        ]
    )


def with_confidence(records: list[dict], p: np.ndarray) -> list[dict]:
    return [{**r, "confidence": float(v)} for r, v in zip(records, p, strict=True)]


def arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stated = np.array([r["confidence"] for r in records], dtype=float)
    y = np.array([answered_correct(r) for r in records], dtype=float)
    declined = np.array([r["declined"] for r in records], dtype=bool)
    return stated, y, declined


def report(
    answers: list[dict],
    reviews: list[dict],
    ids: dict[str, set],
    conf: dict,
    predictions: dict,
    answer_run: str,
    cfg: dict | None = None,
) -> tuple[dict, dict]:
    """(calibration.json, risk_coverage.json) from the winner's answers and the critic's reviews.
    `ids`: the question ids of the calibration (ablation) and held-out splits; `cfg`: the
    evaluation config (tests pass fewer resamples)."""
    cfg = cfg or eval_config()
    b = cfg["bootstrap"]
    boot = partial(bootstrap, count=b["resamples"], confidence=b["confidence"], seed=b["seed"])
    fit_set, test = subset(answers, ids["ablation"]), subset(answers, ids["held_out"])
    if len(fit_set) != len(ids["ablation"]) or len(test) != len(ids["held_out"]):
        raise ValueError("the answers must cover the calibration and held-out splits")
    missing = conf["critic"]["missing_confidence"]

    x_fit, y_fit, d_fit = arrays(fit_set)
    x_test, y_test, d_test = arrays(test)
    stated_fit = answered_confidence(x_fit, d_fit)
    stated_test = answered_confidence(x_test, d_test)
    platt = Platt.fit(stated_fit, y_fit)
    iso = Isotonic.fit(stated_fit, y_fit)
    c_fit = answered_confidence(critic_confidences(fit_set, reviews, missing), d_fit)
    c_test = answered_confidence(critic_confidences(test, reviews, missing), d_test)
    critic_platt = Platt.fit(c_fit, y_fit)
    combined = Logistic.fit(conf["combined"]["inputs"], [stated_fit, c_fit], y_fit)

    def zeroed_without_result(records: list[dict], stated: np.ndarray) -> np.ndarray:
        by_id = {r["question_id"]: r for r in reviews}
        failed = [
            (by_id[r["question_id"]]["not_reviewed"] or "").startswith("final_sql_")
            for r in records
        ]
        return np.where(np.array(failed, dtype=bool), 0.0, stated)

    scores = {
        "stated_raw": stated_test,
        "stated_platt": calibrated(platt(stated_test), d_test),
        "stated_isotonic": calibrated(iso(stated_test), d_test),
        "critic_raw": c_test,
        "critic_platt": calibrated(critic_platt(c_test), d_test),
        "combined_exploratory": calibrated(combined([stated_test, c_test]), d_test),
        "stated_zeroed_without_result": zeroed_without_result(test, stated_test),
    }
    runs = {name: with_confidence(test, p) for name, p in scores.items()}
    summaries = {name: summarise(rs, cfg) for name, rs in runs.items()}

    # the decline threshold, chosen on the calibration split's calibrated confidence
    rule = conf["decline"]
    if rule["calibrator"] != "platt":
        raise ValueError("the pre-registered decline threshold is on the Platt calibration")
    p_fit = calibrated(platt(stated_fit), d_fit)
    choice = choose_threshold(p_fit, y_fit, d_fit, rule["target_accuracy"], rule["min_answered"])
    t = choice["threshold"]
    raw_at = float(stated_fit[~d_fit & (p_fit >= t)].min())
    outcome = threshold_outcome(scores["stated_platt"], y_test, d_test, t, boot)

    reviews_by_id = {r["question_id"]: r for r in reviews}

    def review_counts(rs: list[dict]) -> dict:
        got = [reviews_by_id[r["question_id"]] for r in rs]
        return {
            "questions": len(got),
            "reviewed": sum(r["reviewed"] for r in got),
            "not_reviewed": {
                k: sum(r["not_reviewed"] == k for r in got)
                for k in sorted({r["not_reviewed"] for r in got if r["not_reviewed"]})
            },
            "without_verdict": sum(r["reviewed"] and r["confidence"] is None for r in got),
            "verdicts": {v: sum(r["verdict"] == v for r in got) for v in confidence.VERDICTS},
            "cost_usd": sum(r["cost_usd"] for r in got),
        }

    calibration_out = {
        "note": "one run; calibrators fitted on the calibration split (the ablation set) and "
        "evaluated on the held-out set only; intervals are 95% bootstrap over the held-out "
        "questions with the calibrators and the threshold held fixed",
        "answers": answer_run,
        "calibration_split": {
            "questions": len(fit_set),
            "declined": int(d_fit.sum()),
            "calibrators": {"platt": platt.to_dict(), "isotonic": iso.to_dict()},
            # a positive slope keeps the stated confidence's order, so its ranking measures
            "platt_keeps_order": platt.slope > 0,
        },
        "held_out": {
            "questions": len(test),
            "declined": int(d_test.sum()),
            "execution_accuracy": summaries["stated_raw"]["execution_accuracy"],
            **{
                name: summaries[name]["calibration"]
                for name in ("stated_raw", "stated_platt", "stated_isotonic")
            },
        },
        "decline": {
            "calibrator": rule["calibrator"],
            "target_accuracy": rule["target_accuracy"],
            "min_answered": rule["min_answered"],
            "chosen_on_calibration_split": choice,
            "raw_confidence_at_threshold": raw_at,
            "held_out": outcome,
        },
        "critic": {
            "model": conf["critic"]["model"],
            "missing_confidence": missing,
            "calibration_split": {"platt": critic_platt.to_dict(), **review_counts(fit_set)},
            "held_out": {
                **review_counts(test),
                "critic_raw": summaries["critic_raw"]["calibration"],
                "critic_platt": summaries["critic_platt"]["calibration"],
            },
        },
        "combined_exploratory": {
            "note": "not pre-registered: a logistic regression on the stated and the critic's "
            "confidence, fitted on the calibration split",
            "fit": combined.to_dict(),
            "held_out": summaries["combined_exploratory"]["calibration"],
        },
    }
    calibration_out["predictions"] = with_predictions(
        {
            "P07": {"ece_platt_held_out": summaries["stated_platt"]["calibration"]["ece"]},
            "P08": {
                "reached_target_on_calibration_split": choice["reached_target"],
                "held_out_accuracy": outcome["accuracy"],
                "held_out_coverage": outcome["coverage"],
            },
        },
        predictions,
    )

    def paired(a: str, b: str) -> dict:
        c = compare(runs[a], runs[b], cfg)
        return {"questions": c["questions"], **{k: c[k] for k in SELECTIVE_KEYS}}

    correct_answers = int(y_test.sum())
    risk_out = {
        "note": "one run; held-out set only; declined answers rank below every answered one; "
        "Platt scaling keeps the stated confidence's order, so its ranking measures equal the "
        "raw ones; intervals are 95% bootstrap over the held-out questions, paired for "
        "differences",
        "answers": answer_run,
        "questions": len(test),
        "oracle_aurc": oracle_aurc(correct_answers, len(test)),
        "held_out": {
            name: {
                **{k: s["selective"][k] for k in SELECTIVE_KEYS},
                "auroc": s["calibration"]["auroc"],
                "curve": s["selective"]["curve"],
            }
            for name, s in summaries.items()
        },
        "comparisons": {
            "critic_minus_stated_platt": {
                "pre_registered": True,
                **paired("critic_raw", "stated_platt"),
            },
            "stated_isotonic_minus_stated_platt": {
                "pre_registered": False,
                **paired("stated_isotonic", "stated_platt"),
            },
            "critic_minus_stated_zeroed_without_result": {
                "pre_registered": False,
                **paired("critic_raw", "stated_zeroed_without_result"),
            },
            "combined_minus_stated_platt": {
                "pre_registered": False,
                **paired("combined_exploratory", "stated_platt"),
            },
        },
    }
    return calibration_out, risk_out


def main() -> None:
    preregistration.require()
    conf = confidence.confidence_config()
    path, answer_run = confidence.answers_path(conf)
    answers = read_records(path)
    crit = conf["critic"]
    reviews = []
    for r in crit["runs"]:
        if r["set"] in ("ablation", "held_out"):
            rid = f"{r['set']}-critic-{crit['model']}"
            reviews += confidence.read_records(ROOT / f"results/runs/critic/{rid}.jsonl")
    predictions = {
        p["id"]: p
        for p in preregistration.parse(
            (ROOT / "results/metrics/preregistration.md").read_text(encoding="utf-8")
        )["predictions"]
    }
    ids = {name: split_ids(name) for name in ("ablation", "held_out")}
    cal, risk = report(answers, reviews, ids, conf, predictions, answer_run)
    write_json(CALIBRATION, cal)
    write_json(RISK_COVERAGE, risk)
    ece = cal["held_out"]["stated_platt"]["ece"]
    out = cal["decline"]["held_out"]
    print(
        f"held-out ECE: raw {cal['held_out']['stated_raw']['ece']['estimate']:.3f}, "
        f"Platt {ece['estimate']:.3f} [{ece['low']:.3f}, {ece['high']:.3f}]"
    )
    print(
        f"threshold {out['threshold']:.3f}: held-out accuracy "
        f"{out['accuracy']['estimate']:.3f} at coverage {out['coverage']['estimate']:.3f}"
    )
    d = risk["comparisons"]["critic_minus_stated_platt"]["aurc"]
    print(f"critic - stated AURC: {d['estimate']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]")
    print(f"wrote {CALIBRATION.relative_to(ROOT)}, {RISK_COVERAGE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
