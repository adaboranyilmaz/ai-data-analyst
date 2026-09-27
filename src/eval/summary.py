"""What is reported for a run of per-question records, and how two runs are compared.

A question counts as correct when it was answered (not declined) and its SQL scored correct
(src/eval/records.py `answered_correct`). Every rate carries a bootstrap interval over questions
(configs/eval.yaml `bootstrap`); two runs on the same questions are compared with paired
intervals. A run is summarised with:

- execution accuracy, overall and by database and difficulty; mean Soft-F1 (a prediction too
  large for Soft-F1 counts 0, and the number of such is given). These, and the confidence
  measures below, cover the questions with a gold result (`correct` not null): every benchmark
  question, and the own set's answerable ones;
- selective prediction and calibration, when every record has a confidence: AURC, E-AURC,
  accuracy at the configured coverages, ECE (equal-width and equal-mass bins), Brier score,
  AUROC, the risk-coverage curve and the reliability bins;
- cost (total, per question, per correct answer), latency (median, 95th percentile), steps,
  tool calls, errors by kind, and how the answers were scored.

`own_set_behaviour` scores the hand-written banking set by category: (a) and (b) by execution
accuracy; (c) ambiguous questions by a clarifying question or a stated assumption whose SQL
matches an accepted reading; (d) unanswerable ones by declining with a reason; (e) false
premises by a correction of the premise. Category (f) is scored by the statistical checks.

`select_design` applies the selection rule fixed before any design was run: the lowest AURC
wins, unless a cheaper design is not worse by a paired 95% interval that contains zero; then
the cheapest such design wins.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np

from src.eval import calibration, selective
from src.eval.bootstrap import Interval, bootstrap, paired_difference
from src.eval.config import config
from src.eval.records import answered_correct

OWN_CATEGORIES = ("a", "b", "c", "d", "e", "f")


def _boot(cfg: dict[str, Any]) -> tuple[Callable[..., Interval], Callable[..., Interval]]:
    b = cfg["bootstrap"]
    kw = {"count": b["resamples"], "confidence": b["confidence"], "seed": b["seed"]}
    return partial(bootstrap, **kw), partial(paired_difference, **kw)


def _mean(boot: Callable[..., Interval], values: Sequence[float]) -> dict:
    """The mean of the values, with its interval."""
    v = np.asarray(values, dtype=float)
    return boot(lambda i: v[i].mean(), v.size).to_dict()


def _nan_if_none(x: float | None) -> float:
    return float("nan") if x is None else x


def _percentile(xs: Sequence[float], q: float) -> float | None:
    return float(np.percentile(xs, q)) if len(xs) else None


def _selective_stats(
    conf: np.ndarray, y: np.ndarray, declined: np.ndarray, coverages: Sequence[float]
) -> dict[str, Callable[[np.ndarray], float]]:
    stats: dict[str, Callable[[np.ndarray], float]] = {
        "aurc": lambda i: selective.aurc(conf[i], y[i], declined[i]),
        "e_aurc": lambda i: selective.e_aurc(conf[i], y[i], declined[i]),
    }
    for c in coverages:
        stats[f"accuracy_at_{round(c * 100)}"] = lambda i, c=c: selective.accuracy_at_coverage(
            conf[i], y[i], c, declined[i]
        )
    return stats


def summarise(records: Sequence[dict], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or config()
    boot, _ = _boot(cfg)
    if not records:
        raise ValueError("no records")
    out: dict[str, Any] = {
        "questions": len(records),
        "declined": sum(r["declined"] for r in records),
    }
    scored = [r for r in records if r["correct"] is not None]
    n = len(scored)
    out["scored_questions"] = n
    if n:
        y = np.array([answered_correct(r) for r in scored], dtype=float)
        declined = np.array([r["declined"] for r in scored], dtype=bool)
        f1 = np.array([r["soft_f1"] if r["soft_f1"] is not None else 0.0 for r in scored])
        out["execution_accuracy"] = boot(lambda i: y[i].mean(), n).to_dict()
        out["soft_f1"] = boot(lambda i: f1[i].mean(), n).to_dict()
        out["soft_f1_not_computed"] = sum(r["soft_f1"] is None for r in scored)
        for key in ("db_id", "difficulty"):
            groups: dict[str, list[int]] = defaultdict(list)
            for i, r in enumerate(scored):
                if r[key] is not None:
                    groups[r[key]].append(i)
            out[f"by_{key}"] = {
                g: {"questions": len(ix), "execution_accuracy": _mean(boot, y[ix])}
                for g, ix in sorted(groups.items())
            }

    if n and all(r["confidence"] is not None for r in scored):
        conf = np.array([r["confidence"] for r in scored], dtype=float)
        coverages = cfg["selective"]["coverages"]
        stats = _selective_stats(conf, y, declined, coverages)
        out["selective"] = {name: boot(s, n).to_dict() for name, s in stats.items()}
        coverage, risk = selective.risk_coverage(conf, y, declined)
        out["selective"]["curve"] = {"coverage": coverage.tolist(), "risk": risk.tolist()}
        bins = cfg["calibration"]["ece_bins"]
        out["calibration"] = {
            "ece": boot(lambda i: calibration.ece(conf[i], y[i], bins), n).to_dict(),
            "ece_equal_mass": boot(
                lambda i: calibration.ece(conf[i], y[i], bins, "mass"), n
            ).to_dict(),
            "brier": boot(lambda i: calibration.brier(conf[i], y[i]), n).to_dict(),
            "auroc": boot(lambda i: _nan_if_none(calibration.auroc(conf[i], y[i])), n).to_dict(),
            "reliability": calibration.reliability(conf, y, bins),
            "reliability_equal_mass": calibration.reliability(conf, y, bins, "mass"),
        }

    cost = np.array([r["cost_usd"] for r in records])
    latency = [r["latency_s"] for r in records]
    out["cost"] = {"total_usd": float(cost.sum()), "per_question_usd": float(cost.mean())}
    if n:
        scored_cost = np.array([r["cost_usd"] for r in scored])
        out["cost"]["per_correct_answer_usd"] = boot(
            lambda i: scored_cost[i].sum() / y[i].sum() if y[i].sum() else float("nan"), n
        ).to_dict()
    out["latency_s"] = {"p50": _percentile(latency, 50), "p95": _percentile(latency, 95)}
    out["steps_mean"] = float(np.mean([r["steps"] for r in records]))
    out["tool_calls_mean"] = float(np.mean([r["tool_calls"] for r in records]))
    out["errors_by_kind"] = dict(
        sorted(Counter(e["kind"] for r in records for e in r["errors"]).items())
    )
    out["score_outcomes"] = dict(sorted(Counter(r["score_outcome"] for r in records).items()))
    return out


def _aligned(a: Sequence[dict], b: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    by_b = {r["question_id"]: r for r in b}
    if len(by_b) != len(b) or {r["question_id"] for r in a} != set(by_b) or len(a) != len(b):
        raise ValueError("a paired comparison needs the same questions, once each, in both runs")
    return list(a), [by_b[r["question_id"]] for r in a]


def compare(
    a: Sequence[dict], b: Sequence[dict], cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run a minus run b, paired over the same questions."""
    cfg = cfg or config()
    _, paired = _boot(cfg)
    a, b = _aligned(a, b)
    n = len(a)
    ya = np.array([answered_correct(r) for r in a], dtype=float)
    yb = np.array([answered_correct(r) for r in b], dtype=float)
    ca = np.array([r["cost_usd"] for r in a])
    cb = np.array([r["cost_usd"] for r in b])
    out = {
        "questions": n,
        "execution_accuracy": paired(lambda i: ya[i].mean(), lambda i: yb[i].mean(), n).to_dict(),
        "cost_per_question_usd": paired(
            lambda i: ca[i].mean(), lambda i: cb[i].mean(), n
        ).to_dict(),
    }
    if all(r["confidence"] is not None for r in (*a, *b)):
        conf_a = np.array([r["confidence"] for r in a], dtype=float)
        conf_b = np.array([r["confidence"] for r in b], dtype=float)
        da = np.array([r["declined"] for r in a], dtype=bool)
        db = np.array([r["declined"] for r in b], dtype=bool)
        sa = _selective_stats(conf_a, ya, da, cfg["selective"]["coverages"])
        sb = _selective_stats(conf_b, yb, db, cfg["selective"]["coverages"])
        for name in sa:
            out[name] = paired(sa[name], sb[name], n).to_dict()
    return out


def select_design(runs: dict[str, Sequence[dict]], cfg: dict[str, Any] | None = None) -> dict:
    """The design the selection rule picks among runs on the same questions.

    The lowest AURC is the best design. A design whose mean cost per question is lower than the
    best's, and whose AURC minus the best's has a paired 95% interval containing zero, is not
    shown to be worse; if there are such designs, the cheapest of them wins, otherwise the
    best does. Ties go to the lower cost, then the name.
    """
    cfg = cfg or config()
    summary = {}
    for name, records in runs.items():
        if any(r["confidence"] is None for r in records):
            raise ValueError(f"{name}: every record needs a confidence")
        y = [answered_correct(r) for r in records]
        summary[name] = {
            "aurc": selective.aurc(
                [r["confidence"] for r in records], y, [r["declined"] for r in records]
            ),
            "cost_per_question_usd": float(np.mean([r["cost_usd"] for r in records])),
        }
    order = sorted(
        summary, key=lambda d: (summary[d]["aurc"], summary[d]["cost_per_question_usd"], d)
    )
    best = order[0]
    not_worse = {}
    for name in summary:
        if summary[name]["cost_per_question_usd"] < summary[best]["cost_per_question_usd"]:
            diff = compare(runs[name], runs[best], cfg)["aurc"]
            summary[name]["aurc_minus_best"] = diff
            if diff["low"] <= 0 <= diff["high"]:
                not_worse[name] = summary[name]["cost_per_question_usd"]
    winner = min(not_worse, key=lambda d: (not_worse[d], d)) if not_worse else best
    return {
        "best_aurc": best,
        "winner": winner,
        "cheaper_not_worse": sorted(not_worse),
        "designs": summary,
    }


def own_set_behaviour(records: Sequence[dict], cfg: dict[str, Any] | None = None) -> dict:
    """Success rate per category of the hand-written banking set, with intervals."""
    cfg = cfg or config()
    boot, _ = _boot(cfg)

    def success(r: dict) -> int | None:
        c = r["category"]
        if c in ("a", "b"):
            return answered_correct(r)
        if c == "c":  # asked, or assumed and answered one accepted reading
            asked = bool(r["clarifying_question"])
            return int(asked or (bool(r["assumptions"]) and answered_correct(r) == 1))
        if c == "d":
            return int(r["declined"] and bool(r["decline_reason"]))
        if c == "e":
            return int(bool(r["premise_correction"]))
        return None  # (f): scored by the statistical checks

    out: dict[str, Any] = {}
    for cat in OWN_CATEGORIES:
        rs = [r for r in records if r["source"] == "own" and r["category"] == cat]
        if not rs:
            continue
        s = [success(r) for r in rs]
        row: dict[str, Any] = {"questions": len(rs), "declined": sum(r["declined"] for r in rs)}
        if s[0] is not None:
            row["success"] = _mean(boot, s)
        out[cat] = row
    answerable = [r for r in records if r["source"] == "own" and r["category"] in ("a", "b")]
    if answerable:
        out["false_decline_rate_ab"] = _mean(boot, [r["declined"] for r in answerable])
    return out
