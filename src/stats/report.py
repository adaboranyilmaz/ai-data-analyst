"""The statistical guardrail's results: classification, the banking set's comparative and causal
questions before and after, and the planted effects at both levels, with the predictions they
are checked against.

Every rate comes with a Wilson 95% interval. The primary planted measure (claims of an effect on
copies without one, guarded against numbers only) is a paired difference with a percentile
bootstrap interval over the copies, which resamples the same copies for both inputs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from src.eval.bootstrap import paired_difference

Z = 1.959963984540054
NO_EFFECT = ("none", "confounded")
EFFECT = ("small", "large", "reversed")
FIRST_FAMILIES = ("rates", "means", "trend")


def wilson(k: int, n: int, z: float = Z) -> list[float] | None:
    """Wilson score interval for k of n; None when n is 0."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return [round(lo, 4), round(hi, 4)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci": wilson(k, n)}


def count(flags: Iterable[bool]) -> dict:
    flags = list(flags)
    return rate(sum(flags), len(flags))


# --- classification -----------------------------------------------------------------------------

METHODS = {"rules": "rule_flag", "model": "model_flag", "or": "statistical"}


def classification(records: list[dict], review: dict) -> dict:
    """Sensitivity on the banking set's category f and the planted questions; false positives on
    the banking set's other questions and on the benchmark, as flagged and after reading."""
    f = [r for r in records if r["source"] == "own" and r["category"] == "f"]
    other = [r for r in records if r["source"] == "own" and r["category"] != "f"]
    planted = [r for r in records if r["source"] == "planted"]
    bird = [r for r in records if r["source"] == "bird"]
    labels = {x["id"]: x["label"] for x in review["benchmark_flagged"]}
    out: dict = {
        "questions": {
            "own_f": len(f),
            "own_other": len(other),
            "planted": len(planted),
            "benchmark": len(bird),
        }
    }
    for name, key in METHODS.items():
        flagged = [r for r in bird if r[key]]
        statistical = {r["id"] for r in flagged if labels.get(r["id"]) == "statistical"}
        out[name] = {
            "sensitivity_own_f": count(r[key] for r in f),
            "sensitivity_planted": count(r[key] for r in planted),
            "false_positive_own_other": count(r[key] for r in other),
            "false_positive_benchmark_as_flagged": count(r[key] for r in bird),
            "false_positive_benchmark_after_reading": count(
                r[key] for r in bird if r["id"] not in statistical
            ),
        }
    out["flagged_own_other"] = [
        {"id": r["id"], "rules": r["rule_flag"], "model": r["model_flag"]}
        for r in other
        if r["statistical"]
    ]
    out["flagged_benchmark"] = [
        {
            "id": r["id"],
            "rules": r["rule_flag"],
            "model": r["model_flag"],
            "label": labels.get(r["id"]),
        }
        for r in bird
        if r["statistical"]
    ]
    unread = [x["id"] for x in out["flagged_benchmark"] if x["label"] is None]
    if unread:
        raise ValueError(f"flagged benchmark questions without a reading: {unread}")
    return out


# --- the banking set's comparative and causal questions -----------------------------------------


def small_sample(analysis: dict) -> bool:
    """Whether a group meets the fixed rule: under 30 units, or under 5 with or without the
    outcome (a numeric outcome has no events)."""
    for g in analysis.get("groups", []):
        if g["n"] < 30:
            return True
        events = g.get("events")
        if events is not None and min(events, g["n"] - events) < 5:
            return True
    return False


def warned_small(analysis: dict) -> bool:
    return any(w["kind"] in ("small_group", "few_events") for w in analysis.get("warnings", []))


def banking_after(record: dict, reading: dict) -> dict:
    """The three criteria on the guarded answer: (1) the path ran to the end; (2) the caveat is
    delivered (with every guarded answer) and the model's own text makes no causal claim, as
    decided by reading; (3) the interval, and the small-sample warning where the rule applies."""
    analysis = record.get("analysis")
    ran = analysis is not None and record.get("guarded") is not None
    c1 = ran
    c2 = ran and not reading["causal_claim_after"]
    c3 = ran and (warned_small(analysis) or not small_sample(analysis))
    return {
        "ran_to_the_end": ran,
        "failed_at": None if ran else (record.get("plan_error") or record.get("analysis_error")),
        "interval": c1,
        "no_causal_claim_and_caveat": c2,
        "uncertainty": c3,
        "flagged_causal_sentences": (record.get("guarded") or {})
        .get("checks", {})
        .get("causal_sentences", []),
        "success": c1 and c2 and c3,
        "reviewed_success": bool(reading["reviewed_success"]) and c1 and c2 and c3,
    }


def banking_before(reading: dict) -> dict:
    b = reading["before"]
    c1 = b["interval"]
    c2 = b["no_causal_claim"] and b["caveat"]
    c3 = c1 and b["uncertainty"]
    return {
        "interval": c1,
        "no_causal_claim_and_caveat": c2,
        "uncertainty": c3,
        "success": c1 and c2 and c3,
    }


def banking(own_records: list[dict], classified: list[dict], review: dict) -> dict:
    flagged = {r["id"]: r["statistical"] for r in classified}
    by_id = {r["id"]: r for r in own_records}
    rows = []
    for reading in review["banking_f"]:
        qid = f"own:{reading['id']}"
        rec = by_id[qid]
        after = (
            banking_after(rec, reading)
            if flagged.get(qid)
            else {
                "ran_to_the_end": False,
                "failed_at": "not flagged",
                "interval": False,
                "no_causal_claim_and_caveat": False,
                "uncertainty": False,
                "flagged_causal_sentences": [],
                "success": False,
                "reviewed_success": False,
            }
        )
        rows.append(
            {
                "id": reading["id"],
                "question": rec["question"],
                "before": banking_before(reading),
                "after": after,
                "note": reading["note"],
            }
        )
    criteria = ("interval", "no_causal_claim_and_caveat", "uncertainty", "success")
    totals = {
        side: {c: sum(r[side][c] for r in rows) for c in criteria} for side in ("before", "after")
    }
    totals["after"]["reviewed_success"] = sum(r["after"]["reviewed_success"] for r in rows)
    return {
        "questions": len(rows),
        "totals": totals,
        "per_question": rows,
        "other_flagged": review["banking_other"],
    }


# --- planted effects ----------------------------------------------------------------------------


def level1_cells(records: list[dict], powers: dict[str, float]) -> list[dict]:
    """Per question, condition and plan: detection, in the planted direction and opposite."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["template"], r["family"], r["condition"], r["plan"])].append(r)
    cells = []
    for (tid, family, cond, plan), rs in sorted(groups.items()):
        ok = [r for r in rs if r["ok"]]
        cells.append(
            {
                "template": tid,
                "family": family,
                "condition": cond,
                "plan": plan,
                "copies": len(rs),
                "failed": len(rs) - len(ok),
                "detected": count(r["detected"] for r in ok),
                "planted_direction": count(r.get("direction") == "planted" for r in ok),
                "opposite_direction": count(r.get("direction") == "opposite" for r in ok),
                "theoretical_power": powers.get(cond) if family in FIRST_FAMILIES else None,
                "planted_effect": next((r.get("planted_effect") for r in rs), None),
            }
        )
    return cells


def pooled(records: list[dict]) -> list[dict]:
    """Per family, condition and plan, pooled over the family's questions."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r["ok"]:
            groups[(r["family"], r["condition"], r["plan"])].append(r)
    return [
        {
            "family": fam,
            "condition": cond,
            "plan": plan,
            "detected": count(r["detected"] for r in rs),
            "planted_direction": count(r.get("direction") == "planted" for r in rs),
        }
        for (fam, cond, plan), rs in sorted(groups.items())
    ]


def headline(records: list[dict]) -> dict:
    """Level 1's main figures by name, pooled over questions: false alarms where nothing was
    planted (the first three families), and for the confounded family what each plan finds in
    the confounded and the reversed copies."""
    ok = [r for r in records if r["ok"]]

    def of(plan: str, cond: str, families: tuple[str, ...], key: str = "detected") -> dict:
        rs = [
            r
            for r in ok
            if r["plan"] == plan and r["condition"] == cond and r["family"] in families
        ]
        if key == "detected":
            return count(r["detected"] for r in rs)
        return count(r.get("direction") == key for r in rs)

    conf = ("confounded",)
    return {
        "false_alarms_none": {p: of(p, "none", FIRST_FAMILIES) for p in ("reference", "analyst")},
        "confounded_detected": {
            p: of(p, "confounded", conf) for p in ("reference", "crude", "analyst")
        },
        "reversed_planted_direction": {
            p: of(p, "reversed", conf, "planted") for p in ("reference", "crude", "analyst")
        },
        "reversed_opposite_direction": {
            p: of(p, "reversed", conf, "opposite") for p in ("reference", "crude", "analyst")
        },
    }


def _cell(cells: list[dict], tid: str, cond: str, plan: str) -> dict | None:
    return next(
        (c for c in cells if (c["template"], c["condition"], c["plan"]) == (tid, cond, plan)), None
    )


def power_checks(cells: list[dict]) -> dict:
    """G7: reference detection against the theoretical power; G8: analyst against reference."""
    g7, g8 = [], []
    templates = sorted({c["template"] for c in cells if c["family"] in FIRST_FAMILIES})
    for tid in templates:
        for cond in ("small", "large"):
            ref = _cell(cells, tid, cond, "reference")
            ana = _cell(cells, tid, cond, "analyst")
            gap = ref["detected"]["rate"] - ref["theoretical_power"]
            g7.append(
                {
                    "template": tid,
                    "condition": cond,
                    "reference": ref["detected"]["rate"],
                    "power": ref["theoretical_power"],
                    "gap": round(gap, 4),
                    "within": abs(gap) <= 0.08 + 1e-9,
                }
            )
            a = ana["detected"]["rate"] if ana and ana["detected"]["n"] else None
            gap8 = None if a is None else a - ref["detected"]["rate"]
            g8.append(
                {
                    "template": tid,
                    "condition": cond,
                    "analyst": a,
                    "reference": ref["detected"]["rate"],
                    "gap": None if gap8 is None else round(gap8, 4),
                    "within": gap8 is not None and abs(gap8) <= 0.10 + 1e-9,
                }
            )
    return {"reference_vs_power": g7, "analyst_vs_reference": g8}


def level2_summary(
    records: list[dict], seed: int, resamples: int = 10_000, reading: dict | None = None
) -> dict:
    """`reading`: per input, how many flagged sentences make a causal claim when read."""
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_arm[r["arm"]].append(r)

    def claims(r: dict) -> bool:
        return r["claims_effect"] == "yes"

    def planted_dir(r: dict) -> bool:
        return r["claimed_direction"] is not None and r["claimed_direction"] == r["planted_sign"]

    arms = {}
    for arm, every in sorted(by_arm.items()):
        rs = [r for r in every if r["finding"] is not None]  # a failed call is left out
        no = [r for r in rs if r["condition"] in NO_EFFECT]
        eff = [r for r in rs if r["condition"] in EFFECT]
        arms[arm] = {
            "answers": len(every),
            "failed": [
                {
                    "template": r["template"],
                    "condition": r["condition"],
                    "copy": r["copy"],
                    "errors": r["errors"],
                }
                for r in every
                if r["finding"] is None
            ],
            "claims_on_no_effect": count(claims(r) for r in no),
            "claims_on_no_effect_by_condition": {
                c: count(claims(r) for r in no if r["condition"] == c) for c in NO_EFFECT
            },
            "planted_direction_on_effect": count(planted_dir(r) for r in eff),
            "planted_direction_by_condition": {
                c: count(planted_dir(r) for r in eff if r["condition"] == c) for c in EFFECT
            },
            "claims_unclear": count(r["claims_effect"] == "unclear" for r in rs),
            "disagrees_with_test": count(
                r["checks"]["unsupported_claim"] or r["checks"]["missed_effect"] for r in rs
            ),
            "direction_mismatch": count(r["checks"]["direction_mismatch"] for r in rs),
            "causal_wording_flagged": count(bool(r["checks"]["causal_sentences"]) for r in rs),
            "causal_sentences_flagged": sum(len(r["checks"]["causal_sentences"]) for r in rs),
            "causal_claims_on_reading": (reading or {}).get(arm),
        }
    # the primary measure: paired over the copies without an effect
    keyed = {(r["template"], r["condition"], r["copy"], r["arm"]): r for r in records}
    units = sorted({k[:3] for k in keyed if k[1] in NO_EFFECT})
    paired = [
        u
        for u in units
        if all(
            keyed.get((*u, arm), {}).get("finding") is not None
            for arm in ("guarded", "numbers_only")
        )
    ]
    g = np.array([claims(keyed[(*u, "guarded")]) for u in paired], dtype=float)
    n = np.array([claims(keyed[(*u, "numbers_only")]) for u in paired], dtype=float)
    diff = paired_difference(
        lambda i: float(g[i].mean()),
        lambda i: float(n[i].mean()),
        len(paired),
        count=resamples,
        seed=seed,
    )
    return {
        "arms": arms,
        "primary": {
            "measure": "claims of an effect on copies without one, guarded minus numbers only",
            "copies": len(paired),
            "guarded": round(float(g.mean()), 4),
            "numbers_only": round(float(n.mean()), 4),
            "difference": diff.to_dict(),
        },
    }


# --- predictions --------------------------------------------------------------------------------


def _between(x: float | None, lo: float, hi: float) -> bool:
    return x is not None and lo - 1e-9 <= x <= hi + 1e-9


def predictions(
    cls: dict, bank: dict, l1_cells: list[dict], checks: dict, l2: dict, review: dict
) -> list[dict]:
    ref_none = [
        c
        for c in l1_cells
        if c["plan"] == "reference" and c["condition"] == "none" and c["family"] in FIRST_FAMILIES
    ]
    fa_k = sum(c["detected"]["k"] for c in ref_none)
    fa_n = sum(c["detected"]["n"] for c in ref_none)
    g7 = sum(x["within"] for x in checks["reference_vs_power"])
    g8 = sum(x["within"] for x in checks["analyst_vs_reference"])
    g9 = sum(bool(x["stratifies"]) for x in review["confounder_strata"])
    arms = l2["arms"]
    no = arms.get("numbers_only", {})
    gu = arms.get("guarded", {})

    def rev(arm: dict) -> float | None:
        return arm.get("planted_direction_by_condition", {}).get("reversed", {}).get("rate")

    rows = [
        (
            "G1",
            "The classifier (OR) flags the banking set's comparative and causal questions",
            "9 of 9 (8 to 9)",
            cls["or"]["sensitivity_own_f"]["k"],
            _between(cls["or"]["sensitivity_own_f"]["k"], 8, 9),
        ),
        (
            "G2",
            "The model alone flags the banking set's other questions",
            "0 to 3 of 51",
            cls["model"]["false_positive_own_other"]["k"],
            _between(cls["model"]["false_positive_own_other"]["k"], 0, 3),
        ),
        (
            "G3",
            "The classifier (OR) flags the 470 benchmark questions, as flagged",
            "2% to 8%",
            cls["or"]["false_positive_benchmark_as_flagged"]["rate"],
            _between(cls["or"]["false_positive_benchmark_as_flagged"]["rate"], 0.02, 0.08),
        ),
        (
            "G4",
            "Banking set comparative and causal questions succeeding, before",
            "0 of 9",
            bank["totals"]["before"]["success"],
            bank["totals"]["before"]["success"] == 0,
        ),
        (
            "G5",
            "The same, after",
            "7 to 9 of 9",
            bank["totals"]["after"]["success"],
            _between(bank["totals"]["after"]["success"], 7, 9),
        ),
        (
            "G6",
            "Reference plans: false alarms in none (the first three families, pooled)",
            "3% to 7%",
            round(fa_k / fa_n, 4) if fa_n else None,
            fa_n > 0 and _between(fa_k / fa_n, 0.03, 0.07),
        ),
        (
            "G7",
            "Reference plans: detection within 8 points of the theoretical power",
            "12 of 12 cells",
            g7,
            g7 == 12,
        ),
        (
            "G8",
            "Analyst plans: detection within 10 points of the reference plan's",
            "at least 9 of 12 cells",
            g8,
            g8 >= 9,
        ),
        (
            "G9",
            "The analyst's plan stratifies by the planted confounder, or a variable carrying it",
            "1 of 2 (1 to 2)",
            g9,
            _between(g9, 1, 2),
        ),
        (
            "G10",
            "Level 2, numbers only: claims of an effect on copies without one",
            "25% to 60%",
            no.get("claims_on_no_effect", {}).get("rate"),
            _between(no.get("claims_on_no_effect", {}).get("rate"), 0.25, 0.60),
        ),
        (
            "G11",
            "Level 2, guarded: the same",
            "3% to 15%",
            gu.get("claims_on_no_effect", {}).get("rate"),
            _between(gu.get("claims_on_no_effect", {}).get("rate"), 0.03, 0.15),
        ),
        (
            "G12",
            "Level 2, reversed: claims in the planted direction, numbers only / guarded",
            "at most 20% / 40% to 75%",
            [rev(no), rev(gu)],
            _between(rev(no), 0, 0.20) and _between(rev(gu), 0.40, 0.75),
        ),
        (
            "G13",
            "Guarded answers whose claim disagrees with the test",
            "at most 5%",
            gu.get("disagrees_with_test", {}).get("rate"),
            _between(gu.get("disagrees_with_test", {}).get("rate"), 0, 0.05),
        ),
    ]
    return [
        {"id": i, "prediction": p, "range": rng, "observed": obs, "held": bool(held)}
        for i, p, rng, obs, held in rows
    ]
