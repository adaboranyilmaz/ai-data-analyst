"""The router on the held-out set: the winning design on Claude Sonnet 5, escalated to Claude
Opus 5.5 where Sonnet's calibrated confidence is below the decline threshold.

Refuses to run unless the pre-registration is frozen and unchanged, and unless the escalation model
was adopted (results/metrics/escalation.json). From the winning design's run on all questions
(results/runs/main/), the escalation model's run on the held-out set (results/runs/router/) and the
calibration (results/metrics/calibration.json):

- the routed system, per held-out question: Sonnet's answer where its calibrated confidence reaches
  the decline threshold, Opus's where it does not (a declined Sonnet answer is routed too). Sonnet
  always answers first, since its confidence decides the route, so the routed system's cost is
  Sonnet's on every question plus Opus's on the routed ones;
- its execution accuracy and cost per correct answer, against Sonnet alone on the same questions
  (paired), as pre-registered;
- exploratory, not pre-registered: Opus alone on every held-out question, against the routed
  system and against Sonnet (the escalation gain on questions nothing was chosen on), and each
  model's accuracy on the routed and the kept questions.

Confidence measures are not reported for the routed system: its answers carry two models'
confidences, calibrated for neither together. Every result is from one run.

Costs are at batch prices throughout. Most answers came through batches, but part of the
escalation model's run was made of direct calls, at twice the price; every answer is repriced
from its recorded tokens at the batch price, so that the systems are compared at one price, and
what was actually spent is reported beside it. Writes results/metrics/router.json.

Usage:
    uv run python scripts/60_router_report.py
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.confidence import (  # noqa: E402
    answers_path,
    confidence_config,
    routed_ids,
    run_name,
    winning_design,
)
from src.data.bird import write_json  # noqa: E402
from src.eval import preregistration  # noqa: E402
from src.eval.records import read_records  # noqa: E402
from src.eval.reports import NOT_MEASURED, cost_row, split_ids, subset  # noqa: E402
from src.eval.summary import compare, summarise  # noqa: E402
from src.llm.ledger import load_ledger  # noqa: E402
from src.llm.types import TokenUsage  # noqa: E402

OUT = ROOT / "results/metrics/router.json"
SONNET = "claude-sonnet-5"
KEEP = ("questions", "declined", "execution_accuracy", "by_difficulty", "cost", "score_outcomes")


def summary(records: list[dict], cfg: dict | None) -> dict:
    s = summarise(records, cfg)
    out = {k: s[k] for k in KEEP if k in s}
    out["latency_s"] = {
        **NOT_MEASURED,
        "not_measured": "most model calls were batched, and a batched call has no latency",
    }
    return out


def ex_only(records: list[dict], cfg: dict | None) -> dict:
    s = summarise(records, cfg)
    return {"questions": s["questions"], "execution_accuracy": s["execution_accuracy"]}


def at_batch_price(records: list[dict], cost: Callable[[str, TokenUsage, bool], float]) -> dict:
    """The records repriced at the batch price from their recorded tokens, and what they cost."""
    repriced = [
        {**r, "cost_usd": cost(r["model"], TokenUsage(**r["tokens"]), True)} for r in records
    ]
    return {
        "records": repriced,
        "spent_usd": round(sum(r["cost_usd"] for r in records), 6),
        "at_batch_price_usd": round(sum(r["cost_usd"] for r in repriced), 6),
        # the recorded cost is rounded to 8 decimals; a direct call costs twice the batch price
        "questions_with_direct_calls": sum(
            abs(r["cost_usd"] - b["cost_usd"]) > 1e-6
            for r, b in zip(records, repriced, strict=True)
        ),
    }


def report(
    sonnet: list[dict],
    opus: list[dict],
    routed: set,
    opus_model: str,
    design: str,
    cfg: dict | None = None,
    spent: dict | None = None,
) -> dict:
    """`sonnet`, `opus`: both models' records on the same held-out questions, priced alike;
    `routed`: the ids the router sends to Opus; `spent`: what each model's run actually cost."""
    by_opus = {r["question_id"]: r for r in opus}
    if set(by_opus) != {r["question_id"] for r in sonnet}:
        raise ValueError("both models must have answered the same questions")
    system = []
    for s in sonnet:
        if s["question_id"] in routed:
            o = by_opus[s["question_id"]]
            system.append({**o, "cost_usd": s["cost_usd"] + o["cost_usd"]})
        else:
            system.append(s)
    opus_aligned = [by_opus[s["question_id"]] for s in sonnet]
    kept = [s for s in sonnet if s["question_id"] not in routed]
    s_routed = [s for s in sonnet if s["question_id"] in routed]
    o_routed = [by_opus[s["question_id"]] for s in s_routed]
    o_kept = [by_opus[s["question_id"]] for s in kept]

    def paired(a: list[dict], b: list[dict]) -> dict:
        c = compare(a, b, cfg)
        return {"questions": c["questions"], "execution_accuracy": c["execution_accuracy"]}

    out = {
        "note": "one run per model; the held-out set; intervals are 95% bootstrap over questions, "
        "paired for differences; costs at batch prices",
        "spent": spent,
        "design": design,
        "escalation_model": opus_model,
        "questions": len(sonnet),
        "routed": len(s_routed),
        "kept": len(kept),
        "routed_system": summary(system, cfg),
        "sonnet_alone": summary(sonnet, cfg),
        "routed_minus_sonnet": paired(system, sonnet),
        "exploratory": {
            "note": "not pre-registered",
            "opus_alone": summary(opus_aligned, cfg),
            "opus_minus_sonnet": paired(opus_aligned, sonnet),
            "opus_alone_minus_routed": paired(opus_aligned, system),
            "on_routed_questions": {
                "sonnet": ex_only(s_routed, cfg),
                "opus": ex_only(o_routed, cfg),
            },
            "on_kept_questions": (
                {"sonnet": ex_only(kept, cfg), "opus": ex_only(o_kept, cfg)} if kept else None
            ),
        },
    }
    out["cost_per_correct"] = [
        cost_row(out["routed_system"], system="routed", design=design, set="held_out"),
        cost_row(out["sonnet_alone"], system=SONNET, design=design, set="held_out"),
        cost_row(
            out["exploratory"]["opus_alone"], system=opus_model, design=design, set="held_out"
        ),
    ]
    return out


def main() -> None:
    preregistration.require()
    if not json.loads((ROOT / "results/metrics/escalation.json").read_text(encoding="utf-8"))[
        "rule"
    ]["adopted"]:
        sys.exit("the escalation model was not adopted: there is no router to report")
    conf = confidence_config()
    esc = conf["escalation"]
    design = winning_design()
    path, _ = answers_path(conf)
    held = subset(read_records(path), split_ids(esc["router"]["set"]))
    rid = run_name(esc["router"]["set"], design, esc["model"], esc["evidence"])
    opus = read_records(ROOT / "results/runs/router" / f"{rid}.jsonl")
    calibration = json.loads(
        (ROOT / "results/metrics/calibration.json").read_text(encoding="utf-8")
    )
    cost = load_ledger(conf["phase"]).cost
    priced = {"sonnet": at_batch_price(held, cost), "opus": at_batch_price(opus, cost)}
    spent = {
        m: {k: v for k, v in priced[who].items() if k != "records"}
        for who, m in (("sonnet", SONNET), ("opus", esc["model"]))
    }
    out = report(
        priced["sonnet"]["records"],
        priced["opus"]["records"],
        routed_ids(held, calibration),
        esc["model"],
        design,
        spent=spent,
    )
    write_json(OUT, out)
    d = out["routed_minus_sonnet"]["execution_accuracy"]
    ex = out["routed_system"]["execution_accuracy"]
    print(
        f"routed {out['routed']}/{out['questions']}: EX {ex['estimate']:.3f}; "
        f"minus Sonnet {d['estimate']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]"
    )
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
