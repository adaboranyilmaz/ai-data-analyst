"""A live smoke test of the finished service: two real questions through the live runner, under the
project's ledger and the product's own cap.

`--count` counts every first request with the free token-counting endpoint and prices the run
(central and upper), writing results/metrics/phase8_dry_run.json. `--run` asks the two questions
(an ordinary count, and a comparison that the statistical guardrail analyzes) and writes
results/metrics/live_smoke.json: what streamed, what it cost and how long it took. Nothing is
run without the count, and the ledger's phase cap stops the run if the count was wrong.

Usage:
    uv run python scripts/83_live_smoke.py --count
    uv run python scripts/83_live_smoke.py --run
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.run import Question, QuestionRun  # noqa: E402
from src.agent.run import config as agent_config  # noqa: E402
from src.llm.ledger import load_ledger  # noqa: E402
from src.serving.live import LiveRunner  # noqa: E402
from src.serving.live_guardrail import LiveGuardrail  # noqa: E402
from src.serving.meter import Meter, config  # noqa: E402
from src.stats import guardrail as gr  # noqa: E402
from src.stats.calls import requests_for  # noqa: E402
from src.tools.toolbox import Toolbox  # noqa: E402

DRY = ROOT / "results/metrics/phase8_dry_run.json"
OUT = ROOT / "results/metrics/live_smoke.json"
PHASE = "phase8"
QUESTIONS = [
    ("descriptive", "How many loans were granted in 1996?"),
    ("comparative", "Did giving clients a card make them more likely to take out a loan?"),
]
# output tokens per call: (central, upper); the upper is each call's max_tokens
OUTPUT = {
    "d1": (450, 2048),
    "classify": (60, 200),
    "plan": (950, 2048),
    "answer": (250, 700),
}


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def count(client, request) -> int:
    kw = {
        "model": request.model,
        "messages": strip(request.messages),
        "system": strip(request.system),
    }
    if request.tools:
        kw["tools"] = request.tools
    for k in ("tool_choice", "thinking"):
        if k in request.params:
            kw[k] = request.params[k]
    return client.messages.count_tokens(**kw).input_tokens


def do_count() -> None:
    cfg = config()
    stats = gr.stats_cfg if hasattr(gr, "stats_cfg") else None  # noqa: F841
    from src.stats.calls import config as stats_config

    scfg = stats_config()
    ledger = load_ledger(PHASE)
    client = anthropic.Anthropic()
    acfg = agent_config()
    live = cfg["live"]
    lines = []
    with Toolbox(live["database"]) as box:
        run = QuestionRun(
            Question("smoke", "live", live["database"], QUESTIONS[0][1], None),
            live["design"],
            live["model"],
            False,
            box,
            ledger.cost,
            None,
            acfg,
            None,
        )
        d1 = count(client, run.pending()[0][1])
        q = {
            "id": "x",
            "source": "live",
            "db_id": gr.DB,
            "category": None,
            "question": QUESTIONS[1][1],
        }
        classify = count(client, requests_for(gr.classify_items([q]), "classify", scfg)[0])
        plan = count(client, requests_for(gr.plan_items([q], box), "plan", scfg)[0])
    answer = 1500  # the answer call's prompt measured in Phase 7 (about 1,430 tokens)

    def cost(model: str, tokens_in: int, tokens_out: int, cached: bool) -> float:
        p = ledger.price(model)
        per_in = p.cache_read if cached else p.input
        return (tokens_in * per_in + tokens_out * p.output) / 1e6

    sonnet, haiku = live["model"], scfg["calls"]["classify"]["settings"]["model"]
    central = {
        "q1": cost(sonnet, d1, OUTPUT["d1"][0], False)
        + d1 * 0.25 * ledger.price(sonnet).input / 1e6,
        "q2_d1": cost(sonnet, d1, OUTPUT["d1"][0], True),
        "q2_classify": cost(haiku, classify, OUTPUT["classify"][0], False),
        "q2_plan": cost(sonnet, plan, OUTPUT["plan"][0], False)
        + plan * 0.25 * ledger.price(sonnet).input / 1e6,
        "q2_answers": 2 * cost(sonnet, answer, OUTPUT["answer"][0], False),
    }
    upper = {
        "q1": cost(sonnet, d1, OUTPUT["d1"][1], False),
        "q2_d1": cost(sonnet, d1, OUTPUT["d1"][1], False),
        "q2_classify": cost(haiku, classify, OUTPUT["classify"][1], False),
        "q2_plan": cost(sonnet, plan, OUTPUT["plan"][1], False),
        "q2_answers": 2 * cost(sonnet, answer, OUTPUT["answer"][1], False),
    }
    out = {
        "note": "counted with the token-counting endpoint ($0); cache writes counted at 1.25x for "
        "the first call of a prefix; outputs assumed (central, upper = max_tokens)",
        "prompt_tokens": {"d1": d1, "classify": classify, "plan": plan, "answer_assumed": answer},
        "central_usd": {
            **{k: round(v, 5) for k, v in central.items()},
            "total": round(sum(central.values()), 4),
        },
        "upper_usd": {
            **{k: round(v, 5) for k, v in upper.items()},
            "total": round(sum(upper.values()), 4),
        },
        "phase_cap_usd": ledger.phase_cap_usd,
        "counted_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    DRY.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    lines.append(json.dumps(out, indent=1))
    print("\n".join(lines))


def do_run() -> None:
    if not DRY.exists():
        sys.exit("count first: scripts/83_live_smoke.py --count")
    dry = json.loads(DRY.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(config())
    cfg["live"]["cache_dir"] = "data/serving/smoke/cache"
    cfg["live"]["guardrail_cache_dir"] = "data/serving/smoke/guardrail"
    cfg["live"]["runs_dir"] = "data/serving/smoke/runs"
    meter = Meter.load(cfg)
    ledger = load_ledger(PHASE)  # the project's ledger and the phase's cap
    runner = LiveRunner(cfg, meter, ledger=ledger)
    runner.guardrail = LiveGuardrail(cfg["live"], ledger)
    results = []
    for kind, question in QUESTIONS:
        events, sink = [], []
        for e in runner.run(f"smoke-{kind}", question, sink):
            events.append(e)
        types = [e["type"] for e in events]
        answer = next((e for e in events if e["type"] == "answer"), None)
        done = events[-1]
        results.append(
            {
                "kind": kind,
                "question": question,
                "event_types": types,
                "status": done.get("status"),
                "error": done if done["type"] == "error" else None,
                "had_statistics": "statistics" in types,
                "answer": None if answer is None else answer["text"],
                "notice": None if answer is None else answer.get("notice"),
                "confidence": next((e for e in events if e["type"] == "confidence"), None),
            }
        )
    out = {
        "note": "two real questions through the live service's runner, under the project ledger's "
        "phase cap; the guardrail ran in a sandbox container. The spend is the ledger's; a "
        "re-run of this script replays the cached responses, so it reports no new spend and no "
        "timings",
        "counted_upper_usd": dry["upper_usd"]["total"],
        "phase_spent_usd": round(ledger.phase_spent(), 5),
        "phase_cap_usd": ledger.phase_cap_usd,
        "results": results,
    }
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1))
    for r in results:
        print(
            r["kind"],
            r["status"],
            "statistics" if r["had_statistics"] else "",
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", action="store_true")
    g.add_argument("--run", action="store_true")
    a = p.parse_args()
    load_dotenv(ROOT / ".env")
    do_count() if a.count else do_run()
