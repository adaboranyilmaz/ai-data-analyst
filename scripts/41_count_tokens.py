"""The counted dry run: what a set of runs will cost, before any is paid for.

For each question, the first request of each design is built exactly as the agent sends it and
counted with the token-counting endpoint (free). A single-shot request (design 1, and design 5's
narrowing call) is then known exactly apart from its output. A multi-turn conversation's later
turns depend on the model's replies, so they are projected from a number of turns and the
tokens each turn adds: from assumptions until the pilot has run, then from the pilot's measured
records (`--measured`), which replace the assumptions.

Costs use the ledger's prices (configs/budget.yaml). The central estimate assumes prompt caching
works as the agent requests it: in a multi-turn conversation each turn's new tokens written once
and read on every later turn. The upper estimate assumes no cache hits at all (caching inside a
batch is best effort) and the longest measured conversation for every question. Designs 4 and
5's samples are projected from design 3's measured loop; design 4's first sample is design 3's
run, so it is free when design 3 is in the same estimate. Batched calls cost half.

With `--stage`, every arm of a configured stage is estimated as it will run (its set, evidence,
mode, and `winner` replaced by the chosen design), and a question whose first requests are all
in the response caches the stage reads (src/agent/stages.py `READS`) is counted as free: its
answers will be replayed, not paid for. Writes results/metrics/phase4_dry_run.json.

Usage:
    uv run python scripts/41_count_tokens.py --set pilot --models claude-sonnet-5 \
        --designs d1 d3 [--measured results/runs/pilot]
    uv run python scripts/41_count_tokens.py --stage main --measured results/runs/ablation
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.agent.evaluate import banking_set, benchmark_set  # noqa: E402
from src.agent.run import QuestionRun, config  # noqa: E402
from src.agent.stages import READS, WINNER, resolve_arms, winner  # noqa: E402
from src.data.bird import write_json  # noqa: E402
from src.llm.cache import ResponseCache  # noqa: E402
from src.llm.ledger import load_ledger  # noqa: E402
from src.llm.types import TokenUsage  # noqa: E402
from src.tools.toolbox import Toolbox  # noqa: E402

OUT = ROOT / "results/metrics/phase4_dry_run.json"
# Until the pilot is measured: turns per conversation and tokens each turn adds (the tool
# results and the model's own turn), central and upper.
ASSUMED = {"turns": (5, 10), "added_per_turn": (1200, 2500), "output_per_turn": (250, 600)}


def _strip(obj):
    """The request without cache markers (they do not change the count)."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def count(client, request) -> int:
    kw = {
        "model": request.model,
        "messages": _strip(request.messages),
        "system": _strip(request.system),
    }
    if request.tools:
        kw["tools"] = request.tools
    for k in ("tool_choice", "thinking"):
        if k in request.params:
            kw[k] = request.params[k]
    return client.messages.count_tokens(**kw).input_tokens


def conversation_cost(price, first: int, turns: float, added: float, out: float) -> dict:
    """Tokens and dollars of one conversation. With caching: the first request's prefix is
    written, then each turn reads what came before and writes what it adds. Uncached (caching
    in a batch is best effort): every turn pays its whole prompt at the input price."""
    n = max(1, round(turns))
    writes = first + (n - 1) * added
    reads = sum(first + (t - 1) * added for t in range(1, n))
    output = n * out
    usage = TokenUsage(
        input=0, output=int(output), cache_write_5m=int(writes), cache_read=int(reads)
    )
    usd = (
        usage.cache_write_5m * price.cache_write_5m
        + usage.cache_read * price.cache_read
        + usage.output * price.output
    ) / 1e6
    uncached = ((writes + reads) * price.input + output * price.output) / 1e6
    return {
        "input_tokens": int(writes + reads),
        "output_tokens": int(output),
        "usd": usd,
        "uncached_usd": uncached,
    }


def single_cost(price, prompt: int, out: float, cached_prefix: int = 0) -> float:
    return (
        (prompt - cached_prefix) * price.input
        + cached_prefix * price.cache_read
        + out * price.output
    ) / 1e6


def measured(path: Path, cfg: dict) -> dict:
    """Per design and model, from pilot records and their traces: turns per conversation, the
    tokens each turn adds, and output tokens per turn. Only single-conversation designs are
    measured (a trace lists its requests in one sequence); the samples of designs 4 and 5 run
    design 3's loop and are projected from it."""
    acc: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(path.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            d = cfg["designs"][r["design"]]
            if d["samples"] != 1 or d["narrow"]:
                continue
            trace = json.loads((ROOT / r["trace"]).read_text(encoding="utf-8"))
            prompts = [
                sum(
                    req["usage"].get(k) or 0
                    for k in (
                        "input_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    )
                )
                for req in trace["requests"]
            ]
            outs = [req["usage"].get("output_tokens") or 0 for req in trace["requests"]]
            k = (r["design"], r["model"])
            acc[k]["turns"].append(len(prompts))
            acc[k]["output"].extend(outs)
            if len(prompts) > 1:
                acc[k]["added"].append((prompts[-1] - prompts[0]) / (len(prompts) - 1))
    return {
        f"{d}/{m}": {
            "turns": (statistics.mean(v["turns"]), max(v["turns"])),
            "added_per_turn": (
                statistics.mean(v["added"] or [0]),
                max(v["added"] or [0]),
            ),
            "output_per_turn": (statistics.mean(v["output"]), max(v["output"])),
            "questions": len(v["turns"]),
        }
        for (d, m), v in acc.items()
    }


@dataclass(frozen=True)
class Arm:
    set: str
    model: str
    design: str
    evidence: bool
    batch: bool
    d3_too: bool  # design 3 runs in the same estimate (design 4's first sample is its run)
    cached_in: tuple[str, ...] = ()  # stages whose stored responses this arm reads


def estimate(arm: Arm, client, cfg: dict, ledger, shape: dict, boxes: dict) -> dict:
    """The counted estimate of one arm."""
    items = banking_set() if arm.set == "own" else benchmark_set(arm.set)
    caches = [ResponseCache(ROOT / f"data/cache/llm_{s}") for s in arm.cached_in]
    price = ledger.price(arm.model)
    factor = ledger.batch_discount if arm.batch else 1.0
    d = cfg["designs"][arm.design]
    firsts, narrow, cached = [], [], 0
    for q, _ in items:
        box = boxes.setdefault(q.db_id, Toolbox(q.db_id))
        run = QuestionRun(q, arm.design, arm.model, arm.evidence, box, ledger.cost, None, cfg)
        pending = run.pending()
        if caches and all(any(c.has(r.cache_key) for c in caches) for _, r in pending):
            cached += 1  # replayed from an earlier stage's responses
            continue
        (conv, req), *_ = pending
        n = count(client, req)
        if d["narrow"]:
            narrow.append(n)
            # the samples' first request: the table list of the full schema, as an upper
            # bound on the narrowed one
            run.stop(conv, "dry_run", "not sent")
            n = count(client, run.pending()[0][1])
        firsts.append(n)
    key = f"{arm.design}/{arm.model}"
    # the samples of designs 4 and 5 run design 3's loop
    loop_key = f"d3/{arm.model}" if d["samples"] > 1 or d["narrow"] else key
    s = shape.get(loop_key) or (
        dict(ASSUMED)
        if d["tools"]
        else {
            "turns": (1, 1),
            "added_per_turn": (0, 0),
            "output_per_turn": ASSUMED["output_per_turn"],
        }
    )
    new_samples = d["samples"] - (1 if arm.design == "d4" and arm.d3_too else 0)
    central = upper = 0.0
    for n in firsts:
        if d["tools"]:
            c = conversation_cost(
                price, n, s["turns"][0], s["added_per_turn"][0], s["output_per_turn"][0]
            )
            u = conversation_cost(
                price, n, s["turns"][1], s["added_per_turn"][1], s["output_per_turn"][1]
            )
            central += c["usd"] * new_samples
            upper += u["uncached_usd"] * new_samples
        else:
            central += single_cost(price, n, s["output_per_turn"][0])
            upper += single_cost(price, n, cfg["models"][arm.model]["max_tokens"])
    for n in narrow:
        central += single_cost(price, n, 300)
        upper += single_cost(price, n, cfg["models"][arm.model]["max_tokens"])
    return {
        "set": arm.set,
        "evidence": arm.evidence,
        "questions": len(items),
        "cached_questions": cached,
        "first_request_tokens": {
            "mean": statistics.mean(firsts) if firsts else None,
            "max": max(firsts, default=None),
            "total": sum(firsts),
        },
        "narrowing_tokens_mean": statistics.mean(narrow) if narrow else None,
        "samples": d["samples"],
        "new_samples": new_samples,
        "projection": {
            "source": f"measured ({loop_key})" if loop_key in shape else "assumed",
            **s,
        },
        "central_usd": central * factor,
        "upper_usd": upper * factor,
        "batch": arm.batch,
    }


def stage_arms(stage: str, cfg: dict) -> list[Arm]:
    stages = cfg["stages"]
    arms, notes = resolve_arms(stages[stage], stages["ablation"])
    for note in notes:
        print(note)
    designs = {(a["model"], a["set"], a["evidence"], a["design"]) for a in arms}
    return [
        Arm(
            a["set"],
            a["model"],
            a["design"],
            a["evidence"],
            a["mode"] == "batch",
            (a["model"], a["set"], a["evidence"], "d3") in designs,
            (stage, *READS.get(stage, ())),
        )
        for a in arms
        if cfg["models"][a["model"]]["backend"] == "anthropic"  # the local model is free
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stage", help="estimate this configured stage's arms (overrides the rest)")
    p.add_argument("--set", choices=("pilot", "ablation", "held_out", "all", "own"))
    p.add_argument("--models", nargs="+")
    p.add_argument("--designs", nargs="+", help="designs, or `winner`")
    p.add_argument("--no-evidence", action="store_true")
    p.add_argument("--batch", action="store_true", help="price at the batch discount")
    p.add_argument("--measured", type=Path, help="records to project turns from")
    p.add_argument("--label", default=None, help="key in the output file (default: set/stage)")
    a = p.parse_args()

    cfg = config()
    if a.stage:
        arms = stage_arms(a.stage, cfg)
        label = a.label or f"stage_{a.stage}"
    else:
        if not (a.set and a.models and a.designs):
            p.error("give --stage, or --set, --models and --designs")
        designs = [winner() if x == WINNER else x for x in a.designs]
        arms = [
            Arm(a.set, m, x, not a.no_evidence, a.batch, "d3" in designs)
            for m in a.models
            for x in designs
        ]
        label = a.label or a.set

    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic()
    ledger = load_ledger("phase4")
    shape = measured(a.measured, cfg) if a.measured else {}
    boxes: dict[str, Toolbox] = {}
    out: dict = {"runs": {}}
    try:
        for arm in arms:
            key = f"{arm.set}/{arm.design}/{arm.model}/" + (
                "evidence" if arm.evidence else "no-evidence"
            )
            r = out["runs"][key] = estimate(arm, client, cfg, ledger, shape, boxes)
            mean = r["first_request_tokens"]["mean"]
            print(
                key,
                f"{r['cached_questions']}/{r['questions']} questions cached;",
                f"first request mean {mean or 0:.0f} tokens;",
                f"central ${r['central_usd']:.2f}, upper ${r['upper_usd']:.2f}",
            )
    finally:
        for box in boxes.values():
            box.close()
    out["central_usd"] = sum(r["central_usd"] for r in out["runs"].values())
    out["upper_usd"] = sum(r["upper_usd"] for r in out["runs"].values())
    doc = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    doc[label] = out
    write_json(OUT, doc)
    total = f"total central ${out['central_usd']:.2f}, upper ${out['upper_usd']:.2f}"
    print(f"{total}; wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
