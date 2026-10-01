"""Count the statistical guardrail's model calls before any is paid for (the free token-counting
endpoint), and price them.

Every request of each kind is built as it will be sent and counted: the classifier on every
question it reads, the plan call on every question it may plan (the whole banking set as the
upper bound, since which of its questions are flagged is not known before the classifier runs),
and the answer call on the planted copies' results, written from the reference plans' results
(the analyst's plans do not exist yet; the inputs have the same shape). Outputs are assumed, with
a central and an upper value per kind. Writes results/metrics/phase7_dry_run.json.

Usage:
    uv run python scripts/72_count_guardrail.py --label probes|main
"""

from __future__ import annotations

import argparse
import datetime as dt
import statistics
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.bird import write_json  # noqa: E402
from src.llm.ledger import load_ledger  # noqa: E402
from src.stats import guardrail as gr  # noqa: E402
from src.stats import planted as pl  # noqa: E402
from src.stats.answer import answer_text  # noqa: E402
from src.stats.calls import Item, config, requests_for  # noqa: E402
from src.tools.toolbox import Toolbox  # noqa: E402

OUT = ROOT / "results/metrics/phase7_dry_run.json"
# output tokens per call: (central, upper). The plan's central is design 1's measured output on
# the banking set's comparative questions (952 on average); its upper is the call's max_tokens.
OUTPUT = {"classify": (60, 200), "plan": (950, 2048), "answer": (250, 700)}


def _strip(obj):
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


def price(
    kind: str,
    prompts: list[int],
    calls: int,
    cfg: dict,
    ledger,
    batch: bool,
    cached_prefix: int = 0,
) -> dict:
    """Central: the mean counted prompt per call, a cached prefix read after the first call;
    upper: the largest counted prompt, no cache hit, the upper output."""
    model = cfg["calls"][kind]["settings"]["model"]
    p = ledger.price(model)
    factor = ledger.batch_discount if batch else 1.0
    out_c, out_u = OUTPUT[kind]
    mean, top = statistics.mean(prompts), max(prompts)
    rest = mean - cached_prefix
    central = (
        (
            cached_prefix * (p.cache_write_5m + (calls - 1) * p.cache_read)
            + calls * (rest * p.input + out_c * p.output)
        )
        * factor
        / 1e6
    )
    upper = calls * (top * p.input + out_u * p.output) * factor / 1e6
    return {
        "model": model,
        "calls": calls,
        "batch": batch,
        "prompt_tokens": {"counted": len(prompts), "mean": round(mean, 1), "max": top},
        "cached_prefix_tokens": cached_prefix,
        "output_tokens": {"central": out_c, "upper": out_u},
        "central_usd": round(central, 4),
        "upper_usd": round(upper, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", required=True, choices=["probes", "main"])
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic()
    cfg = config()
    ledger = load_ledger(cfg["phase"])
    probe = True  # the prompts are not frozen before the probes
    runs: dict[str, dict] = {}

    # --- the classifier
    qs = gr.classify_questions(cfg, probe=a.label == "probes")
    reqs = requests_for(gr.classify_items(qs), "classify", cfg, probe)
    counts = [count(client, r) for r in reqs]
    runs["classify"] = price(
        "classify", counts, len(reqs), cfg, ledger, cfg["calls"]["classify"]["mode"] == "batch"
    )

    # --- the plan call
    if a.label == "probes":
        plan_qs = gr.probe_questions(cfg)
        central_calls = upper_calls = len(plan_qs)
    else:
        own = gr.own_questions()
        plan_qs = own + gr.planted_questions(cfg)
        f = sum(q["category"] == "f" for q in own)
        # central: the comparative questions plus five of the others; upper: every question
        central_calls = f + 5 + len(gr.planted_questions(cfg))
        upper_calls = len(plan_qs)
    with Toolbox(gr.DB) as box:
        reqs = requests_for(gr.plan_items(plan_qs, box), "plan", cfg, probe)
    counts = [count(client, r) for r in reqs]
    # the schema and the instructions are the cached prefix (everything but the question)
    prefix = min(counts) - 40
    direct = cfg["calls"]["plan"]["mode"] != "batch"
    central = price("plan", counts, central_calls, cfg, ledger, not direct, prefix)
    upper = price("plan", counts, upper_calls, cfg, ledger, not direct, prefix)
    runs["plan"] = {**central, "upper_usd": upper["upper_usd"], "upper_calls": upper_calls}

    # --- the answer call: inputs built from the reference plans' results on planted copies
    pcfg = cfg["planted"]
    pl.setup_database(pcfg)
    units = pl.Units.load()
    _, _, kept = pl.run_level1(
        pcfg, units, {}, 1, keep=1, keep_plan="reference", progress=lambda m: None
    )
    questions = {t["id"]: t["question"] for t in pcfg["templates"]}
    items = []
    for (tid, cond, copy), (plan, pulled, result) in kept.items():
        for arm, res in (("guarded", result), ("numbers_only", None)):
            items.append(
                Item(
                    f"{tid}/{cond}/{copy}#{arm}",
                    f"Database: {gr.DB}",
                    answer_text(questions[tid], plan, pulled, res),
                )
            )
    reqs = requests_for(items, "answer", cfg, probe)
    counts = [count(client, r) for r in reqs]
    n_templates = len(pcfg["templates"])
    if a.label == "probes":
        calls = 2 * len(cfg["probes"]["questions"])
        runs["answer"] = price("answer", counts, calls, cfg, ledger, False)
    else:
        level2 = n_templates * 3 * pcfg["copies"]["level2"] * 2
        own_central, own_upper = 2 * central_calls, 2 * upper_calls
        runs["answer_planted"] = price(
            "answer", counts, level2, cfg, ledger, cfg["calls"]["answer"]["planted_mode"] == "batch"
        )
        oc = price("answer", counts, own_central, cfg, ledger, False)
        ou = price("answer", counts, own_upper, cfg, ledger, False)
        runs["answer_own"] = {**oc, "upper_usd": ou["upper_usd"], "upper_calls": own_upper}

    total_c = round(sum(r["central_usd"] for r in runs.values()), 4)
    total_u = round(sum(r["upper_usd"] for r in runs.values()), 4)
    doc = OUT.exists() and __import__("json").loads(OUT.read_text(encoding="utf-8")) or {}
    doc[a.label] = {
        "counted_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "note": "prompt tokens counted with the token-counting endpoint ($0); outputs assumed; "
        "central: mean prompt, cached prefix read after the first call; upper: largest prompt, "
        "no cache hit, upper output, at the price of the run's mode",
        "runs": runs,
        "central_usd": total_c,
        "upper_usd": total_u,
        "upper_usd_if_all_direct": round(
            sum(
                r["upper_usd"] / (ledger.batch_discount if r["batch"] else 1) for r in runs.values()
            ),
            4,
        ),
    }
    write_json(OUT, doc)
    for k, r in runs.items():
        print(
            f"{k:16} calls {r['calls']:4}  prompt mean {r['prompt_tokens']['mean']:8} "
            f"max {r['prompt_tokens']['max']:6}  central ${r['central_usd']:.3f}  "
            f"upper ${r['upper_usd']:.3f}"
        )
    print(f"total central ${total_c:.2f}, upper ${total_u:.2f}")


if __name__ == "__main__":
    main()
