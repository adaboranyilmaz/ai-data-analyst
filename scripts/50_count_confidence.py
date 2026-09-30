"""The counted dry run of the critic and escalation runs: what they will cost, before any is paid.

Every run configured in configs/confidence.yaml is built exactly as it will be sent: the critic's
review of each answer (src/agent/critic.py) and the winning design's first request on the
escalation model (src/agent/escalation.py). Each request is counted with the token-counting
endpoint (free); a request already in the stage's response cache is counted as free (it will be
replayed), and an answer the critic does not review costs nothing. The output is not known until
the model replies, so it is assumed, or measured from earlier runs' records (`--measured`, e.g.
the probes on the pilot questions): the mean for the central estimate; for the upper, the larger of
the largest measured and the assumed upper.

Costs use the ledger's prices (configs/budget.yaml), at the batch discount for batched runs.
The central estimate assumes each request's prompt is read from the prompt cache in the share
measured on the batched design 1 runs (60-71% of prompt tokens; 60% is used) and written
otherwise; the upper estimate assumes no cache hit at all (every prompt token written at the
5-minute cache price). An escalation reply without the tool call costs a second request, which
neither estimate includes (the probes measure how often it happens).

Writes results/metrics/phase5_dry_run.json under a label.

Usage:
    uv run python scripts/50_count_confidence.py --label probes --sets pilot
    uv run python scripts/50_count_confidence.py --label main --sets ablation held_out \
        --measured results/runs/critic results/runs/escalation
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.agent.confidence import answers_path, confidence_config  # noqa: E402
from src.agent.critic import CriticRun, items_for, load_prompt  # noqa: E402
from src.agent.critic import make_spec as critic_spec  # noqa: E402
from src.agent.escalation import AutoToolRun, agent_config_with, router_spec  # noqa: E402
from src.agent.evaluate import benchmark_set  # noqa: E402
from src.agent.run import config  # noqa: E402
from src.agent.stages import WINNER, make_spec, winner  # noqa: E402
from src.data.bird import write_json  # noqa: E402
from src.eval.records import read_records  # noqa: E402
from src.llm.cache import ResponseCache  # noqa: E402
from src.llm.ledger import load_ledger  # noqa: E402
from src.tools.toolbox import Toolbox  # noqa: E402

OUT = ROOT / "results/metrics/phase5_dry_run.json"
CACHE_READ_SHARE = 0.60
# Output tokens per request until measured: (central, upper). The critic's verdict is short; the
# escalation model's reply is design 1's answer (Claude Sonnet 5 wrote 472 tokens on average)
# plus its thinking at low effort, not known before the probes.
ASSUMED_OUTPUT = {"critic": (300, 800), "escalation": (1000, 2500)}


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


def output_per_request(dirs: list[Path], model: str, arm: str) -> tuple[float, float]:
    """(central, upper) output tokens per request: the measured mean, and the larger of the
    measured maximum and the assumed upper (a few probes can miss the longest replies)."""
    got = measured_output(dirs, model)
    assumed = ASSUMED_OUTPUT[arm]
    return assumed if got is None else (got[0], max(got[1], assumed[1]))


def measured_output(dirs: list[Path], model: str) -> tuple[float, float] | None:
    """Mean and largest output tokens per model call in earlier records of this model."""
    per_call = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                calls = r.get("steps") or (1 if r.get("reviewed") else 0)
                if r["model"] == model and calls:
                    per_call.append(r["tokens"]["output"] / calls)
    return (statistics.mean(per_call), max(per_call)) if per_call else None


def price(prompts: list[int], out: tuple[float, float], p, factor: float) -> tuple[float, float]:
    central = upper = 0.0
    for n in prompts:
        central += (
            n * (1 - CACHE_READ_SHARE) * p.cache_write_5m
            + n * CACHE_READ_SHARE * p.cache_read
            + out[0] * p.output
        )
        upper += n * p.cache_write_5m + out[1] * p.output
    return central * factor / 1e6, upper * factor / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--sets", nargs="*", help="only the runs on these question sets")
    ap.add_argument("--measured", nargs="*", type=Path, default=[])
    ap.add_argument("--router", action="store_true", help="count the router's run only")
    a = ap.parse_args()

    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic()
    conf = confidence_config()
    ledger = load_ledger(conf["phase"])
    agent_cfg = config()
    boxes: dict[str, Toolbox] = {}
    runs: dict[str, dict] = {}

    def box(db: str) -> Toolbox:
        return boxes.setdefault(db, Toolbox(db))

    try:
        crit = conf["critic"]
        system, _ = load_prompt(ROOT / crit["prompt"])
        path, answer_run = answers_path(conf)
        answers = read_records(path)
        out = output_per_request(a.measured, crit["model"], "critic")
        for r in [] if a.router else crit["runs"]:
            if a.sets and r["set"] not in a.sets:
                continue
            items = items_for(r["set"], answers, r.get("limit"))
            spec = critic_spec(
                r["set"], r.get("limit"), crit["model"], items, answer_run, conf["phase"], "batch"
            )
            cache = ResponseCache(spec.cache_dir)
            prompts, cached, skipped = [], 0, 0
            for q, ans in items:
                run = CriticRun(
                    q, ans, answer_run, box(q.db_id), crit, system, agent_cfg, ledger.cost
                )
                pending = run.pending()
                if not pending:
                    skipped += 1
                elif cache.has(pending[0][1].cache_key):
                    cached += 1
                else:
                    prompts.append(count(client, pending[0][1]))
            factor = ledger.batch_discount if crit["mode"] == "batch" else 1.0
            central, upper = price(prompts, out, ledger.price(crit["model"]), factor)
            runs[spec.name] = {
                "questions": len(items),
                "not_reviewed": skipped,
                "cached": cached,
                "counted": len(prompts),
                "prompt_tokens": {
                    "mean": statistics.mean(prompts) if prompts else None,
                    "max": max(prompts, default=None),
                    "total": sum(prompts),
                },
                "output_per_request": {"central": out[0], "upper": out[1]},
                "central_usd": central,
                "upper_usd": upper,
            }

        esc = conf["escalation"]
        cfg = agent_config_with(esc["model"], esc["settings"])
        design = winner() if esc["design"] == WINNER else esc["design"]
        out = output_per_request(a.measured, esc["model"], "escalation")
        planned = []
        if a.router:
            spec = router_spec(conf, design)
            planned.append((spec, spec.items))
        for r in [] if a.router else esc["runs"]:
            if a.sets and r["set"] not in a.sets:
                continue
            spec = make_spec(
                "escalation",
                r["set"],
                design,
                esc["model"],
                esc["evidence"],
                esc["mode"],
                r.get("limit"),
                phase=conf["phase"],
            )
            planned.append((spec, benchmark_set(r["set"])[: r.get("limit") or None]))
        for spec, items in planned:
            cache = ResponseCache(spec.cache_dir)
            prompts, cached = [], 0
            for q, _gold in items:
                run = AutoToolRun(
                    q, design, esc["model"], esc["evidence"], box(q.db_id), ledger.cost, None, cfg
                )
                ((_, req),) = run.pending()
                if cache.has(req.cache_key):
                    cached += 1
                else:
                    prompts.append(count(client, req))
            factor = ledger.batch_discount if esc["mode"] == "batch" else 1.0
            central, upper = price(prompts, out, ledger.price(esc["model"]), factor)
            runs[spec.name] = {
                "questions": len(items),
                "cached": cached,
                "counted": len(prompts),
                "prompt_tokens": {
                    "mean": statistics.mean(prompts) if prompts else None,
                    "max": max(prompts, default=None),
                    "total": sum(prompts),
                },
                "output_per_request": {"central": out[0], "upper": out[1]},
                "central_usd": central,
                "upper_usd": upper,
            }
    finally:
        for b in boxes.values():
            b.close()

    result = {
        "assumptions": {
            "cache_read_share_central": CACHE_READ_SHARE,
            "upper": "no cache hit: every prompt token written at the 5-minute cache price",
            "output": "measured from " + ", ".join(map(str, a.measured))
            if a.measured
            else "assumed (before any probe)",
            "not_included": "an escalation reply without the tool call costs a second request",
        },
        "runs": runs,
        "central_usd": sum(r["central_usd"] for r in runs.values()),
        "upper_usd": sum(r["upper_usd"] for r in runs.values()),
        "phase_spent_usd": ledger.phase_spent(),
    }
    doc = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    doc[a.label] = result
    write_json(OUT, doc)
    for name, r in runs.items():
        print(
            f"{name}: {r['counted']} counted, {r['cached']} cached; "
            f"central ${r['central_usd']:.2f}, upper ${r['upper_usd']:.2f}"
        )
    print(
        f"total central ${result['central_usd']:.2f}, upper ${result['upper_usd']:.2f}; "
        f"wrote {OUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
