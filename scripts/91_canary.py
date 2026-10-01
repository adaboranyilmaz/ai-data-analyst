"""The canary: re-send a fixed set of recorded requests to the models and compare the answers.

  --dry-run   count every request's tokens with the free token-counting endpoint, price the run
              (central and upper, at the direct price) and write
              results/metrics/phase9_dry_run.json. Nothing is paid for.
  --run       send the requests (no response cache) under the phase's ledger cap and `--max-usd`,
              compare each answer with the recorded one and write the run to --out (default
              results/canary/<utc>.json). `--score` also checks, on the loaded benchmark database,
              that the new SQL returns the rows the recorded SQL returned.
  --merge F   append a run file (for example the one a manual workflow produced) to the history,
              results/metrics/canary_history.json.

Usage:
    uv run python scripts/91_canary.py --dry-run
    uv run python scripts/91_canary.py --run --max-usd 0.75 [--score] [--label first]
    uv run python scripts/91_canary.py --merge results/canary/<utc>.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.bird import write_json  # noqa: E402
from src.llm.cache import ResponseCache  # noqa: E402
from src.llm.ledger import load_ledger  # noqa: E402
from src.tracking import canary, gate, otlp, registry  # noqa: E402

DRY_RUN = ROOT / "results/metrics/phase9_dry_run.json"
# Output tokens per call: (central, upper, ceiling). Central: the evaluated runs' mean output for
# the model (sonnet 470-553, opus 477); upper: above the largest output any evaluated call produced
# (1,049 to 1,139 tokens); ceiling: the model's configured max_tokens, the most one call can make.
OUTPUT_TOKENS = {"claude-sonnet-5": (520, 1150, 2048), "claude-opus-5-5": (480, 1150, 8192)}


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def count_tokens(client, request) -> int:
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


def build(entry):
    run, box = gate._run(entry, ROOT)
    try:
        ((_, request),) = run.pending()
        return request
    finally:
        box.close()


def dry_run() -> None:
    import anthropic

    load_dotenv(ROOT / ".env")
    cfg = canary.config()
    ledger = load_ledger("phase8")  # prices only; no call is made
    client = anthropic.Anthropic()
    rows, seen = [], set()
    for entry in canary.items(cfg):
        n = count_tokens(client, build(entry))
        p = ledger.price(entry["model"])
        db = entry["question"]["db_id"]
        first = (entry["model"], db) not in seen  # the first call on a schema writes the cache
        seen.add((entry["model"], db))
        out_c, out_u, out_max = OUTPUT_TOKENS[entry["model"]]
        prefix = max(n - 150, 0)  # the schema and prompt: cached; the question and hint are not
        central = (
            prefix * (p.cache_write_5m if first else p.cache_read)
            + (n - prefix) * p.input
            + out_c * p.output
        ) / 1e6
        upper = (n * p.cache_write_5m + out_u * p.output) / 1e6
        ceiling = (n * p.cache_write_1h + out_max * p.output) / 1e6
        rows.append(
            {
                "id": entry["id"],
                "prompt_tokens": n,
                "central_usd": round(central, 5),
                "upper_usd": round(upper, 5),
                "ceiling_usd": round(ceiling, 5),
            }
        )
    out = {
        "note": "counted with the token-counting endpoint (free), priced at the direct list "
        "price; central: the mean evaluated output, the first call on each schema writing the "
        "cache and the rest reading it; upper: every call writes the cache (five minutes) and "
        "produces more than any evaluated call did; ceiling: every call writes the cache for an "
        "hour and produces max_tokens, which no call has come near",
        "calls": len(rows),
        "prompt_tokens": {
            "mean": round(statistics.mean(r["prompt_tokens"] for r in rows), 1),
            "max": max(r["prompt_tokens"] for r in rows),
        },
        "central_usd": round(sum(r["central_usd"] for r in rows), 4),
        "upper_usd": round(sum(r["upper_usd"] for r in rows), 4),
        "ceiling_usd": round(sum(r["ceiling_usd"] for r in rows), 4),
        "per_call": rows,
    }
    write_json(DRY_RUN, out)
    print(
        f"{out['calls']} calls: central ${out['central_usd']:.3f}, upper ${out['upper_usd']:.3f}, "
        f"ceiling ${out['ceiling_usd']:.3f}"
    )
    print(f"wrote {DRY_RUN.relative_to(ROOT)}")


def scorer():
    """Rows of the new SQL against the recorded SQL's rows, on the loaded benchmark database."""
    from src.agent import verify
    from src.tools.toolbox import Toolbox, evaluation_limits

    boxes: dict[str, Toolbox] = {}

    def score(db: str, new_sql: str | None, recorded_sql: str | None) -> bool | None:
        if new_sql is None or recorded_sql is None:
            return new_sql == recorded_sql
        box = boxes.setdefault(db, Toolbox(db))
        limits = evaluation_limits(box.cfg)
        a = verify.fetch(box.sql.guard, box.executor, new_sql, limits)
        b = verify.fetch(box.sql.guard, box.executor, recorded_sql, limits)
        if not (a.ok and b.ok):
            return False
        return {tuple(map(repr, r)) for r in a.rows} == {tuple(map(repr, r)) for r in b.rows}

    return score


def run(max_usd: float, label: str, do_score: bool, out: Path | None) -> None:
    from src.llm.backends import make_backend

    load_dotenv(ROOT / ".env")
    if os.environ.get("ANALYST_REPLAY_ONLY") == "1":
        sys.exit("ANALYST_REPLAY_ONLY is set: a canary run is a live run")
    cfg = canary.config()
    ledger = load_ledger(cfg["phase"])
    started_at = canary.now()
    cache = ResponseCache(ROOT / cfg["cache_root"] / started_at.replace(":", "").replace("-", ""))
    backend = make_backend("anthropic")

    from src.llm.generate import generate_cached

    exporters = otlp.exporters_from_env(dict(os.environ))
    provider = otlp.provider(exporters) if exporters else None
    tracer = provider.get_tracer("src.agent") if provider else None
    score = scorer() if do_score else None
    spent0, results, stopped = ledger.phase_spent(), [], None

    def call(request):
        response, _, _ = generate_cached(backend, request, cache, ledger)
        return response

    for entry in canary.items(cfg):
        if ledger.phase_spent() - spent0 >= max_usd:
            stopped = f"the run's cap of ${max_usd:.2f} was reached"
            break
        results.append(canary.run_item(entry, call, ROOT, tracer, score))
        r = results[-1]
        print(f"  {r['id']}: valid={r['valid_answer']} sql_unchanged={r['sql_unchanged']}")
    if provider:
        provider.shutdown()
    state = registry.read_state()
    record = {
        "utc": started_at,
        "label": label,
        "champion": registry.champion_name(state),
        "champion_sha256": state["versions"][registry.champion_name(state)]["config_sha256"],
        "canary_config_sha256": registry.file_sha256(ROOT / canary.CONFIG),
        "spent_usd": round(ledger.phase_spent() - spent0, 6),
        "cap_usd": max_usd,
        "stopped": stopped,
        "scored_on_database": do_score,
        "summary": canary.summarize(results, cfg["flags"]),
        "results": results,
    }
    target = out or ROOT / "results/canary" / (
        started_at.replace(":", "").replace("-", "") + ".json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    s = record["summary"]
    print(f"spent ${record['spent_usd']:.4f} of ${max_usd:.2f}; flags: {s['flags'] or 'none'}")
    print(f"wrote {target}")


def merge(path: Path) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    cfg = canary.config()
    history = canary.append(ROOT / cfg["history"], record)
    print(f"history holds {len(history['runs'])} run(s)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--merge", type=Path)
    p.add_argument("--max-usd", type=float, default=0.0)
    p.add_argument("--label", default="")
    p.add_argument("--score", action="store_true")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.dry_run:
        dry_run()
    elif a.run:
        if a.max_usd <= 0:
            sys.exit("--run needs --max-usd")
        run(a.max_usd, a.label, a.score, a.out)
    elif a.merge:
        merge(a.merge)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
