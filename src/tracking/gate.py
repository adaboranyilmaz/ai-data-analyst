"""The evaluation gate: a replay of recorded model calls that fails on any unexpected change.

The bundle (results/regression/bundle.json) holds, for a fixed set of recorded model calls, the
request's cache key and the model's response. The gate rebuilds every request from what is in the
repository now (the prompts, the schema snapshot and dictionary, the model's request settings, the
design) and checks two things, at no cost and with no model, database or key:

1. the request is the same one: its cache key equals the recorded key, so a changed prompt,
   schema description, model setting or design is caught here;
2. the recorded response still parses to the answer that was evaluated: its SQL, answer text,
   confidence, declined flag and the rest equal the evaluation's scored record, so a change in how
   a response is read is caught here.

`build` makes the bundle from the local response caches (not committed); `replay` is what CI runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.llm.types import LLMResponse
from src.tracking import registry

ROOT = Path(__file__).resolve().parent.parent.parent
BUNDLE = "results/regression/bundle.json"
ANSWER_FIELDS = (
    ("final_sql", lambda a: None if a.declined else a.sql),
    ("answer", lambda a: a.answer),
    ("confidence", lambda a: a.confidence or 0.0),
    ("declined", lambda a: a.declined),
    ("decline_reason", lambda a: a.decline_reason),
    ("clarifying_question", lambda a: a.clarifying_question),
    ("assumptions", lambda a: a.assumptions),
    ("premise_correction", lambda a: a.premise_correction),
)
MIN_ENTRIES_PER_MODEL = 10


def _records(root: Path, source_run: str) -> dict[Any, dict]:
    from src.eval.records import read_records

    stage, name = source_run.split("/", 1)
    return {
        r["question_id"]: r for r in read_records(root / "results/runs" / stage / f"{name}.jsonl")
    }


def _run(entry: dict[str, Any], root: Path, tracer=None):
    """The question's run, built as the evaluation built it, but without a database: the first
    request of a single-shot design reads only the committed schema snapshot and dictionary."""
    from src.agent.run import Question, QuestionRun, config
    from src.serving.larger import AutoToolRun, agent_config_with
    from src.tools.toolbox import Toolbox

    q = Question(**entry["question"])
    box = Toolbox(q.db_id)
    if entry["model"] == "claude-sonnet-5":
        return QuestionRun(
            q,
            entry["design"],
            entry["model"],
            entry["evidence"],
            box,
            lambda *a: 0.0,
            tracer,
            config(),
        ), box
    esc = yaml.safe_load((root / "configs/confidence.yaml").read_text(encoding="utf-8"))[
        "escalation"
    ]
    if entry["model"] != esc["model"]:
        raise ValueError(f"no request settings known for {entry['model']}")
    cfg = agent_config_with(entry["model"], esc["settings"])
    run = AutoToolRun(
        q,
        entry["design"],
        entry["model"],
        entry["evidence"],
        box,
        lambda *a: 0.0,
        tracer,
        cfg,
        None,
    )
    return run, box


def replay_entry(entry: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    run, box = _run(entry, root)
    try:
        ((conv, request),) = run.pending()
        out: dict[str, Any] = {
            "id": entry["id"],
            "model": entry["model"],
            "request_key_ok": request.cache_key == entry["request_key"],
            "differences": [],
        }
        if not out["request_key_ok"]:
            return out  # the response answers another request: nothing more to compare
        run.feed(conv, request.cache_key, LLMResponse(**entry["response"]))
        expected = _records(root, entry["source_run"])[entry["question"]["question_id"]]
        answer = conv.answer
        for name, get in ANSWER_FIELDS:
            if get(answer) != expected[name]:
                out["differences"].append(name)
        return out
    finally:
        box.close()


def replay(root: Path = ROOT) -> dict[str, Any]:
    bundle = json.loads((root / BUNDLE).read_text(encoding="utf-8"))
    results = [replay_entry(e, root) for e in bundle["entries"]]
    ok = [r for r in results if r["request_key_ok"] and not r["differences"]]
    return {
        "entries": len(results),
        "unchanged": len(ok),
        "changed_request": [r["id"] for r in results if not r["request_key_ok"]],
        "changed_answer": {r["id"]: r["differences"] for r in results if r["differences"]},
        "ok": len(ok) == len(results),
    }


def coverage_problems(root: Path = ROOT) -> list[str]:
    """The registered configurations' models must be covered by enough recorded calls."""
    bundle = json.loads((root / BUNDLE).read_text(encoding="utf-8"))
    have: dict[str, int] = {}
    for e in bundle["entries"]:
        have[e["model"]] = have.get(e["model"], 0) + 1
    state = registry.read_state(root)
    out = []
    for name, v in state["versions"].items():
        cfg = v["config"]
        for model in {cfg["model"], (cfg["router"] or {}).get("model")} - {None}:
            if have.get(model, 0) < MIN_ENTRIES_PER_MODEL:
                out.append(
                    f"{name}: only {have.get(model, 0)} recorded calls of {model} in the "
                    f"regression bundle (at least {MIN_ENTRIES_PER_MODEL} needed)"
                )
    return out


def build(root: Path, cache_dirs: list[Path]) -> dict[str, Any]:
    """The bundle, from the curated runs the demo serves and the response caches on this machine."""
    from src.llm.cache import ResponseCache

    caches = [ResponseCache(d) for d in cache_dirs]
    entries: list[dict[str, Any]] = []
    for path in sorted((root / "results/demo/runs").glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["kind"] not in ("benchmark", "banking"):
            continue  # a guarded comparison has no single recorded call to replay
        source = run["source_run"]
        qid = (
            int(run["id"].split("-")[1])
            if run["kind"] == "benchmark"
            else run["id"][len("bank-") :]
        )
        base = {
            "question": {
                "question_id": qid,
                "source": "bird" if run["kind"] == "benchmark" else "own",
                "db_id": run["db_id"],
                "question": run["question"],
                "evidence": run["hint"],
            },
            "design": run["design"],
            "evidence": "no-evidence" not in source,
        }
        variants = [("claude-sonnet-5", source)]
        if run["kind"] == "benchmark":
            variants.append(("claude-opus-5-5", "router/held_out-d1-claude-opus-5-5-evidence"))
        for model, src in variants:
            entry = {"id": f"{run['id']}/{model}", "model": model, "source_run": src, **base}
            run_obj, box = _run(entry, root)
            try:
                ((conv, request),) = run_obj.pending()
            finally:
                box.close()
            hit = next((c.get(request.cache_key) for c in caches if c.has(request.cache_key)), None)
            if hit is None:
                raise LookupError(f"no recorded response for {entry['id']}")
            entry["request_key"] = request.cache_key
            entry["response"] = {k: getattr(hit, k) for k in LLMResponse.__dataclass_fields__}
            entries.append(entry)
    return {
        "note": "recorded model calls of the curated demo questions: each request's cache key and "
        "the response the model gave; scripts/89_eval_gate.py rebuilds the requests and checks "
        "them (src/tracking/gate.py)",
        "entries": entries,
    }


def promotion_problems(root: Path = ROOT) -> list[str]:
    """The promotion that put the current champion in place must follow from the committed
    results under the rule now in force: made under this rule, about these versions of the two
    configurations, and the decision recomputed from the held-out answers equal to the logged one.
    (Earlier promotions are history; only the latest one is checked.)"""
    state = registry.read_state(root)
    passed = [p for p in registry.read_promotions(root) if p["decision"]["promote"]]
    if not passed:
        return []
    p = passed[-1]
    label = f"the promotion of {p['challenger']} over {p['champion']}"
    rule_path = root / "configs/promotion.yaml"
    if p["rule_sha256"] != registry.file_sha256(rule_path):
        return [f"{label} was decided under another version of the promotion rule"]
    versions = state["versions"]
    for role in ("champion", "challenger"):
        v = versions.get(p[role])
        if v is None or v["config_sha256"] != p[f"{role}_sha256"]:
            return [f"{label} was decided on another version of {p[role]}"]
    if registry.champion_name(state) != p["challenger"]:
        return [f"{label} is the latest, but {registry.champion_name(state)} is the champion"]

    from src.eval import promotion
    from src.llm.ledger import load_ledger

    rule = yaml.safe_load(rule_path.read_text(encoding="utf-8"))
    cost = load_ledger("phase5").cost
    champ, _ = promotion.system_records(versions[p["champion"]]["config"], root, cost)
    chall, _ = promotion.system_records(versions[p["challenger"]]["config"], root, cost)
    comparison = promotion.compare_systems(chall, champ)
    decision = promotion.decide(rule, comparison, promotion.summary(chall))
    out = []
    if decision["promote"] != p["decision"]["promote"]:
        out.append(f"{label}: recomputed from the results, the decision is not the logged one")
    for key in ("execution_accuracy", "aurc"):
        logged = p["challenger_minus_champion"][key]["estimate"]
        if abs(comparison[key]["estimate"] - logged) > 1e-9:
            out.append(f"{label}: the {key} difference no longer matches the logged one")
    return out
