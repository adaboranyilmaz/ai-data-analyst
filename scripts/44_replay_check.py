"""Rebuild every agent run from the response cache alone and compare it with the committed records.

Each stage's runs (configs/agent.yaml `stages`, and the critic and escalation runs of
configs/confidence.yaml) whose records exist under results/runs/ are run again with
ANALYST_REPLAY_ONLY=1: every model response must come from the cache (a missing one
is an error, never a call), and the tools, the checks and the scoring run again on the
database. The rebuilt records go to a temporary directory and are compared with the committed
ones field by field, leaving out what differs on every run by nature: the wall-clock latency,
the time of scoring, and the trace file's path. Costs are compared too: a replay recomputes them
from the cached token counts.

Writes results/metrics/replay_check.json and exits non-zero if any record differs.

Usage:
    uv run python scripts/44_replay_check.py [--stages pilot ablation ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.bird import write_json  # noqa: E402
from src.eval.records import read_records  # noqa: E402

OUT = ROOT / "results/metrics/replay_check.json"
AGENT_STAGES = ("pilot", "ablation", "main", "own")
STAGES = (*AGENT_STAGES, "critic", "escalation", "router")
IGNORED = ("latency_s", "evaluated_at", "trace")
EXAMPLES = 5  # differing records shown per run


def compare(original: list[dict], replayed: list[dict]) -> dict:
    """How two record lists of one run differ, question by question."""
    before = {r["question_id"]: r for r in original}
    after = {r["question_id"]: r for r in replayed}
    differing = []
    for qid in sorted(before.keys() & after.keys(), key=str):
        fields = sorted(
            k
            for k in before[qid].keys() | after[qid].keys()
            if k not in IGNORED and before[qid].get(k) != after[qid].get(k)
        )
        if fields:
            differing.append({"question_id": qid, "fields": fields})
    missing = sorted(before.keys() - after.keys(), key=str)
    extra = sorted(after.keys() - before.keys(), key=str)
    return {
        "records": len(original),
        "identical": len(before.keys() & after.keys()) - len(differing),
        "differing": len(differing),
        "missing_in_replay": missing,
        "extra_in_replay": extra,
        "examples": differing[:EXAMPLES],
        "ok": not differing and not missing and not extra,
    }


def replay_stage(stage: str, scratch: Path) -> dict[str, dict]:
    """Replay a stage's runs that have committed records; the result by run name."""
    from src.agent.evaluate import execute, execute_many
    from src.agent.run import config
    from src.agent.stages import make_spec, resolve_arms

    stages = config()["stages"]
    arms, _ = resolve_arms(stages[stage], stages["ablation"])
    specs = []
    for arm in arms:
        spec = make_spec(
            stage,
            arm["set"],
            arm["design"],
            arm["model"],
            arm["evidence"],
            arm["mode"],
            arm.get("limit"),
        )
        if not spec.records.exists():
            continue
        rel = spec.records.relative_to(ROOT / "results/runs")
        specs.append(
            (
                spec,
                dataclasses.replace(
                    spec,
                    records=scratch / "runs" / rel,
                    traces_dir=scratch / "traces" / rel.with_suffix(""),
                    spans=scratch / "spans" / rel,
                ),
            )
        )
    out = {}
    batched = [(s, r) for s, r in specs if r.mode == "batch"]
    direct = [(s, r) for s, r in specs if r.mode != "batch"]
    if batched:
        rebuilt = execute_many([r for _, r in batched], log=lambda _: None)
        for (s, _), records in zip(batched, rebuilt, strict=True):
            out[s.name] = compare(read_records(s.records), records)
    for s, r in direct:
        out[s.name] = compare(read_records(s.records), execute(r, log=lambda _: None))
    return out


def _scratch(spec, scratch: Path):
    """The same run, writing its records, traces and spans under the scratch directory."""
    rel = spec.records.relative_to(ROOT / "results/runs")
    return dataclasses.replace(
        spec,
        records=scratch / "runs" / rel,
        traces_dir=scratch / "traces" / rel.with_suffix(""),
        spans=scratch / "spans" / rel,
    )


def replay_critic(scratch: Path) -> dict[str, dict]:
    """Replay the critic's runs that have committed records."""
    from src.agent import critic
    from src.agent.confidence import answers_path, confidence_config
    from src.agent.confidence import read_records as read_reviews

    conf = confidence_config()
    crit = conf["critic"]
    path, answer_run = answers_path(conf)
    answers = read_records(path)
    specs = []
    for r in crit["runs"]:
        items = critic.items_for(r["set"], answers, r.get("limit"))
        spec = critic.make_spec(
            r["set"],
            r.get("limit"),
            crit["model"],
            items,
            answer_run,
            conf["phase"],
            crit["mode"],
            conf["answers"]["evidence"],
        )
        if spec.records.exists():
            specs.append((spec, _scratch(spec, scratch)))
    if not specs:
        return {}
    rebuilt = critic.execute([r for _, r in specs], crit, log=lambda _: None)
    return {
        s.name: compare(read_reviews(s.records), records)
        for (s, _), records in zip(specs, rebuilt, strict=True)
    }


def replay_escalation(scratch: Path) -> dict[str, dict]:
    """Replay the escalation runs that have committed records."""
    from src.agent import escalation
    from src.agent.confidence import confidence_config
    from src.agent.stages import WINNER, make_spec, winner

    conf = confidence_config()
    esc = conf["escalation"]
    cfg = escalation.agent_config_with(esc["model"], esc["settings"])
    design = winner() if esc["design"] == WINNER else esc["design"]
    specs = []
    for r in esc["runs"]:
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
        if spec.records.exists():
            specs.append((spec, _scratch(spec, scratch)))
    if not specs:
        return {}
    rebuilt = escalation.execute([r for _, r in specs], cfg, log=lambda _: None)
    return {
        s.name: compare(read_records(s.records), records)
        for (s, _), records in zip(specs, rebuilt, strict=True)
    }


def replay_router(scratch: Path) -> dict[str, dict]:
    """Replay the router's run of the escalation model, if it has committed records."""
    from src.agent import escalation
    from src.agent.confidence import confidence_config, winning_design

    conf = confidence_config()
    esc = conf["escalation"]
    spec = escalation.router_spec(conf, winning_design())
    if not spec.records.exists():
        return {}
    cfg = escalation.agent_config_with(esc["model"], esc["settings"])
    (records,) = escalation.execute([_scratch(spec, scratch)], cfg, log=lambda _: None)
    return {spec.name: compare(read_records(spec.records), records)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    a = p.parse_args()
    os.environ["ANALYST_REPLAY_ONLY"] = "1"  # a missing response is an error, never a call

    runs: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for stage in a.stages:
            if not any((ROOT / "results/runs" / stage).glob("*.jsonl")):
                print(f"{stage}: no records, skipped", flush=True)
                continue
            if stage == "critic":
                got = replay_critic(Path(tmp))
            elif stage == "escalation":
                got = replay_escalation(Path(tmp))
            elif stage == "router":
                got = replay_router(Path(tmp))
            else:
                got = replay_stage(stage, Path(tmp))
            runs.update(got)
            print(f"{stage}: {len(got)} runs replayed", flush=True)
    ok = all(r["ok"] for r in runs.values())
    write_json(
        OUT,
        {
            "note": "every run rebuilt from cached responses (ANALYST_REPLAY_ONLY=1); records "
            f"compared field by field, except {', '.join(IGNORED)}",
            "stages": a.stages,
            "all_identical": ok,
            "runs": runs,
        },
    )
    for name, r in runs.items():
        print(f"  {name}: {r['identical']}/{r['records']} identical", flush=True)
    print(f"wrote {OUT.relative_to(ROOT)}")
    if not ok:
        sys.exit("some replayed records differ from the committed ones")


if __name__ == "__main__":
    main()
