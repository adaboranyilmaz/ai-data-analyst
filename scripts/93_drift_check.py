"""How the drift monitor behaves on question sets it should and should not alarm on.

The monitor (src/serving/drift.py) compares the last `window` calibrated confidences, and the
share of answers withheld or declined, with the held-out table the confidence meter rests on. This
runs the real monitor class over windows drawn from sets of recorded answers and reports the
population stability index (PSI) it reads, against the thresholds in configs/serving.yaml:

- the held-out questions themselves, the reference's own population: how often does a window of
  200 drawn from them cross a threshold by chance (the false-alarm rate);
- subsets of the held-out questions that really are different (the challenging ones, the simple
  ones): a shift in the mix of questions;
- another domain: the hand-written banking set (Claude Sonnet 5 only; the larger model was not
  run on it);
- two synthetic shifts, labelled as such: every stated confidence lowered (a changed prompt or
  model that makes the answers less sure) or raised.

Windows are drawn at random with replacement (1,000 per scenario, fixed seed), so a set smaller
than the window is a distribution to sample, not a window to replay. Replayed traffic against a
running service is checked separately (scripts/94_alert_check.py).

Writes results/metrics/drift_check.json.

Usage:
    uv run python scripts/93_drift_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.bird import write_json  # noqa: E402
from src.eval import promotion  # noqa: E402
from src.eval.records import read_records  # noqa: E402
from src.serving import champion as champion_mod  # noqa: E402
from src.serving.drift import DriftMonitor  # noqa: E402
from src.serving.meter import Meter, config  # noqa: E402
from src.tracking import registry  # noqa: E402

OUT = ROOT / "results/metrics/drift_check.json"
DRAWS = 1000
SEED = 20261001
SYNTHETIC = {"lowered": -0.25, "raised": 0.25}  # added to the stated confidence, within [0, 1]


def read(split: str, name: str) -> list[dict]:
    return read_records(ROOT / "results/runs" / split / f"{name}.jsonl")


def windows(values: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(values, size=(DRAWS, size), replace=True)


def scenario(
    kind: str, values: np.ndarray, meter: Meter, serving: dict, rng: np.random.Generator
) -> dict:
    cfg = serving["drift"]
    psis, below = [], []
    for window in windows(values, cfg["window"], rng):
        m = DriftMonitor.from_meter(meter, **cfg)
        for v in window:
            m.observe(float(v))
        snap = m.snapshot()
        psis.append(snap["psi"])
        below.append(snap["below_psi"])
    psis, below = np.array(psis), np.array(below)
    ref = DriftMonitor.from_meter(meter, **cfg)
    return {
        "kind": kind,
        "questions": int(len(values)),
        "psi": {
            "mean": round(float(psis.mean()), 4),
            "p5": round(float(np.percentile(psis, 5)), 4),
            "p95": round(float(np.percentile(psis, 95)), 4),
        },
        "share_of_windows_above_warn": round(float((psis > cfg["warn"]).mean()), 4),
        "share_of_windows_above_alert": round(float((psis > cfg["alert"]).mean()), 4),
        "withheld_share_psi_mean": round(float(below.mean()), 4),
        "share_below_threshold": round(float((values < meter.threshold).mean()), 4),
        "reference_share_below_threshold": round(ref.reference_below / ref.reference_total, 4),
    }


def system_scenarios(name: str, meter: Meter, serving: dict, own: bool) -> dict:
    resolved = registry.read_state()["versions"][name]["config"]
    held, _ = promotion.system_records(resolved, ROOT, split="held_out")
    rng = np.random.default_rng(SEED)
    conf = np.array([r["confidence"] for r in held])
    out = {
        "held_out (the reference's own population)": scenario(
            "reference", conf, meter, serving, rng
        )
    }
    for difficulty in ("simple", "moderate", "challenging"):
        sub = np.array([r["confidence"] for r in held if r["difficulty"] == difficulty])
        out[f"held_out, {difficulty} questions only"] = scenario(
            "subset of the held-out questions", sub, meter, serving, rng
        )
    if own:  # the banking set, answered by Claude Sonnet 5 alone, calibrated by the same curve
        recs = read("own", "own-d1-claude-sonnet-5-no-evidence")
        vals = np.array([0.0 if r["declined"] else meter.calibrate(r["confidence"]) for r in recs])
        out["banking questions (another domain)"] = scenario(
            "another domain", vals, meter, serving, rng
        )
    if name == "d1-sonnet-5":  # the stated confidences are the baseline's own
        ids = {r["question_id"] for r in held}
        stated = [
            r for r in read("main", "all-d1-claude-sonnet-5-evidence") if r["question_id"] in ids
        ]
        for label, delta in SYNTHETIC.items():
            vals = np.array(
                [
                    0.0
                    if r["declined"]
                    else meter.calibrate(min(max(r["confidence"] + delta, 0.0), 1.0))
                    for r in stated
                ]
            )
            out[f"synthetic: every stated confidence {label} by 0.25"] = scenario(
                "synthetic", vals, meter, serving, rng
            )
    return out


def main() -> None:
    serving = config()
    state = registry.read_state()
    systems = {}
    for name in sorted(state["versions"]):
        v = state["versions"][name]
        evaluation = json.loads((ROOT / v["evaluation"]).read_text(encoding="utf-8"))
        meter = Meter.from_registry(v["config"], evaluation, serving)
        systems[name] = {
            "decline_threshold": round(meter.threshold, 4),
            "scenarios": system_scenarios(name, meter, serving, own=v["config"]["router"] is None),
        }
    out = {
        "note": "the real drift monitor over windows drawn at random with replacement from "
        "recorded answers (1,000 windows per scenario, fixed seed); PSI against the held-out "
        "table of each system's own meter; one run of recorded answers, so a rate is the share of "
        "windows",
        "settings": {**serving["drift"], "draws_per_scenario": DRAWS, "seed": SEED},
        "champion": champion_mod.load().name,
        "systems": systems,
    }
    write_json(OUT, out)
    for name, s in systems.items():
        print(name)
        for label, r in s["scenarios"].items():
            warn, alert = r["share_of_windows_above_warn"], r["share_of_windows_above_alert"]
            print(f"  {label:<52} PSI {r['psi']['mean']:.3f}  >warn {warn:.0%}  >alert {alert:.0%}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
