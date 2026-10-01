"""The curated replay set: which recorded runs the public demo serves.

The rule is in configs/serving.yaml and was fixed before any run was looked at. Each slot (for
example "a benchmark answer that was wrong and fell under the decline threshold") is filled by a
seeded random draw among the runs that fit it, so no run is chosen for what it says. Wrong
answers and withheld ones are in the set by design. A slot with fewer runs than it asks for takes
what exists, and the shortfall is reported.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.eval.summary import own_success


@dataclass(frozen=True)
class Pick:
    slot: str
    key: str  # the record's question id, as a string
    record: dict[str, Any]


def draw(seed: int, slot: str, candidates: list[Any], k: int, key: Callable[[Any], str]) -> list:
    """`k` of the candidates (all if fewer), the same ones for the same seed and slot, whatever
    the order the candidates arrive in."""
    ordered = sorted(candidates, key=key)
    rng = random.Random(f"{seed}:{slot}")
    return rng.sample(ordered, min(k, len(ordered)))


def held_out_slots(
    records: list[dict[str, Any]],
    held_out_ids: set[int],
    calibrate: Callable[[float], float],
    threshold: float,
    slots: dict[str, int],
    seed: int,
) -> tuple[list[Pick], dict[str, int]]:
    """Benchmark answers on the held-out split, by whether they were right and whether their
    calibrated confidence reached the decline threshold."""
    pool: dict[str, list[dict]] = {name: [] for name in slots}
    for r in records:
        if r["question_id"] not in held_out_ids or r["declined"]:
            continue
        above = calibrate(r["confidence"]) >= threshold
        name = f"{'correct' if r['correct'] else 'wrong'}_{'above' if above else 'below'}_threshold"
        if name in pool:
            pool[name].append(r)
    picks, short = [], {}
    for name, k in slots.items():
        got = draw(seed, f"held_out:{name}", pool[name], k, lambda r: str(r["question_id"]))
        picks += [Pick(name, str(r["question_id"]), r) for r in got]
        if len(got) < k:
            short[name] = k - len(got)
    return picks, short


def banking_slots(
    records: list[dict[str, Any]], categories: list[str], wrong_extra: int, seed: int
) -> tuple[list[Pick], dict[str, int]]:
    """One question per category, then some more among those scored wrong."""
    picks: list[Pick] = []
    short: dict[str, int] = {}
    for c in categories:
        got = draw(
            seed,
            f"banking:{c}",
            [r for r in records if r["category"] == c],
            1,
            lambda r: str(r["question_id"]),
        )
        picks += [Pick(f"category_{c}", str(r["question_id"]), r) for r in got]
        if not got:
            short[f"category_{c}"] = 1
    taken = {p.key for p in picks}
    wrong = [
        r
        for r in records
        if r["category"] in categories
        and own_success(r) == 0
        and str(r["question_id"]) not in taken
    ]
    got = draw(seed, "banking:wrong", wrong, wrong_extra, lambda r: str(r["question_id"]))
    picks += [Pick("wrong_extra", str(r["question_id"]), r) for r in got]
    if len(got) < wrong_extra:
        short["wrong_extra"] = wrong_extra - len(got)
    return picks, short


def guardrail_slots(
    records: list[dict[str, Any]],
    review: list[dict[str, Any]],
    successes: int,
    failures: int,
    seed: int,
) -> tuple[list[Pick], dict[str, int]]:
    """Guarded answers to the banking set's comparative and causal questions, split by the
    author's close reading. A question the guardrail did not analyze has nothing to show."""
    verdict = {r["id"]: bool(r["reviewed_success"]) for r in review}
    shown = [r for r in records if r.get("guarded") and r["id"].split(":")[-1] in verdict]
    picks, short = [], {}
    for name, want, k in (
        ("reviewed_success", True, successes),
        ("reviewed_failure", False, failures),
    ):
        pool = [r for r in shown if verdict[r["id"].split(":")[-1]] is want]
        got = draw(seed, f"guardrail:{name}", pool, k, lambda r: r["id"])
        picks += [Pick(name, r["id"].split(":")[-1], r) for r in got]
        if len(got) < k:
            short[name] = k - len(got)
    return picks, short
