"""Disjoint question sets drawn in proportion to each stratum (database x difficulty).

The benchmark's 500 questions are split into a pilot set (prompts are tuned on it; never part
of a reported comparison), an ablation set (the designs are chosen on it, and confidence is
calibrated on it) and the held-out rest (the test set: nothing is chosen or fitted on it).

Each set's share of every stratum follows the stratum's share of the benchmark. A set of s
questions gives stratum h, of size n_h out of N, s * n_h / N questions, rounded by the largest
remainder so the set has exactly s (Hamilton's method; equal remainders go to the larger
stratum, then the stratum's name). Within a stratum, questions are shuffled by a generator
seeded with the configured seed and the stratum's name, and taken in that order: first the
pilot's share, then the ablation's; the rest are held out.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

HELD_OUT = "held_out"


def _stratum(q: dict[str, Any]) -> str:
    return f"{q['db_id']}/{q['difficulty']}"


def allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder shares of `total` over strata of the given sizes."""
    n = sum(sizes.values())
    exact = {h: total * m / n for h, m in sizes.items()}
    share = {h: int(x) for h, x in exact.items()}
    left = total - sum(share.values())
    for h in sorted(exact, key=lambda h: (-(exact[h] - share[h]), -sizes[h], h))[:left]:
        share[h] += 1
    return share


def split(
    questions: Sequence[dict[str, Any]], sizes: dict[str, int], seed: int
) -> dict[str, list[Any]]:
    """{set name: sorted question ids} for each named set in `sizes` (in order) and the
    held-out rest."""
    strata: dict[str, list[Any]] = defaultdict(list)
    for q in questions:
        strata[_stratum(q)].append(q["question_id"])
    counts = {h: len(ids) for h, ids in strata.items()}
    shares = {name: allocate(counts, k) for name, k in sizes.items()}
    out: dict[str, list[Any]] = {name: [] for name in [*sizes, HELD_OUT]}
    for h in sorted(strata):
        ids = sorted(strata[h])
        random.Random(f"{seed}/{h}").shuffle(ids)
        start = 0
        for name in sizes:
            take = shares[name][h]
            if start + take > len(ids):
                raise ValueError(f"stratum {h} is too small for the requested sets")
            out[name].extend(ids[start : start + take])
            start += take
        out[HELD_OUT].extend(ids[start:])
    return {name: sorted(ids) for name, ids in out.items()}


def stratum_counts(
    questions: Sequence[dict[str, Any]], sets: dict[str, list[Any]]
) -> dict[str, dict[str, int]]:
    by_id = {q["question_id"]: _stratum(q) for q in questions}
    return {
        name: dict(sorted(Counter(by_id[i] for i in ids).items())) for name, ids in sets.items()
    }


def sets_sha256(sets: dict[str, list[Any]]) -> str:
    """Identifies the split: the sets' names and their sorted ids."""
    text = json.dumps({k: sorted(v) for k, v in sorted(sets.items())}, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
