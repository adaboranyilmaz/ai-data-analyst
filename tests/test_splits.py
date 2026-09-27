"""Stratified, disjoint question sets."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from src.eval.config import ROOT, config
from src.eval.splits import HELD_OUT, allocate, sets_sha256, split, stratum_counts


def questions() -> list[dict]:
    """500 questions in strata of uneven size, as in the benchmark."""
    sizes = {("db1", "simple"): 3, ("db1", "moderate"): 97, ("db2", "simple"): 150,
             ("db2", "challenging"): 102, ("db3", "moderate"): 148}  # fmt: skip
    qs, i = [], 0
    for (db, difficulty), m in sizes.items():
        for _ in range(m):
            qs.append({"question_id": i, "db_id": db, "difficulty": difficulty})
            i += 1
    return qs


SIZES = {"pilot": 30, "ablation": 150}


def test_sets_are_disjoint_complete_and_of_the_asked_size():
    s = split(questions(), SIZES, seed=1)
    assert {k: len(v) for k, v in s.items()} == {"pilot": 30, "ablation": 150, HELD_OUT: 320}
    ids = [i for v in s.values() for i in v]
    assert len(ids) == len(set(ids)) == 500


def test_each_stratum_is_represented_in_proportion():
    qs = questions()
    s = split(qs, SIZES, seed=1)
    counts = stratum_counts(qs, s)
    total = Counter(f"{q['db_id']}/{q['difficulty']}" for q in qs)
    for name, k in SIZES.items():
        for h, m in total.items():
            assert abs(counts[name].get(h, 0) - k * m / 500) < 1


def test_the_seed_fixes_the_sets():
    qs = questions()
    assert split(qs, SIZES, seed=1) == split(list(reversed(qs)), SIZES, seed=1)
    assert split(qs, SIZES, seed=1) != split(qs, SIZES, seed=2)
    assert sets_sha256(split(qs, SIZES, seed=1)) == sets_sha256(split(qs, SIZES, seed=1))


def test_largest_remainder_allocation():
    assert allocate({"x": 3, "y": 97}, 30) == {"x": 1, "y": 29}  # 0.9 and 29.1
    assert sum(allocate({"a": 1, "b": 1, "c": 1}, 2).values()) == 2
    assert allocate({"a": 5, "b": 5}, 5) == {"a": 3, "b": 2}  # equal: by name


def test_a_stratum_too_small_is_refused():
    qs = [{"question_id": 0, "db_id": "x", "difficulty": "simple"}]
    with pytest.raises(ValueError):
        split(qs, {"a": 1, "b": 1}, seed=0)


def test_the_committed_splits_match_their_hash_and_configuration():
    path = ROOT / config()["splits_file"]
    if not path.exists():
        pytest.skip("results/metrics/splits.json is not written yet")
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = config()["splits"]
    assert data["seed"] == cfg["seed"]
    assert {k: len(v) for k, v in data["sets"].items()} == {
        "pilot": cfg["pilot"],
        "ablation": cfg["ablation"],
        HELD_OUT: data["questions"] - cfg["pilot"] - cfg["ablation"],
    }
    assert sets_sha256(data["sets"]) == data["sets_sha256"]


@pytest.mark.bird
def test_the_committed_splits_are_reproduced(bird_ready):
    from src.data import bird

    data = json.loads((ROOT / config()["splits_file"]).read_text(encoding="utf-8"))
    cfg = config()["splits"]
    fresh = split(bird.questions(), {"pilot": cfg["pilot"], "ablation": cfg["ablation"]},
                  cfg["seed"])  # fmt: skip
    assert fresh == data["sets"]
